import argparse
import features
import math
import model as M
import numpy
import nnue_bin_dataset
import struct
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import reduce
import operator
import os
import matplotlib.pyplot as plt
import datetime

NNUE_BINARY_EXTENSIONS = (".nnue", ".bin")

def _get_runtime_feature_set(feature_set):
  runtime_name = feature_set.name[:-1] if feature_set.name.endswith("^") else feature_set.name
  return features.get_feature_set_from_name(runtime_name)

def _get_engine_feature_name(feature_set):
  runtime_feature_set = _get_runtime_feature_set(feature_set)
  if runtime_feature_set.name.startswith("HalfKPE9"):
    return "HalfKPE9(Friend)"
  if runtime_feature_set.name.startswith("HalfKP"):
    return "HalfKP(Friend)"
  return runtime_feature_set.name

def ascii_hist(name, x, bins=6):
  N,X = numpy.histogram(x, bins=bins)
  total = 1.0*len(x)
  width = 50
  nmax = N.max()

  print(name)
  for (xi, n) in zip(X,N):
    bar = '#'*int(n*1.0*width/nmax)
    xi = '{0: <8.4g}'.format(xi).ljust(10)
    print('{0}| {1}'.format(xi,bar))

# hardcoded for now
VERSION = 0x7AF32F16

class NNUEWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model, output_directory_path, layer_stacks=1):
    self.output_directory_path = output_directory_path
    if not self.output_directory_path:
      self.output_directory_path = '.'
    os.makedirs(self.output_directory_path, exist_ok=True)
    self.figure_index = 0
    self.buf = bytearray()
    self.layer_stacks = layer_stacks

    runtime_feature_set = _get_runtime_feature_set(model.feature_set)
    fc_hash = self.fc_hash(model, layer_stacks)
    l1_size = model.l1_stack[0].in_features // 2
    l2_size = model.l1_stack[0].out_features
    l3_size = model.l2.out_features
    num_features = runtime_feature_set.num_real_features
    feature_name = _get_engine_feature_name(model.feature_set)

    # YaneuraOu file format:
    # 1. VERSION (4 bytes)
    # 2. fc_hash (4 bytes)
    # 3. description.size (4 bytes)
    # 4. description string (N bytes)
    # 5. Feature transformer (input.weight, input.bias)
    # 6. fc_hash for verification (4 bytes)
    # 7. fc_0, fc_1, fc_2

    # Write header
    self.write_header(model, fc_hash, runtime_feature_set, layer_stacks)

    # Write feature transformer
    self.write_feature_transformer(model.input)

    # Write fc_hash for verification
    self.int32(fc_hash)

    if layer_stacks > 1:
      self.write_fc_layer_layerstack(model)
    else:
      self.write_fc_layers_standard(model)

  def write_feature_transformer(self, layer):
    """Write the feature transformer (input layer) weights and biases."""
    kWeightScaleBits = 6
    kActivationScale = 127.0
    kBiasScale = (1 << kWeightScaleBits) * kActivationScale  # = 8128
    kWeightScale = kBiasScale / kActivationScale  # = 64.0

    # Write input bias (int16)
    bias = (layer.bias.data / kBiasScale).to(torch.float16).numpy().round().astype(numpy.int16)
    self.buf.extend(bias.tobytes())

    # Write input weight (int16)
    non_padded_shape = layer.weight.shape
    weight = (layer.weight.data / kWeightScale).to(torch.float16).numpy().round().astype(numpy.int16)
    self.buf.extend(weight.tobytes())

  @staticmethod
  def fc_hash(model, layer_stacks):
    """Compute network hash matching YaneuraOu's GetHashValue().

    For LayerStacks=1 (HALFKP_768X2_16_64):
      HiddenLayer1 = ClippedReLU<AffineTransformSparseInput<InputLayer, l2_size>>
      HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, l3_size>>
      OutputLayer = AffineTransform<HiddenLayer2, 1>

      Hash chain: InputSlice -> HiddenLayer1(ReLU) -> HiddenLayer2(ReLU) -> OutputLayer(AffineTransform)
      Key: HiddenLayer1/2 are just ClippedReLU wrappers, so only ReLU hash (0x538D24C7) is added.
    """
    # InputSlice hash
    prev_hash = 0xEC42E90D
    prev_hash ^= model.l1_stack[0].in_features

    if layer_stacks > 1:
      # LayerStack hash (0xB58B6A8D base for LayerStack)
      layer_hash = 0xB58B6A8D
      layer_hash += layer_stacks
      layer_hash ^= prev_hash >> 1
      layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
      prev_hash = layer_hash

      # A1: ClippedReLU after stacked layer
      a1_hash = 0x538D24C7 + prev_hash
      a1_hash &= 0xFFFFFFFF
      prev_hash = a1_hash

      # L2: AffineTransform (no ReLU) - this is fc_1
      l2_hash = 0xCC03DAE4
      l2_hash += model.l2.out_features
      l2_hash ^= prev_hash >> 1
      l2_hash ^= (prev_hash << 31) & 0xFFFFFFFF
      l2_hash &= 0xFFFFFFFF
      prev_hash = l2_hash

      # A2: ClippedReLU after L2
      a2_hash = 0x538D24C7 + prev_hash
      a2_hash &= 0xFFFFFFFF
      prev_hash = a2_hash

      # L3: AffineTransform output (no ReLU) - this is fc_2
      l3_hash = 0xCC03DAE4
      l3_hash += model.output.out_features
      l3_hash ^= prev_hash >> 1
      l3_hash ^= (prev_hash << 31) & 0xFFFFFFFF
      l3_hash &= 0xFFFFFFFF
      return l3_hash
    else:
      # HALFKP_768X2_16_64 (LayerStacks=1) format:
      # HiddenLayer1 = ClippedReLU<AffineTransformSparseInput>  -- just ReLU hash
      # HiddenLayer2 = ClippedReLU<AffineTransform>             -- just ReLU hash
      # OutputLayer = AffineTransform                            -- AffineTransform hash

      # HiddenLayer1 = ClippedReLU (0x538D24C7 + prev)
      prev_hash = (prev_hash + 0x538D24C7) & 0xFFFFFFFF
      # HiddenLayer2 = ClippedReLU (0x538D24C7 + prev)
      prev_hash = (prev_hash + 0x538D24C7) & 0xFFFFFFFF
      # OutputLayer = AffineTransform
      fc_hash = 0xCC03DAE4 + model.output.out_features
      fc_hash ^= prev_hash >> 1
      fc_hash ^= (prev_hash << 31) & 0xFFFFFFFF
      fc_hash &= 0xFFFFFFFF
      return fc_hash

  def write_header(self, model, fc_hash, runtime_feature_set, layer_stacks):
    self.int32(VERSION) # version
    self.int32(fc_hash) # Network hash

    l1_size = model.l1_stack[0].in_features // 2
    l2_size = model.l1_stack[0].out_features
    l3_size = model.l2.out_features
    num_features = runtime_feature_set.num_features

    feature_name = _get_engine_feature_name(model.feature_set)

    if layer_stacks > 1:
      # halfkp_256x2-32-32 format: matches actual YaneuraOu nn.bin description
      desc = f"Features={feature_name}[{num_features}->{l1_size}x2],LayerStack[{layer_stacks}],".encode('ascii')
      desc += f"Network=AffineTransform[1<-{l3_size}](".encode('ascii')
      desc += f"ClippedReLU[{l3_size}](".encode('ascii')
      desc += f"AffineTransform[{l3_size}<-{l2_size}](".encode('ascii')
      desc += f"ClippedReLU[{l2_size}](".encode('ascii')
      desc += f"AffineTransform[{l2_size}<-{l1_size*2}](".encode('ascii')
      desc += f"InputSlice[{l1_size*2}(0:{l1_size*2})])".encode('ascii')
      desc += b")" * 4
    else:
      desc = f"Features={feature_name}[{num_features}->{l1_size}x2],".encode('ascii')
      desc += f"Network=AffineTransform[1<-{l3_size}](".encode('ascii')
      desc += f"ClippedReLU[{l3_size}](".encode('ascii')
      desc += f"AffineTransform[{l3_size}<-{l2_size}](".encode('ascii')
      desc += f"ClippedReLU[{l2_size}](".encode('ascii')
      desc += f"AffineTransform[{l2_size}<-{l1_size*2}](".encode('ascii')
      desc += f"InputSlice[{l1_size*2}(0:{l1_size*2})]".encode('ascii')
      desc += b")" * 4

    self.int32(len(desc)) # Network definition
    self.buf.extend(desc)

  def coalesce_ft_weights(self, model, layer):
    weight = layer.weight.data
    indices = model.feature_set.get_virtual_to_real_features_gather_indices()
    weight_coalesced = weight.new_zeros((weight.shape[0], model.feature_set.num_real_features))
    for i_real, is_virtual in enumerate(indices):
      weight_coalesced[:, i_real] = sum(weight[:, i_virtual] for i_virtual in is_virtual)

    return weight_coalesced

  def save_histogram(self, file_name, data, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(xlabel)
    bins = min(256, data.numel())
    frequency, value = data.to(torch.float).histogram(bins=bins)
    value += (value[1] - value[0]) * 0.5
    width = value[1] - value[0]
    value = value[:-1]
    ax.bar(value, frequency, width=width)
    ax.set_title(title)
    fig.savefig(os.path.join(self.output_directory_path, file_name))
    print(f'Saved a histogram to {file_name}')

    mean = data.to(torch.float).mean().item()
    std = data.to(torch.float).std().item()
    print(f'{mean=} {std=}')

  def round_away_from_zero(self, x):
      return torch.where(x >= 0, torch.ceil(x), torch.floor(x))

  def stochastic_round_cpp(self, x):
      integer_part = torch.trunc(x)
      n = torch.abs(x - integer_part)

      return torch.where(n > 0.5,
                         torch.where(x >= 0, integer_part + 1, integer_part - 1),
                         torch.where(x >= 0, integer_part, integer_part))

  def tensor(self, dtype, shape):
    if dtype == numpy.int8:
      return torch.from_numpy(numpy.zeros(shape, dtype=numpy.int8))
    elif dtype == numpy.int16:
      return torch.from_numpy(numpy.zeros(shape, dtype=numpy.int16))
    elif dtype == numpy.int32:
      return torch.from_numpy(numpy.zeros(shape, dtype=numpy.int32))
    else:
      raise Exception('Unsupported dtype')

  def write_fc_layers_standard(self, model):
    """Write fc_0, fc_1, fc_2 for LayerStacks=1 (HALFKP_768X2_16_64).

    YaneuraOu format: bias(int32) then weight(int8) for each layer.
    - fc_0: AffineTransformSparseInput - bias[int32, l2_size], weight[int8, l2_size x l1.in_features]
    - fc_1: AffineTransform - bias[int32, l3_size], weight[int8, l3_size x l2_size]
    - fc_2: AffineTransform output - bias[int32, 1], weight[int8, 1 x l3_size]
    """
    kWeightScaleBits = 6
    kActivationScale = 127.0
    kBiasScale = (1 << kWeightScaleBits) * kActivationScale  # = 8128
    kWeightScale = kBiasScale / kActivationScale  # = 64.0

    # fc_0: AffineTransformSparseInput - bias int32[l2_size], weight int8[l2_size, l1.in_features]
    bias = (model.l1_stack[0].bias.data / kBiasScale).to(torch.float32).numpy().round().astype(numpy.int32)
    self.buf.extend(bias.tobytes())
    weight = (model.l1_stack[0].weight.data / kWeightScale).to(torch.float32).numpy().round().astype(numpy.int8)
    self.buf.extend(weight.tobytes())

    # fc_1: AffineTransform - bias int32[l3_size], weight int8[l3_size, l2_size]
    kBiasScale_fc1 = (1 << kWeightScaleBits) * kActivationScale
    kWeightScale_fc1 = kBiasScale_fc1 / kActivationScale
    non_padded_shape = model.l2.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    bias_fc1 = (model.l2.bias.data / kBiasScale_fc1).to(torch.float32).numpy().round().astype(numpy.int32)
    self.buf.extend(bias_fc1.tobytes())
    # Write padded weight to match YaneuraOu format (padded to 32 for SIMD)
    weight_fc1_full = numpy.zeros(padded_shape, dtype=numpy.int8)
    weight_fc1_full[:non_padded_shape[0], :non_padded_shape[1]] = (
        (model.l2.weight.data / kWeightScale_fc1).to(torch.float32).numpy().round().astype(numpy.int8)
    )
    self.buf.extend(weight_fc1_full.tobytes())

    # fc_2: output - bias int32[1], weight int8[1, l3_size]
    kBiasScale_out = 9600.0
    kWeightScale_out = kBiasScale_out / kActivationScale

    bias_out = (model.output.bias.data / kBiasScale_out).to(torch.float32).numpy().round().astype(numpy.int32)
    self.buf.extend(bias_out.tobytes())
    weight_out = (model.output.weight.data / kWeightScale_out).to(torch.float32).numpy().round().astype(numpy.int8)
    self.buf.extend(weight_out.tobytes())

  def write_fc_layer_layerstack(self, model):
    """Write fc_0 in LayerStack format (int8 weights + int32 bias) + fc_1, fc_2 (int8 weights)

    For the halfkp_256x2-32-32 architecture with LayerStacks=8:
    - fc_0[bucket]: int32 bias [l2_size] + int8 weights [l2_size][kInputDimensions]
    - kInputDimensions = 2 * l1_size (from InputSlice)
    """
    kWeightScaleBits = 6
    kActivationScale = 127.0

    kBiasScale = (1 << kWeightScaleBits) * kActivationScale  # = 8128
    kWeightScale = kBiasScale / kActivationScale  # = 64.0

    # Write fc_0 for each stack (int8 weights and int32 bias)
    for s in range(self.layer_stacks):
      # fc_0 bias: int32, shape (l2_size,)
      bias_int32 = (model.l1_stack[s].bias.data / kBiasScale).to(torch.float32).numpy().round().astype(numpy.int32)
      self.buf.extend(bias_int32.tobytes())

      # fc_0 weight: int8, shape (l2_size, kInputDimensions)
      weight_int8 = (model.l1_stack[s].weight.data / kWeightScale).to(torch.float32).numpy().round().astype(numpy.int8)
      self.buf.extend(weight_int8.tobytes())

    # Write fc_1 (l2): int8 weight, int32 bias
    kBiasScale_fc1 = (1 << kWeightScaleBits) * kActivationScale
    kWeightScale_fc1 = kBiasScale_fc1 / kActivationScale
    non_padded_shape = model.l2.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    bias_fc1 = (model.l2.bias.data / kBiasScale_fc1).to(torch.float32).numpy().round().astype(numpy.int32)
    self.buf.extend(bias_fc1.tobytes())

    # Write padded weight to match YaneuraOu format (padded to 32 for SIMD)
    weight_fc1_full = numpy.zeros(padded_shape, dtype=numpy.int8)
    weight_fc1_full[:non_padded_shape[0], :non_padded_shape[1]] = (
        (model.l2.weight.data / kWeightScale_fc1).to(torch.float32).numpy().round().astype(numpy.int8)
    )
    self.buf.extend(weight_fc1_full.tobytes())

    # Write fc_2 (output): int8 weight, int32 bias
    kBiasScale_out = 9600.0
    kWeightScale_out = kBiasScale_out / kActivationScale

    bias_out = (model.output.bias.data / kBiasScale_out).to(torch.float32).numpy().round().astype(numpy.int32)
    self.buf.extend(bias_out.tobytes())

    weight_out = (model.output.weight.data / kWeightScale_out).to(torch.float32).numpy().round().astype(numpy.int8)
    self.buf.extend(weight_out.tobytes())

  def write_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers

    # FC inputs are padded to 32 elements for simd.
    non_padded_shape = layer.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    layer.bias.data = self.tensor(numpy.int32, layer.bias.shape).divide(kBiasScale)
    layer.weight.data = self.tensor(numpy.int8, padded_shape).divide(kWeightScale)

    # Strip padding.
    layer.weight.data = layer.weight.data[:non_padded_shape[0], :non_padded_shape[1]]

  def read_fc_layer(self, layer, is_output=False):
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers

    non_padded_shape = layer.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    layer.bias.data = self.tensor(numpy.int32, layer.bias.shape).divide(kBiasScale)
    layer.weight.data = self.tensor(numpy.int8, padded_shape).divide(kWeightScale)

    # Strip padding.
    layer.weight.data = layer.weight.data[:non_padded_shape[0], :non_padded_shape[1]]

  def read_int32(self, expected=None):
    v = struct.unpack("<I", self.f.read(4))[0]
    if expected is not None and v != expected:
      raise Exception("Expected: %x, got %x" % (expected, v))
    return v

  def int8(self, v):
    self.buf.extend(struct.pack("<b", int(v)))

  def int16(self, v):
    self.buf.extend(struct.pack("<h", int(v)))

  def int32(self, v):
    self.buf.extend(struct.pack("<I", v & 0xFFFFFFFF))

  def int64(self, v):
    self.buf.extend(struct.pack("<q", v))

def _is_nnue_binary_path(path):
  return path.endswith(NNUE_BINARY_EXTENSIONS)

def _infer_feature_set_name(num_features, requested_features=None):
  if requested_features is not None:
    requested_feature_set = features.get_feature_set_from_name(requested_features)
    if requested_feature_set.num_features == num_features:
      return requested_features

  matches = []
  for feature_name in features.get_available_feature_blocks_names():
    feature_set = features.get_feature_set_from_name(feature_name)
    if feature_set.num_features == num_features:
      matches.append(feature_name)

  if len(matches) == 1:
    return matches[0]

  if len(matches) > 1:
    raise Exception(
      f"Multiple feature sets match input.weight width {num_features}: {matches}. "
      "Specify --features explicitly.")

  raise Exception(
    f"Could not infer feature set from input.weight width {num_features}. "
    "Specify --features explicitly.")

def _load_model_from_state_dict(state_dict, requested_features=None):
  required_keys = [
    "input.weight",
    "input.bias",
    "l2.weight",
    "l2.bias",
    "output.weight",
    "output.bias",
  ]
  missing_keys = [key for key in required_keys if key not in state_dict]
  if missing_keys:
    raise Exception(f"Checkpoint is missing required keys: {missing_keys}")

  feature_name = _infer_feature_set_name(
    state_dict["input.weight"].shape[1], requested_features=requested_features)
  l1_size = state_dict["input.weight"].shape[0]
  l2_size = state_dict["l2.weight"].shape[0]
  l3_size = state_dict["l2.weight"].shape[0]

  # Detect layer_stacks from state dict keys
  layer_stacks = 1
  l1_key_count = 0
  for key in state_dict:
    if key.startswith("l1_stack."):
      l1_key_count += 1
      # Extract stack index from key like "l1_stack.0.weight"
      try:
        idx = int(key.split(".")[1].split(".")[0])
        layer_stacks = max(layer_stacks, idx + 1)
      except (ValueError, IndexError):
        pass

  # Also check for old l1.weight/bias keys (backward compatibility with single stack)
  has_old_l1 = "l1.weight" in state_dict and "l1.bias" in state_dict
  if has_old_l1:
    layer_stacks = 1

  model = M.NNUE(
    feature_name,
    l1_size=self.l1_size,
    l2_size=self.l2_size,
    l3_size=self.l3_size,
    layer_stacks=layer_stacks
  )

  # Handle both old format (l1.weight/bias) and new format (l1_stack.N.weight/bias)
  new_state_dict = {}
  for key, value in state_dict.items():
    if key == "l1.weight" or key == "l1.bias":
      # Convert old l1 keys to new l1_stack.0 keys
      new_key = "l1_stack.0." + key[4:]  # "l1.weight" -> "l1_stack.0.weight"
      new_state_dict[new_key] = value
    else:
      new_state_dict[key] = value

  model.load_state_dict(new_state_dict)
  return model

def load_model(path, requested_features=None):
  try:
    obj = torch.load(path, map_location="cpu", weights_only=False)
  except TypeError:
    obj = torch.load(path, map_location="cpu")

  if isinstance(obj, M.NNUE):
    return obj

  if isinstance(obj, dict):
    if "state_dict" in obj:
      return _load_model_from_state_dict(
        obj["state_dict"], requested_features=requested_features)

    if "model" in obj and isinstance(obj["model"], M.NNUE):
      return obj["model"]

  if hasattr(obj, "state_dict"):
    return _load_model_from_state_dict(
      obj.state_dict(), requested_features=requested_features)

  raise Exception(f"Unsupported model format in {path}")


class NNUEReader():
  """Read .nnue binary files into a PyTorch model."""

  def __init__(self, f, feature_set, l1_size=1024, l2_size=8, l3_size=96):
    self.f = f
    self.feature_set = feature_set
    self.l1_size = l1_size
    self.l2_size = l2_size
    self.l3_size = l3_size
    self.layer_stacks = 1
    self.read_header()

  def read_header(self):
    """Read version, hash, and description to determine architecture."""
    import re
    self.version = self.read_int32()
    self.fc_hash = self.read_int32()
    desc_size = self.read_int32()
    self.desc = self.f.read(desc_size)
    
    # Parse architecture dimensions from description string
    desc_str = self.desc.decode('ascii')
    
    # Extract l1_size from Features section: Features=...[num_features->l1_size x2]
    feat_match = re.search(r'Features=.*?\[(\d+)->(\d+)x2\]', desc_str)
    if feat_match:
      self.l1_size = int(feat_match.group(2))
    
    # Extract l3_size from Network=AffineTransform[1<-l3_size]
    net_match = re.search(r'Network=AffineTransform\[1<-(\d+)\]', desc_str)
    if net_match:
      self.l3_size = int(net_match.group(1))
    
    # Extract l2_size from first ClippedReLU+AffineTransform pair
    # Format: ClippedReLU[l3_size](AffineTransform[l3_size<-l2_size](
    af_match = re.search(r'ClippedReLU\[(\d+)\]\(AffineTransform\[\d+<-(\d+)\]\(', desc_str)
    if af_match:
      self.l2_size = int(af_match.group(2))
    
    # Auto-detect layer_stacks from description
    # LayerStacks=8 files explicitly include "LayerStack" in the description
    if "LayerStack" in desc_str:
      self.layer_stacks = 8

  def read_feature_transformer(self, model):
    """Read the feature transformer (input layer) weights and biases."""
    kWeightScaleBits = 6
    kActivationScale = 127.0
    kBiasScale = (1 << kWeightScaleBits) * kActivationScale  # = 8128
    kWeightScale = kBiasScale / kActivationScale  # = 64.0

    # Read input bias (int16)
    bias_data = numpy.fromfile(self.f, dtype=numpy.int16, count=model.input.bias.shape[0])
    model.input.bias.data = torch.from_numpy(bias_data.astype(numpy.float32)) * kBiasScale

    # Read input weight (int16)
    num_in_features = model.input.weight.shape[1]
    num_out_features = model.input.weight.shape[0]
    weight_data = numpy.fromfile(self.f, dtype=numpy.int16, count=num_in_features * num_out_features)
    weight_data = weight_data.reshape((num_out_features, num_in_features))
    model.input.weight.data = torch.from_numpy(weight_data.astype(numpy.float32)) * kWeightScale

    # Skip fc_hash verification (YaneuraOu format)
    self.f.read(4)

  def read_fc_layers_standard(self, model):
    """Read fc_0, fc_1, fc_2 for LayerStacks=1 (HALFKP_768X2_16_64)."""
    kWeightScaleBits = 6
    kActivationScale = 127.0
    kBiasScale = (1 << kWeightScaleBits) * kActivationScale  # = 8128
    kWeightScale = kBiasScale / kActivationScale  # = 64.0

    # fc_0: AffineTransformSparseInput - bias int32[l2_size], weight int8[l2_size, l1_size*2]
    bias = numpy.fromfile(self.f, dtype=numpy.int32, count=model.l1_stack[0].bias.shape[0])
    model.l1_stack[0].bias.data = torch.from_numpy(bias.astype(numpy.float32)) * kBiasScale

    weight = numpy.fromfile(self.f, dtype=numpy.int8, count=model.l1_stack[0].weight.numel())
    weight = weight.reshape(model.l1_stack[0].weight.shape)
    model.l1_stack[0].weight.data = torch.from_numpy(weight.astype(numpy.float32)) * kWeightScale

    # fc_1: AffineTransform - bias int32[l3_size], weight int8[l3_size, l2_size]
    kBiasScale_fc1 = (1 << kWeightScaleBits) * kActivationScale
    kWeightScale_fc1 = kBiasScale_fc1 / kActivationScale
    non_padded_shape = model.l2.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    bias_fc1 = numpy.fromfile(self.f, dtype=numpy.int32, count=non_padded_shape[0])
    model.l2.bias.data = torch.from_numpy(bias_fc1.astype(numpy.float32)) * kBiasScale_fc1

    weight_fc1 = numpy.fromfile(self.f, dtype=numpy.int8, count=padded_shape[0] * padded_shape[1])
    weight_fc1 = weight_fc1.reshape(padded_shape)
    # Only keep the non-padded portion for the model
    model.l2.weight.data = torch.from_numpy(weight_fc1[:non_padded_shape[0], :non_padded_shape[1]].astype(numpy.float32)) * kWeightScale_fc1

    # fc_2: output - bias int32[1], weight int8[1, l3_size]
    kBiasScale_out = 9600.0
    kWeightScale_out = kBiasScale_out / kActivationScale

    bias_out = numpy.fromfile(self.f, dtype=numpy.int32, count=1)
    model.output.bias.data = torch.from_numpy(bias_out.astype(numpy.float32)) * kBiasScale_out

    weight_out = numpy.fromfile(self.f, dtype=numpy.int8, count=model.output.weight.numel())
    weight_out = weight_out.reshape(model.output.weight.shape)
    model.output.weight.data = torch.from_numpy(weight_out.astype(numpy.float32)) * kWeightScale_out

  def read_fc_layer_layerstack(self, model):
    """Read fc_0[bucket] for each stack + shared fc_1, fc_2."""
    kWeightScaleBits = 6
    kActivationScale = 127.0
    kBiasScale = (1 << kWeightScaleBits) * kActivationScale  # = 8128
    kWeightScale = kBiasScale / kActivationScale  # = 64.0

    # Read fc_0 for each stack (int8 weights and int32 bias)
    for s in range(self.layer_stacks):
      bias_int32 = numpy.fromfile(self.f, dtype=numpy.int32, count=model.l1_stack[0].bias.shape[0])
      model.l1_stack[s].bias.data = torch.from_numpy(bias_int32.astype(numpy.float32)) * kBiasScale

      weight_int8 = numpy.fromfile(self.f, dtype=numpy.int8, count=model.l1_stack[0].weight.numel())
      weight_int8 = weight_int8.reshape(model.l1_stack[s].weight.shape)
      model.l1_stack[s].weight.data = torch.from_numpy(weight_int8.astype(numpy.float32)) * kWeightScale

    # Read shared fc_1 (l2): int32 bias, int8 weight
    kBiasScale_fc1 = (1 << kWeightScaleBits) * kActivationScale
    kWeightScale_fc1 = kBiasScale_fc1 / kActivationScale
    non_padded_shape = model.l2.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    bias_fc1 = numpy.fromfile(self.f, dtype=numpy.int32, count=non_padded_shape[0])
    model.l2.bias.data = torch.from_numpy(bias_fc1.astype(numpy.float32)) * kBiasScale_fc1

    weight_fc1 = numpy.fromfile(self.f, dtype=numpy.int8, count=padded_shape[0] * padded_shape[1])
    weight_fc1 = weight_fc1.reshape(padded_shape)
    # Only keep the non-padded portion for the model
    model.l2.weight.data = torch.from_numpy(weight_fc1[:non_padded_shape[0], :non_padded_shape[1]].astype(numpy.float32)) * kWeightScale_fc1

    # Read shared fc_2 (output): int32 bias, int8 weight
    kBiasScale_out = 9600.0
    kWeightScale_out = kBiasScale_out / kActivationScale

    bias_out = numpy.fromfile(self.f, dtype=numpy.int32, count=1)
    model.output.bias.data = torch.from_numpy(bias_out.astype(numpy.float32)) * kBiasScale_out

    weight_out = numpy.fromfile(self.f, dtype=numpy.int8, count=model.output.weight.numel())
    weight_out = weight_out.reshape(model.output.weight.shape)
    model.output.weight.data = torch.from_numpy(weight_out.astype(numpy.float32)) * kWeightScale_out

  def read_int32(self):
    v = struct.unpack("<I", self.f.read(4))[0]
    return v

  def load(self, requested_features=None):
    """Read the network and return a loaded NNUE model."""
    feature_name = _infer_feature_set_name(
      self.feature_set.num_features, requested_features=requested_features)
    
    # Use dimensions detected from description by read_header()
    model = M.NNUE(
      feature_name,
      l1_size=self.l1_size,
      l2_size=self.l2_size,
      l3_size=self.l3_size,
      layer_stacks=self.layer_stacks
    )
    
    # Read feature transformer
    self.read_feature_transformer(model)
    
    # Read fully connected layers
    if self.layer_stacks > 1:
      self.read_fc_layer_layerstack(model)
    else:
      self.read_fc_layers_standard(model)
    
    return model



def main():
  parser = argparse.ArgumentParser(description="Converts files between ckpt and nnue format.")
  parser.add_argument("--architecture", type=str, default=None, choices=['HALFKP_768X2_16_64', 'halfkp_768x2-16_64', 'halfkp_256x2-32-32'],
                      help="YaneuraOu architecture to use for compatibility")
  parser.add_argument("source", help="Source file (can be .ckpt, .pt, .nnue or .bin)")
  parser.add_argument("target", help="Target file (can be .pt, .nnue or .bin)")
  features.add_argparse_args(parser)
  parser.add_argument("--l1_size", type=int, default=1024)
  parser.add_argument("--l2_size", type=int, default=8)
  parser.add_argument("--l3_size", type=int, default=96)
  parser.add_argument("--num_buckets", type=int, default=1, help="Number of layer stacks (buckets)")
  args = parser.parse_args()

  feature_set = features.get_feature_set_from_name(args.features)

  # Validate config for YaneuraOu compatibility
  if args.num_buckets > 1:
    # LayerStacks=8 case (halfkp_256x2-32-32 architecture)
    if args.l1_size != 256:
      print(f"WARNING: For LayerStacks=8 (num_buckets={args.num_buckets}), l1_size should be 256 (halfkp_256x2-32-32 architecture).")
    if args.l2_size != 32:
      print(f"WARNING: For LayerStacks=8 (num_buckets={args.num_buckets}), l2_size should be 32 (halfkp_256x2-32-32 architecture).")
    if args.l3_size != 32:
      print(f"WARNING: For LayerStacks=8 (num_buckets={args.num_buckets}), l3_size should be 32 (halfkp_256x2-32-32 architecture).")
  else:
    # LayerStacks=1 case (HALFKP_768X2_16_64 architecture)
    if args.l1_size != 768:
      print(f"WARNING: For LayerStacks=1 (num_buckets=1), l1_size should be 768 (HALFKP_768X2_16_64 architecture).")
    if args.l2_size != 16:
      print(f"WARNING: For LayerStacks=1 (num_buckets=1), l2_size should be 16 (HALFKP_768X2_16_64 architecture).")
    if args.l3_size != 64:
      print(f"WARNING: For LayerStacks=1 (num_buckets=1), l3_size should be 64 (HALFKP_768X2_16_64 architecture).")

  print('Converting %s to %s' % (args.source, args.target))

  if args.source.endswith(".pt") or args.source.endswith(".ckpt"):
    if not _is_nnue_binary_path(args.target):
      raise Exception("Target file must end with .nnue or .bin")
    nnue = load_model(args.source, requested_features=args.features)
    nnue.cpu()
    nnue.eval()
    writer = NNUEWriter(nnue, os.path.dirname(args.target), layer_stacks=args.num_buckets)
    with open(args.target, 'wb') as f:
      f.write(writer.buf)
  elif _is_nnue_binary_path(args.source):
    if not args.target.endswith(".pt"):
      raise Exception("Target file must end with .pt")
    with open(args.source, 'rb') as f:
      reader = NNUEReader(f, feature_set, l1_size=args.l1_size, l2_size=args.l2_size, l3_size=args.l3_size)
      reader.layer_stacks = args.num_buckets
      nnue = reader.load(requested_features=args.features)
    torch.save(nnue, args.target)
  else:
    raise Exception('Invalid filetypes: ' + str(args))

if __name__ == '__main__':
  main()
