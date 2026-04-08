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
import sys

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
YANE_LAYERSTACK_HASH_SEED = 0xB58B6A8D
YANE_HALFKP_FRIEND_HASH = 0x5D69D5B8
YANE_MOE_HASH_SEED = 0x94A3EDE7


def _infer_features_from_input_dim(input_dim):
  for feature_name in features.get_available_feature_blocks_names():
    feature_set = features.get_feature_set_from_name(feature_name)
    if feature_set.num_features == input_dim:
      return feature_name
  raise ValueError(f"Could not infer feature set from input dimension {input_dim}")


def _infer_model_args_from_state_dict(state_dict):
  layer_indices = sorted({
      int(key.split(".")[2])
      for key in state_dict
      if key.startswith("l1.layers.") and key.endswith(".weight")
  })

  if layer_indices:
    first_l1_key = f"l1.layers.{layer_indices[0]}.weight"
    l2_size = int(state_dict[first_l1_key].shape[0])
    num_buckets = len(layer_indices)
  else:
    l2_size = int(state_dict["l1.weight"].shape[0])
    num_buckets = 1

  return {
      "features": _infer_features_from_input_dim(state_dict["input.weight"].shape[1]),
      "l1_size": int(state_dict["input.weight"].shape[0]),
      "l2_size": l2_size,
      "l3_size": int(state_dict["l2.weight"].shape[0]),
      "num_buckets": num_buckets,
  }


def _load_checkpoint_extras(model, checkpoint):
  if hasattr(model, "on_load_checkpoint"):
    model.on_load_checkpoint(checkpoint)


def _affine_hash(prev_hash, out_features):
  hash_value = 0xCC03DAE4
  hash_value += out_features
  hash_value ^= prev_hash >> 1
  hash_value ^= (prev_hash << 31) & 0xFFFFFFFF
  return hash_value & 0xFFFFFFFF


def _clipped_relu_hash(prev_hash):
  return (0x538D24C7 + prev_hash) & 0xFFFFFFFF


def _input_slice_hash(output_dimensions, offset=0):
  return (0xEC42E90D ^ output_dimensions ^ (offset << 10)) & 0xFFFFFFFF


def _layer_stack_hash(prev_hash, num_buckets):
  hash_value = YANE_LAYERSTACK_HASH_SEED
  hash_value += num_buckets
  hash_value ^= prev_hash >> 1
  hash_value ^= (prev_hash << 31) & 0xFFFFFFFF
  return hash_value & 0xFFFFFFFF


def _moe_hash(prev_hash, num_experts, out_features):
  hash_value = YANE_MOE_HASH_SEED
  hash_value += num_experts
  hash_value ^= prev_hash >> 1
  hash_value ^= (prev_hash << 31) & 0xFFFFFFFF
  hash_value = _affine_hash(hash_value & 0xFFFFFFFF, out_features)
  return hash_value & 0xFFFFFFFF


def _yaneuraou_network_hash(model):
  input_dims = model.input.out_features * 2
  if isinstance(model.l1, M.MoELinear):
    hidden1_dims = model.l1.experts[0].out_features
  elif isinstance(model.l1, M.StackedLinear):
    hidden1_dims = model.l1.layers[0].out_features
  else:
    hidden1_dims = model.l1.out_features
  hidden2_dims = model.l2.out_features

  if isinstance(model.l1, M.MoELinear):
    hash_value = _affine_hash(_input_slice_hash(input_dims), model.l1.num_experts)
    hash_value = _moe_hash(hash_value, model.l1.num_experts, hidden1_dims)
  else:
    hash_value = _layer_stack_hash(_input_slice_hash(input_dims), getattr(model, 'num_buckets', 1))
  hash_value = _affine_hash(_clipped_relu_hash(hash_value), hidden2_dims)
  hash_value = _affine_hash(_clipped_relu_hash(hash_value), 1)
  return hash_value & 0xFFFFFFFF


def _yaneuraou_feature_name(feature_set_name):
  if feature_set_name.startswith("HalfKP"):
    return "HalfKP(Friend)"
  return feature_set_name


def _yaneuraou_feature_hash(feature_set):
  if feature_set.name.startswith("HalfKP"):
    return YANE_HALFKP_FRIEND_HASH
  return feature_set.hash


def _build_stockfish_description(model):
  l1_size = model.input.out_features
  l2_size = model.l2.in_features
  l3_size = model.l2.out_features
  num_features = model.feature_set.num_features
  num_buckets = model.num_buckets if hasattr(model, 'num_buckets') else 1

  description = f"Features={model.feature_set.name}[{num_features}->{l1_size}x2],".encode('ascii')
  if isinstance(model.l1, M.MoELinear):
    description += (
        f"Network=AffineTransform[1<-{l3_size}]"
        f"(ClippedReLU[{l3_size}](AffineTransform[{l3_size}<-{l2_size}]"
        f"(ClippedReLU[{l2_size}](MoE[{num_buckets}x{l2_size}<-{l1_size * 2}]"
        f"(AffineTransform[{num_buckets}<-{l1_size * 2}](InputSlice[{l1_size * 2}(0:{l1_size * 2})]))))))"
    ).encode('ascii')
  elif num_buckets > 1:
    description += (
        f"Network=AffineTransform[1<-{l3_size}]"
        f"(ClippedReLU[{l3_size}](AffineTransform[{l3_size}<-{l2_size}]"
        f"(ClippedReLU[{l2_size}](LayerStack[{num_buckets}x{l2_size}<-{l1_size * 2}]"
        f"(InputSlice[{l1_size * 2}(0:{l1_size * 2})])))))"
    ).encode('ascii')
  else:
    description += (
        f"Network=AffineTransform[1<-{l3_size}]"
        f"(ClippedReLU[{l3_size}](AffineTransform[{l3_size}<-{l2_size}]"
        f"(ClippedReLU[{l2_size}](AffineTransform[{l2_size}<-{l1_size * 2}]"
        f"(InputSlice[{l1_size * 2}(0:{l1_size * 2})])))))"
    ).encode('ascii')
  return description


def _build_yaneuraou_description(model):
  l1_size = model.input.out_features
  l2_size = model.l2.in_features
  l3_size = model.l2.out_features
  num_features = model.feature_set.num_features
  num_buckets = model.num_buckets if hasattr(model, 'num_buckets') else 1

  feature_name = _yaneuraou_feature_name(model.feature_set.name)
  description = f"Features={feature_name}[{num_features}->{l1_size}x2],".encode('ascii')
  if isinstance(model.l1, M.MoELinear):
    description += (
        f"Network=AffineTransform[1<-{l3_size}]"
        f"(ClippedReLU[{l3_size}](AffineTransform[{l3_size}<-{l2_size}]"
        f"(ClippedReLU[{l2_size}](MoE[{num_buckets}x{l2_size}<-{l1_size * 2}]"
        f"(AffineTransform[{num_buckets}<-{l1_size * 2}](InputSlice[{l1_size * 2}(0:{l1_size * 2})]))))))"
    ).encode('ascii')
  elif num_buckets > 1:
    description += (
        f"Network=AffineTransform[1<-{l3_size}]"
        f"(ClippedReLU[{l3_size}](AffineTransform[{l3_size}<-{l2_size}]"
        f"(ClippedReLU[{l2_size}](LayerStack[{num_buckets}x{l2_size}<-{l1_size * 2}]"
        f"(InputSlice[{l1_size * 2}(0:{l1_size * 2})])))))"
    ).encode('ascii')
  else:
    description += (
        f"Network=AffineTransform[1<-{l3_size}]"
        f"(ClippedReLU[{l3_size}](AffineTransform[{l3_size}<-{l2_size}]"
        f"(ClippedReLU[{l2_size}](AffineTransform[{l2_size}<-{l1_size * 2}]"
        f"(InputSlice[{l1_size * 2}(0:{l1_size * 2})])))))"
    ).encode('ascii')
  return description

if sys.version_info < (3, 10):
  raise RuntimeError(f"Python 3.10+ is required. Detected: {sys.version}")

class NNUEWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model, output_directory_path, target_engine='stockfish', yane_network_hash=None):
    self.output_directory_path = output_directory_path
    if not self.output_directory_path:
      self.output_directory_path = '.'
    os.makedirs(self.output_directory_path, exist_ok=True)
    self.figure_index = 0
    self.buf = bytearray()
    self.target_engine = target_engine
    self.yane_network_hash = _yaneuraou_network_hash(model) if yane_network_hash is None else yane_network_hash

    fc_hash = self.fc_hash(model)
    self.write_header(model, fc_hash)
    self.int32(self.feature_transformer_hash(model)) # Feature transformer hash
    self.write_feature_transformer(model)
    self.int32(fc_hash) # FC layers hash
    
    if isinstance(model.l1, M.MoELinear):
        self.write_fc_layer(model.l1.router)
        for expert in model.l1.experts:
            self.write_fc_layer(expert)
    elif isinstance(model.l1, M.StackedLinear):
        for i in range(model.num_buckets):
            layer = model.l1.layers[i]
            # Bias
            bias = layer.bias.data
            bias = self.stochastic_round_cpp(bias * 8128.0).to(torch.int32)
            self.buf.extend(bias.flatten().numpy().tobytes())
            
            # Weight
            weight = layer.weight.data
            kWeightScale = 8128.0 / 127.0
            kMaxWeight = 127.0 / kWeightScale
            weight = self.stochastic_round_cpp(weight.clamp(-kMaxWeight, kMaxWeight) * kWeightScale).to(torch.int8)
            
            # Padding to 32
            num_input = weight.shape[1]
            if num_input % 32 != 0:
                num_input += 32 - (num_input % 32)
                new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
                new_w[:, :weight.shape[1]] = weight
                weight = new_w
            self.buf.extend(weight.flatten().numpy().tobytes())
    else:
        self.write_fc_layer(model.l1)
        
    self.write_fc_layer(model.l2)
    self.write_fc_layer(model.output, is_output=True)

  @staticmethod
  def fc_hash(model):
    if isinstance(model.l1, M.MoELinear):
      return _yaneuraou_network_hash(model)

    # InputSlice hash
    prev_hash = 0xEC42E90D
    l1_first = model.l1.layers[0] if isinstance(model.l1, M.StackedLinear) else model.l1
    prev_hash ^= l1_first.in_features # This is l1_size * 2

    # Fully connected layers
    # Note: For StackedLinear, we use the first layer for hash calculation
    # as they all have the same dimensions.
    l1_first = model.l1.layers[0] if isinstance(model.l1, M.StackedLinear) else model.l1
    layers = [l1_first, model.l2, model.output]
    for layer in layers:
      layer_hash = 0xCC03DAE4
      layer_hash += layer.out_features
      layer_hash ^= prev_hash >> 1
      layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
      if layer.out_features != 1:
        # Clipped ReLU hash
        layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
      prev_hash = layer_hash
    return layer_hash

  def feature_transformer_hash(self, model):
    if self.target_engine == 'yaneuraou':
      # YaneuraOu: RawFeatureHash ^ kOutputDimensions (kOutputDimensions = l1_size * 2)
      return _yaneuraou_feature_hash(model.feature_set) ^ (model.input.out_features * 2)
    # Stockfish-style default used by this serializer previously
    return model.feature_set.hash ^ model.input.in_features

  def header_hash(self, model, fc_hash):
    if self.target_engine == 'yaneuraou':
      return self.feature_transformer_hash(model) ^ self.yane_network_hash
    return fc_hash ^ model.feature_set.hash ^ model.input.in_features

  def write_header(self, model, fc_hash):
    self.int32(VERSION) # version
    self.int32(self.header_hash(model, fc_hash)) # hash
    if self.target_engine == 'yaneuraou':
      description = _build_yaneuraou_description(model)
    else:
      description = _build_stockfish_description(model)
    self.int32(len(description)) # Network definition
    self.buf.extend(description)

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
    ax.set_ylabel(ylabel)
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
      # C++: v >= 0 ? ceil(v) : floor(v)
      return torch.where(x >= 0, torch.ceil(x), torch.floor(x))

  def stochastic_round_cpp(self, x):
      """
      Stochastic rounding that matches the C++ implementation:

          auto n = abs(v - int(v));
          if (rand > n)
              v = trunc(v);
          else
              v = round_away_from_zero(v);

      """
      # fractional part (absolute)
      integer_part = torch.trunc(x)
      n = torch.abs(x - integer_part)

      # random uniform [0,1)
      rand = torch.rand_like(x)

      # if (rand > n) → trunc
      # else → round_away_from_zero
      return torch.where(rand > n, integer_part, self.round_away_from_zero(x))

  def write_feature_transformer(self, model):
    # int16 bias = round(x * 127)
    # int16 weight = round(x * 127)
    layer = model.input
    bias = layer.bias.data
    bias = self.stochastic_round_cpp(bias * 127).to(torch.int16)
    ascii_hist('ft bias:', bias.numpy())
    self.save_histogram(f'{self.figure_index:02}_feature_transformer_bias.png', bias, 'bias', 'frequency', 'feature transformer bias')
    self.figure_index += 1
    self.buf.extend(bias.flatten().numpy().tobytes())

    print(datetime.datetime.now())
    weight = self.coalesce_ft_weights(model, layer)
    weight = self.stochastic_round_cpp(weight * 127).to(torch.int16)
    ascii_hist('ft weight:', weight.numpy())
    self.save_histogram(f'{self.figure_index:02}_feature_transformer_weight.png', weight, 'weight', 'frequency', 'feature transformer weight')
    self.figure_index += 1
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    self.buf.extend(weight.transpose(0, 1).flatten().numpy().tobytes())
    print(datetime.datetime.now())
    print()

  def write_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
    kMaxWeight = 127.0 / kWeightScale # roughly 2.0

    # int32 bias = round(x * kBiasScale)
    # int8 weight = round(x * kWeightScale)
    bias = layer.bias.data
    bias = self.stochastic_round_cpp(bias * kBiasScale).to(torch.int32)
    ascii_hist('fc bias:', bias.numpy())
    self.save_histogram(f'{self.figure_index:02}_fully_connected_layer_bias.png', bias, 'bias', 'frequency', 'fully connected layer bias')
    self.figure_index += 1
    self.buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight.data
    clipped = torch.count_nonzero(weight.clamp(-kMaxWeight, kMaxWeight) - weight)
    total_elements = torch.numel(weight)
    clipped_max = torch.max(torch.abs(weight.clamp(-kMaxWeight, kMaxWeight) - weight))
    print(f"layer has {clipped}/{total_elements} clipped weights. Exceeding by {clipped_max} the maximum {kMaxWeight}.")
    weight = self.stochastic_round_cpp(weight.clamp(-kMaxWeight, kMaxWeight) * kWeightScale).to(torch.int8)
    ascii_hist('fc weight:', weight.numpy())
    self.save_histogram(f'{self.figure_index:02}_fully_connected_layer_weight.png', weight, 'weight', 'frequency', 'fully connected layer weight')
    self.figure_index += 1
    # FC inputs are padded to 32 elements for simd.
    num_input = weight.shape[1]
    if num_input % 32 != 0:
      num_input += 32 - (num_input % 32)
      new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
      new_w[:, :weight.shape[1]] = weight
      weight = new_w
    # Stored as [outputs][inputs], so we can flatten
    self.buf.extend(weight.flatten().numpy().tobytes())
    print()

  def int32(self, v):
    self.buf.extend(struct.pack("<I", v))

class NNUEReader():
  def __init__(self, f, feature_set, l1_size=256, l2_size=32, l3_size=32, num_buckets=8, target_engine='stockfish', yane_network_hash=None):
    self.f = f
    self.feature_set = feature_set
    self.model = M.NNUE(feature_set.name, l1_size=l1_size, l2_size=l2_size, l3_size=l3_size, num_buckets=num_buckets)
    self.target_engine = target_engine
    self.yane_network_hash = _yaneuraou_network_hash(self.model) if yane_network_hash is None else yane_network_hash
    fc_hash = NNUEWriter.fc_hash(self.model)

    self.read_header(feature_set, fc_hash)
    self.read_int32(self.feature_transformer_hash()) # Feature transformer hash
    self.read_feature_transformer(self.model.input)
    self.read_int32(fc_hash) # FC layers hash
    
    if isinstance(self.model.l1, M.MoELinear):
        self.read_fc_layer(self.model.l1.router)
        for expert in self.model.l1.experts:
            self.read_fc_layer(expert)
    elif isinstance(self.model.l1, M.StackedLinear):
        for i in range(self.model.num_buckets):
            layer = self.model.l1.layers[i]
            # Bias (int32)
            kBiasScale = 8128.0
            layer.bias.data = self.tensor(numpy.int32, layer.bias.shape).divide(kBiasScale)
            
            # Weight (int8)
            kWeightScale = 8128.0 / 127.0
            non_padded_shape = layer.weight.shape
            padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)
            layer.weight.data = self.tensor(numpy.int8, padded_shape).divide(kWeightScale)
            # Strip padding
            layer.weight.data = layer.weight.data[:non_padded_shape[0], :non_padded_shape[1]]
    else:
        self.read_fc_layer(self.model.l1)
        
    self.read_fc_layer(self.model.l2)
    self.read_fc_layer(self.model.output, is_output=True)

  def feature_transformer_hash(self):
    if self.target_engine == 'yaneuraou':
      return _yaneuraou_feature_hash(self.feature_set) ^ (self.model.input.out_features * 2)
    return self.feature_set.hash ^ self.model.input.in_features

  def expected_header_hash(self, fc_hash):
    if self.target_engine == 'yaneuraou':
      return self.feature_transformer_hash() ^ self.yane_network_hash
    return fc_hash ^ self.feature_set.hash ^ self.model.input.in_features

  def read_header(self, feature_set, fc_hash):
    self.read_int32(VERSION) # version
    self.read_int32(self.expected_header_hash(fc_hash)) # hash
    desc_len = self.read_int32() # Network definition
    description = self.f.read(desc_len)

  def tensor(self, dtype, shape):
    d = numpy.fromfile(self.f, dtype, reduce(operator.mul, shape, 1))
    d = torch.from_numpy(d.astype(numpy.float32))
    d = d.reshape(shape)
    return d

  def read_feature_transformer(self, layer):
    layer.bias.data = self.tensor(numpy.int16, layer.bias.shape).divide(127.0)
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    weights = self.tensor(numpy.int16, layer.weight.shape[::-1])
    layer.weight.data = weights.divide(127.0).transpose(0, 1)

  def read_fc_layer(self, layer, is_output=False):
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

  def read_int32(self, expected=None):
    v = struct.unpack("<I", self.f.read(4))[0]
    if expected is not None and v != expected:
      raise Exception(f"Expected: 0x{expected:x}, got 0x{v:x}")
    return v

def main():
  parser = argparse.ArgumentParser(description="Converts files between ckpt and nnue/bin format.")
  parser.add_argument("source", help="Source file (can be .ckpt, .pt, .nnue or .bin)")
  parser.add_argument("target", help="Target file (can be .pt, .nnue or .bin)")
  features.add_argparse_args(parser)
  parser.set_defaults(features=None)
  parser.add_argument("--l1_size", type=int, default=None)
  parser.add_argument("--l2_size", type=int, default=None)
  parser.add_argument("--l3_size", type=int, default=None)
  parser.add_argument("--num_buckets", type=int, default=None)
  parser.add_argument("--target-engine", choices=["stockfish", "yaneuraou"], default=None)
  parser.add_argument("--yane-network-hash", type=lambda x: int(x, 0), default=None,
                      help="Used only when --target-engine yaneuraou.")
  args = parser.parse_args()

  default_features = "HalfKP"
  default_l1_size = 256
  default_l2_size = 32
  default_l3_size = 32
  default_num_buckets = 8

  def resolve_model_args(hparams=None, inferred=None):
    hparams = hparams or {}
    inferred = inferred or {}
    resolved_features = args.features if args.features is not None else hparams.get("features", inferred.get("features", default_features))
    resolved_l1_size = args.l1_size if args.l1_size is not None else hparams.get("l1_size", inferred.get("l1_size", default_l1_size))
    resolved_l2_size = args.l2_size if args.l2_size is not None else hparams.get("l2_size", inferred.get("l2_size", default_l2_size))
    resolved_l3_size = args.l3_size if args.l3_size is not None else hparams.get("l3_size", inferred.get("l3_size", default_l3_size))
    resolved_num_buckets = args.num_buckets if args.num_buckets is not None else hparams.get("num_buckets", inferred.get("num_buckets", default_num_buckets))
    return resolved_features, resolved_l1_size, resolved_l2_size, resolved_l3_size, resolved_num_buckets

  print(f"Converting {args.source} to {args.target}")

  def is_nnue_binary_path(path):
    return path.endswith(".nnue") or path.endswith(".bin")

  def resolve_target_engine(source_path, target_path):
    if args.target_engine is not None:
      return args.target_engine
    if target_path.endswith(".bin") or source_path.endswith(".bin"):
      return "yaneuraou"
    return "stockfish"

  target_engine = resolve_target_engine(args.source, args.target)

  if args.source.endswith(".pt") or args.source.endswith(".ckpt"):
    if not is_nnue_binary_path(args.target):
      raise Exception("Target file must end with .nnue or .bin")
    if args.source.endswith(".pt"):
      nnue = torch.load(args.source)
    else:
      checkpoint = torch.load(args.source, map_location="cpu")
      hyper_parameters = checkpoint.get("hyper_parameters", {})
      inferred_args = _infer_model_args_from_state_dict(checkpoint["state_dict"])
      resolved_features, resolved_l1_size, resolved_l2_size, resolved_l3_size, resolved_num_buckets = resolve_model_args(hyper_parameters, inferred_args)
      nnue = M.NNUE(
        features=resolved_features,
        l1_size=resolved_l1_size,
        l2_size=resolved_l2_size,
        l3_size=resolved_l3_size,
        num_buckets=resolved_num_buckets,
      )
      nnue.load_state_dict(checkpoint["state_dict"])
      _load_checkpoint_extras(nnue, checkpoint)
    nnue.cpu()
    nnue.eval()
    writer = NNUEWriter(
      nnue,
      os.path.dirname(args.target),
      target_engine=target_engine,
      yane_network_hash=args.yane_network_hash,
    )
    with open(args.target, 'wb') as f:
      f.write(writer.buf)
  elif is_nnue_binary_path(args.source):
    if not args.target.endswith(".pt"):
      raise Exception("Target file must end with .pt")
    resolved_features, resolved_l1_size, resolved_l2_size, resolved_l3_size, resolved_num_buckets = resolve_model_args()
    feature_set = features.get_feature_set_from_name(resolved_features)
    with open(args.source, 'rb') as f:
      reader = NNUEReader(
        f,
        feature_set,
        l1_size=resolved_l1_size,
        l2_size=resolved_l2_size,
        l3_size=resolved_l3_size,
        num_buckets=resolved_num_buckets,
        target_engine=target_engine,
        yane_network_hash=args.yane_network_hash,
      )
    torch.save(reader.model, args.target)
  else:
    raise Exception('Invalid filetypes: ' + str(args))

if __name__ == '__main__':
  main()
