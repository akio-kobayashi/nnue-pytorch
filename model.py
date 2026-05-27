import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
from collections.abc import Callable, Iterator
from typing import Any
from torch.optim import Optimizer
import features as features_module

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(
      self, features: str, lambda_: list[float] | None = None, lr: list[float] | None = None,
      label_smoothing_eps: float = 0.0, num_batches_warmup: int = 10000, newbob_decay: float = 0.5,
      num_epochs_to_adjust_lr: int = 500, score_scaling: float = 361.0, min_newbob_scale: float = 1e-5,
      momentum: float = 0.0, ply_begin_threshold: float = 100.0, ply_end_threshold: float = 120.0,
      l1_size: int = 1024, l2_size: int = 8, l3_size: int = 96,
      layer_stacks: int = 1,
      factorization_rank: int = 0, factorization_weight_decay: float = 1e-4):
    super().__init__()
    if lambda_ is None:
      lambda_ = [1.0]
    if lr is None:
      lr = [1.0]

    self.NNUE_TO_SCORE = 600.0
    self.EPSILON = 1e-12
    self.WEIGHT_SCALE_BITS = 6
    self.ACTIVATION_SCALE = 127.0
    self.FV_SCALE = 16.0

    feature_set = features_module.get_feature_set_from_name(features)
    self.input = nn.Linear(feature_set.num_features, l1_size)
    self.feature_set = feature_set
    self.l2 = nn.Linear(l2_size, l3_size)
    self.output = nn.Linear(l3_size, 1)
    
    # Per-stack l1 layers for LayerStacks > 1
    self.l1_stack = nn.ModuleList([
        nn.Linear(2 * l1_size, l2_size) for _ in range(layer_stacks)
    ])
    self.layer_stacks = layer_stacks
    self.lambda_ = lambda_
    self.lr = lr
    self.label_smoothing_eps = label_smoothing_eps
    self.num_batches_warmup = num_batches_warmup
    self.newbob_scale = 1.0
    self.newbob_decay = newbob_decay
    self.best_loss = 1e10
    self.num_epochs_to_adjust_lr = num_epochs_to_adjust_lr
    self.latest_loss_sum = 0.0
    self.latest_loss_count = 0
    self.score_scaling = score_scaling
    # Warmupを開始するステップ数
    self.warmup_start_global_step = 0
    self.min_newbob_scale = min_newbob_scale
    self.parameter_index = 0
    self.momentum = momentum
    self.layer_stacks = layer_stacks
    self.ply_begin_threshold = ply_begin_threshold
    self.ply_end_threshold = ply_end_threshold
    self.validation_step_outputs = []

    self._zero_virtual_feature_weights()

    # Tensor decomposition state
    self.factorization_rank = factorization_rank
    self.factorization_weight_decay = factorization_weight_decay
    self._init_factorization()

  def _init_factorization(self) -> None:
    """Initialize CP decomposition parameters for the main feature factor."""
    if self.factorization_rank <= 0:
      return
    feature_set = self.feature_set
    # Only single feature block supported
    assert len(feature_set.features) == 1, 'Tensor decomposition requires a single feature block'
    main_factor_name = feature_set.features[0].get_main_factor_name()
    num_real_features = feature_set.features[0].num_real_features

    # Determine decomposition dimensions from the main factor
    l1_size = self.l1_stack[0].in_features // 2
    main_factor_block = feature_set.features[0]
    is_halfkpe9 = main_factor_name.startswith('HalfKPE9')
    if is_halfkpe9:
      from halfkpe9 import NUM_SQ, NUM_PLANES, EFFECT_STATES
      shape = (l1_size, NUM_SQ, NUM_PLANES, EFFECT_STATES)
      num_factors = 4
    else:
      from halfkp import NUM_SQ, NUM_PLANES
      shape = (l1_size, NUM_SQ, NUM_PLANES)
      num_factors = 3

    # Scale rank by number of factors to keep total parameter count roughly proportional
    scaled_rank = self.factorization_rank

    # Create CP decomposition parameters with Xavier initialization
    self.register_parameter('tf_A', nn.Parameter(torch.empty(shape[0], scaled_rank)))
    fan_in_A, fan_out_A = shape[0], scaled_rank
    self.tf_A.data.uniform_(-1 / (1.0 + min(fan_in_A, fan_out_A)), 1 / (1.0 + max(fan_in_A, fan_out_A)))

    for i in range(1, num_factors):
      dim_i = shape[i]
      param_name = f'tf_{chr(65 + i)}'
      param = nn.Parameter(torch.empty(dim_i, scaled_rank))
      fan_in_i, fan_out_i = dim_i, scaled_rank
      param.data.uniform_(-1 / (1.0 + min(fan_in_i, fan_out_i)), 1 / (1.0 + max(fan_in_i, fan_out_i)))
      self.register_parameter(param_name, param)

  '''
  We zero all virtual feature weights because during serialization to .nnue
  we compute weights for each real feature as being the sum of the weights for
  the real feature in question and the virtual features it can be factored to.
  This means that if we didn't initialize the virtual feature weights to zero
  we would end up with the real features having effectively unexpected values
  at initialization - following the bell curve based on how many factors there are.
  '''
  def _zero_virtual_feature_weights(self) -> None:
    weights = self.input.weight
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  def get_materialized_cp_weight(self) -> Tensor:
    """Materialize the CP-decomposed weight for the main feature factor.

    Returns a tensor of shape (l1_size, num_real_features) containing
    the outer product of the factor matrices reshaped to 2D.
    """
    tf_params = [self.tf_A, self.tf_B, self.tf_C]
    if hasattr(self, 'tf_D'):
      tf_params.append(self.tf_D)
    num_factors = len(tf_params)

    # Build einsum string
    # Need num_factors chars for unique dims + 1 char for shared rank
    dim_chars = [chr(110 + i) for i in range(num_factors)]  # e.g., n,o,p for 3 factors
    rank_char = chr(110 + num_factors)  # e.g., q
    einsum_parts = [f'{dim_chars[0]}{rank_char}']
    for i in range(1, num_factors):
      einsum_parts.append(f'{dim_chars[i]}{rank_char}')
    einsum_str = ','.join(einsum_parts)
    output_indices = ''.join(dim_chars)
    einsum_expr = einsum_str + '->' + output_indices

    # Compute outer product via einsum
    weight_4d = torch.einsum(einsum_expr, *tf_params)

    # Reshape to 2D: (l1_size, num_real_features)
    real_features = self.feature_set.features[0].num_real_features
    if weight_4d.numel() != self.tf_A.shape[0] * real_features:
      weight_4d = weight_4d.view(self.tf_A.shape[0], real_features)
    else:
      weight_4d = weight_4d.reshape(self.tf_A.shape[0], real_features)

    return weight_4d

  '''
  This method attempts to convert the model from using the self.feature_set
  to new_feature_set.
  '''
  def set_feature_set(self, new_feature_set: Any) -> None:
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # next(iter(new_feature_block.factors)) is the way to get the
    # first item in a OrderedDict. (the ordered dict being str : int
    # mapping of the factor name to its size).
    # It is our new_feature_factor_name.
    # For example old_feature_block.name == "HalfKP"
    # and new_feature_factor_name == "HalfKP^"
    # We assume here that the "^" denotes factorized feature block
    # and we would like feature block implementers to follow this convention.
    # So if our current feature_set matches the first factor in the new_feature_set
    # we only have to add the virtual feature on top of the already existing real ones.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # We can just extend with zeros since it's unfactorized -> factorized
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us: Tensor, them: Tensor, w_in: Tensor, b_in: Tensor,
              bucket_index: Tensor | None = None) -> Tensor:
    """Forward pass through the network.
    
    Args:
        us: Color tensor for the side to move (size: [batch, 1])
        them: Color tensor for the opponent (size: [batch, 1])
        w_in: White active features sparse tensor (size: [batch, num_features])
        b_in: Black active features sparse tensor (size: [batch, num_features])
        bucket_index: Optional bucket indices for LayerStacks > 1 (size: [batch])
            If None, defaults to bucket 0 (for LayerStacks=1 compatibility).
    """
    if self.factorization_rank > 0:
      cp_weight = self.get_materialized_cp_weight()
      num_real_features = self.feature_set.num_real_features
      effective_weight = self.input.weight.clone()
      effective_weight[:, :num_real_features] += cp_weight
      effective_bias = self.input.bias
      w = F.linear(w_in, effective_weight, effective_bias)
      b = F.linear(b_in, effective_weight, effective_bias)
    else:
      w = self.input(w_in)
      b = self.input(b_in)
    
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    
    if self.layer_stacks > 1:
      # LayerStacks > 1: select per-stack weights based on bucket_index
      if bucket_index is None:
        bucket_index = torch.zeros(l0_.size(0), dtype=torch.long, device=l0_.device)
      bucket_index = torch.clamp(bucket_index, 0, self.layer_stacks - 1)
      
      # Gather the correct l1 weights for each sample using advanced indexing
      l1_weights = torch.stack([self.l1_stack[s].weight for s in range(self.layer_stacks)], dim=0)
      l1_biases = torch.stack([self.l1_stack[s].bias for s in range(self.layer_stacks)], dim=0)
      
      # Select weights for each sample: l1_weights[bucket_index[i], :, :]
      selected_l1_weight = l1_weights[bucket_index]  # [batch, l2_size, 2*l1_size]
      selected_l1_bias = l1_biases[bucket_index]      # [batch, l2_size]
      
      # Batched matrix multiply: [batch, 2*l1_size] @ [batch, 2*l1_size, l2_size].transpose(-1,-2) + bias
      l1_ = torch.bmm(l0_.unsqueeze(1), selected_l1_weight.transpose(-1, -2)).squeeze(1) + selected_l1_bias
      l1_ = torch.clamp(l1_, 0.0, 1.0)
    else:
      # LayerStacks = 1: use shared l1 layer
      l1_ = torch.clamp(self.l1_stack[0](l0_), 0.0, 1.0)
    
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def step_(self, batch: tuple[Tensor, ...], batch_idx: int, loss_type: str) -> Tensor:
    # Extract bucket_index from batch data (8-tuple when available)
    if len(batch) == 8:
      us, them, white, black, outcome, score, ply, bucket_index = batch
    else:
      us, them, white, black, outcome, score, ply = batch
      bucket_index = None

    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    scaling = self.score_scaling

    q = self(us, them, white, black, bucket_index) * self.NNUE_TO_SCORE / scaling
    t = outcome * (1.0 - self.label_smoothing_eps * 2.0) + self.label_smoothing_eps
    p = (score / scaling).sigmoid()

    teacher_entropy = -(p * (p + self.EPSILON).log() + (1.0 - p) * (1.0 - p + self.EPSILON).log())
    outcome_entropy = -(t * (t + self.EPSILON).log() + (1.0 - t) * (1.0 - t + self.EPSILON).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    if self.lambda_[self.parameter_index] >= 0.0:
      lambda_ = self.lambda_[self.parameter_index]
    else:
      lambda_ = (self.ply_end_threshold - ply) / (self.ply_end_threshold - self.ply_begin_threshold)
      lambda_ = torch.clamp(lambda_ , 0.0, 1.0)
    result  = lambda_ * teacher_loss    + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()
    self.log(loss_type, loss)
    return loss

    # MSE Loss function for debugging
    # Scale score by 600.0 to match the expected NNUE scaling factor
    # output = self(us, them, white, black) * 600.0
    # loss = F.mse_loss(output, score)

  def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
    loss = self.step_(batch, batch_idx, 'val_loss')
    self.validation_step_outputs.append(loss)
    return loss
  
  def on_validation_epoch_end(self) -> None:
    if not self.validation_step_outputs:
      return
    outputs = self.validation_step_outputs
    self.latest_loss_sum += float(sum(outputs)) / len(outputs)
    self.latest_loss_count += 1

    if self.newbob_decay != 1.0 and self.current_epoch > 0 and self.current_epoch % self.num_epochs_to_adjust_lr == 0:
      latest_loss = self.latest_loss_sum / self.latest_loss_count
      self.latest_loss_sum = 0.0
      self.latest_loss_count = 0
      if latest_loss < self.best_loss:
        self.print(f"{self.current_epoch=}, {latest_loss=} < {self.best_loss=}, accepted, {self.newbob_scale=}")
        sys.stdout.flush()
        self.best_loss = latest_loss
      else:
        self.newbob_scale *= self.newbob_decay
        self.print(f"{self.current_epoch=}, {latest_loss=} >= {self.best_loss=}, rejected, {self.newbob_scale=}")
        sys.stdout.flush()
    
    if self.newbob_scale < self.min_newbob_scale:
      self.parameter_index += 1
      if self.parameter_index < len(self.lr):
        self.best_loss = 1e10
        self.newbob_scale = 1.0
      else:
        self.trainer.should_stop = True
        self.print(f"{self.current_epoch=}, early stopping")
    
    self.validation_step_outputs.clear()

  def test_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> None:
    self.step_(batch, batch_idx, 'test_loss')

  # learning rate warm-up
  def optimizer_step(
      self,
      epoch: int,
      batch_idx: int,
      optimizer: Optimizer,
      optimizer_closure: Callable[[], Any],
  ) -> None:
    # manually warm up lr without a scheduler
    if self.trainer.global_step - self.warmup_start_global_step < self.num_batches_warmup:
      warmup_scale = min(1.0, float(self.trainer.global_step - self.warmup_start_global_step + 1) / self.num_batches_warmup)
    else:
      warmup_scale = 1.0
    for pg in optimizer.param_groups:
      pg["lr"] = self.lr[self.parameter_index] * warmup_scale * self.newbob_scale
      self.log("lr", pg["lr"])

    # update params
    optimizer.step(closure=optimizer_closure)

    # clip parameters
    for child in self.children():
      if not isinstance(child, nn.Linear):
        continue

      if child == self.input:
        continue

      # FC layers are stored as int8 weights, and int32 biases
      if child != self.output:
        kBiasScale = (1 << self.WEIGHT_SCALE_BITS) * self.ACTIVATION_SCALE
      else:
        kBiasScale = self.NNUE_TO_SCORE * self.FV_SCALE
      kWeightScale = kBiasScale / self.ACTIVATION_SCALE
      kMaxWeight = self.ACTIVATION_SCALE / kWeightScale
      child.weight.data.clamp_(-kMaxWeight, kMaxWeight)

    # Clip CP decomposition factors to prevent weight explosion
    if self.factorization_rank > 0:
      kWeightScale = (1 << self.WEIGHT_SCALE_BITS) * self.ACTIVATION_SCALE / self.ACTIVATION_SCALE
      kMaxWeight = self.ACTIVATION_SCALE / kWeightScale
      rank = self.factorization_rank
      # Scale by rank to account for CP sum accumulation
      factor_clip = (kMaxWeight / rank) ** (1 / 3)
      for param in [self.tf_A, self.tf_B, self.tf_C]:
        param.data.clamp_(-factor_clip, factor_clip)
      if hasattr(self, 'tf_D'):
        factor_clip = (kMaxWeight / rank) ** (1 / 4)
        self.tf_D.data.clamp_(-factor_clip, factor_clip)

  def configure_optimizers(self) -> Optimizer:
    return torch.optim.SGD(self.parameters(), lr=self.lr[0], momentum=self.momentum)

  def get_layers(self, filt: Callable[[nn.Module], bool]) -> Iterator[nn.Parameter]:
    """
    Returns a list of layers.
    filt: Return true to include the given layer.
    """
    for i in self.children():
      if filt(i):
        if isinstance(i, nn.Linear):
          for p in i.parameters():
            if p.requires_grad:
              yield p
