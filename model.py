import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from collections.abc import Callable, Iterator
from typing import Any
from torch.optim import Optimizer
import features as features_module

Batch = tuple[Tensor, ...]
TensorDict = dict[str, Tensor]


class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(
      self, features: str, lambda_: list[float] | None = None, lr: list[float] | None = None,
      label_smoothing_eps: float = 0.0, num_batches_warmup: int = 10000,
      score_scaling: float = 361.0,
      momentum: float = 0.0, ply_begin_threshold: float = 100.0, ply_end_threshold: float = 120.0,
      l1_size: int = 1024, l2_size: int = 8, l3_size: int = 96,
      ema_enabled: bool = False, ema_decay: float = 0.9995, ema_update_every: int = 1, ema_start_step: int = 1000,
      teacher_temperature: float = 1.0, entropy_coef: float = 1.0, outcome_pos_weight: float = 1.0,
      corn_aux_weight: float = 0.0, corn_aux_thresholds: list[float] | None = None):
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
    self.l1 = nn.Linear(2 * l1_size, l2_size)
    self.l2 = nn.Linear(l2_size, l3_size)
    self.output = nn.Linear(l3_size, 1)
    self.lambda_ = lambda_
    self.lr = lr
    self.label_smoothing_eps = label_smoothing_eps
    self.num_batches_warmup = num_batches_warmup
    self.score_scaling = score_scaling
    # Warmupを開始するステップ数
    self.warmup_start_global_step = 0
    self.momentum = momentum
    self.ply_begin_threshold = ply_begin_threshold
    self.ply_end_threshold = ply_end_threshold
    self.validation_step_outputs = []
    self.ema_enabled = ema_enabled
    self.ema_decay = ema_decay
    self.ema_update_every = max(1, int(ema_update_every))
    self.ema_start_step = max(0, int(ema_start_step))
    self._ema_state: TensorDict = {}
    self._ema_backup: TensorDict | None = None
    self.teacher_temperature = max(float(teacher_temperature), self.EPSILON)
    self.entropy_coef = float(entropy_coef)
    self.outcome_pos_weight = max(float(outcome_pos_weight), self.EPSILON)
    self.corn_aux_weight = max(float(corn_aux_weight), 0.0)
    if corn_aux_thresholds is None:
      corn_aux_thresholds = []
    self.corn_aux_thresholds = sorted(float(v) for v in corn_aux_thresholds)

    self._zero_virtual_feature_weights()

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

  def forward(self, us: Tensor, them: Tensor, w_in: Tensor, b_in: Tensor) -> Tensor:
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def _compute_lambda(self, ply: Tensor) -> Tensor | float:
    lambda_base = self.lambda_[0]
    if lambda_base >= 0.0:
      return lambda_base
    lambda_ = (self.ply_end_threshold - ply) / (self.ply_end_threshold - self.ply_begin_threshold)
    return torch.clamp(lambda_, 0.0, 1.0)

  def _compute_loss_terms(
      self,
      q: Tensor,
      outcome: Tensor,
      score: Tensor,
  ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    t = outcome * (1.0 - self.label_smoothing_eps * 2.0) + self.label_smoothing_eps
    p = (score / (self.score_scaling * self.teacher_temperature)).sigmoid()
    teacher_entropy = -(p * (p + self.EPSILON).log() + (1.0 - p) * (1.0 - p + self.EPSILON).log())
    outcome_entropy = -(t * (t + self.EPSILON).log() + (1.0 - t) * (1.0 - t + self.EPSILON).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(
        self.outcome_pos_weight * t * F.logsigmoid(q)
        + (1.0 - t) * F.logsigmoid(-q)
    )
    return teacher_entropy, outcome_entropy, teacher_loss, outcome_loss

  def _compute_corn_aux_loss(self, q: Tensor, score: Tensor) -> Tensor:
    """
    A cumulative ordinal auxiliary loss with fixed logit thresholds.

    Thresholds live in the same teacher-logit space as the softened teacher:
    score / (score_scaling * teacher_temperature).
    """
    if self.corn_aux_weight <= 0.0 or not self.corn_aux_thresholds:
      return torch.zeros_like(q)

    thresholds = q.new_tensor(self.corn_aux_thresholds).view(1, -1)
    logits = (q / self.teacher_temperature).unsqueeze(-1) - thresholds
    score_logits = score.unsqueeze(-1) / (self.score_scaling * self.teacher_temperature)
    targets = (score_logits >= thresholds).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').mean(dim=-1)

  def step_(self, batch: Batch, batch_idx: int, loss_type: str) -> Tensor:
    us, them, white, black, outcome, score, ply = batch
    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    q = self(us, them, white, black) * self.NNUE_TO_SCORE / self.score_scaling
    teacher_entropy, outcome_entropy, teacher_loss, outcome_loss = self._compute_loss_terms(q, outcome, score)
    lambda_ = self._compute_lambda(ply)
    result = lambda_ * teacher_loss + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    corn_aux_loss = self._compute_corn_aux_loss(q, score)
    loss = result.mean() - self.entropy_coef * entropy.mean() + self.corn_aux_weight * corn_aux_loss.mean()
    self.log(loss_type, loss)
    if self.corn_aux_weight > 0.0 and self.corn_aux_thresholds:
      self.log(f"{loss_type}_corn_aux", corn_aux_loss.mean())
    return loss

  def _iter_ema_parameters(self) -> Iterator[tuple[str, Tensor]]:
    for name, param in self.named_parameters():
      if param.requires_grad and torch.is_floating_point(param):
        yield name, param

  def _initialize_ema_state(self) -> None:
    if self._ema_state:
      return
    self._ema_state = {
        name: param.detach().clone()
        for name, param in self._iter_ema_parameters()
    }

  def _update_ema_state(self) -> None:
    self._initialize_ema_state()
    decay = float(self.ema_decay)
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      for name, param in self._iter_ema_parameters():
        self._ema_state[name].mul_(decay).add_(param.detach(), alpha=one_minus_decay)

  def apply_ema_weights(self) -> bool:
    if not self._ema_state:
      return False
    if self._ema_backup is not None:
      return True
    self._ema_backup = {}
    with torch.no_grad():
      for name, param in self._iter_ema_parameters():
        ema_weight = self._ema_state.get(name)
        if ema_weight is None:
          continue
        self._ema_backup[name] = param.detach().clone()
        param.copy_(ema_weight.to(device=param.device, dtype=param.dtype))
    return True

  def restore_original_weights(self) -> None:
    if self._ema_backup is None:
      return
    with torch.no_grad():
      for name, param in self._iter_ema_parameters():
        original = self._ema_backup.get(name)
        if original is None:
          continue
        param.copy_(original.to(device=param.device, dtype=param.dtype))
    self._ema_backup = None

  def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
    loss = self.step_(batch, batch_idx, 'val_loss')
    self.validation_step_outputs.append(loss)
    return loss

  def on_fit_start(self) -> None:
    if self.ema_enabled:
      self._initialize_ema_state()

  def on_train_batch_end(self, outputs: Any, batch: Batch, batch_idx: int) -> None:
    if not self.ema_enabled:
      return
    global_step = self.trainer.global_step
    if global_step < self.ema_start_step:
      return
    if (global_step - self.ema_start_step) % self.ema_update_every != 0:
      return
    self._update_ema_state()

  def on_validation_epoch_start(self) -> None:
    if self.ema_enabled:
      self.apply_ema_weights()
  
  def on_validation_epoch_end(self) -> None:
    try:
      if not self.validation_step_outputs:
        return
      self.validation_step_outputs.clear()
    finally:
      self.restore_original_weights()

  def test_step(self, batch: Batch, batch_idx: int) -> None:
    self.step_(batch, batch_idx, 'test_loss')

  def on_test_epoch_start(self) -> None:
    if self.ema_enabled:
      self.apply_ema_weights()

  def on_test_epoch_end(self) -> None:
    self.restore_original_weights()

  def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
    if self._ema_state:
      checkpoint["ema_state"] = {
          name: tensor.detach().cpu()
          for name, tensor in self._ema_state.items()
      }

  def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
    loaded_ema = checkpoint.get("ema_state")
    if isinstance(loaded_ema, dict):
      self._ema_state = {
          name: tensor.clone()
          for name, tensor in loaded_ema.items()
          if isinstance(tensor, Tensor)
      }

  def _apply_learning_rate(self, optimizer: Optimizer) -> None:
    if self.trainer.global_step - self.warmup_start_global_step < self.num_batches_warmup:
      warmup_scale = min(
          1.0,
          float(self.trainer.global_step - self.warmup_start_global_step + 1) / self.num_batches_warmup,
      )
    else:
      warmup_scale = 1.0

    for pg in optimizer.param_groups:
      pg["lr"] = self.lr[0] * warmup_scale
      self.log("lr", pg["lr"])

  def _clip_linear_weight(self, layer: nn.Linear) -> None:
    if layer != self.output:
      bias_scale = (1 << self.WEIGHT_SCALE_BITS) * self.ACTIVATION_SCALE
    else:
      bias_scale = self.NNUE_TO_SCORE * self.FV_SCALE
    weight_scale = bias_scale / self.ACTIVATION_SCALE
    max_weight = self.ACTIVATION_SCALE / weight_scale
    layer.weight.data.clamp_(-max_weight, max_weight)

  # learning rate warm-up
  def optimizer_step(
      self,
      epoch: int,
      batch_idx: int,
      optimizer: Optimizer,
      optimizer_closure: Callable[[], Any],
  ) -> None:
    self._apply_learning_rate(optimizer)

    # update params
    optimizer.step(closure=optimizer_closure)

    # clip parameters
    for child in self.children():
      if not isinstance(child, nn.Linear):
        continue

      if child == self.input:
        continue

      # FC layers are stored as int8 weights, and int32 biases
      self._clip_linear_weight(child)

  def configure_optimizers(self) -> Optimizer:
    return torch.optim.SGD(self.parameters(), lr=self.lr[0], momentum=self.momentum)

  def get_layers(self, filt: Callable[[nn.Module], bool]) -> Iterator[nn.Parameter]:
    """
    Returns a list of layers.
    filt: Return true to include the given layer.
    """
    for module in self.children():
      if filt(module) and isinstance(module, nn.Linear):
        for param in module.parameters():
          if param.requires_grad:
            yield param
