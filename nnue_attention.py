import math
from collections.abc import Callable, Iterator
from typing import Any

import features as features_module
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer


Batch = tuple[Tensor, ...]
TensorDict = dict[str, Tensor]


class NNUEAttention(pl.LightningModule):
  """
  Uses the standard NNUE feature transformer as the embedding stage and replaces
  the downstream MLP with a lightweight attention mixer plus a scalar head.
  """
  def __init__(
      self, features: str, lambda_: list[float] | None = None, lr: list[float] | None = None,
      label_smoothing_eps: float = 0.0, num_batches_warmup: int = 10000,
      score_scaling: float = 361.0,
      momentum: float = 0.0, ply_begin_threshold: float = 100.0, ply_end_threshold: float = 120.0,
      l1_size: int = 1024,
      token_dim: int = 64, attention_dim: int = 128, attention_heads: int = 4,
      attention_layers: int = 2, attention_ff_mult: int = 4, attention_dropout: float = 0.1,
      head_hidden_dim: int = 128,
      freeze_embedding: bool = True, input_lr_scale: float = 0.1, weight_decay: float = 0.01,
      ema_enabled: bool = False, ema_decay: float = 0.9995, ema_update_every: int = 1, ema_start_step: int = 1000,
      teacher_temperature: float = 1.0, entropy_coef: float = 1.0, outcome_pos_weight: float = 1.0,
      corn_aux_weight: float = 0.0, corn_aux_thresholds: list[float] | None = None):
    super().__init__()
    if lambda_ is None:
      lambda_ = [1.0]
    if lr is None:
      lr = [1.0]
    self.save_hyperparameters()

    self.NNUE_TO_SCORE = 600.0
    self.EPSILON = 1e-12
    self.ACTIVATION_SCALE = 127.0

    feature_set = features_module.get_feature_set_from_name(features)
    self.feature_set = feature_set
    self.input = nn.Linear(feature_set.num_features, l1_size)

    total_dim = 2 * l1_size
    if total_dim % token_dim != 0:
      raise ValueError(
          f"2 * l1_size ({total_dim}) must be divisible by token_dim ({token_dim})."
      )
    self.num_tokens = total_dim // token_dim
    self.token_dim = token_dim
    self.attention_dim = attention_dim

    self.token_proj = nn.Linear(token_dim, attention_dim)
    self.cls_token = nn.Parameter(torch.zeros(1, 1, attention_dim))
    self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens + 1, attention_dim))
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=attention_dim,
        nhead=attention_heads,
        dim_feedforward=attention_dim * attention_ff_mult,
        dropout=attention_dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    self.attention = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)
    self.final_norm = nn.LayerNorm(attention_dim)
    self.head_hidden = nn.Linear(attention_dim, head_hidden_dim)
    self.output = nn.Linear(head_hidden_dim, 1)

    self.lambda_ = lambda_
    self.lr = lr
    self.weight_decay = float(weight_decay)
    self.label_smoothing_eps = label_smoothing_eps
    self.num_batches_warmup = num_batches_warmup
    self.score_scaling = score_scaling
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
    self.freeze_embedding = bool(freeze_embedding)
    self.input_lr_scale = float(input_lr_scale)
    if corn_aux_thresholds is None:
      corn_aux_thresholds = []
    self.corn_aux_thresholds = sorted(float(v) for v in corn_aux_thresholds)

    self._zero_virtual_feature_weights()
    self._init_attention_parameters()
    self._set_embedding_trainable(not self.freeze_embedding)

  def _init_attention_parameters(self) -> None:
    nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
    nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
    nn.init.xavier_uniform_(self.token_proj.weight, gain=0.5)
    nn.init.zeros_(self.token_proj.bias)
    nn.init.xavier_uniform_(self.head_hidden.weight, gain=0.5)
    nn.init.zeros_(self.head_hidden.bias)
    nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
    nn.init.zeros_(self.output.bias)

  def _zero_virtual_feature_weights(self) -> None:
    weights = self.input.weight
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  def _set_embedding_trainable(self, trainable: bool) -> None:
    self.input.weight.requires_grad = trainable
    if self.input.bias is not None:
      self.input.bias.requires_grad = trainable

  def set_feature_set(self, new_feature_set: Any) -> None:
    if self.feature_set.name == new_feature_set.name:
      return
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    if old_feature_block.name == next(iter(new_feature_block.factors)):
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
      self._zero_virtual_feature_weights()
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def _nnue_embedding(self, us: Tensor, them: Tensor, w_in: Tensor, b_in: Tensor) -> Tensor:
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    return torch.clamp(l0_, 0.0, 1.0)

  def _forward_hidden(self, us: Tensor, them: Tensor, w_in: Tensor, b_in: Tensor) -> Tensor:
    l0_ = self._nnue_embedding(us, them, w_in, b_in)
    if not torch.isfinite(l0_).all():
      raise RuntimeError("Non-finite values detected in NNUE embedding output.")
    tokens = l0_.reshape(l0_.shape[0], self.num_tokens, self.token_dim)
    tokens = self.token_proj(tokens)
    cls = self.cls_token.expand(tokens.shape[0], -1, -1)
    tokens = torch.cat([cls, tokens], dim=1)
    tokens = tokens + self.pos_embedding
    tokens = self.attention(tokens)
    pooled = self.final_norm(tokens[:, 0])
    hidden = torch.clamp(self.head_hidden(pooled), 0.0, 1.0)
    if not torch.isfinite(hidden).all():
      raise RuntimeError("Non-finite values detected in NNUE attention hidden activations.")
    return hidden

  def forward(self, us: Tensor, them: Tensor, w_in: Tensor, b_in: Tensor) -> Tensor:
    return self.output(self._forward_hidden(us, them, w_in, b_in))

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
    if self.corn_aux_weight <= 0.0 or not self.corn_aux_thresholds:
      return torch.zeros_like(q)

    thresholds = q.new_tensor(self.corn_aux_thresholds).view(1, -1)
    logits = (q / self.teacher_temperature).unsqueeze(-1) - thresholds
    score_logits = score.unsqueeze(-1) / (self.score_scaling * self.teacher_temperature)
    targets = (score_logits >= thresholds).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').mean(dim=-1)

  def _unpack_batch(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if len(batch) != 7:
      raise ValueError(f"Unexpected batch size: {len(batch)}")
    return batch

  def step_(self, batch: Batch, batch_idx: int, loss_type: str) -> Tensor:
    us, them, white, black, outcome, score, ply = self._unpack_batch(batch)
    hidden = self._forward_hidden(us, them, white, black)
    q = self.output(hidden) * self.NNUE_TO_SCORE / self.score_scaling
    if not torch.isfinite(q).all():
      raise RuntimeError("Non-finite values detected in NNUE attention logits.")
    teacher_entropy, outcome_entropy, teacher_loss, outcome_loss = self._compute_loss_terms(q, outcome, score)
    lambda_ = self._compute_lambda(ply)
    result = lambda_ * teacher_loss + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    corn_aux_loss = self._compute_corn_aux_loss(q, score)
    loss = result.mean() - self.entropy_coef * entropy.mean() + self.corn_aux_weight * corn_aux_loss.mean()
    if not torch.isfinite(loss):
      raise RuntimeError(
          "Non-finite loss detected. "
          f"q_range=({q.min().item():.4f},{q.max().item():.4f}) "
          f"score_range=({score.min().item():.4f},{score.max().item():.4f})"
      )
    self.log(loss_type, loss, on_step=(loss_type == 'train_loss'), on_epoch=True, prog_bar=(loss_type != 'train_loss'))
    if self.corn_aux_weight > 0.0 and self.corn_aux_thresholds:
      self.log(f"{loss_type}_corn_aux", corn_aux_loss.mean(), on_step=False, on_epoch=True)
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
    self.log("lr", float(optimizer.param_groups[0]["lr"]), on_step=False, on_epoch=True)

  def optimizer_step(
      self,
      epoch: int,
      batch_idx: int,
      optimizer: Optimizer,
      optimizer_closure: Callable[[], Any],
  ) -> None:
    self._apply_learning_rate(optimizer)
    optimizer.step(closure=optimizer_closure)

  def configure_optimizers(self) -> Optimizer:
    if math.isclose(self.momentum, 0.0):
      if self.freeze_embedding:
        return torch.optim.AdamW(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.lr[0],
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
      base_lr = self.lr[0]
      embedding_params = list(self.input.parameters())
      other_params = [
          param for name, param in self.named_parameters()
          if param.requires_grad and not name.startswith("input.")
      ]
      return torch.optim.AdamW(
          [
              {"params": embedding_params, "lr": base_lr * self.input_lr_scale, "weight_decay": self.weight_decay},
              {"params": other_params, "lr": base_lr, "weight_decay": self.weight_decay},
          ],
          lr=base_lr,
          betas=(0.9, 0.95),
      )
    return torch.optim.SGD(self.parameters(), lr=self.lr[0], momentum=self.momentum)
