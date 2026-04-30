import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import pytorch_lightning as pl
from collections.abc import Callable, Iterator
from typing import Any
from torch.optim import Optimizer
import cshogi
import features as features_module
import nnue_dataset

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
      corn_aux_weight: float = 0.0, corn_aux_thresholds: list[float] | None = None,
      king_zone_aux_weight: float = 0.0, major_safety_aux_weight: float = 0.0,
      preference_route: str = "none", preference_weight: float = 1.0, preference_beta: float = 1.0,
      fixed_ref_max_legal_moves: int = 0, preference_num_contexts: int = 8,
      preference_delta_scale: float = 1.0, base_ckpt: str = "", use_ema_weights: bool = False,
      input_adapter: str = "none", input_adapter_rank: int = 8, input_adapter_alpha: float = 1.0,
      input_adapter_init_std: float = 1e-3, freeze_base_input: bool = False):
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
    self.king_zone_head = nn.Linear(l3_size, 2)
    self.major_safety_head = nn.Linear(l3_size, 2)
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
    self.king_zone_aux_weight = max(float(king_zone_aux_weight), 0.0)
    self.major_safety_aux_weight = max(float(major_safety_aux_weight), 0.0)
    if preference_route not in {"none", "fixed_ref"}:
      raise ValueError(f"Unsupported preference_route: {preference_route}")
    self.preference_route = preference_route
    self.preference_weight = max(float(preference_weight), 0.0)
    self.preference_beta = float(preference_beta)
    self.fixed_ref_max_legal_moves = max(int(fixed_ref_max_legal_moves), 0)
    self.preference_num_contexts = max(int(preference_num_contexts), 1)
    self.preference_delta_scale = float(preference_delta_scale)
    self.context_embedding = (
        nn.Embedding(self.preference_num_contexts, l3_size)
        if self.preference_route == "fixed_ref"
        else None
    )
    self.input_adapter = "none"
    self.input_adapter_rank = max(1, int(input_adapter_rank))
    self.input_adapter_alpha = float(input_adapter_alpha)
    self.input_adapter_init_std = max(float(input_adapter_init_std), 0.0)
    self.freeze_base_input = bool(freeze_base_input)
    self.input_lora_a: nn.Module | None = None
    self.input_lora_b: nn.Module | None = None
    self._fixed_ref_state: TensorDict = {}

    self._zero_virtual_feature_weights()
    self.configure_input_adapter(
        input_adapter=input_adapter,
        input_adapter_rank=input_adapter_rank,
        input_adapter_alpha=input_adapter_alpha,
        input_adapter_init_std=input_adapter_init_std,
        freeze_base_input=freeze_base_input,
    )
    if base_ckpt:
      self._load_base_checkpoint(base_ckpt, use_ema_weights=use_ema_weights)

  def _load_checkpoint_state(self, base_ckpt: str) -> tuple[TensorDict, dict[str, Any]]:
    checkpoint = torch.load(base_ckpt, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
      return checkpoint["state_dict"], checkpoint
    if isinstance(checkpoint, dict):
      return checkpoint, {}
    if hasattr(checkpoint, "state_dict"):
      return checkpoint.state_dict(), {}
    raise TypeError(f"Unsupported checkpoint format for {base_ckpt}")

  def _load_base_checkpoint(self, base_ckpt: str, use_ema_weights: bool = False) -> None:
    state_dict, checkpoint = self._load_checkpoint_state(base_ckpt)
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("context_embedding.")
    }
    incompatible = self.load_state_dict(state_dict, strict=False)
    allowed_missing = set()
    if self.context_embedding is not None:
      allowed_missing.add("context_embedding.weight")
    if self.input_adapter == "halfkp_lora":
      allowed_missing.update({"input_lora_a.weight", "input_lora_b.weight"})
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected or missing:
      raise RuntimeError(
          "base_ckpt is incompatible with the current NNUE model: "
          f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
      )

    if checkpoint:
      self.on_load_checkpoint(checkpoint)

    if use_ema_weights:
      if not self.apply_ema_weights():
        raise RuntimeError("Requested use_ema_weights=True but no EMA weights were found in base_ckpt.")
      # Training should start from the EMA parameters themselves.
      self._ema_backup = None

  def _set_base_input_trainable(self, trainable: bool) -> None:
    self.input.weight.requires_grad = trainable
    if self.input.bias is not None:
      self.input.bias.requires_grad = trainable

  def _clear_input_adapter_parameters(self) -> None:
    for name in ["input_lora_a", "input_lora_b"]:
      if name in self._parameters:
        del self._parameters[name]
      if name in self._modules:
        del self._modules[name]
      setattr(self, name, None)

  def configure_input_adapter(
      self,
      input_adapter: str = "none",
      input_adapter_rank: int = 8,
      input_adapter_alpha: float = 1.0,
      input_adapter_init_std: float = 1e-3,
      freeze_base_input: bool = False,
  ) -> None:
    adapter = input_adapter.lower()
    if adapter not in {"none", "halfkp_lora"}:
      raise ValueError(f"Unsupported input_adapter: {input_adapter}")

    self.input_adapter = adapter
    self.input_adapter_rank = max(1, int(input_adapter_rank))
    self.input_adapter_alpha = float(input_adapter_alpha)
    self.input_adapter_init_std = max(float(input_adapter_init_std), 0.0)
    self.freeze_base_input = bool(freeze_base_input)
    self._clear_input_adapter_parameters()

    if adapter == "halfkp_lora":
      in_features = self.input.in_features
      out_features = self.input.out_features
      # Sparse embedding-based LoRA: each feature gets a rank-dim update vector
      self.input_lora_a = nn.Embedding(in_features, self.input_adapter_rank)
      self.input_lora_b = nn.Embedding(out_features, self.input_adapter_rank)
      
      if self.input_adapter_init_std > 0.0:
        nn.init.normal_(self.input_lora_a.weight, mean=0.0, std=self.input_adapter_init_std)
        nn.init.normal_(self.input_lora_b.weight, mean=0.0, std=self.input_adapter_init_std)
      else:
        nn.init.zeros_(self.input_lora_a.weight)
        nn.init.zeros_(self.input_lora_b.weight)

    self._set_base_input_trainable(not self.freeze_base_input)

  def get_effective_input_weight(self) -> Tensor:
    weight = self.input.weight
    if self.input_adapter == "halfkp_lora" and self.input_lora_a is not None:
      a = self.input_lora_a.weight                                    # [in_features, rank]
      b = self.input_lora_b.weight                                    # [out_features, rank]
      scale = self.input_adapter_alpha / float(self.input_adapter_rank)
      return weight + scale * (b @ a.T)

    return weight

  def get_effective_input_bias(self) -> Tensor:
    return self.input.bias

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
      self.input.in_features = new_feature_set.num_features
      self.feature_set = new_feature_set
      if self.input_adapter != "none":
        self.configure_input_adapter(
            input_adapter=self.input_adapter,
            input_adapter_rank=self.input_adapter_rank,
            input_adapter_alpha=self.input_adapter_alpha,
            input_adapter_init_std=self.input_adapter_init_std,
            freeze_base_input=self.freeze_base_input,
        )
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def _apply_input_adapter(self, x: Tensor) -> Tensor:
    if self.input_adapter == "none" or self.input_lora_a is None or self.input_lora_b is None:
      return x.new_zeros((x.shape[0], self.input.out_features))
    
    # x: sparse [batch, in_features]
    # Look up LoRA vectors for active features and aggregate per sample
    a_vecs = self.input_lora_a(x.indices()[1])                          # [nnz, rank]
    b_vecs = self.input_lora_b.weight.unsqueeze(1)                       # [out_features, 1, rank]
    
    scale = self.input_adapter_alpha / float(self.input_adapter_rank)
    
    # For each feature j at index (i, j) with value v:
    #   output[i] += v * (a_vecs[k] @ b_vecs[out_feature])
    # Use scatter_add to aggregate contributions per sample and output dim
    b_weights = self.input_lora_b.weight                              # [out_features, rank]
    a_contrib = a_vecs @ b_weights.T                                   # [nnz, out_features]
    
    batch_idx = x.indices()[0].unsqueeze(1).expand_as(a_contrib)       # [nnz, out_features]
    output = torch.zeros(x.shape[0], self.input.out_features, device=x.device, dtype=x.dtype)
    output.scatter_add_(0, batch_idx, a_contrib * scale)
    
    return output

  def _forward_hidden(
      self,
      us: Tensor,
      them: Tensor,
      w_in: Tensor,
      b_in: Tensor,
  ) -> Tensor:
    # Base linear pass — manual scatter_add to avoid sparse->dense expansion
    w_idx = w_in.indices()[0]
    b_idx = b_in.indices()[0]
    w_feat = w_in.indices()[1]
    b_feat = b_in.indices()[1]
    w_val = w_in.values()
    b_val = b_in.values()
    batch_size = w_in.shape[0]
    l1_size = self.input.out_features
    iw = self.input.weight
    ib = self.input.bias
    w_vecs = iw[w_feat]  # [nnz, l1_size]
    b_vecs = iw[b_feat]  # [nnz, l1_size]
    w_out = torch.zeros(batch_size, l1_size, device=w_in.device, dtype=w_in.dtype)
    b_out = torch.zeros(batch_size, l1_size, device=b_in.device, dtype=b_in.dtype)
    w_out.scatter_reduce_(0, w_idx.unsqueeze(1).expand_as(w_vecs), w_vecs * w_val.unsqueeze(1), reduce='sum', include_self=False)
    b_out.scatter_reduce_(0, b_idx.unsqueeze(1).expand_as(b_vecs), b_vecs * b_val.unsqueeze(1), reduce='sum', include_self=False)
    if ib is not None:
      w_out = w_out + ib
      b_out = b_out + ib
    w_base = w_out
    b_base = b_out
    
    # Shared LoRA pass
    w_adapter = self._apply_input_adapter(w_in)
    b_adapter = self._apply_input_adapter(b_in)
    
    w = w_base + w_adapter
    b = b_base + b_adapter
    
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    return l2_

  def forward(self, us: Tensor, them: Tensor, w_in: Tensor, b_in: Tensor) -> Tensor:
    return self.output(self._forward_hidden(us, them, w_in, b_in))

  def forward_with_context(
      self,
      us: Tensor,
      them: Tensor,
      w_in: Tensor,
      b_in: Tensor,
  ) -> Tensor:
    hidden = self._forward_hidden(us, them, w_in, b_in)
    q = self.output(hidden)
    return q

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

  def _compute_structural_aux_losses(self, hidden: Tensor, aux_targets: Tensor | None) -> tuple[Tensor, Tensor]:
    zero = hidden.new_zeros(hidden.shape[0])
    if aux_targets is None:
      return zero, zero

    king_zone_loss = zero
    major_safety_loss = zero

    if self.king_zone_aux_weight > 0.0:
      logits = self.king_zone_head(hidden)
      king_zone_loss = F.binary_cross_entropy_with_logits(
          logits,
          aux_targets[:, 0:2],
          reduction='none',
      ).mean(dim=-1)

    if self.major_safety_aux_weight > 0.0:
      logits = self.major_safety_head(hidden)
      major_safety_loss = F.binary_cross_entropy_with_logits(
          logits,
          aux_targets[:, 2:4],
          reduction='none',
      ).mean(dim=-1)

    return king_zone_loss, major_safety_loss

  def _unpack_batch(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    if len(batch) == 7:
      us, them, white, black, outcome, score, ply = batch
      aux_targets = None
    elif len(batch) == 8:
      us, them, white, black, outcome, score, ply, aux_targets = batch
    else:
      raise ValueError(f"Unexpected batch size: {len(batch)}")
    return us, them, white, black, outcome, score, ply, aux_targets

  def step_(self, batch: Batch, batch_idx: int, loss_type: str) -> Tensor:
    if isinstance(batch, dict):
      if self.preference_route == "fixed_ref":
        return self._step_fixed_ref(batch, loss_type)
      raise ValueError("Dictionary batches require model.preference_route=fixed_ref")

    us, them, white, black, outcome, score, ply, aux_targets = self._unpack_batch(batch)
    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    hidden = self._forward_hidden(us, them, white, black)
    q = self.output(hidden) * self.NNUE_TO_SCORE / self.score_scaling
    teacher_entropy, outcome_entropy, teacher_loss, outcome_loss = self._compute_loss_terms(q, outcome, score)
    lambda_ = self._compute_lambda(ply)
    result = lambda_ * teacher_loss + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    corn_aux_loss = self._compute_corn_aux_loss(q, score)
    king_zone_aux_loss, major_safety_aux_loss = self._compute_structural_aux_losses(hidden, aux_targets)
    loss = (
        result.mean()
        - self.entropy_coef * entropy.mean()
        + self.corn_aux_weight * corn_aux_loss.mean()
        + self.king_zone_aux_weight * king_zone_aux_loss.mean()
        + self.major_safety_aux_weight * major_safety_aux_loss.mean()
    )
    self.log(loss_type, loss)
    if self.corn_aux_weight > 0.0 and self.corn_aux_thresholds:
      self.log(f"{loss_type}_corn_aux", corn_aux_loss.mean())
    if self.king_zone_aux_weight > 0.0 and aux_targets is not None:
      self.log(f"{loss_type}_king_zone_aux", king_zone_aux_loss.mean())
    if self.major_safety_aux_weight > 0.0 and aux_targets is not None:
      self.log(f"{loss_type}_major_safety_aux", major_safety_aux_loss.mean())
    return loss

  def _ensure_fixed_ref_state(self) -> None:
    if self._fixed_ref_state:
      return
    self._fixed_ref_state = {
        key: value.detach().clone()
        for key, value in self.state_dict().items()
        if torch.is_floating_point(value)
    }

  def _zero_loss_with_grad(self) -> Tensor:
    for param in self.parameters():
      if param.requires_grad and torch.is_floating_point(param):
        return param.sum() * 0.0
    raise RuntimeError("No trainable floating-point parameters are available to anchor the loss.")

  def _make_sparse_tensors_from_fens(
      self,
      fens: list[str],
      plies: list[int],
      device: torch.device,
  ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    scores = [0 for _ in fens]
    results = [0 for _ in fens]
    batch = nnue_dataset.make_sparse_batch_from_fens(self.feature_set, fens, scores, plies, results)
    try:
      tensors = batch.contents.get_tensors(device)
      us, them, white, black, _outcome, _score, _ply = tensors[:7]
    finally:
      nnue_dataset.destroy_sparse_batch(batch)
    return us, them, white, black

  def _fixed_ref_forward(self, us: Tensor, them: Tensor, white: Tensor, black: Tensor) -> Tensor:
    self._ensure_fixed_ref_state()
    s = self._fixed_ref_state
    input_weight = s["input.weight"].to(device=us.device, dtype=us.dtype)
    input_bias = s["input.bias"].to(device=us.device, dtype=us.dtype)
    l1_weight = s["l1.weight"].to(device=us.device, dtype=us.dtype)
    l1_bias = s["l1.bias"].to(device=us.device, dtype=us.dtype)
    l2_weight = s["l2.weight"].to(device=us.device, dtype=us.dtype)
    l2_bias = s["l2.bias"].to(device=us.device, dtype=us.dtype)
    output_weight = s["output.weight"].to(device=us.device, dtype=us.dtype)
    output_bias = s["output.bias"].to(device=us.device, dtype=us.dtype)
    
    w_base = F.linear(white, input_weight, input_bias)
    b_base = F.linear(black, input_weight, input_bias)
    w = w_base + b_base
    b = b_base + w_base
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(F.linear(l0_, l1_weight, l1_bias), 0.0, 1.0)
    l2_ = torch.clamp(F.linear(l1_, l2_weight, l2_bias), 0.0, 1.0)
    return F.linear(l2_, output_weight, output_bias)


  def _iter_candidate_after_fens(self, sfen: str, actual_move: int) -> list[tuple[int, str]]:
    board = cshogi.Board(sfen)
    candidates: list[tuple[int, str]] = []
    for move in board.legal_moves:
      move_int = int(move)
      if move_int == int(actual_move):
        continue
      board.push(move_int)
      candidates.append((move_int, board.sfen()))
      board.pop()
      if self.fixed_ref_max_legal_moves > 0 and len(candidates) >= self.fixed_ref_max_legal_moves:
        break
    return candidates

  def _make_after_sfen(self, sfen: str, move: int) -> str | None:
    board = cshogi.Board(sfen)
    move_int = int(move)
    if not board.is_legal(move_int):
      return None
    board.push(move_int)
    return board.sfen()

  def _build_fixed_ref_pairs(self, batch: dict[str, Any], device: torch.device) -> tuple[list[str], list[str], list[int], list[int], Tensor]:
    actual_fens: list[str] = []
    ref_fens: list[str] = []
    pair_plies: list[int] = []
    pair_context_ids: list[int] = []
    pair_weights: list[float] = []
    
    sfens = batch["sfen"]
    actual_moves = batch["actual_move"].detach().cpu().tolist()
    plies = batch["ply"].reshape(-1).detach().cpu().int().tolist()
    context_ids = batch["context_id"].detach().cpu().int().tolist()
    weights = batch["weight"].detach().cpu().tolist()

    for sfen, actual_move, ply, context_id, weight in zip(sfens, actual_moves, plies, context_ids, weights):
      actual_after = self._make_after_sfen(sfen, int(actual_move))
      if actual_after is None:
        continue

      candidates = self._iter_candidate_after_fens(sfen, int(actual_move))
      if not candidates:
        continue

      candidate_fens = [candidate_sfen for _move, candidate_sfen in candidates]
      candidate_plies = [int(ply) + 1 for _ in candidate_fens]
      
      # Batch evaluate all candidates at once using C++ sparse batch generation
      all_fens = candidate_fens
      all_plies = candidate_plies
      us, them, white, black = self._make_sparse_tensors_from_fens(all_fens, all_plies, device)
      with torch.no_grad():
        q = self._fixed_ref_forward(us, them, white, black).reshape(-1)
      ref_scores = q
      
      # The network output is side-to-move oriented. After one move, the side to
      # move is the opponent, so the original mover's utility is the negative score.
      ref_utilities = -ref_scores
      best_index = int(torch.argmax(ref_utilities).item())

      actual_fens.append(actual_after)
      ref_fens.append(candidate_fens[best_index])
      pair_plies.append(int(ply) + 1)
      pair_context_ids.append(int(context_id))
      pair_weights.append(float(weight))

    return actual_fens, ref_fens, pair_plies, pair_context_ids, torch.tensor(pair_weights, device=device)

  def _score_fens_with_current_context(
      self,
      fens: list[str],
      plies: list[int],
      context_ids: list[int],
      device: torch.device,
  ) -> Tensor:
    us, them, white, black = self._make_sparse_tensors_from_fens(fens, plies, device)
    # context_ids ignored: LoRA adapter is now shared across all contexts
    return self.forward(us, them, white, black).reshape(-1)

  def _forward_context_delta(self, hidden: Tensor, context_id: Tensor) -> Tensor:
    if self.context_embedding is None:
      raise RuntimeError("context_embedding is only available for preference_route=fixed_ref")
    if hidden.shape[0] != context_id.shape[0]:
      raise ValueError("hidden and context_id batch sizes must match")
    if torch.any(context_id < 0) or torch.any(context_id >= self.preference_num_contexts):
      raise ValueError(
          f"context_id is outside [0, {self.preference_num_contexts}); "
          "increase model.preference_num_contexts"
      )
    embedding = self.context_embedding(context_id.to(device=hidden.device, dtype=torch.long))
    return (hidden * embedding).sum(dim=1, keepdim=True) * self.preference_delta_scale

  def _step_fixed_ref(self, batch: dict[str, Any], loss_type: str) -> Tensor:
    param = next(self.parameters())
    device = param.device
    actual_fens, ref_fens, pair_plies, pair_context_ids, weights = self._build_fixed_ref_pairs(batch, device)
    if not actual_fens:
      loss = self._zero_loss_with_grad()
      self.log(loss_type, loss)
      self.log(f"{loss_type}_fixed_ref_pairs", 0.0)
      return loss

    actual_q = self._score_fens_with_current_context(actual_fens, pair_plies, pair_context_ids, device)
    ref_q = self._score_fens_with_current_context(ref_fens, pair_plies, pair_context_ids, device)
    actual_utility = -actual_q
    ref_utility = -ref_q
    logits = self.preference_beta * (actual_utility - ref_utility)
    
    raw_loss = -F.logsigmoid(logits)
    # Weighted average
    pref_loss = (raw_loss * weights).sum() / (weights.sum() + self.EPSILON)

    loss = self.preference_weight * pref_loss
    self.log(loss_type, loss)
    self.log(f"{loss_type}_fixed_ref_pref", pref_loss)
    self.log(f"{loss_type}_fixed_ref_pairs", float(len(actual_fens)))
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
    self.validation_step_outputs.append(loss.detach())
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
      if self.validation_step_outputs:
        del self.validation_step_outputs[:]
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
