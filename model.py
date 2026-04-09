import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Callable, Iterator
from torch.optim import Optimizer
import features as features_module

class StackedLinear(nn.Module):
    """
    バケットごとに独立した重みを持つ全結合層
    """
    def __init__(self, num_buckets: int, in_features: int, out_features: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.layers = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_buckets)])

    def forward(self, x: torch.Tensor, bucket_indices: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, in_features]
        # bucket_indices: [batch_size, 1]
        
        batch_size = x.shape[0]
        out_features = self.layers[0].out_features
        output = torch.zeros(batch_size, out_features, device=x.device, dtype=x.dtype)
        
        # バケットごとにバッチ内の該当する局面を処理
        for i in range(self.num_buckets):
            mask = (bucket_indices == i).flatten()
            if mask.any():
                output[mask] = self.layers[i](x[mask])
        return output

    def copy_weights_from_first_bucket(self):
        """全てのバケットに最初のバケットの重みをコピーして初期化します"""
        first_layer = self.layers[0]
        with torch.no_grad():
            for i in range(1, self.num_buckets):
                self.layers[i].weight.copy_(first_layer.weight)
                self.layers[i].bias.copy_(first_layer.bias)


class MoELinear(nn.Module):
    """
    最初の 512->32 層だけを expert 化したシンプルな MoE
    """
    def __init__(self, num_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(in_features, num_experts)
        self.experts = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        probs = torch.softmax(logits, dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(probs.unsqueeze(-1) * expert_outputs, dim=1)
        return output, probs

    def forward_top1(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        expert_indices = logits.argmax(dim=1)
        batch_size = x.shape[0]
        out_features = self.experts[0].out_features
        output = torch.zeros(batch_size, out_features, device=x.device, dtype=x.dtype)

        for i in range(self.num_experts):
            mask = expert_indices == i
            if mask.any():
                output[mask] = self.experts[i](x[mask])
        return output, expert_indices

    def copy_weights_from_first_expert(self):
        first_expert = self.experts[0]
        with torch.no_grad():
            for i in range(1, self.num_experts):
                self.experts[i].weight.copy_(first_expert.weight)
                self.experts[i].bias.copy_(first_expert.bias)

    def load_balancing_loss(self, route_probs: torch.Tensor) -> torch.Tensor:
        importance = route_probs.mean(dim=0)
        return self.num_experts * torch.sum(importance * importance) - 1.0

class NNUE(pl.LightningModule):
  """
  将棋の局面評価のためのNNUE (Efficiently Updatable Neural Network) モデル
  先頭 512->32 層を dense / LayerStack / MoE で切り替え可能な版
  """
  def __init__(
      self,
      features: str,
      lambda_: list[float] | None = None,
      lr: list[float] | None = None,
      label_smoothing_eps: float = 0.0,
      num_batches_warmup: int = 10000,
      gamma: float = 0.992,
      score_scaling: float = 361.0,
      momentum: float = 0.0,
      ply_begin_threshold: float = 100.0,
      ply_end_threshold: float = 120.0,
      moe_aux_loss_weight: float = 0.01,
      l1_mode: str = "moe",
      l1_size: int = 256,
      l2_size: int = 32,
      l3_size: int = 32,
      num_buckets: int = 8
  ) -> None:
    super().__init__()
    self.save_hyperparameters()
    if lambda_ is None:
        lambda_ = [1.0]
    if lr is None:
        lr = [1.0]

    self.NNUE_TO_SCORE_CONSTANT: float = 600.0
    self.EPSILON: float = 1e-12
    self.WEIGHT_CLIP_BITS: int = 6
    self.ACTIVATION_SCALE: float = 127.0
    self.FV_SCALE: int = 16

    feature_set = features_module.get_feature_set_from_name(features)
    self.feature_set = feature_set
    self.num_buckets = num_buckets
    self.l1_mode = l1_mode

    self.input = nn.Linear(feature_set.num_features, l1_size)
    if l1_mode == "moe":
      self.l1 = MoELinear(num_buckets, 2 * l1_size, l2_size)
    elif l1_mode == "layerstack":
      self.l1 = StackedLinear(num_buckets, 2 * l1_size, l2_size)
    elif l1_mode == "dense":
      self.l1 = nn.Linear(2 * l1_size, l2_size)
    else:
      raise ValueError(f"Unsupported l1_mode: {l1_mode}")
    self.l2 = nn.Linear(l2_size, l3_size)
    self.output = nn.Linear(l3_size, 1)

    self.lambda_ = lambda_
    self.lr = lr
    self.label_smoothing_eps = label_smoothing_eps
    self.num_batches_warmup = num_batches_warmup
    self.gamma = gamma
    self.score_scaling = score_scaling
    self.warmup_start_global_step = 0
    self.momentum = momentum
    self.ply_begin_threshold = ply_begin_threshold
    self.ply_end_threshold = ply_end_threshold
    self.moe_aux_loss_weight = moe_aux_loss_weight
    self.validation_step_outputs = []

    self._zero_virtual_feature_weights()
    if isinstance(self.l1, MoELinear):
      self.l1.copy_weights_from_first_expert()
    elif isinstance(self.l1, StackedLinear):
      self.l1.copy_weights_from_first_bucket()

  def _zero_virtual_feature_weights(self) -> None:
    weights = self.input.weight
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  def forward(
      self,
      us: torch.Tensor,
      them: torch.Tensor,
      w_in: torch.Tensor,
      b_in: torch.Tensor,
      use_top1: bool = False,
      return_aux: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    w_out = self.input(w_in)
    b_out = self.input(b_in)

    l0_input = (us * torch.cat([w_out, b_out], dim=1)) + \
               (them * torch.cat([b_out, w_out], dim=1))

    l0_output = torch.clamp(l0_input, 0.0, 1.0)
    if isinstance(self.l1, MoELinear) and use_top1:
      l1_preact, _expert_indices = self.l1.forward_top1(l0_output)
      moe_aux_loss = l0_output.new_zeros(())
    elif isinstance(self.l1, MoELinear):
      l1_preact, route_probs = self.l1(l0_output)
      moe_aux_loss = self.l1.load_balancing_loss(route_probs)
    elif isinstance(self.l1, StackedLinear):
      raise ValueError("LayerStack forward requires bucket indices; call forward_layerstack().")
    else:
      l1_preact = self.l1(l0_output)
      moe_aux_loss = l0_output.new_zeros(())

    l1_output = torch.clamp(l1_preact, 0.0, 1.0)
    l2_output = torch.clamp(self.l2(l1_output), 0.0, 1.0)
    output = self.output(l2_output)
    if return_aux:
      return output, moe_aux_loss
    return output

  def forward_layerstack(
      self,
      us: torch.Tensor,
      them: torch.Tensor,
      w_in: torch.Tensor,
      b_in: torch.Tensor,
      bucket_indices: torch.Tensor,
  ) -> torch.Tensor:
    w_out = self.input(w_in)
    b_out = self.input(b_in)

    l0_input = (us * torch.cat([w_out, b_out], dim=1)) + \
               (them * torch.cat([b_out, w_out], dim=1))
    l0_output = torch.clamp(l0_input, 0.0, 1.0)

    if isinstance(self.l1, StackedLinear):
      l1_preact = self.l1(l0_output, bucket_indices)
    elif isinstance(self.l1, nn.Linear):
      l1_preact = self.l1(l0_output)
    else:
      raise ValueError("forward_layerstack() is not valid for MoE mode.")

    l1_output = torch.clamp(l1_preact, 0.0, 1.0)
    l2_output = torch.clamp(self.l2(l1_output), 0.0, 1.0)
    return self.output(l2_output)

  def step_(self, batch: Tuple, batch_idx: int, loss_type: str) -> torch.Tensor:
    if len(batch) == 8:
      us_indices, them_indices, white_features, black_features, game_outcome, search_score, current_ply, npm = batch
    elif len(batch) == 7:
      us_indices, them_indices, white_features, black_features, game_outcome, search_score, npm = batch
      current_ply = None
    else:
      raise ValueError(f'Unexpected batch format (len={len(batch)}). Expected 7 or 8 tensors.')

    if isinstance(self.l1, StackedLinear):
      ls_indices = torch.clamp((16384.0 - npm) * float(self.num_buckets) / 16384.0, 0, self.num_buckets - 1).long()
      model_raw_output = self.forward_layerstack(us_indices, them_indices, white_features, black_features, ls_indices)
      moe_aux_loss = model_raw_output.new_zeros(())
    else:
      model_raw_output, moe_aux_loss = self(us_indices, them_indices, white_features, black_features, return_aux=True)
    scaled_model_output = model_raw_output * self.NNUE_TO_SCORE_CONSTANT / self.score_scaling

    smoothed_outcome_prob = game_outcome * (1.0 - self.label_smoothing_eps * 2.0) + self.label_smoothing_eps
    scaled_search_score_prob = (search_score / self.score_scaling).sigmoid()

    teacher_entropy = -(scaled_search_score_prob * (scaled_search_score_prob + self.EPSILON).log() +
                        (1.0 - scaled_search_score_prob) * (1.0 - scaled_search_score_prob + self.EPSILON).log())
    outcome_entropy = -(smoothed_outcome_prob * (smoothed_outcome_prob + self.EPSILON).log() +
                        (1.0 - smoothed_outcome_prob) * (1.0 - smoothed_outcome_prob + self.EPSILON).log())
    
    teacher_loss_term = -(scaled_search_score_prob * F.logsigmoid(scaled_model_output) +
                           (1.0 - scaled_search_score_prob) * F.logsigmoid(-scaled_model_output))
    outcome_loss_term = -(smoothed_outcome_prob * F.logsigmoid(scaled_model_output) +
                           (1.0 - smoothed_outcome_prob) * F.logsigmoid(-scaled_model_output))
    
    current_lambda: float
    if self.lambda_[0] >= 0.0:
      current_lambda = self.lambda_[0]
    else:
      if current_ply is None:
        raise ValueError('Dynamic lambda is enabled (lambda_ < 0), but ply is missing from the batch.')
      current_lambda = (self.ply_end_threshold - current_ply) / (self.ply_end_threshold - self.ply_begin_threshold)
      current_lambda = torch.clamp(current_lambda , 0.0, 1.0)
    
    combined_loss_result  = current_lambda * teacher_loss_term + (1.0 - current_lambda) * outcome_loss_term
    combined_entropy_result = current_lambda * teacher_entropy + (1.0 - current_lambda) * outcome_entropy
    
    final_loss = combined_loss_result.mean() - combined_entropy_result.mean()
    final_loss = final_loss + self.moe_aux_loss_weight * moe_aux_loss
    self.log(loss_type, final_loss)
    self.log(f'{loss_type}_moe_aux', moe_aux_loss)
    return final_loss

  def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
    loss = self.step_(batch, batch_idx, 'val_loss')
    self.validation_step_outputs.append(loss)
    return loss
  
  def on_validation_epoch_end(self) -> None:
    self.validation_step_outputs.clear()

  def test_step(self, batch: Tuple, batch_idx: int) -> None:
    self.step_(batch, batch_idx, 'test_loss')

  def optimizer_step(
      self,
      epoch: int,
      batch_idx: int,
      optimizer: Optimizer,
      optimizer_closure: Callable[[], None],
  ) -> None:
    if self.trainer.global_step - self.warmup_start_global_step < self.num_batches_warmup:
      warmup_scale = min(1.0, float(self.trainer.global_step - self.warmup_start_global_step + 1) / self.num_batches_warmup)
      for pg in optimizer.param_groups:
        pg["lr"] = self.lr[0] * warmup_scale
        self.log("lr", pg["lr"])

    optimizer.step(closure=optimizer_closure)

    for module in self.modules():
      if not isinstance(module, nn.Linear):
        continue
      if module == self.input:
        continue

      if module == self.output:
        bias_scale = self.NNUE_TO_SCORE_CONSTANT * self.FV_SCALE
      else:
        bias_scale = (1 << self.WEIGHT_CLIP_BITS) * self.ACTIVATION_SCALE

      weight_scale = bias_scale / self.ACTIVATION_SCALE
      max_weight_value = self.ACTIVATION_SCALE / weight_scale
      module.weight.data.clamp_(-max_weight_value, max_weight_value)

  def configure_optimizers(self) -> Tuple[list[Optimizer], list[object]]:
    optimizer = torch.optim.SGD(self.parameters(), lr=self.lr[0], momentum=self.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
    return [optimizer], [scheduler]

  def get_layers(self, filt: Callable[[nn.Module], bool]) -> Iterator[nn.Parameter]:
    for module in self.children():
      if filt(module):
        if isinstance(module, (nn.Linear, StackedLinear, MoELinear)):
          for param in module.parameters():
            if param.requires_grad:
              yield param
