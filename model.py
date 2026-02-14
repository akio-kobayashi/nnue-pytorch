import chess
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
from typing import Tuple, Callable, Iterator
from torch.optim import Optimizer
import features as features_module

class SCReLU(nn.Module):
    """
    Squared Clipped ReLU (SCReLU) 活性化関数
    
    x = clamp(x, 0, 1)
    y = x^2 * (255/256)
    
    将棋エンジンの整数演算（ビットシフト）との互換性を保つためのスケーリングを含みます。
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        return torch.pow(x, 2.0) * (255.0 / 256.0)

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

class NNUE(pl.LightningModule):
  """
  将棋の局面評価のためのNNUE (Efficiently Updatable Neural Network) モデル
  LayerStack (Bucketing) と SCReLU を搭載した進化版
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
      l1_size: int = 1024,
      l2_size: int = 8,
      l3_size: int = 96,
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

    # 入力層 (Feature Transformer)
    self.input = nn.Linear(feature_set.num_features, l1_size)
    
    # LayerStack (Bucketed L1)
    # 2 * l1_size は自軍と敵軍の入力を結合したサイズ
    self.l1 = StackedLinear(num_buckets, 2 * l1_size, l2_size)
    
    # 以降の隠れ層
    self.l2 = nn.Linear(l2_size, l3_size)
    
    # 出力層
    self.output = nn.Linear(l3_size, 1)
    
    # PSQT用パス (wpsqt - bpsqt)
    # 特徴量から直接評価値に加算される線形層
    self.psqt = nn.Linear(feature_set.num_features, 1, bias=False)

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
    self.validation_step_outputs = []

    self._zero_virtual_feature_weights()
    # 全てのバケットを初期状態で同じ重みにする
    self.l1.copy_weights_from_first_bucket()

  def _zero_virtual_feature_weights(self) -> None:
    weights = self.input.weight
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  def forward(self, us: torch.Tensor, them: torch.Tensor, w_in: torch.Tensor, b_in: torch.Tensor, ls_indices: torch.Tensor) -> torch.Tensor:
    # Feature Transformer
    w_out = self.input(w_in)
    b_out = self.input(b_in)

    # 視点に応じた特徴の結合
    l0_input = (us * torch.cat([w_out, b_out], dim=1)) + \
               (them * torch.cat([b_out, w_out], dim=1))
    
    # 第1層活性化: SCReLU
    l0_output = torch.clamp(l0_input, 0.0, 1.0)
    l0_output = torch.pow(l0_output, 2.0) * (255.0 / 256.0)
    
    # Stacked L1
    l1_output = self.l1(l0_output, ls_indices)
    l1_output = torch.clamp(l1_output, 0.0, 1.0) # 中間層は通常のClipped ReLU
    
    # L2
    l2_output = torch.clamp(self.l2(l1_output), 0.0, 1.0)
    
    # Main output
    nnue_output = self.output(l2_output)
    
    # PSQT Path: (wpsqt - bpsqt)
    # w_in, b_in は sparse tensor の場合があるが、ここでは dense 前提か 
    # Datasetからは sparse で来るので、psqt.weight との行列演算が必要
    # 簡易のため、w_in と b_in の差分に対して適用
    w_psqt = F.linear(w_in, self.psqt.weight)
    b_psqt = F.linear(b_in, self.psqt.weight)
    psqt_output = (us * (w_psqt - b_psqt)) + (them * (b_psqt - w_psqt))
    
    return nnue_output + psqt_output

  def step_(self, batch: Tuple, batch_idx: int, loss_type: str) -> torch.Tensor:
    us_indices, them_indices, white_features, black_features, game_outcome, search_score, npm = batch

    # バケットインデックスの計算 (NPMに基づく 8バケット分割)
    # bucket_index = (16384 - total_non_pawn_material) * 8 / 16384
    ls_indices = torch.clamp((16384.0 - npm) * 8.0 / 16384.0, 0, self.num_buckets - 1).long()

    model_raw_output = self(us_indices, them_indices, white_features, black_features, ls_indices)
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
      current_lambda = (self.ply_end_threshold - current_ply) / (self.ply_end_threshold - self.ply_begin_threshold)
      current_lambda = torch.clamp(current_lambda , 0.0, 1.0)
    
    combined_loss_result  = current_lambda * teacher_loss_term + (1.0 - current_lambda) * outcome_loss_term
    combined_entropy_result = current_lambda * teacher_entropy + (1.0 - current_lambda) * outcome_entropy
    
    final_loss = combined_loss_result.mean() - combined_entropy_result.mean()
    self.log(loss_type, final_loss)
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

    for child in self.children():
      if isinstance(child, nn.Linear):
          if child == self.input or child == self.psqt:
            continue
          
          if child != self.output:
            bias_scale = (1 << self.WEIGHT_CLIP_BITS) * self.ACTIVATION_SCALE
          else:
            bias_scale = self.NNUE_TO_SCORE_CONSTANT * self.FV_SCALE
          
          weight_scale = bias_scale / self.ACTIVATION_SCALE
          max_weight_value = self.ACTIVATION_SCALE / weight_scale
          child.weight.data.clamp_(-max_weight_value, max_weight_value)
      
      elif isinstance(child, StackedLinear):
          # StackedLinear の各バケットに対しても同様にクリッピング
          bias_scale = (1 << self.WEIGHT_CLIP_BITS) * self.ACTIVATION_SCALE
          weight_scale = bias_scale / self.ACTIVATION_SCALE
          max_weight_value = self.ACTIVATION_SCALE / weight_scale
          for layer in child.layers:
              layer.weight.data.clamp_(-max_weight_value, max_weight_value)

  def configure_optimizers(self) -> Tuple[list[Optimizer], list[object]]:
    optimizer = torch.optim.SGD(self.parameters(), lr=self.lr[0], momentum=self.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
    return [optimizer], [scheduler]

  def get_layers(self, filt: Callable[[nn.Module], bool]) -> Iterator[nn.Parameter]:
    for module in self.children():
      if filt(module):
        if isinstance(module, (nn.Linear, StackedLinear)):
          for param in module.parameters():
            if param.requires_grad:
              yield param
