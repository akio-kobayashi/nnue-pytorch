import chess
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
import math
import features as features_module
from typing import Tuple, Callable, Iterator
from torch.optim import Optimizer

class NNUE(pl.LightningModule):
  """
  将棋の局面評価のためのNNUE (Efficiently Updatable Neural Network) モデル

  このモデルは、現代の将棋エンジン（例：YaneuraOu）で採用されているものと
  同様のニューラルネットワークアーキテクチャを用いて、将棋の局面を評価することを目的としています。
  自己対戦や教師エンジンから得られるゲーム結果と探索スコアの両方を用いて学習できるように設計されており、
  `lambda_`パラメータによって損失の割合を柔軟に変更できます。

  主な特徴:
  - 自軍（us）と敵軍（them）の特徴に対して重みを共有する入力層
  - Clipped ReLU（0.0と1.0の間に制限されたReLU）を活性化関数として持つ複数の隠れ層
  - ゲーム結果（result）と探索スコア（teacher）を`lambda_`に基づいて組み合わせる損失計算
  - "NewBob"と呼ばれる減衰メカニズムを用いた学習率スケジューリング
  - 学習中の量子化を意識した重みクリッピング

  `lambda_` パラメータは損失関数のバランスを制御します:
  - `lambda_ = 0.0`: 損失は純粋にゲーム結果（例：勝ち/負け/引き分け）に基づきます。
  - `lambda_ = 1.0`: 損失は純粋に探索スコア（例：強力な教師エンジンによる評価値）に基づきます。
  - 0.0と1.0の間の値は、これら2つの極端なケースを補間します。
  また動的ラムダを使うと、手数に応じてラムダの値を変えることができます。
  """
  def __init__(
      self,
      features: str,
      lambda_: list[float] | None = None,
      lr: list[float] | None = None,
      label_smoothing_eps: float = 0.0,
      num_batches_warmup: int = 10000,
      newbob_decay: float = 0.5,
      num_epochs_to_adjust_lr: int = 500,
      score_scaling: float = 361.0,
      min_newbob_scale: float = 1e-5,
      momentum: float = 0.0,
      ply_begin_threshold: float = 100.0,
      ply_end_threshold: float = 120.0,
      l1_size: int = 1024,
      l2_size: int = 8,
      l3_size: int = 96
  ) -> None:
    """
    NNUEモデルを初期化します。

    引数:
        features (str): 使用する特徴量のセット名（例: 'HalfKP'）。
        lambda_ (list[float], optional): 損失補間に使用するラムダ値のリスト。デフォルトは [1.0]。
        lr (list[float], optional): 学習率のリスト。デフォルトは [1.0]。
        label_smoothing_eps (float, optional): ラベルスムージングのイプシロン値。デフォルトは 0.0。
        num_batches_warmup (int, optional): 学習率ウォームアップのためのバッチ数。デフォルトは 10000。
        newbob_decay (float, optional): NewBob学習率スケジューリングの減衰係数。デフォルトは 0.5。
        num_epochs_to_adjust_lr (int, optional): 学習率を調整するエポック数。デフォルトは 500。
        score_scaling (float, optional): 評価値のスケーリング係数。デフォルトは 361.0。
        min_newbob_scale (float, optional): 停止する前の最小newbobスケール。デフォルトは 1e-5。
        momentum (float, optional): オプティマイザのモーメンタム。デフォルトは 0.0。
        ply_begin_threshold (float, optional): 動的ラムダの開始手数。デフォルトは 100.0。
        ply_end_threshold (float, optional): 動的ラムダの終了手数。デフォルトは 120.0。
        l1_size (int, optional): 最初の隠れ層（入力後）のサイズ。デフォルトは 1024。
        l2_size (int, optional): 2番目の隠れ層のサイズ。デフォルトは 8。
        l3_size (int, optional): 3番目の隠れ層のサイズ。デフォルトは 96。
    """
    super().__init__()
    if lambda_ is None:
        lambda_ = [1.0]
    if lr is None:
        lr = [1.0]

    # モデルで使用される定数。主に損失計算で使用されます。
    # NNUE_TO_SCORE_CONSTANT: 生のNNUE出力をスコアに変換するための係数。
    self.NNUE_TO_SCORE_CONSTANT: float = 600.0
    # EPSILON: エントロピー計算でlog(0)を防ぐための微小な値。
    self.EPSILON: float = 1e-12

    # NNUEの量子化と重みクリッピングのための定数
    # WEIGHT_CLIP_BITS: 重みスケーリング計算に使用されるビット数。
    self.WEIGHT_CLIP_BITS: int = 6
    # ACTIVATION_SCALE: 量子化された層の活性化値のスケールファクター。
    self.ACTIVATION_SCALE: float = 127.0
    # FV_SCALE (Feature Value Scale): 特徴量のスケール。NNUEの文脈ではしばしば16。
    self.FV_SCALE: int = 16

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
    self.ply_begin_threshold = ply_begin_threshold
    self.ply_end_threshold = ply_end_threshold
    self.validation_step_outputs = []

    self._zero_virtual_feature_weights()

  def _zero_virtual_feature_weights(self) -> None:
    """
    入力層内の「仮想的な特徴」に対応する重みをゼロで初期化します。

    【背景】
    この関数は主に、`HalfKP^`（因子分解された特徴量）のような、
    「実質特徴量」と「仮想特徴量」が明確に区別される特徴セットで使用されます。
    「仮想特徴量」とは、玉の位置や駒の状態といった独立した要素を表す特徴量です。

    【目的】
    重みのシリアライズ（.nnue形式への変換）時に、実質特徴量の重みが
    仮想特徴量の重みの合計として計算される場合などに、学習開始時点での
    初期値を制御するために利用されます。仮想特徴量の重みをゼロにすることで、
    初期のモデル出力が意図しない値になることを防ぎます。

    【注意】
    もし`HalfKP`のような因子分解されていない特徴セットを使用している場合、
    この関数は実質的に何もしません（対応する仮想特徴量が存在しないため）。
    """
    weights = self.input.weight
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  def set_feature_set(self, new_feature_set: 'features_module.FeatureSet') -> None:
    """
    モデルが現在使用している特徴セットを新しいものに変換しようと試みます。

    このメソッドは現在、分解されていない特徴ブロックを、
    ゼロで初期化された仮想特徴量を追加することで分解されたものに拡張する
    特徴セットの変換に主に対応しています。

    引数:
        new_feature_set (features_module.FeatureSet): モデルに適用する新しい特徴セット。

    例外:
        Exception: 変換がサポートされていない場合（例：複数の特徴ブロックがある場合、
                   または互換性のない特徴セット名の場合）。
    """
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception(f'Cannot change feature set from {self.feature_set.name} to {new_feature_set.name}. '
                      'Conversion not supported for multiple feature blocks.')

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # Check if the old feature block name matches the first factor of the new feature block.
    # This typically implies converting from an unfactorized to a factorized feature set
    # (e.g., "HalfKP" to "HalfKP^").
    # If so, we can extend the existing weights with zeros for the new virtual features.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # Extend weights with zeros since it's unfactorized -> factorized conversion
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception(f'Cannot change feature set from {self.feature_set.name} to {new_feature_set.name}. '
                      'Feature block names do not match for supported conversion types.')

  def forward(self, us: torch.Tensor, them: torch.Tensor, w_in: torch.Tensor, b_in: torch.Tensor) -> torch.Tensor:
    """
    NNUEモデルの順伝播処理を定義します。

    引数:
        us (torch.Tensor): 各特徴に対する「自軍」（手番側）の視点を示すテンソル。
        them (torch.Tensor): 各特徴に対する「敵軍」（相手側）の視点を示すテンソル。
        w_in (torch.Tensor): 「先手」（または1番目のプレイヤー）側の入力特徴量。
        b_in (torch.Tensor): 「後手」（または2番目のプレイヤー）側の入力特徴量。

    戻り値:
        torch.Tensor: 与えられた局面に対する単一の評価値。
    """
    # Apply input layer to both sides' features
    w_out = self.input(w_in)
    b_out = self.input(b_in)

    # Combine features based on 'us' and 'them' perspectives
    # This creates the L0 layer input with a 'flipped' perspective for the opponent
    l0_input = (us * torch.cat([w_out, b_out], dim=1)) + \
               (them * torch.cat([b_out, w_out], dim=1))
    
    # Apply Clipped ReLU (clamp to 0.0-1.0) for the first activation
    l0_output = torch.clamp(l0_input, 0.0, 1.0)
    
    # Pass through subsequent hidden layers with Clipped ReLU
    l1_output = torch.clamp(self.l1(l0_output), 0.0, 1.0)
    l2_output = torch.clamp(self.l2(l1_output), 0.0, 1.0)
    
    # Final output layer
    evaluation_score = self.output(l2_output)
    return evaluation_score


# ========================================================================================
# ▼▼▼▼▼▼▼▼▼▼ ここから下はモデルの学習に関する、より高度なコードです ▼▼▼▼▼▼▼▼▼▼
# ========================================================================================
#
# 上記の __init__ と forward がモデルの「構造」を定義する部分です。
#
# ここから下のコードは、そのモデルを「どのように学習させるか」を定義する部分であり、
# PyTorch Lightningの専門的な機能も多く含まれます。
#
# 最初は、ここは「学習部分のコーディングなのであとまわし」として、
# まずはモデルの構造（どんな層がどう繋がっているか）の理解に集中してください。
#
# ※ 注意: この学習部分のコードは、将来的にリファクタリング（大規模な変更）
#          される可能性が高いです。現在の形に固執せず、PyTorch Lightningの
# #          高度な使い方の一例として参照する程度に留めてください。
#
# ========================================================================================

  def step_(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, loss_type: str) -> torch.Tensor:
    """
    単一の学習/バリデーション/テストステップを実行します。

    ゲーム結果と探索スコアに基づき損失を計算し、
    ラベルスムージングと動的なラムダ重み付けを適用します。

    引数:
        batch (Tuple[torch.Tensor, ...]): 以下のデータを含むバッチ:
            us_indices (torch.Tensor): 手番側の特徴インデックス。
            them_indices (torch.Tensor): 相手側の特徴インデックス。
            white_features (torch.Tensor): 先手側の特徴活性化。
            black_features (torch.Tensor): 後手側の特徴活性化。
            game_outcome (torch.Tensor): ゲーム結果（-1:負け、0:引き分け、1:勝ち）。
            search_score (torch.Tensor): 局面に対する探索スコア。
            current_ply (torch.Tensor): 現在の手数（ply）。
        batch_idx (int): 現在のバッチのインデックス。
        loss_type (str): 損失のタイプ（'train_loss'、'val_loss'、'test_loss'）。

    戻り値:
        torch.Tensor: バッチに対する計算された損失。
    """
    us_indices, them_indices, white_features, black_features, game_outcome, search_score, current_ply = batch

    model_raw_output = self(us_indices, them_indices, white_features, black_features)
    scaled_model_output = model_raw_output * self.NNUE_TO_SCORE_CONSTANT / self.score_scaling

    # Smoothed game outcome probability using label smoothing
    smoothed_outcome_prob = game_outcome * (1.0 - self.label_smoothing_eps * 2.0) + self.label_smoothing_eps
    # Scaled search score probability using sigmoid
    scaled_search_score_prob = (search_score / self.score_scaling).sigmoid()

    # Calculate entropy for teacher (search score) and outcome (game result)
    teacher_entropy = -(scaled_search_score_prob * (scaled_search_score_prob + self.EPSILON).log() +
                        (1.0 - scaled_search_score_prob) * (1.0 - scaled_search_score_prob + self.EPSILON).log())
    outcome_entropy = -(smoothed_outcome_prob * (smoothed_outcome_prob + self.EPSILON).log() +
                        (1.0 - smoothed_outcome_prob) * (1.0 - smoothed_outcome_prob + self.EPSILON).log())
    
    # Calculate loss terms using logsigmoid
    teacher_loss_term = -(scaled_search_score_prob * F.logsigmoid(scaled_model_output) +
                           (1.0 - scaled_search_score_prob) * F.logsigmoid(-scaled_model_output))
    outcome_loss_term = -(smoothed_outcome_prob * F.logsigmoid(scaled_model_output) +
                           (1.0 - smoothed_outcome_prob) * F.logsigmoid(-scaled_model_output))
    
    # Determine lambda for loss weighting (static or dynamic based on ply)
    current_lambda: float
    if self.lambda_[self.parameter_index] >= 0.0:
      current_lambda = self.lambda_[self.parameter_index]
    else:
      # Dynamic lambda: interpolates between game outcome and search score based on ply
      current_lambda = (self.ply_end_threshold - current_ply) / (self.ply_end_threshold - self.ply_begin_threshold)
      current_lambda = torch.clamp(current_lambda , 0.0, 1.0)
    
    # Combine loss and entropy terms using the determined lambda
    combined_loss_result  = current_lambda * teacher_loss_term + (1.0 - current_lambda) * outcome_loss_term
    combined_entropy_result = current_lambda * teacher_entropy + (1.0 - current_lambda) * outcome_entropy
    
    # Final loss is the mean of combined loss minus the mean of combined entropy
    final_loss = combined_loss_result.mean() - combined_entropy_result.mean()
    self.log(loss_type, final_loss)
    return final_loss

  def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
    """
    共通の `step_` ロジックを使用して、単一の学習ステップを実行します。

    引数:
        batch (Tuple[torch.Tensor, ...]): 学習データのバッチ。
        batch_idx (int): 現在のバッチのインデックス。

    戻り値:
        torch.Tensor: 計算された学習損失。
    """
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
    """
    共通の `step_` ロジックを使用して、単一の検証ステップを実行します。

    引数:
        batch (Tuple[torch.Tensor, ...]): 検証データのバッチ。
        batch_idx (int): 現在のバッチのインデックス。

    戻り値:
        torch.Tensor: 計算された検証損失。
    """
    loss = self.step_(batch, batch_idx, 'val_loss')
    self.validation_step_outputs.append(loss)
    return loss
  
  def on_validation_epoch_end(self) -> None:
    """
    検証エポックの終了時に呼び出され、結果を集計し、
    NewBobスケジューリング戦略に基づいて学習率を調整します。
    """
    if not self.validation_step_outputs:
      return
    
    # エポックの平均検証損失を計算
    epoch_outputs = self.validation_step_outputs
    current_epoch_avg_loss = sum(output.item() for output in epoch_outputs) / len(epoch_outputs) # Convert tensors to floats for sum
    self.latest_loss_sum += current_epoch_avg_loss
    self.latest_loss_count += 1

    # NewBob学習率調整を適用
    if self.newbob_decay != 1.0 and self.current_epoch > 0 and self.current_epoch % self.num_epochs_to_adjust_lr == 0:
      aggregated_avg_loss = self.latest_loss_sum / self.latest_loss_count
      self.latest_loss_sum = 0.0
      self.latest_loss_count = 0
      
      if aggregated_avg_loss < self.best_loss:
        # 損失が改善した場合、ベスト損失を更新
        self.print(f"{self.current_epoch=}, {aggregated_avg_loss=} < {self.best_loss=}, accepted, {self.newbob_scale=}")
        sys.stdout.flush()
        self.best_loss = aggregated_avg_loss
      else:
        # 損失が改善しなかった場合、学習率のスケールを減衰
        self.newbob_scale *= self.newbob_decay
        self.print(f"{self.current_epoch=}, {aggregated_avg_loss=} >= {self.best_loss=}, rejected, {self.newbob_scale=}")
        sys.stdout.flush()
    
    # パラメータセットの変更または早期停止を確認
    if self.newbob_scale < self.min_newbob_scale:
      self.parameter_index += 1
      if self.parameter_index < len(self.lr):
        # 次の学習率/ラムダパラメータセットへ移行
        self.best_loss = float('inf') # 新しいパラメータセットのためにベスト損失をリセット
        self.newbob_scale = 1.0
        self.print(f"Moved to parameter set {self.parameter_index} with LR {self.lr[self.parameter_index]}")
        sys.stdout.flush()
      else:
        # パラメータセットがもうない場合、早期停止を開始
        self.trainer.should_stop = True
        self.print(f"{self.current_epoch=}, early stopping initiated.")
        sys.stdout.flush()
    
    self.validation_step_outputs.clear()

  def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
    """
    共通の `step_` ロジックを使用して、単一のテストステップを実行します。

    引数:
        batch (Tuple[torch.Tensor, ...]): テストデータのバッチ。
        batch_idx (int): 現在のバッチのインデックス。
    """
    self.step_(batch, batch_idx, 'test_loss')

  def optimizer_step(
      self,
      epoch: int,
      batch_idx: int,
      optimizer: Optimizer,
      optimizer_closure: Callable[[], None],
  ) -> None:
    """
    学習率のウォームアップと重みクリッピングを含む、単一の最適化ステップを実行します。

    引数:
        epoch (int): 現在のエポック番号。
        batch_idx (int): 現在のバッチのインデックス。
        optimizer (Optimizer): 使用中のオプティマイザ。
        optimizer_closure (Callable[[], None]): モデルを再評価し、損失を返すクロージャ。
    """
    # スケジューラなしで学習率をウォームアップ
    if self.trainer.global_step - self.warmup_start_global_step < self.num_batches_warmup:
      warmup_scale = min(1.0, float(self.trainer.global_step - self.warmup_start_global_step + 1) / self.num_batches_warmup)
    else:
      warmup_scale = 1.0
    for pg in optimizer.param_groups:
      pg["lr"] = self.lr[self.parameter_index] * warmup_scale * self.newbob_scale
      self.log("lr", pg["lr"])

    # パラメータを更新
    optimizer.step(closure=optimizer_closure)

    # パラメータが過度に大きくならないようにクリッピング（量子化を意識）
    for child in self.children():
      if not isinstance(child, nn.Linear):
        continue

      if child == self.input:
        continue

      # 全結合層の重みはint8、バイアスはint32で保存されます。
      if child != self.output:
        bias_scale = (1 << self.WEIGHT_CLIP_BITS) * self.ACTIVATION_SCALE
      else:
        # 出力層のバイアスの場合: kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
        bias_scale = self.NNUE_TO_SCORE_CONSTANT * self.FV_SCALE
      
      weight_scale = bias_scale / self.ACTIVATION_SCALE
      max_weight_value = self.ACTIVATION_SCALE / weight_scale # クリッピング前の最大重み値
      child.weight.data.clamp_(-max_weight_value, max_weight_value)

  def configure_optimizers(self) -> Optimizer:
    """
    モデルのオプティマイザを設定します。

    戻り値:
        torch.optim.Optimizer: 設定されたSGDオプティマイザ。
    """
    # 設定されたリストの初期学習率とモーメンタムを使用するSGDオプティマイザ。
    return torch.optim.SGD(self.parameters(), lr=self.lr[0], momentum=self.momentum)

  def get_layers(self, filt: Callable[[nn.Module], bool]) -> Iterator[nn.Parameter]:
    """
    選択された全結合層の学習可能なパラメータに対するイテレータを返します。

    引数:
        filt (Callable[[nn.Module], bool]): モジュールを受け取り、そのパラメータを含めるべきであれば
                                            Trueを返すフィルター関数。

    戻り値:
        torch.nn.Parameter: フィルターされた全結合層からの学習可能なパラメータ。
    """
    for module in self.children():
      if filt(module):
        if isinstance(module, nn.Linear):
          for param in module.parameters():
            if param.requires_grad:
              yield param
