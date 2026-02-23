# NNUE学習ガイド（従来戦略 + 拡張戦略）

この文書は、`nnue-pytorch` の学習を段階的に理解して実験できるようにまとめたものです。
対象は `train.py` / `model.py` / `config.template.yaml` の現在実装です。

## 1. まず押さえる全体像

このプロジェクトの学習は、次の2本立てです。

1. 従来戦略（ベースライン）
2. 拡張戦略（EMA・損失の軽微改良・サンプリング拡張）

最初に従来戦略で「普通に学習できる状態」を作り、その後に拡張戦略を1つずつ足して比較します。

### 1.1 なぜ段階的に変えるのか

学習戦略は「良くなること」だけでなく「壊れる原因」を一緒に持ち込みます。  
複数の要素を同時に変えると、うまくいった理由・失敗した理由を切り分けられません。  
そのため、このガイドでは次を原則にします。

1. まずベースラインを作る
2. 1回の実験で1要素だけ変える
3. 指標とログで差分を確認する

## 2. 従来戦略（ベースライン）

### 2.1 モデルと目的

- 入力: HalfKP系特徴（`data.features`）
- 出力: 1つの評価値（スカラー）
- 損失: 
  - 探索スコア由来の教師信号（teacher）
  - 対局結果由来の教師信号（outcome）
  - この2つを `model.lambda_` で混合

意味:

- `teacher` は「探索器がその局面をどう見たか」を学びます。
- `outcome` は「最終的に勝ったか負けたか」を学びます。
- `lambda_` は、短期的な探索評価と最終結果のどちらを重く見るかのつまみです。

### 2.1.1 出力は何を表すか（評価値と勝率）

このモデルの生出力はスカラー評価値ですが、学習時は最終的に「勝率」と整合する形で扱います。

1. 探索スコア由来の教師確率  
   `p = sigmoid(score / (score_scaling * teacher_temperature))`
2. 対局結果由来の教師確率  
   `t = outcome`（必要なら label smoothing を適用）
3. モデル出力側も sigmoid を通すと勝率として解釈可能  
   `q_prob = sigmoid(model_score_scaled)`

要点:

- `score_scaling` は評価値スケールを確率変換に合わせるための係数です。
- 実験結果を読むときは、`loss` だけでなく「勝率解釈で不自然な偏りがないか」も確認します。

### 2.2 最適化

- Optimizer: SGD + Momentum
- 学習率:
  - warmup (`model.num_batches_warmup`)
- 量子化互換の重みクリップ: 学習ステップごとに実施

意味:

- warmup は、学習初期の不安定な更新を抑える安全装置です。
- 重みクリップは、最終的に `.nnue` へ量子化する前提で「表現可能な範囲」に学習を保つためです。

### 2.3 データ使用

- C++ローダー経路（高速）: `data.py_data: false`
- Pythonローダー経路（拡張しやすい）: `data.py_data: true`
- 既存の間引き:
  - `data.smart_fen_skipping`
  - `data.random_fen_skipping`

各オプションの意味:

1. `smart_fen_skipping`  
   C++ローダー経路（`py_data=false`）で有効なフィルタです。  
   コード上は次のどちらかを満たす局面をスキップします。
   - `e.isCapturingMove()`（最善手が駒取り）
   - `e.isInCheck()`（王手がかかっている）
   つまり「非静的（tactical）になりやすい局面」を除外し、静かな局面中心で学習させる設定です。
2. `random_fen_skipping`  
   C++ローダー経路（`py_data=false`）で有効です。  
   `N` を指定すると、各局面を確率 `N/(N+1)` でスキップします。  
   例: `N=10` なら約 90.9% を飛ばし、約 9.1% を使います。

補足（2つを同時に使うとき）:

- 実装上の条件は `random skip OR smart filter` です。  
  つまり、`smart_fen_skipping` に該当する局面は必ず除外され、残り局面に `random_fen_skipping` がかかります。

## 3. 拡張戦略（今回追加）

### 3.1 EMA（Exponential Moving Average）

目的: 評価時の重みを滑らかにして安定化する。

設定（`model`）:

- `ema_enabled`（true/false）
- `ema_decay`（例: 0.9995）
- `ema_update_every`（何stepごとに更新するか）
- `ema_start_step`（EMA更新開始step）

実装上の挙動:

- 学習中は通常重みを更新
- 検証/テスト開始時に EMA 重みへ一時切替
- 検証/テスト終了時に元の重みへ復元
- checkpointには `ema_state` を保存
- `serialize.py --use_ema` で EMA 重みを `.nnue` へ出力可能

意味:

- 直近の重みはミニバッチごとに揺れます。EMAはその揺れを平均化した「安定版の重み」です。
- そのため、学習中の一時的なノイズに引きずられにくくなります。

期待される変化:

1. `val_loss` の振れ幅が小さくなる
2. 実戦評価の再現性が上がる

悪化のサイン:

1. `ema_decay` が高すぎて追従が遅く、改善が鈍る
2. `ema_start_step` が早すぎて初期ノイズを平均してしまう

### 3.2 損失の軽微改良

目的: 学習信号の強さを小さく調整し、データ条件に合わせる。

追加設定（`model`）:

- `teacher_temperature`（既定 1.0）
  - 探索スコアの sigmoid 変換を緩める/鋭くする
- `entropy_coef`（既定 1.0）
  - エントロピー項の重み
- `outcome_pos_weight`（既定 1.0）
  - outcomeの正例側重み

重要:

- 既定値はすべて 1.0 なので、従来挙動と互換です。
- まずは1項目ずつ変更し、同時に多くを動かさないこと。

意味:

- `teacher_temperature` は、探索スコアを確率へ変換するときの「強調度」です。  
  大きくすると教師信号がなだらかになり、過信を抑えます。
- `entropy_coef` は、過度に極端な出力に寄りすぎるのを抑える強さです。
- `outcome_pos_weight` は、勝ち側信号の重みを調整してクラス不均衡に対応します。

期待される変化:

1. 収束が安定する
2. 特定局面への過適合が減る

悪化のサイン:

1. 温度が高すぎて学習が鈍化
2. `entropy_coef` が低すぎて過信、または高すぎて学習不足
3. `outcome_pos_weight` が大きすぎて予測バランスが崩れる

### 3.3 サンプリング拡張（Pythonローダー経路）

目的: 局面分布の偏りを抑える。

追加設定（`data`）:

- `py_data_sampling_mode`: `uniform` or `ply_balanced`
- `py_data_sampling_bins`: ビン数（例: 8）
- `py_data_sampling_max_positions`: サンプリング対象の上限（0で全件）
- `py_data_sampling_seed`: 乱数seed

注意:

- この拡張は `data.py_data: true` のとき有効です。
- `uniform` は従来相当（互換モード）です。

各オプションの意味:

1. `py_data_sampling_mode=uniform`  
   単純一様サンプリング。基準比較用のモードです。
2. `py_data_sampling_mode=ply_balanced`  
   手数（ply）帯ごとの偏りを減らすモードです。序中終盤の学習バランス改善を狙います。
3. `py_data_sampling_bins`  
   ply を何分割するか。増やすと粒度は上がりますが、各binのサンプル不足に注意が必要です。
4. `py_data_sampling_max_positions`  
   サンプリング用に参照する局面上限。0 は全件参照です。大規模データでは前処理時間短縮に有効です。
5. `py_data_sampling_seed`  
   再現性確保用の乱数種です。比較実験では固定します。

意味:

- 学習データに偏りがあると、モデルは「よく出る局面」だけ得意になります。
- `ply_balanced` は手数帯ごとの偏りを減らし、序盤/中盤/終盤のバランスを取りやすくします。

期待される変化:

1. 特定手数帯での性能の偏りが減る
2. 汎化が改善する可能性がある

悪化のサイン:

1. Pythonローダー利用により学習速度が落ちる
2. バランスを強くしすぎて、実運用分布とずれる

## 4. 実験の進め方（推奨順）

### Step 0: ベースラインを固定

1. 既定設定で1本学習
2. 指標（`val_loss`、学習時間、`.nnue` 変換可否）を記録
3. この結果を比較基準にする

### Step 1: EMAのみON

- 例:
  - `model.ema_enabled: true`
  - `model.ema_decay: 0.9995`
  - `model.ema_start_step: 1000`

確認:

- 学習が止まらない
- `serialize.py --use_ema` で出力できる
- `val_loss` が悪化しにくい

### Step 2: 損失を微調整

- 変更は1つずつ
- 例:
  - `model.teacher_temperature`: 1.0 → 1.2
  - `model.entropy_coef`: 1.0 → 0.8
  - `model.outcome_pos_weight`: 1.0 → 1.1

確認:

- 損失曲線が暴れない
- 収束速度と最終 `val_loss` を比較

### Step 3: サンプリングを変更

- `data.py_data: true`
- `data.py_data_sampling_mode: ply_balanced`

確認:

- 学習速度低下が許容範囲か
- `uniform` より `val_loss` または実戦評価が改善するか

## 5. 実験ログの残し方（最低限）

各runで次を記録します。

1. 変更したパラメータ
2. 学習開始/終了時刻
3. `val_loss` 推移
4. 変換コマンド（`serialize.py`）
5. 得られた `.nnue` の評価結果

## 6. よくある失敗

1. 一度に多項目を変更して原因が分からなくなる
2. ベースラインを記録せず比較不能になる
3. `py_data` と C++ローダー経路を混同する
4. EMAをONにしたのに `--use_ema` なしでエクスポートしてしまう
5. 「lossが少し改善した」だけで採用し、対局評価を確認しない

## 7. 最小構成の実験例（YAML）

```yaml
model:
  lambda_: [1.0]
  lr: [1.0]
  momentum: 0.0
  label_smoothing_eps: 0.0

  ema_enabled: true
  ema_decay: 0.9995
  ema_update_every: 1
  ema_start_step: 1000

  teacher_temperature: 1.0
  entropy_coef: 1.0
  outcome_pos_weight: 1.0

data:
  py_data: true
  py_data_sampling_mode: uniform
  py_data_sampling_bins: 8
  py_data_sampling_max_positions: 0
  py_data_sampling_seed: 42
```

この設定を基準に、1項目ずつ変更して比較します。

## 8. 外部情報に基づく推奨プロファイル

以下は、NNUE系トレーナーで公開されている推奨設定を本実装向けに読み替えたものです。

### 8.1 推奨プロファイルA（まず最初に試す）

狙い: まず安定して回し、比較基準を作る。

1. Optimizer: `SGD + momentum + warmup`
2. `batch_size`: 16384
3. `smart_fen_skipping: true`
4. `random_fen_skipping: 10`
5. `ema_enabled: true`（`ema_decay: 0.9995`）
6. 損失調整は既定値（`teacher_temperature=1.0`, `entropy_coef=1.0`, `outcome_pos_weight=1.0`）

### 8.2 推奨プロファイルB（データが非常に大きい場合）

狙い: 学習時間を抑えつつ、多様な局面を維持する。

1. Aをベースにする
2. `random_fen_skipping` を 10 前後へ増やす
3. 必要なら `py_data=false`（C++ローダー）を優先して処理速度を確保

### 8.3 なぜこの値なのか

1. `batch_size=16384` は、公開NNUEトレーナー系で繰り返し使われる実務値です。
2. `smart_fen_skipping + random_fen_skipping` は、局面の偏りとデータ量の両方を制御しやすい組み合わせです。  
   特に nodchip の shogi ブランチ README では `random-fen-skipping 10` が推奨例です。
3. EMAは推論時の揺れを抑えるため、学習戦略の最初の拡張として入れやすいです。

### 8.4 使うときの注意

1. まずAで1本回し、次にBへ進む（同時変更しない）。
2. 比較は `val_loss` だけでなく、変換後 `.nnue` の対局結果で判断する。
3. 損失調整パラメータは、A/Bの差が見えてから触る。

## 9. 参照ソース

1. nodchip `nnue-pytorch` shogi ブランチ README（推奨実行例、`random-fen-skipping 10`）  
   https://github.com/nodchip/nnue-pytorch/tree/shogi.2023-10-29.halfkp_1024x2-8-32
2. nodchip ブログ（将棋向け実験設定の詳細）  
   https://nodchip.hatenablog.com/entry/2025/04/20/000000
3. Lightning公式ドキュメント（`optimizer_step` による warmup 実装方針）  
   https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
4. HalfKP/NNUE Training Guide（歴史的なNNUE学習設定の背景）  
   https://github.com/HalfKP/NNUE
