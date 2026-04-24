# Conditional Preference Fixed-Ref: Initial Implementation Notes

## Branch policy

- branch:
  - `conditional-preference-fixed-ref`
- purpose:
  - まず fixed-reference route を先に成立させる
  - Elo / player を同一の context interface で扱えることを確認する

## Why fixed-ref first

- 学習時に外部エンジンを呼ばない
- 学習中モデルそのものを候補生成器に使わない
- `V0` を固定参照モデルとして使うことで、候補生成と学習対象を分離できる

## Current constraints in this repository

- 既存の高速 C++ loader は `.bin` 前提であり、HDF5 decision-point 学習にはそのまま使えない
- 既存 `py_data` は PackedSfenValue `.bin` を対象とする経路だが、fixed-ref route では
  HDF5 の `psv` と `actual_move` を直接読みたい
- 学習用 feature 生成そのものは `nnue_dataset.make_sparse_batch_from_fens()` を使う方が自然
- したがって fixed-ref route の最初の実装は Python Dataset + custom collate が入口になる

## Minimal shared contract

Elo と player を共通化するため、dataset は以下の形で sample を返す。

- `sfen`
- `actual_move`
- `ply`
- `game_result`
- `context_type`
- `context_id`
- `context_label`
- `metadata`

ここで

- `context_type`:
  - `elo`
  - `player`
- `context_id`:
  - 埋め込み lookup に使う整数 id
- `context_label`:
  - デバッグや評価時の可読ラベル

## Route boundary

fixed-ref route の責務を更新する。

- dataset:
  - `psv` とメタを読む
  - `actual_move` と context を返す
  - Elo に基づく学習重み `sample_weight` を計算する
- collate / batch builder:
  - 現局面の合法手を列挙する
  - `V0` で候補手を選ぶ
  - pairwise 学習に必要な `actual` / `ref` 遷移先を構成する
- model:
  - 最下層 (FT) に Elo 条件付き LoRA アダプタを適用する
  - 損失計算時に `sample_weight` による加重平均を行う
  - `V(s, c) = (W0 + DeltaW_c) * x + b` を学習する

## First code slice

最初のコードスライスでは、基盤となるデータ構造と LoRA の条件付き化を成立させる。

1. HDF5 から Elo 加重を計算して返せる
2. モデルの入力層 LoRA が `context_id` (Eloバケット) に依存して切り替わる
3. 損失計算が `sample_weight` を受け取れる

## Next slices

### Slice F1: Weighted Dataset

- HDF5 fixed-ref dataset への `sample_weight` 計算ロジック追加
- Elo レーティングから 0.0〜1.0 程度の重みへのマッピング

### Slice F2: Context-conditioned Input LoRA

- `model.input_lora_a` / `b` を `nn.Embedding` 化
- `_forward_hidden` での `context_id` を用いた動的な LoRA 適用
- バッチ内での効率的なアダプタ計算ロジック

### Slice F3: Weighted Preference Loss

- `_step_fixed_ref` での加重損失計算の実装
- `V0` 推論用の current / actual / candidate SFEN 構成と統合

### Slice F4

- player 条件を同じ経路に追加
- context encoder の共有を確認する

## Initial command shape

Elo 条件で fixed-ref route を起動する最小形は以下である。

```bash
python train.py fit \
  --data.train /path/to/train.h5 \
  --data.val /path/to/val.h5 \
  --data.preference_data true \
  --data.preference_context_type elo \
  --data.batch_size 128 \
  --model.base_ckpt /path/to/pretrained.ckpt \
  --model.preference_route fixed_ref \
  --model.preference_weight 1.0 \
  --model.preference_beta 1.0 \
  --model.preference_num_contexts 8 \
  --model.input_adapter halfkp_lora \
  --model.input_adapter_rank 8 \
  --model.input_adapter_alpha 1.0 \
  --model.freeze_base_input true
```

player 条件では `preference_context_type` だけを差し替える。

```bash
python train.py fit \
  --data.train /path/to/train.h5 \
  --data.val /path/to/val.h5 \
  --data.preference_data true \
  --data.preference_context_type player \
  --data.batch_size 128 \
  --model.base_ckpt /path/to/pretrained.ckpt \
  --model.preference_route fixed_ref \
  --model.preference_num_contexts 4096 \
  --model.input_adapter halfkp_lora \
  --model.freeze_base_input true
```

## Current implementation state

- HDF5 の `psv` / `actual_move` / context を読む dataset を追加済み
- `preference_data=true` で DataModule が HDF5 dataset を使う
- `preference_route=fixed_ref` で model が合法手を列挙し、固定参照スナップショット `V0` により候補を選ぶ
- `context_id` は `Delta(s,c)` に入る context embedding として使う
- pairwise loss は one-ply 後局面を side-to-move score として評価し、元の着手者視点では符号を反転して比較する
- `model.base_ckpt` を指定すると、学習開始時に既存 checkpoint / state dict / `.pt` model から通常 NNUE 本体を読み込む
- `model.input_adapter=halfkp_lora` と `model.freeze_base_input=true` により、ベース FT を固定した低ランク差分学習を行える
- `serialize.py` は従来互換 `nn.bin` しか出力しないため、`context_embedding.*` は export 時に明示的に除外する
- LoRA adapter は `W + BA` として feature transformer 重みに畳み込めるため、static export に反映される

制約:

- 初期実装では候補生成を batch 内で逐次実行するため、速度はまだ最適化していない
- ローカル確認環境に `torch` がない場合、構文確認のみ可能
- 現段階の static export は条件付き補正を直接埋め込まない。条件固定 export / 蒸留 export は後続スライスで検討する
- `model.base_ckpt` は初期化用であり、学習再開には Lightning の checkpoint resume を使う

## Deferred Plan: Tensor-Factorized Adapters

現段階では matrix LoRA を低ランク差分学習の first baseline とする。
HalfKP の構造を使った tensor-factorized adapter は、fixed-ref route の有効性確認後の後続スライスとして扱う。

現在の matrix LoRA:

```text
W_eff = W0 + (alpha / r) B A
```

後続候補:

- `halfkp_tensor_cp`
- `halfkp_tensor_tucker`
- context-conditioned tensor adapter

設計制約:

- HalfKP 入力を flat vector ではなく `king square x piece plane` 構造として扱う
- adapter は feature transformer 側に限定する
- export 時には dense な `W_eff` に畳み込めることを必須条件にする
- matrix LoRA と同じ `input_adapter` 切替で実験できるようにする

想定スライス:

### Slice T0: Tensor adapter design

- HalfKP の `king square x piece plane` 構造を明示する
- CP / Tucker のどちらを先に実装するかを決める
- export 時の dense fold-in 条件を定義する

### Slice T1: CP adapter prototype

CP 分解:

```text
DeltaW[o,k,p] = sum_r A[o,r] B[k,r] C[p,r]
```

- `input_adapter=halfkp_tensor_cp` として追加する
- まず context 非依存の adapter として matrix LoRA と比較する

### Slice T2: Context-conditioned tensor adapter

context によって rank component を重み付けする。

```text
DeltaW_c[o,k,p] = sum_r g_r(c) A[o,r] B[k,r] C[p,r]
```

- Elo bucket と player を同じ context interface で扱う
- 条件固定 export または蒸留 export を前提にする

### Slice T3: Export strategy

- 条件なし export:
  - `W0 + average(DeltaW_c)`
- 条件固定 export:
  - `W0 + DeltaW_c`
- 蒸留 export:
  - 条件付きモデルの出力を static NNUE に蒸留する

この計画は、現行 fixed-ref 実装の matrix LoRA を置き換えるものではなく、
有効性確認後に追加する比較実験として扱う。
