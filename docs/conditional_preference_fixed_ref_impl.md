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

fixed-ref route の責務を先に固定する。

- dataset:
  - `psv` とメタを読む
  - `actual_move` と context を返す
- collate / batch builder:
  - 現局面の合法手を列挙する
  - `V0` で候補手を選ぶ
  - pairwise 学習に必要な `actual` / `ref` 遷移先を構成する
- model:
  - `V(s, c) = V0(s) + Delta(s, c)` を学習する

## First code slice

最初のコードスライスでは、まだ `V0` 候補選別までは入れない。
先に次を成立させる。

1. HDF5 を index 化して decision point を返せる
2. Elo / player の両方を同じ sample schema で返せる
3. 学習時に必要な最小メタを失わない

## Next slices

### Slice F1

- HDF5 fixed-ref dataset の追加
- lazy-open
- `context_type` 切替
- Elo bucket 化

### Slice F2

- fixed-ref collate / batch builder
- `V0` 推論用の current / actual / candidate SFEN 構成

### Slice F3

- model 側に route-gated preference loss を追加
- Elo 条件を先に通す

### Slice F4

- player 条件を同じ経路に追加
- context encoder の共有を確認する
