# Conditional Preference Learning: Training and Loss Design

## Purpose

本メモは、条件付き選好学習の学習設計と損失関数を、

- 数式レベル
- 実装関数レベル
- 初期実験設定レベル

で参照できるように整理したものである。

対象は以下の設計である。

- 最終成果物は従来互換の static `nn.bin`
- 学習時のみ `player` / `elo` を条件コンテキストとして用いる
- 学習単位は独立局面ではなく意思決定点とする

## Model

評価関数は以下で定義する。

\[
V(s, c) = V_0(s) + \Delta(s, c)
\]

ここで

- \(s\):
  - 局面
- \(c\):
  - 条件コンテキスト
  - `player` または `elo bucket`
- \(V_0(s)\):
  - ベース評価
- \(\Delta(s, c)\):
  - 条件依存補正

解釈:

- `V0` は強さを担保する基礎評価
- `Delta` は人間の選好や強さ差に由来する補正を担う

## Decision Point

1 サンプルは意思決定点 `i` として定義する。

- 現局面:
  - \(s_i\)
- 実着手:
  - \(a_i^{act}\)
- 実着手後局面:
  - \(s_i^{act} = T(s_i, a_i^{act})\)
- 探索最善候補:
  - \(a_i^{best}\)
- 探索最善候補後局面:
  - \(s_i^{best} = T(s_i, a_i^{best})\)

必要に応じて top-k 候補を用いる場合は、

- \(a_{ij}^{alt}\)
- \(s_{ij}^{alt} = T(s_i, a_{ij}^{alt})\)

へ拡張する。

## Multitask Objective

全体損失は以下で定義する。

\[
L
=
L_{\mathrm{value}}
+
\alpha_{\mathrm{elo}} L_{\mathrm{elo\_pref}}
+
\alpha_{\mathrm{player}} L_{\mathrm{player\_pref}}
+
\lambda_{\Delta}\Omega_{\Delta}
+
\lambda_{\theta}\Omega_{\theta}
\]

ここで

- \(L_{\mathrm{value}}\):
  - 既存の value / outcome 混合損失
- \(L_{\mathrm{elo\_pref}}\):
  - Elo 条件選好損失
- \(L_{\mathrm{player\_pref}}\):
  - プレーヤー条件選好損失
- \(\Omega_{\Delta}\):
  - 条件依存補正の大きさ制約
- \(\Omega_{\theta}\):
  - 条件依存パラメータへの L2 正則化

初期段階では、簡約版として

\[
L
=
L_{\mathrm{value}}
+
\alpha_{\mathrm{elo}} L_{\mathrm{elo\_pref}}
+
\alpha_{\mathrm{player}} L_{\mathrm{player\_pref}}
\]

でもよい。

## Value Loss

`L_value` は既存 `nnue-pytorch` の主損失をそのまま使う。

すなわち、

- 検索評価値に対応する teacher 項
- 勝敗に対応する outcome 項

を混合した loss を再利用する。

実装上は既存の `_compute_primary_loss()` 相当をそのまま用いる。

## Pairwise Preference Loss

基本となる pairwise logistic loss は以下である。

\[
\ell_{\mathrm{pref}}(i;c)
=
-\log \sigma\left(
\beta \left(
V(s_i^{act}, c) - V(s_i^{best}, c)
\right)
\right)
\]

ここで

- \(\sigma\):
  - sigmoid
- \(\beta\):
  - 温度または鋭さ

解釈:

- 条件 \(c\) の下で
- 実着手後局面の評価が
- 探索最善候補後局面より高くなるように学習する

## Alternate Route: Fixed-Reference Static Preference

計算量対策として、探索ベース教師とは独立に、固定参照モデル `V0` の即値のみを用いる別ルートを置く。

このルートでは、学習時に外部エンジンを呼ばず、候補生成に学習中モデルそのものも使わない。
自己参照による不安定化を避けるため、候補選別は固定された `V0` によって行う。

### Reference utility

\[
U_{\mathrm{ref}}(s, a) = V_0(T(s, a))
\]

ここで

- \(T(s, a)\):
  - 局面 `s` に手 `a` を適用した遷移先局面

### Candidate construction

現局面 \(s_i\) に対し、合法手集合を \(\mathcal{A}(s_i)\) とする。

\[
a_i^{ref}
=
\arg\max_{a \in \mathcal{A}(s_i), a \neq a_i^{act}}
U_{\mathrm{ref}}(s_i, a)
\]

対応する候補後局面は

\[
s_i^{ref} = T(s_i, a_i^{ref})
\]

top-k に拡張する場合は、

\[
\mathcal{A}_k^{ref}(s_i)
=
\operatorname{TopK}_{a \in \mathcal{A}(s_i) \setminus \{a_i^{act}\}}
U_{\mathrm{ref}}(s_i, a)
\]

とする。

### Fixed-reference pairwise loss

学習対象は引き続き

\[
V(s, c) = V_0(s) + \Delta(s, c)
\]

であり、pairwise loss は

\[
\ell_{\mathrm{pref}}^{ref}(i;c)
=
-\log \sigma\left(
\beta \left(
V(s_i^{act}, c) - V(s_i^{ref}, c)
\right)
\right)
\]

と定める。

解釈:

- 候補の選別基準は固定された `V0`
- 学習されるのは条件付き補正を含む `V`
- 候補生成器と学習対象を分離することで、外部探索なしでも安定した軽量 route を構成する

### Route distinction

本メモでは pairwise 比較関数を次のように分ける。

- search-based route:
  - `U_search(s, a) = Q_d(s, a)`
- fixed-reference route:
  - `U_ref(s, a) = V0(T(s, a))`

両者は同じ pairwise loss の骨格を共有し、違いは

- 候補の生成元
- 比較に使う補助スコア

のみに限定する。

## Elo Preference Loss

Elo 条件コンテキストを \(c_i^{elo}\) とする。

ここで `elo` は連続値のまま扱わず、帯域分割した bucket id を用いる。

例:

- `[-inf, 1200)`
- `[1200, 1600)`
- `[1600, 2000)`
- `[2000, 2400)`
- `[2400, inf)`

実装上は `player` と同様に embedding lookup で扱えるようにする。

\[
L_{\mathrm{elo\_pref}}
=
\frac{1}{\sum_i m_i^{elo} + \varepsilon}
\sum_i
m_i^{elo}
\,
w_i^{elo}
\,
\ell_{\mathrm{pref}}(i;c_i^{elo})
\]

ここで

- \(m_i^{elo}\):
  - Elo 用局面マスク
- \(w_i^{elo}\):
  - Elo 用重み

初期実装では

\[
w_i^{elo} = 1
\]

としてよい。

必要なら不確実性重みを

\[
w_i^{elo}
=
\exp\left(
-\frac{|q_i^{(1)} - q_i^{(2)}|}{\tau_{\mathrm{elo}}}
\right)
\]

のように導入する。

## Player Preference Loss

プレーヤー条件コンテキストを \(c_i^{player}\) とする。

\[
L_{\mathrm{player\_pref}}
=
\frac{1}{\sum_i m_i^{player} + \varepsilon}
\sum_i
m_i^{player}
\,
\ell_{\mathrm{pref}}(i;c_i^{player})
\]

プレーヤー条件では、局面を序盤寄りに絞ることが重要である。

## Opening-Focused Masks

### Opening mask

\[
m_i^{opening}
=
\mathbf{1}[p_{\min} \le \mathrm{ply}_i \le p_{\max}]
\]

### Quiet mask

\[
m_i^{quiet}
=
\mathbf{1}[\mathrm{in\_check}_i = 0]
\cdot
\mathbf{1}[\mathrm{tactical}_i = 0]
\]

### Close-candidate mask

\[
m_i^{close}
=
\mathbf{1}[|q_i^{(1)} - q_i^{(2)}| \le \tau_q]
\]

### Final player mask

\[
m_i^{player}
=
m_i^{opening}
\cdot
m_i^{quiet}
\cdot
m_i^{close}
\]

### Final elo mask

\[
m_i^{elo}
=
\mathbf{1}[p_{\min}^{elo} \le \mathrm{ply}_i \le p_{\max}^{elo}]
\cdot
\mathbf{1}[\mathrm{valid}_i = 1]
\]

## Initial Simplified Training Setup

最初の実験では、以下の簡約版を採用する。

- 比較対象は `actual_move` と `candidates[0]` のみ
- player 用マスク:
  - `8 <= ply <= 40`
  - `in_check == 0`
- `w_elo = 1`
- top-k 比較はまだ導入しない

数式としては

\[
m_i^{player}
=
\mathbf{1}[8 \le \mathrm{ply}_i \le 40]
\cdot
\mathbf{1}[\mathrm{in\_check}_i = 0]
\]

とする。

## Regularization

### Delta magnitude penalty

\[
\Omega_{\Delta}
=
\frac{1}{N}
\sum_i \Delta(s_i, c_i)^2
\]

### Parameter L2 penalty

\[
\Omega_{\theta}
=
\|\theta_{\Delta}\|_2^2
\]

初期段階では \(\Omega_{\theta}\) のみでもよい。

## Recommended Initial Hyperparameters

初期値の推奨は以下とする。

- `alpha_elo = 0.1`
- `alpha_player = 0.05`
- `alpha_elo > alpha_player`
- `beta`:
  - おおむね `1/200` から `1/400` 相当
- `lambda_theta = 1e-5` 程度

理由:

- `player` の方がノイズが大きい
- `elo` の方が広い傾向差を学びやすい

## Function-Level Expansion for `model.py`

実装上は以下の関数分割を想定する。

### 1. `compute_pref_logits(actual_value, best_value, beta)`

役割:

- pairwise 比較用の logit

定義:

- `beta * (actual_value - best_value)`

### 2. `compute_pref_loss(actual_value, best_value, beta)`

役割:

- pairwise logistic loss を計算する

対応数式:

- `-log sigmoid(beta * (actual_value - best_value))`

### 3. `compute_player_pref_mask(batch)`

役割:

- player 用マスク `m_player` を返す

入力候補:

- `ply`
- `in_check`
- `tactical`
- top1-top2 差

### 4. `compute_elo_pref_mask(batch)`

役割:

- Elo 用マスク `m_elo` を返す

入力候補:

- `ply`
- 有効サンプルフラグ

### 5. `compute_player_pref_loss(batch, outputs)`

役割:

- player 用マスクを掛けた `L_player_pref` を返す

### 6. `compute_elo_pref_loss(batch, outputs)`

役割:

- Elo 用マスクと重みを掛けた `L_elo_pref` を返す

### 7. `compute_multitask_loss(batch, outputs)`

役割:

- `L_value`
- `L_elo_pref`
- `L_player_pref`
- 正則化

を合成して最終損失を返す

## Data Requirements

この損失を実装するために、Dataset は少なくとも以下を返す必要がある。

- `s_t`
- `s_t_actual`
- `s_t_best`
- `context_player`
- `context_elo`
- `ply`
- `in_check`
- 候補手評価差
- validity mask

トップ 1 のみを使う初期実装では、`candidates[0]` があれば足りる。

## Training Schedule

初期の段階的な学習戦略は以下でよい。

### Stage 1: Elo bucket only

1. `V0` の既存学習または既存重みを用意する
2. `L_value` のみで基本挙動を確認する
3. `L_elo_pref` を追加する
4. `value + elo_pref` のみで validation / evaluation を回す
5. 必要なら top-k 比較や Elo bucket 設計を調整する

### Stage 2: Add player condition on a new branch

1. Stage 1 の安定版から新ブランチを切る
2. `player` context を追加する
3. `L_player_pref` を追加する
4. `value + elo_pref + player_pref` で再評価する

意図:

- 先に `elo bucket` で条件付き選好学習の実装を成立させる
- その後 `player` を追加して差分を評価しやすくする

## Practical Notes

- 最初は速度より正しさを優先する
- top-k 比較や不確実性重みは後段でよい
- `player` と `elo` は共通 loss 形式を保つ
- `elo` は bucket 化し、`player` と同様のカテゴリ条件として扱う
- プレーヤー条件は序盤寄り、Elo 条件はより広い局面を扱う

## Relationship to Evaluation

この損失設計は、評価指標と次のように対応する。

- `L_player_pref`
  - 個人選好の pairwise accuracy 改善を狙う
- `L_elo_pref`
  - 強さ帯に応じた選好差の再現を狙う
- opening-focused masks
  - 個人差が出やすい局面に学習を集中させる
- multitask learning
  - 強さを維持しつつ選好再現性を上げる

## Open Design Questions

未確定の設計課題も残る。

- `Delta(s, c)` をどの層に入れるか
- `player` と `elo` の context encoder を完全共有するか
- static NNUE への export を条件固定で行うか蒸留で行うか
- top-k 候補比較をどの時点で導入するか

## Summary

今回の学習設計は、

- 既存 value 学習を保持しつつ
- 意思決定点単位の pairwise 選好学習を追加し
- `player` と `elo` を同じ条件付き枠組みで扱う

ものである。

最初は `actual vs best` の 2 手比較に限定し、
序盤駒組中心の player マスクを導入して PoC を進めるのがよい。
