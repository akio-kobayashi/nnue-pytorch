# Conditional Preference Learning Slices

## Goal

最終成果物は従来互換の `YaneuraOu` 用 `nn.bin` とする。
学習時のみ `shogi_ai/wsl2` の `.h5` に含まれるメタ情報を用い、`nnue-pytorch` 側で
条件付き選好学習を実装する。

## Current Assumptions

- `shogi_ai/wsl2` の `build-h5` は対局メタを `game_group.attrs` に保存する
- 各局面には `psv`、`actual_move`、`candidates` が保存される
- `.bin` 生成経路は既存のままとし、HDF5 拡張の影響を受けない
- 選好学習のサンプル単位は独立局面ではなく「意思決定点」とする

## Data Model

1 サンプルは以下で構成する。

- pre-state: `s_t`
- actual transition: `actual_move`, `s_t_actual`
- alternative transitions: `candidate_moves`, `s_t_alt[j]`
- context:
  - player context: current player name
  - elo context: current player elo bucket
- meta:
  - `ply`
  - `game_result`
  - `black_player`, `white_player`
  - `rating_b`, `rating_w`
  - `file_path`, `kif_index`

## Implementation Slices

### Slice 0: HDF5 actual move persistence

目的:

- `build-h5` の各局面に棋譜の実着手を明示保存する

完了条件:

- `positions[*].actual_move` が存在する
- `.bin` 生成経路に変更が入っていない

Status:

- done

### Slice 1: PackedSfen adapter

目的:

- HDF5 内の `psv` 40 バイトを decode し、`cshogi.Board` を復元する

完了条件:

- `positions[i]['psv']` から `Board` を復元できる
- 既存 `feature_set.get_active_features()` に通せる

メモ:

- 最初は正しさ優先でよい
- HDF5 ハンドルは worker ごと lazy open にする

### Slice 2: Sequential decision-point dataset

目的:

- 独立局面ではなく、意思決定点を 1 サンプルとして返す Dataset を実装する

返り値:

- `s_t`
- `s_t_actual`
- `s_t_alt[j]`
- `context`
- `meta`

完了条件:

- `actual_move` から実遷移を生成できる
- `candidates[*].move` から代替遷移を生成できる
- current player の名前と Elo を context として返せる

### Slice 3: DataLoader and collate

目的:

- 重い HDF5 読み込みと遷移生成を DataLoader 前提で回せるようにする

完了条件:

- `num_workers > 0` で動く
- `persistent_workers=True` で安定動作する
- 可変長 alternatives を `collate_fn` でまとめられる

メモ:

- まずは `num_workers=0` で正しく動くことを優先する
- 速度不足なら前展開済み training shard を後段で検討する

### Slice 4: Unified context interface

目的:

- player name と Elo を同じ条件付き選好インターフェースで扱う

設計:

- `context_type in {player, elo}`
- `context_value` を encoder に渡して埋め込み化する
- `elo` は連続値ではなく帯域分割した bucket id として扱う

完了条件:

- player 指定と Elo 指定で同じ Dataset/forward/loss を通せる

### Slice 5: Pairwise preference loss

目的:

- 実際に選ばれた遷移先と候補遷移先の比較学習を追加する

基本形:

- `V(s_t_actual, c) > V(s_t_alt, c)`

完了条件:

- player preference loss と elo preference loss が同じコードパスで計算できる
- 既存 value loss と併用できる

定式化:

- 意思決定点 `i` について
  - 現局面: `s_i`
  - 実着手遷移先: `s_i_actual = T(s_i, a_i_actual)`
  - 深さ `d` 探索の最善候補遷移先: `s_i_best = T(s_i, a_i_best)`
- 条件付き評価関数:
  - `V(s, c) = V0(s) + Δ(s, c)`
- 基本 pairwise loss:
  - `l_pref(i; c) = -log sigmoid(beta * (V(s_i_actual, c) - V(s_i_best, c)))`

player preference 用マスク:

- `m_opening(i) = 1[p_min <= ply_i <= p_max]`
- `m_quiet(i) = 1[in_check_i = 0] * 1[tactical_i = 0]`
- `m_close(i) = 1[abs(q_i^(1) - q_i^(2)) <= tau_q]`
- `m_player(i) = m_opening(i) * m_quiet(i) * m_close(i)`

player preference loss:

- `L_player_pref = (sum_i m_player(i) * l_pref(i; c_i_player)) / (sum_i m_player(i) + eps)`

elo preference 用マスク:

- `m_elo(i) = 1[p_min_elo <= ply_i <= p_max_elo] * 1[valid_i = 1]`

elo preference loss:

- `L_elo_pref = (sum_i m_elo(i) * w_elo(i) * l_pref(i; c_i_elo)) / (sum_i m_elo(i) + eps)`

全体損失:

- `L = L_value + alpha_elo * L_elo_pref + alpha_player * L_player_pref + lambda_theta * Omega_theta`

実装上の最初の簡約版:

- `m_player(i) = 1[8 <= ply_i <= 40] * 1[in_check_i = 0]`
- 比較対象は `actual_move` と `candidates[0]` のみ
- `w_elo(i) = 1`

`model.py` への関数展開:

- `compute_pref_logits(actual_value, best_value, beta)`
  - `beta * (actual_value - best_value)` を返す
- `compute_pref_loss(actual_value, best_value, beta)`
  - pairwise logistic loss を返す
- `compute_player_pref_mask(batch)`
  - `ply`、`in_check`、必要なら `tactical` と top1-top2 差から `m_player` を返す
- `compute_elo_pref_mask(batch)`
  - `ply` と有効性条件から `m_elo` を返す
- `compute_player_pref_loss(batch, outputs)`
  - `m_player` を掛けた `L_player_pref` を返す
- `compute_elo_pref_loss(batch, outputs)`
  - `m_elo` と `w_elo` を掛けた `L_elo_pref` を返す
- `compute_multitask_loss(batch, outputs)`
  - `L_value`、`L_elo_pref`、`L_player_pref`、正則化を合成して最終 loss を返す

関数分離の意図:

- mask の定義を loss 本体から分離する
- player と elo のコードパスを最大限共通化する
- 最初は `actual vs best` の2手比較に限定し、後から top-k 比較へ拡張しやすくする

### Slice 6: Training integration

目的:

- 既存学習フローに選好学習を統合する

完了条件:

- `value / elo_pref / player_pref` を別ログで確認できる
- 既存設定を壊さず feature flag で切り替えられる

### Slice 7: Static NNUE export

目的:

- 学習中は条件付きでも、最終的に従来互換の static `nn.bin` に落とす

候補:

- 条件固定 export
- 蒸留 export

完了条件:

- `YaneuraOu` 側に追加の条件入力を要求しない

## Alternate Route: Fixed-Reference Static Preference

探索ベース候補生成は本線として維持するが、計算量対策として別ルートを独立に持つ。
この別ルートでは、学習時に外部エンジンを呼ばず、固定参照モデル `V0` の即値だけを使う。

定義:

- 参照モデル:
  - `V0`
- 学習対象:
  - `V(s, c) = V0(s) + Delta(s, c)`
- 候補スコア:
  - `U_ref(s, a) = V0(T(s, a))`

設計意図:

- 学習中の `nnue-pytorch` から YaneuraOu エンジンを呼ばない
- 学習中モデル自身を候補生成に使う自己参照を避ける
- `V0` は固定、`Delta` だけが条件付き選好を学ぶ

### Slice A0: Fixed-reference route spec

目的:

- 本線の search-based route とは別に、fixed-reference static route の入出力仕様を定める

入出力:

- input:
  - `s_t`
  - `actual_move`
  - `context`
  - optional metadata
- derived inside training:
  - `s_t_actual = T(s_t, actual_move)`
  - `candidate_moves_ref`
  - `s_t_ref[j] = T(s_t, candidate_moves_ref[j])`

完了条件:

- `create_dataset.py` に探索値保存を追加しなくても route が成立する
- route A と route B の切替点が `candidate source / score source` で明示されている

### Slice A1: Legal-move expansion and reference scoring

目的:

- 学習時に現局面 `s_t` から合法手を列挙し、固定参照モデル `V0` で 1 手後局面を採点する

処理:

- `s_t` から合法手 `a` を列挙
- 各 `a` について `s_t_a = T(s_t, a)` を生成
- `V0(s_t_a)` を計算
- 上位 `k` 手を `candidate_moves_ref` として採用

完了条件:

- 外部探索器なしで候補手集合を作れる
- `actual_move` が候補上位に入らない場合も学習サンプルを構成できる

メモ:

- 最初は top-1 のみでよい
- 速度不足時のみ合法手全列挙からヒューリスティクス前選別を入れる

### Slice A2: Fixed-reference pairwise dataset path

目的:

- HDF5 の `psv` と `actual_move` だけから、fixed-reference route 用の pairwise サンプルを構成する

返り値:

- `s_t`
- `s_t_actual`
- `s_t_ref_best`
- `context`
- `meta`

完了条件:

- `candidates` フィールドが空でも route が動く
- route A の search-based dataset 実装と大部分のコードを共通化できる

### Slice A3: Fixed-reference preference loss integration

目的:

- `V0` で候補を選び、`V0 + Delta` で pairwise 学習する loss を既存学習系へ追加する

基本形:

- `U_ref(s, a) = V0(T(s, a))`
- `l_pref_ref(i; c) = -log sigmoid(beta * (V(s_i_actual, c) - V(s_i_ref_best, c)))`

完了条件:

- 候補選別に使う関数と学習対象の関数が分離されている
- `context_type = elo` で route B が先に完結できる

### Slice A4: Route toggle and experiments

目的:

- search-based route と fixed-reference route を設定で切り替えられるようにする

設定例:

- `preference_route = search`
- `preference_route = fixed_ref`

比較項目:

- HDF5 生成時間
- 学習 1 epoch 時間
- pairwise accuracy
- 条件再現性
- 最終 static `nn.bin` への悪影響の有無

完了条件:

- route B を Elo 条件のみで先行実装できる
- route A と route B の比較実験が可能になる

### Slice A5: Matrix LoRA baseline

目的:

- fixed-reference route の最初の低ランク差分学習として matrix LoRA を使う

定義:

- `W_eff = W0 + (alpha / r) B A`
- 適用先は feature transformer input weight を基本とする
- `freeze_base_input = true` により `W0` を保ち、adapter のみを主に学習する

完了条件:

- `input_adapter = halfkp_lora` で学習できる
- export 時に `W_eff` を static NNUE の feature transformer weight へ畳み込める

### Slice A6: Deferred tensor-factorized adapters

目的:

- HalfKP の `king square x piece plane` 構造を使った tensor-factorized adapter を後続実験として導入する

候補:

- `input_adapter = halfkp_tensor_cp`
- `input_adapter = halfkp_tensor_tucker`
- context-conditioned tensor adapter

制約:

- 現行 fixed-ref の有効性確認後に実装する
- matrix LoRA と同じ `input_adapter` 切替で比較可能にする
- export 時に dense `W_eff` へ畳み込めることを必須条件とする

CP 分解案:

- `DeltaW[o,k,p] = sum_r A[o,r] B[k,r] C[p,r]`

context-conditioned 案:

- `DeltaW_c[o,k,p] = sum_r g_r(c) A[o,r] B[k,r] C[p,r]`

export 案:

- 条件なし export:
  - `W0 + average(DeltaW_c)`
- 条件固定 export:
  - `W0 + DeltaW_c`
- 蒸留 export:
  - 条件付きモデルの出力を static NNUE に蒸留する

## Evaluation

評価はミクロ評価とマクロ評価の 2 段で行う。

### Data split

- プレーヤーごとに棋譜を時系列順に並べる
- 各プレーヤー内で `train / val / test` に分割する
- 評価は `test` のみで行う
- 個人選好評価は主に序盤から中盤入口を用いる
  - 例: `8 <= ply <= 40`
  - 王手中や強戦術局面は除外する

### Micro evaluation

各評価局面について、以下を比較する。

- `actual_move`
- 深さ `d` 探索の `best_move`

主指標:

- pairwise accuracy
  - `V(s_actual, c_player) > V(s_best, c_player)` となる率
- top-k hit rate
  - 実着手が候補上位 `k` に入る率
- mean rank
  - 実着手の順位平均
- true-vs-other margin
  - 真のプレーヤー context のときの実着手スコアが、他プレーヤー context よりどれだけ高いか

解釈:

- ミクロ評価は「その人の手を局面ごとに再現できるか」を測る

### Macro evaluation

教師なしクラスタを以下で定義する。

- クラスタ写像: `z = C(x) in {1, ..., K}`
- `x` は序盤局面特徴、遷移特徴、あるいは候補手適用後局面特徴

プレーヤー `p` の実データ分布:

- `P_real^(p)(z) = (1 / N_p) * sum_i 1[C(x_i_act) = z]`

プレーヤー `p` のモデル分布:

- `P_model^(p)(z) = (1 / N_p) * sum_i 1[C(x_i_pred) = z]`

学習前モデルと学習後モデルを分ける場合:

- `P_base^(p)(z)`
- `P_adapt^(p)(z)`

推奨距離:

- Jensen-Shannon divergence
- `D_JS^(p) = JS(P_model^(p) || P_real^(p))`

学習前後の改善量:

- `Delta_cluster^(p) = JS(P_base^(p) || P_real^(p)) - JS(P_adapt^(p) || P_real^(p))`

解釈:

- `Delta_cluster^(p) > 0` なら学習後モデルの方が実棋譜分布に近い
- マクロ評価は「その人らしい棋風分布を再現できたか」を測る

### Success criteria

成功条件は以下の両方を満たすこととする。

- ミクロ指標が改善する
  - pairwise accuracy 向上
  - mean rank 改善
  - true-vs-other margin 拡大
- マクロ指標が改善する
  - `D_JS` が低下する
  - `Delta_cluster > 0` となるプレーヤーが多数を占める

### Reporting

最低限、以下を出力する。

- プレーヤー別の pairwise accuracy
- プレーヤー別の mean rank
- プレーヤー別の true-vs-other margin
- プレーヤー別の `D_JS`
- 学習前後での平均改善量

補足:

- プレーヤーごとの棋譜数に偏りがあるため、単純平均と件数重み付き平均の両方を見る
- 初期段階ではミクロ評価を主、マクロ評価を補助とする

## Order

実装は 2 段階に分ける。

### Stage 1: Elo bucket conditional learning

最初の実装段階では `elo bucket` のみを扱い、`player` 条件はまだ導入しない。

順序:

1. Slice 1
2. Slice 2
3. Slice 3
4. Slice 4
   - ただし `elo bucket` のみ実装する
5. Slice 5
   - ただし `L_elo_pref` のみ実装する
6. Slice 6
   - `value + elo_pref` の統合まで行う
7. Slice 7
   - Elo 条件から static NNUE への落とし方を確認する

この段階では、以下を完了条件とする。

- `elo bucket` 条件で Dataset / forward / loss / evaluation が一通り動く
- `player` に依存しない形で条件付き選好学習の有効性を確認できる

推奨ブランチ:

- 現ブランチを `elo` 実装用として使う
- 例: `codex/elo-conditional-preference`

### Stage 2: Player conditional learning

`Elo` 段階が安定した後、新しいブランチを切って `player` 条件を追加する。

順序:

1. Stage 1 完了時点から新ブランチを切る
2. Slice 4 を拡張し、`player` context を追加する
3. Slice 5 を拡張し、`L_player_pref` を追加する
4. Slice 6 を拡張し、`value + elo_pref + player_pref` を統合する
5. Evaluation に `true-vs-other margin` と player 別指標を追加する

この段階では、以下を完了条件とする。

- `player` 条件が `elo bucket` 条件と同じコードパスで動く
- `player` 追加による改善が、`elo` のみの場合よりも確認できる

推奨ブランチ:

- `Elo` 実装の安定版から切り直す
- 例: `codex/player-conditional-preference`

## Non-goals For First Pass

- `YaneuraOu` 側の推論コード改修
- `.bin` 学習経路の置き換え
- 完全な高速化最適化

## Notes

- HDF5 は教師生成と研究用の中間形式と考える
- 学習時にのみメタ情報を使い、配布モデルは従来互換に保つ
- 最初の PoC では速度よりデータ整合性を優先する
