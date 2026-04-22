# Conditional Preference Learning Background

## Purpose

本メモは、条件付き選好学習の設計に関する背景、関連研究、今回の手法の新規性、
および評価設計の要点を、参照リンク付きで整理した資料である。

対象は以下の問題設定である。

- 将棋プレーヤーの棋譜に現れる個人差や強さ差をモデル化したい
- 学習時にはメタ情報を活用したい
- 最終成果物は従来互換の `YaneuraOu` 用 `nn.bin` にしたい

## Problem Setting

今回の基本方針は以下である。

- ベース評価:
  - `V(s, c) = V0(s) + Delta(s, c)`
- `s`:
  - 局面
- `c`:
  - 条件コンテキスト
  - `player` または `elo bucket`

ここで重要なのは、最終成果物として配布するモデルは static NNUE としつつ、
学習時のみ条件付き補正 `Delta(s, c)` を導入する点である。

## Why Preference Learning

従来の value 学習は、局面を独立サンプルとして扱うことが多い。
しかし個人選好を見たい場合、学習単位は独立局面ではなく「意思決定点」に変わる。

1 サンプルは以下で構成される。

- pre-state:
  - `s_t`
- actual transition:
  - `a_t_actual`, `s_t_actual`
- alternative transition:
  - `a_t_best`, `s_t_best`
- context:
  - `player` または `elo`

このため、損失も局面ごとの回帰ではなく、遷移先どうしの pairwise 比較が自然である。

## Related Work

### 1. Individual move prediction in shogi

最も近い将棋の先行研究として、
山内智晴・鶴岡慶雅「将棋における個人に適応した着手推定モデルの構築」がある。

この研究は、

- 特定プレーヤーの小規模な棋譜
- 大規模な一般棋譜

を併用して評価関数を学習し、個人向けの着手推定精度を改善している。
さらに、探索深さにもプレーヤー差を反映させる発想を含む。

この研究は次の点で今回の設計を直接支える。

- 棋譜には個人差が現れる
- 個人適応は着手予測精度を改善しうる
- 評価関数だけでなく探索条件も個人差に関係する

References:

- [CiNii: Developing Move Prediction Models for Individual Players in Shogi](https://cir.nii.ac.jp/crid/1050574047106947840)
- [IPSJ PDF: 将棋における個人に適応した着手推定モデルの構築](https://ipsj.ixsq.nii.ac.jp/record/175347/files/IPSJ-GPWS2016018.pdf)

### 2. Human move prediction and personalization in chess

McIlroy-Young らによる Maia 系研究は、
「最善手を出す AI」ではなく「人が実際に選ぶ手を当てる AI」という方向を強く打ち出した。

主なポイントは次である。

- 人間の granular decision-making を直接モデル化する
- レーティング帯ごとに人間らしい手を予測する
- 個人の棋譜で fine-tuning すると、さらに個別の着手予測精度が上がる

この流れは今回の設計に対して、

- move prediction を主要評価とする妥当性
- 個人化前後の比較を取る妥当性
- 強さ別モデルと個人別モデルを比較する妥当性

を与える。

References:

- [Aligning Superhuman AI with Human Behavior: Chess as a Model System](https://www.microsoft.com/en-us/research/?p=706345)
- [KDD 2020 accepted paper page](https://www.kdd.org/kdd2020/accepted-papers/view/aligning-superhuman-ai-with-human-behavior-chess-as-a-model-system.html)
- [Project Maia technical deep dive](https://www.microsoft.com/en-us/research/project/project-maia/technical-deep-dive/?lang=zh-cn)
- [Maia Chess project page](https://www.maiachess.com/)

### 3. Move prediction difficulty and human-centered evaluation

将棋では、「着手予測モデルが予測しづらい局面の考察・分類と確信度を利用した一致率の向上」
のように、人間らしい着手予測モデルの難しさや評価を扱う研究もある。

この流れは今回の設計に対して、

- どの局面を評価に使うべきか
- 難しい局面とそうでない局面を分ける必要性
- 一致率だけでなく、予測困難性や局面分類が重要であること

を示唆する。

References:

- [CiNii: Consideration and classification of positions that are difficult for move prediction models to predict](https://cir.nii.ac.jp/crid/1050856970555541376)
- [JAIST repository entry](https://dspace.jaist.ac.jp/dspace/handle/10119/18237?mode=full)

### 4. Broader player modeling background

ゲーム AI における player modeling / opponent modeling は、
プレーヤーの傾向や戦略を予測して、それに合わせて行動する研究領域である。

今回の研究は完全情報ゲームかつオフライン棋譜学習という点で特殊だが、
「プレーヤーの行動分布をモデル化する」こと自体は広い player modeling の流れに位置づく。

References:

- [Game Player Modeling (overview)](https://www.researchgate.net/publication/314599195_Game_Player_Modeling)

## What Is New In This Work

今回の新規性は、既存研究の単なる焼き直しではなく、以下の組み合わせにある。

### 1. NNUE-compatible conditional preference learning

既存研究は主として

- 着手予測モデル
- policy モデル
- 個人向け fine-tuning

として提示されている。

これに対し今回は、

- 学習時のみ条件付きモデルを使う
- 最終配布物は従来互換の static NNUE とする

という構造を採る。

これは `YaneuraOu` 互換性を保ったまま個人差や強さ差を学習に取り込むという点で実装的に新しい。

### 2. Unified treatment of player and Elo

通常、個人差と強さ差は別問題として扱われがちである。
今回の設計では両者を同じ条件変数 `c` に統一する。

- `c = player`
- `c = elo bucket`

これにより、

- Dataset
- forward
- loss
- 評価

を共通化できる。

### 3. Decision-point based learning instead of independent position learning

従来の NNUE 学習は局面独立の value 学習が中心である。
今回の設計では、学習単位を

- 現局面
- 実着手遷移先
- 探索 best 遷移先

からなる意思決定点に置き換える。

これは「局面の強さ」ではなく「どの遷移先を選ぶか」を学ぶための再定式化である。

### 4. Separation of strength and style through opening-focused masks

個人選好は終盤の最善手強制局面よりも、

- 序盤
- 駒組
- 複数の自然な構想が存在する局面

で現れやすいという仮説を明示的に採用する。

これにより、

- Elo 条件選好
- プレーヤー条件選好

を同じ loss 形式で持ちつつ、局面マスクで役割分担できる。

### 5. Two-level evaluation: micro and macro

評価は以下の 2 段に分ける。

- micro:
  - actual vs best の再現性
  - pairwise accuracy
  - mean rank
  - true-vs-other margin
- macro:
  - 教師なしクラスタ分布
  - Jensen-Shannon divergence

これにより、「局面ごとに当てる」ことと「棋風全体を再現する」ことを分けて評価できる。

## Evaluation Design

### 1. Data split

- プレーヤーごとに棋譜を時系列で並べる
- 各プレーヤー内で `train / val / test` に分割する
- 評価は `test` のみで行う
- 個人選好評価は主に序盤から中盤入口で行う

例:

- `8 <= ply <= 40`
- 王手中除外
- 強戦術局面除外

### 2. Micro metrics

各 test 局面について、

- `actual_move`
- 深さ `d` 探索の `best_move`

を比較する。

主要指標:

- pairwise accuracy
- top-k hit rate
- mean rank
- true-vs-other margin

### 3. Macro metrics

教師なしクラスタ写像 `z = C(x)` を作り、

- `P_real^(p)(z)`
- `P_model^(p)(z)`

の距離を Jensen-Shannon divergence で比較する。

改善量は

- `Delta_cluster^(p) = JS(P_base^(p) || P_real^(p)) - JS(P_adapt^(p) || P_real^(p))`

とする。

### 4. Why Jensen-Shannon divergence

Jensen-Shannon divergence を使う理由は次である。

- 対称である
- 0 頻度に対して KL より安定である
- 有界で解釈しやすい
- 離散クラスタ分布の比較に向く
- 棋風分布の「近さ」を測る目的に合う

## Practical Implications

今回の設計は、学習時にはメタ情報をフル活用しつつ、
配布時には static NNUE に戻すことで、研究と実運用の両立を狙うものである。

したがって、この研究の価値は次の 3 点にある。

1. 棋譜に現れる個人差・強さ差の検証
2. それを NNUE 互換の学習系へ統合する実装的枠組み
3. 局面単位と棋風分布単位の両面から再現性を評価すること

実装順としては、

- まず `elo bucket` 条件のみで条件付き選好学習を成立させる
- その後、別ブランチで `player` 条件を追加する

の 2 段階に分けるのが妥当である。

## Limits and Open Questions

現時点での限界や未確定点も明示しておく。

- NNUE の `Delta(s, c)` を static `nn.bin` にどう落とすかは追加設計が必要
- 個人名と Elo を同じ context として扱う設計の最適形は今後の実験で要確認
- クラスタが強さや戦型を主に反映してしまう場合、個人性評価が曖昧になる可能性がある
- HDF5 直読みによる学習コストが高いため、学習専用 shard 化を後段で検討する余地がある

## Recommended Citation Notes

論文や発表資料でこのテーマを説明する際には、少なくとも以下を引用候補に含めるとよい。

- 山内・鶴岡:
  - 将棋における個人適応着手推定の直接先行研究
- McIlroy-Young et al.:
  - 人間らしい着手予測および個人化の強力な先行例
- player modeling overview:
  - broader background の位置づけ

## Links Summary

- [Shogi individual move prediction (CiNii)](https://cir.nii.ac.jp/crid/1050574047106947840)
- [Shogi individual move prediction (IPSJ PDF)](https://ipsj.ixsq.nii.ac.jp/record/175347/files/IPSJ-GPWS2016018.pdf)
- [Maia / KDD 2020](https://www.microsoft.com/en-us/research/?p=706345)
- [KDD 2020 accepted paper page](https://www.kdd.org/kdd2020/accepted-papers/view/aligning-superhuman-ai-with-human-behavior-chess-as-a-model-system.html)
- [Project Maia technical deep dive](https://www.microsoft.com/en-us/research/project/project-maia/technical-deep-dive/?lang=zh-cn)
- [Maia Chess project page](https://www.maiachess.com/)
- [Move prediction difficulty in shogi (CiNii)](https://cir.nii.ac.jp/crid/1050856970555541376)
- [Move prediction difficulty in shogi (JAIST)](https://dspace.jaist.ac.jp/dspace/handle/10119/18237?mode=full)
- [Game Player Modeling overview](https://www.researchgate.net/publication/314599195_Game_Player_Modeling)
