# AMD Zen2+ / NVIDIA CUDA 12.4 実験パイプライン

この手順は以下を一気通貫で実行します。

1. Docker 上に PyTorch 実験環境を構築（`uv` 仮想環境）
2. `nnue-pytorch` で学習（YaneuraOu デフォルト形状: `1024-8-96`, `num_buckets=8`）
3. `serialize.py` で YaneuraOu 用 `nn.bin` をエクスポート
4. ホスト側で YaneuraOu を AMD 最適化（`TARGET_CPU=ZEN2`）でビルド
5. 学習済み `nn.bin` を読み込んで USI スモークテスト

## 前提

- Linux x86_64
- Docker + NVIDIA Container Toolkit（`docker run --gpus all` が通る）
- AMD Zen2 以降の CPU
- NVIDIA GPU（CUDA 12.4 系ドライバ）
- 学習データ（`bin` / `binpack`）

## 追加されたスクリプト

- `scripts/experiment/build_image.sh`
- `scripts/experiment/train_export_in_docker.sh`
- `scripts/experiment/train_export_inner.sh`
- `scripts/experiment/build_yaneuraou_amd.sh`
- `scripts/experiment/test_yaneuraou_with_nnue.sh`
- `scripts/experiment/full_pipeline.sh`

## 最短実行

```bash
cd nnue-pytorch

# 例: 1epoch の疎通確認
bash scripts/experiment/full_pipeline.sh \
  data/train.binpack \
  data/val.binpack \
  experiments/artifacts \
  1
```

成功すると以下が出力されます。

- チェックポイント: `experiments/artifacts/run_.../...ckpt`
- エクスポート済みモデル: `experiments/artifacts/export/nn.bin`
- マニフェスト: `experiments/artifacts/latest_manifest.env`

## 段階実行

### 1) Docker イメージだけ先に作る

```bash
bash scripts/experiment/build_image.sh
```

### 2) 学習 + エクスポート（Docker 内）

```bash
bash scripts/experiment/train_export_in_docker.sh \
  data/train.binpack \
  data/val.binpack \
  experiments/artifacts \
  5
```

### 3) YaneuraOu を Zen2 最適化ビルド

```bash
TARGET_CPU=ZEN2 bash scripts/experiment/build_yaneuraou_amd.sh
```

### 4) YaneuraOu で `nn.bin` ロードテスト

```bash
bash scripts/experiment/test_yaneuraou_with_nnue.sh \
  YaneuraOu/source/YaneuraOu-by-gcc \
  experiments/artifacts/export/nn.bin
```

## ハッシュ不一致時

YaneuraOu 側の `kHashValue` と `serialize.py` の `--yane-network-hash` が不一致だと読み込みに失敗します。
必要なら以下のように指定して再実行してください。

```bash
export YANE_NETWORK_HASH=0xXXXXXXXX
bash scripts/experiment/train_export_in_docker.sh \
  data/train.binpack data/val.binpack experiments/artifacts 5
```

## 注意

- `train_export_in_docker.sh` は単純化のため、入力データと出力先をリポジトリ配下に限定しています。
- 学習時間短縮の初期確認は `max_epochs=1` を推奨します。
