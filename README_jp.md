# nnue-pytorch README (Japanese)

英語版の原本は [README.md](/Users/akio/Documents/GitHub/nnue-pytorch/README.md) です。  
このファイルは、普段の導入と学習に必要な内容を日本語でまとめたものです。

## 現在の学習方針

- Optimizer: SGD + momentum + warmup (`model.num_batches_warmup`)
- 任意の安定化: EMA (`model.ema_*`)
- 任意の損失調整: `model.teacher_temperature`, `model.entropy_coef`, `model.outcome_pos_weight`
- 任意の CORN 補助損失: `model.corn_aux_weight`, `model.corn_aux_thresholds`
- 任意の構造補助損失 (`py_data=true`): `model.king_zone_aux_weight`, `model.major_safety_aux_weight`
- 任意のサンプリング拡張 (`py_data=true`): `data.py_data_sampling_mode` (`uniform` / `ply_balanced`)

日本語の学習方針ガイドは [docs/training_strategy_guide_ja.md](/Users/akio/Documents/GitHub/nnue-pytorch/docs/training_strategy_guide_ja.md) を参照してください。

## Ubuntu / WSL でのセットアップ

### 事前に必要な apt パッケージ

高速 DataLoader は C++ / CMake ビルドなので、Python 仮想環境だけでは足りません。

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  git \
  python3-dev
```

`uv` が未導入なら:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### `uv` で仮想環境を作る

PyTorch は CPU / CUDA / ROCm ごとに wheel が異なるので、仮想環境も分けてください。

#### CPU

```bash
uv venv .venv-cpu
source .venv-cpu/bin/activate
uv pip install torch torchvision torchaudio
uv pip install -r requirements.txt
```

#### CUDA

PyTorch の GPU wheel は環境により変わるので、基本は公式 selector を参照してください。  
https://pytorch.org/get-started/locally/

例: CUDA 12.8

```bash
uv venv .venv-cu128
source .venv-cu128/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

#### ROCm

ROCm は Linux 専用です。ホスト側に ROCm を導入した上で、専用の仮想環境を作ってください。

例: ROCm 6.3

```bash
uv venv .venv-rocm63
source .venv-rocm63/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
uv pip install -r requirements.txt
```

## `cshogi` について

通常の `nnue-pytorch` 学習パスでは `cshogi` は必須ではありません。
そのため、共通依存の `requirements.txt` にも含めていません。

一方で、`shogi_ai/wsl2/src/create_dataset.py` など `cshogi` に依存するデータ生成フローを使う場合は、その環境に必要な fork 版 `cshogi` を別途インストールしてください。
このリポジトリでは、`cshogi` は `nnue-pytorch` 本体の基本依存ではなく、データ生成パイプライン側の依存として扱います。

## Docker / Apptainer での導入

### Docker

ホストにビルドツールを入れたくない場合は、リポジトリを bind mount して作業できます。

CPU 例:

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  ubuntu:24.04 bash
```

コンテナ内で:

```bash
apt update
apt install -y build-essential cmake ninja-build pkg-config git curl python3 python3-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"
uv venv .venv-cpu
source .venv-cpu/bin/activate
uv pip install torch torchvision torchaudio
uv pip install -r requirements.txt
sh compile_data_loader.sh
```

CUDA 例:

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  ubuntu:24.04 bash
```

この場合も、コンテナ内の PyTorch wheel は CUDA 用を選んでください。  
ホスト側には NVIDIA driver と `nvidia-container-toolkit` が必要です。

### Apptainer

共有サーバーでは Docker より Apptainer の方が使いやすいことがあります。

```bash
apptainer shell \
  --bind "$PWD":/workspace \
  --pwd /workspace \
  docker://ubuntu:24.04
```

GPU を使う場合:

- CUDA: `--nv`
- ROCm: `--rocm`

例:

```bash
apptainer shell --nv --bind "$PWD":/workspace --pwd /workspace docker://ubuntu:24.04
```

コンテナ内のセットアップは Docker と同じです。CPU / CUDA / ROCm で PyTorch wheel が違うので、仮想環境も分けてください。

## 高速 DataLoader のビルド

学習に使う仮想環境を activate した状態でビルドします。

Linux / macOS / WSL:

```bash
source .venv-cpu/bin/activate
sh compile_data_loader.sh
```

Windows:

```bash
bash compile_data_loader.sh
```

## 学習

```bash
source .venv-cpu/bin/activate
python train.py train_data.bin val_data.bin
```

### checkpoint から再開

```bash
python train.py --resume_from_checkpoint <path> ...
```

### GPU を使う

```bash
python train.py --gpus 1 ...
```

### feature set

デフォルトは factorized HalfKP (`HalfKP^`) です。

```bash
python train.py ... --features="HalfKP^"
```

## `shogi_ai` と組み合わせる場合

`.bin` を `shogi_ai/wsl2/src/create_dataset.py` で作るなら、局面選別は `shogi_ai` 側を source of truth としてください。

- quiet / tactical の分離: `classify-sfen` や `generate --quiet-level`
- SFEN 頻度調整: `count-sfen` と `generate --sfen-sampling-mode`
- `nnue-pytorch` 側では追加の skip を基本的に入れない

推奨例:

```bash
python train.py --batch-size 16384 --threads 8 --num-workers 8 --gpus 1 trainingdata validationdata
```

原則として、以下は生データを直接読む場合以外は無効のままにしてください。

- `--smart-fen-skipping`
- `--random-fen-skipping`

## CORN 補助損失の閾値

`model.corn_aux_thresholds` は value loss に cumulative ordinal な補助損失を追加するための閾値です。
基本は、実際に `train.py` が読む `.bin` 分布から作るのが自然です。

```bash
python corn_thresholds.py --input-bin /path/to/train.bin --num-thresholds 7 --weight 0.1
```

`shogi_ai` を使う場合は、`generate` と同じ頻度補正を反映した閾値を作る方が整合的です。

```bash
python shogi_ai/wsl2/src/create_dataset.py corn-thresholds \
  --input-csv eval_sfen.csv \
  --sfen-count-csv sfen_counts.csv \
  --sfen-sampling-mode sqrt \
  --sfen-sampling-min-freq 2 \
  --num-thresholds 7 \
  --score-scaling 361 \
  --teacher-temperature 1.0 \
  --corn-aux-weight 0.1
```

## 補足

その他の補助スクリプト、可視化、シリアライズ、対局自動化などの詳細は英語版 [README.md](/Users/akio/Documents/GitHub/nnue-pytorch/README.md) を参照してください。
