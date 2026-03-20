# Logs (if required)
# Training Strategy Guide (Japanese)
- `docs/training_strategy_guide_ja.md` : 従来戦略と拡張戦略（EMA / 損失調整 / サンプリング）の学習ガイド

# Current training strategy
- Optimizer: SGD + momentum + warmup (`model.num_batches_warmup`)
- Optional stabilization: EMA (`model.ema_*`)
- Optional loss tuning: `model.teacher_temperature`, `model.entropy_coef`, `model.outcome_pos_weight`
- Optional CORN-style auxiliary loss: `model.corn_aux_weight`, `model.corn_aux_thresholds`
- Optional structural auxiliary losses (`py_data=true`): `model.king_zone_aux_weight`, `model.major_safety_aux_weight`
- Optional sampling extension (`py_data=true`): `data.py_data_sampling_mode` (`uniform` / `ply_balanced`)

# Setup

## Ubuntu / WSL prerequisites

The fast data loader is a C++/CMake build, so a Python virtual environment alone is not enough.
Install the toolchain first:

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

If `uv` is not installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new shell afterwards, or ensure `uv` is on `PATH`.

## Python environment with `uv`

PyTorch wheels differ between CPU / CUDA / ROCm, so keep separate virtual environments for each target.
Do not reuse one environment across backends.

### CPU environment

```bash
uv venv .venv-cpu
source .venv-cpu/bin/activate
uv pip install torch torchvision torchaudio
uv pip install -r requirements.txt
```

`requirements.txt` includes `jsonargparse[signatures]`, which is required by `pytorch-lightning`'s `LightningCLI`.
It also includes `tensorboard`, because the default config uses `TensorBoardLogger`.

### CUDA environment

As of March 19, 2026, the PyTorch selector at `https://pytorch.org/get-started/locally/` shows CUDA-specific pip indexes such as `cu118`, `cu126`, and `cu128`.
Pick the one matching your installed NVIDIA driver / CUDA runtime.

Example for CUDA 12.8:

```bash
uv venv .venv-cu128
source .venv-cu128/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

### ROCm environment

ROCm wheels are Linux-only. Install ROCm on the host first, then create a dedicated environment.

Example for ROCm 6.3:

```bash
uv venv .venv-rocm63
source .venv-rocm63/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
uv pip install -r requirements.txt
```

If you need a different backend or newer command, treat the official PyTorch install selector as the source of truth:
https://pytorch.org/get-started/locally/

## About `cshogi`

`nnue-pytorch` itself does not require `cshogi` for the normal training path.
The common Python dependencies in `requirements.txt` therefore do not include it.

If you use `shogi_ai/wsl2/src/create_dataset.py` or other upstream data-generation scripts that depend on `cshogi`, install the required fork separately in that environment.
In this repository, treat `cshogi` as a data-pipeline dependency, not a base `nnue-pytorch` dependency.

## Container-based setup

### Docker

If you do not want to install build tools on the host, you can work inside a container and bind-mount this repository.

CPU example:

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  ubuntu:24.04 bash
```

Inside the container:

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

CUDA example:

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  ubuntu:24.04 bash
```

Inside the container, use the same setup but install the CUDA wheel matching your target backend, for example:

```bash
uv venv .venv-cu128
source .venv-cu128/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
sh compile_data_loader.sh
```

For CUDA containers, the host must already have NVIDIA drivers and `nvidia-container-toolkit` configured.

### Apptainer

Apptainer is often easier on shared servers because it does not require Docker daemon access.

Interactive shell from Ubuntu:

```bash
apptainer shell --fakeroot docker://ubuntu:24.04
```

Or with the repository bound in:

```bash
apptainer shell \
  --bind "$PWD":/workspace \
  --pwd /workspace \
  docker://ubuntu:24.04
```

Inside the container, install the same apt and `uv` dependencies as above, then build in the bound repository.

GPU notes:

- CUDA: use `apptainer shell --nv ...`
- ROCm: use `apptainer shell --rocm ...`

Example:

```bash
apptainer shell --nv --bind "$PWD":/workspace --pwd /workspace docker://ubuntu:24.04
```

If you build a persistent Apptainer image, keep separate images or virtual environments for CPU / CUDA / ROCm for the same reason as bare-metal installs: PyTorch wheels are backend-specific.

# Build the fast DataLoader

Activate the same virtual environment you plan to train with, then build the extension:

Linux / macOS / WSL:
```bash
source .venv-cpu/bin/activate
sh compile_data_loader.sh
```

Windows:
```bash
bash compile_data_loader.sh
```

# Train a network

```
source env/bin/activate
python train.py train_data.bin val_data.bin
```

## Resuming from a checkpoint
```
python train.py --resume_from_checkpoint <path> ...
```

## Training on GPU
```
python train.py --gpus 1 ...
```
## Feature set selection
By default the trainer uses a factorized HalfKP feature set (named "HalfKP^")
If you wish to change the feature set used then you can use the `--features=NAME` option. For the list of available features see `--help`
The default is:
```
python train.py ... --features="HalfKP^"
```

## Skipping certain fens in the training

`--smart-fen-skipping` currently skips over moves where the king is in check, or where the bestMove is a capture (typical of non-quiet positions).
`--random-fen-skipping N` skip N fens on average before using one. Uses fewer fens per game, useful with large data sets.

## Current recommended training invocation

```
python train.py --smart-fen-skipping --random-fen-skipping 10 --batch-size 16384 --threads 8 --num-workers 8 --gpus 1 trainingdata validationdata 
```
best nets have been trained with 16B d9-scored nets, training runs >200 epochs

## Using `shogi_ai` as the data pipeline

If your `.bin` files are generated by `shogi_ai/wsl2/src/create_dataset.py`, treat `shogi_ai` as the source of truth for position selection.

- Quiet/tactical filtering should be done in `shogi_ai` via `classify-sfen` and/or `generate --quiet-level`
- SFEN frequency control should be done in `shogi_ai` via `count-sfen` + `generate --sfen-sampling-mode`
- In this setup, `nnue-pytorch` should usually train on the generated `.bin` files without additional loader-side skipping

Recommended settings when training on `shogi_ai`-generated data:

```
python train.py --batch-size 16384 --threads 8 --num-workers 8 --gpus 1 trainingdata validationdata
```

In particular, keep these disabled unless you are training on raw, unfiltered data:

- `--smart-fen-skipping`
- `--random-fen-skipping`

Why:

- `shogi_ai` uses a stronger and more explicit quiet-position definition than `nnue-pytorch`
- applying filtering in both places makes the effective training distribution harder to reason about
- `shogi_ai` may already apply SFEN-level sampling before writing `.bin`
- `shogi_ai` currently writes `.bin` files with `move=0`, so `nnue-pytorch`'s capture-based skip heuristic is not a reliable replacement for upstream filtering

## CORN auxiliary thresholds

`model.corn_aux_thresholds` can be used to add a cumulative ordinal auxiliary loss on top of the main value loss.
There are two ways to build thresholds:

- `shogi_ai/wsl2/src/create_dataset.py corn-thresholds`
  - preferred when you use `shogi_ai` as the data pipeline
  - mirrors `generate`'s SFEN-frequency correction and should be treated as the source of truth
- `nnue-pytorch/corn_thresholds.py`
  - convenient when you already have final `.bin` files or want a local fallback inside this repository

`nnue-pytorch/corn_thresholds.py` derives thresholds from the actual training-input distribution, preferably from the final PackedSfenValue `.bin` files that `train.py` will read.
The command computes quantiles in cp space and then converts them into the softened teacher-logit space used by `model.py`:
`score / (score_scaling * teacher_temperature)`.
That keeps the ordinal bins from collapsing into heavily imbalanced classes while matching the loss scale.

```bash
python corn_thresholds.py --input-bin /path/to/train.bin --num-thresholds 7 --weight 0.1
```

When using `shogi_ai`, prefer building thresholds there from the same frequency-corrected distribution that `generate` will use:

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

That command mirrors `generate`'s SFEN-frequency correction and prints both cp thresholds and the corresponding `--model.corn_aux_thresholds=[...]` values for `nnue-pytorch`.
If you use `shogi_ai`, this is the recommended path.

To update `config.yaml` directly:

```bash
python corn_thresholds.py --config config.yaml --input-bin /path/to/train.bin --num-thresholds 7 --weight 0.1
```

Multiple inputs are supported, so you can use split outputs before or after `generate`:

```bash
python corn_thresholds.py --input-bin train_part1.bin train_part2.bin --num-thresholds 7
python corn_thresholds.py --input-csv eval_part1.csv eval_part2.csv --num-thresholds 7
```

Explicit thresholds are also supported:

```bash
python corn_thresholds.py --thresholds -400 -200 0 200 400 --weight 0.1
```

Explicit `--thresholds` are interpreted in cp space and converted to logit thresholds before being written to config.

If `--input-bin` and `--input-csv` are omitted, the command falls back to uniform thresholds from `--min-score` to `--max-score`.
That fallback is mainly for quick experiments; distribution-based thresholds are the recommended mode.

## Structural auxiliary losses

Two training-only auxiliary heads are available when `data.py_data=true`:

- `model.king_zone_aux_weight`
  - predicts whether side-to-move major pieces (`ROOK/BISHOP` and promoted aliases when available) attack the opponent king zone, and vice versa
- `model.major_safety_aux_weight`
  - predicts whether either side has a hanging major piece (attacked and not defended)

These labels are derived from the decoded board in `nnue_bin_dataset.py`, so they are currently unavailable on the fast C++ loader path.
Use them only as small regularizers, for example:

```bash
python train.py \
  --data.py_data=true \
  --model.king_zone_aux_weight=0.02 \
  --model.major_safety_aux_weight=0.01
```

Recommended usage:

- keep both weights small
- enable one at a time before combining them
- compare against a `py_data=true` baseline, since the loader path changes

## Reusable training script

If you do not want to repeat the `CORN` and training options manually, use:

```bash
scripts/train_with_corn.sh
```

Useful overrides:

```bash
RUN_NAME=vanilla_halfkp \
TRAIN_BIN=/path/train.bin \
VAL_BIN=/path/val.bin \
FEATURES=HalfKP \
ENABLE_CORN=1 \
CORN_NUM_THRESHOLDS=7 \
CORN_WEIGHT=0.1 \
scripts/train_with_corn.sh
```

The script generates a run-local config from `config.template.yaml`, optionally updates `model.corn_aux_*` via `corn_thresholds.py`, and then launches:

```bash
python train.py fit --config <generated_config>
```



# Export a network

Using either a checkpoint (`.ckpt`) or serialized model (`.pt`),
you can export to the YaneuraOu NN binary format. This will convert `last.ckpt`
to `nn.bin`.
```
python serialize.py last.ckpt nn.bin
```
To export EMA weights:
```
python serialize.py --use_ema last.ckpt nn.bin
```

# Import a network

Import an existing YaneuraOu NN binary to the pytorch network format.
```
python serialize.py nn.bin converted.pt
```

# Visualize a network

Visualize a network from either a checkpoint (`.ckpt`), a serialized model (`.pt`)
or a YaneuraOu NN binary (`.bin`).
```
python visualize.py nn.bin --features="HalfKP"
```

Visualize the difference between two networks from either a checkpoint (`.ckpt`), a serialized model (`.pt`)
or a YaneuraOu NN binary (`.bin`).
```
python visualize.py nn.bin  --features="HalfKP" --ref-model nn.cpkt --ref-features="HalfKP^"
```

# Logging

```
tensorboard --logdir=logs
```
Then, go to http://localhost:6006/

# Automatically run matches to determine the best net generated by a (running) training

```
python run_games.py --concurrency 16 --stockfish_exe ./stockfish.master --c_chess_exe ./c-chess-cli --ordo_exe ./ordo --book_file_name ./noob_3moves.epd run96
```

Automatically converts all `.ckpt` found under `run96` to `.bin` and runs games to find the best net. Games are played using `c-chess-cli` and nets are ranked using `ordo`.
This script runs in a loop, and will monitor the directory for new checkpoints. Can be run in parallel with the training, if idle cores are available.


# Thanks

* Sopel - for the amazing fast sparse data loader
* connormcmonigle - https://github.com/connormcmonigle/seer-nnue, and loss function advice.
* syzygy - http://www.talkchess.com/forum3/viewtopic.php?f=7&t=75506
* https://github.com/DanielUranga/TensorFlowNNUE
* https://hxim.github.io/Stockfish-Evaluation-Guide/
* dkappe - Suggesting ranger (https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
