#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <train.bin|binpack> <val.bin|binpack> <output_dir> [max_epochs]"
  exit 1
fi

TRAIN_DATA="$1"
VAL_DATA="$2"
OUTPUT_DIR="$3"
MAX_EPOCHS="${4:-1}"

REPO_ROOT="/workspace"
cd "${REPO_ROOT}"

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "Train data not found: ${TRAIN_DATA}"
  exit 1
fi
if [[ ! -f "${VAL_DATA}" ]]; then
  echo "Validation data not found: ${VAL_DATA}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# 1) Create Python venv with uv and install dependencies.
uv venv .venv
source .venv/bin/activate
uv pip install -r docker/requirements.train.txt

# 2) Build fast C++ loader and place the shared library where nnue_dataset.py expects it.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

if [[ -f build/libtraining_data_loader.so ]]; then
  cp build/libtraining_data_loader.so ./training_data_loader.so
elif [[ -f build/training_data_loader.dll ]]; then
  cp build/training_data_loader.dll ./training_data_loader.dll
elif [[ -f build/libtraining_data_loader.dylib ]]; then
  cp build/libtraining_data_loader.dylib ./training_data_loader.dylib
else
  echo "training_data_loader shared library was not produced."
  exit 1
fi

# 3) Train with YaneuraOu default network dimensions (1024-8-96, 8 buckets).
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

python train.py fit \
  --config config.yaml \
  --model.l1_size 1024 \
  --model.l2_size 8 \
  --model.l3_size 96 \
  --model.num_buckets 8 \
  --data.features HalfKP \
  --data.train "${TRAIN_DATA}" \
  --data.val "${VAL_DATA}" \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.max_epochs "${MAX_EPOCHS}" \
  --trainer.default_root_dir "${RUN_DIR}"

CKPT_PATH="$(find "${RUN_DIR}" -type f -name '*.ckpt' | sort | tail -n1 || true)"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "Checkpoint was not generated under ${RUN_DIR}"
  exit 1
fi

# 4) Export for YaneuraOu.
EXPORT_DIR="${OUTPUT_DIR}/export"
mkdir -p "${EXPORT_DIR}"
NNUE_PATH="${EXPORT_DIR}/nn.bin"
YANE_NETWORK_HASH="${YANE_NETWORK_HASH:-0x7AF32F16}"

python serialize.py "${CKPT_PATH}" "${NNUE_PATH}" \
  --features HalfKP \
  --l1_size 1024 \
  --l2_size 8 \
  --l3_size 96 \
  --num_buckets 8 \
  --target-engine yaneuraou \
  --yane-network-hash "${YANE_NETWORK_HASH}"

MANIFEST="${OUTPUT_DIR}/latest_manifest.env"
cat > "${MANIFEST}" <<MANIFEST_EOF
RUN_ID=${RUN_ID}
RUN_DIR=${RUN_DIR}
CKPT_PATH=${CKPT_PATH}
NNUE_PATH=${NNUE_PATH}
YANE_NETWORK_HASH=${YANE_NETWORK_HASH}
MANIFEST_EOF

echo "Training and export completed."
echo "Manifest: ${MANIFEST}"
