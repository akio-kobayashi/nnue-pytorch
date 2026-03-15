#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build-halfkpe9-smoke"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_DATA=""
VAL_DATA=""
FEATURES="HalfKPE9^"
RUN_TRAIN=0

usage() {
  cat <<'EOF'
Usage:
  ./run_halfkpe9_smoke_test.sh [--train TRAIN.bin --val VAL.bin] [--python python3] [--features HalfKPE9^]

Behavior:
  1. Configures and builds training_data_loader with CMake
  2. Verifies that the Python feature registry resolves the requested feature set
  3. If --train and --val are provided, runs a tiny training smoke test

Notes:
  - This script does not modify tracked files.
  - It copies the built training_data_loader shared library into the repo root
    because nnue_dataset.py expects it there.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train)
      TRAIN_DATA="$2"
      RUN_TRAIN=1
      shift 2
      ;;
    --val)
      VAL_DATA="$2"
      RUN_TRAIN=1
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --features)
      FEATURES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  if [[ -z "${TRAIN_DATA}" || -z "${VAL_DATA}" ]]; then
    echo "--train and --val must be provided together" >&2
    exit 1
  fi
fi

cd "${ROOT_DIR}"

echo "[1/3] Configure training_data_loader"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_POLICY_VERSION_MINIMUM=3.5

echo "[2/3] Build training_data_loader"
cmake --build "${BUILD_DIR}" --target training_data_loader -j 2

SHARED_LIB="$(find "${BUILD_DIR}" -maxdepth 2 \( -name 'libtraining_data_loader*.dylib' -o -name 'libtraining_data_loader*.so' -o -name 'training_data_loader*.dll' \) | head -n 1)"
if [[ -z "${SHARED_LIB}" ]]; then
  echo "Could not find built training_data_loader shared library" >&2
  exit 1
fi
cp "${SHARED_LIB}" "${ROOT_DIR}/"
echo "Copied shared library: $(basename "${SHARED_LIB}")"

echo "[3/3] Verify Python feature registration"
"${PYTHON_BIN}" - <<PY
import features
fs = features.get_feature_set_from_name("${FEATURES}")
print("feature_set=", fs.name)
print("num_real_features=", fs.num_real_features)
print("num_virtual_features=", fs.num_virtual_features)
PY

if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  echo "[4/4] Run tiny training smoke test"
  "${PYTHON_BIN}" train.py \
    fit \
    --model.features="${FEATURES}" \
    --trainer.max_epochs=1 \
    --trainer.limit_train_batches=1 \
    --trainer.limit_val_batches=1 \
    --trainer.enable_progress_bar=false \
    --data.train="${TRAIN_DATA}" \
    --data.val="${VAL_DATA}" \
    --data.batch_size=128 \
    --data.num_workers=1 \
    --data.epoch_size=128 \
    --data.validation_size=128
fi

echo "Smoke test completed."
