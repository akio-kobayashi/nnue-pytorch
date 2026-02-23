#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <train.bin|binpack> <val.bin|binpack> <output_dir> [max_epochs]"
  exit 1
fi

TRAIN_DATA="$(realpath "$1")"
VAL_DATA="$(realpath "$2")"
OUTPUT_DIR="$(realpath "$3")"
MAX_EPOCHS="${4:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-nnue-pytorch-cu124}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

mkdir -p "${OUTPUT_DIR}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found."
  exit 1
fi

# Train data and val data must be under the repo mount to keep paths simple in-container.
case "${TRAIN_DATA}" in
  "${REPO_ROOT}"/*) ;;
  *) echo "Train data must be under ${REPO_ROOT}: ${TRAIN_DATA}"; exit 1 ;;
esac
case "${VAL_DATA}" in
  "${REPO_ROOT}"/*) ;;
  *) echo "Validation data must be under ${REPO_ROOT}: ${VAL_DATA}"; exit 1 ;;
esac
case "${OUTPUT_DIR}" in
  "${REPO_ROOT}"/*) ;;
  *) echo "Output dir must be under ${REPO_ROOT}: ${OUTPUT_DIR}"; exit 1 ;;
esac

TRAIN_IN_CONTAINER="/workspace/${TRAIN_DATA#${REPO_ROOT}/}"
VAL_IN_CONTAINER="/workspace/${VAL_DATA#${REPO_ROOT}/}"
OUT_IN_CONTAINER="/workspace/${OUTPUT_DIR#${REPO_ROOT}/}"

docker run --rm --gpus all \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  -e YANE_NETWORK_HASH="${YANE_NETWORK_HASH:-0x7AF32F16}" \
  "${IMAGE_NAME}:${IMAGE_TAG}" \
  bash scripts/experiment/train_export_inner.sh \
    "${TRAIN_IN_CONTAINER}" \
    "${VAL_IN_CONTAINER}" \
    "${OUT_IN_CONTAINER}" \
    "${MAX_EPOCHS}"

echo "Docker training/export completed: ${OUTPUT_DIR}"
