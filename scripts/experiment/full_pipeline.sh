#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <train.bin|binpack> <val.bin|binpack> <artifacts_dir> [max_epochs]"
  exit 1
fi

TRAIN_DATA="$1"
VAL_DATA="$2"
ARTIFACTS_DIR="$3"
MAX_EPOCHS="${4:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

"${SCRIPT_DIR}/build_image.sh"
"${SCRIPT_DIR}/train_export_in_docker.sh" "${TRAIN_DATA}" "${VAL_DATA}" "${ARTIFACTS_DIR}" "${MAX_EPOCHS}"
"${SCRIPT_DIR}/build_yaneuraou_amd.sh"

MANIFEST="$(realpath "${ARTIFACTS_DIR}")/latest_manifest.env"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}"
  exit 1
fi

# shellcheck disable=SC1090
source "${MANIFEST}"
ENGINE_BIN="${REPO_ROOT}/YaneuraOu/source/YaneuraOu-by-gcc"

"${SCRIPT_DIR}/test_yaneuraou_with_nnue.sh" "${ENGINE_BIN}" "${NNUE_PATH}"

echo "Pipeline completed successfully."
echo "- Engine: ${ENGINE_BIN}"
echo "- Checkpoint: ${CKPT_PATH}"
echo "- Exported NNUE: ${NNUE_PATH}"
