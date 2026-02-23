#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
YANE_SOURCE="${REPO_ROOT}/YaneuraOu/source"

TARGET_CPU="${TARGET_CPU:-ZEN2}"
EDITION="${YANEURAOU_EDITION:-YANEURAOU_ENGINE_NNUE}"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -d "${YANE_SOURCE}" ]]; then
  echo "YaneuraOu source not found: ${YANE_SOURCE}"
  exit 1
fi

make -C "${YANE_SOURCE}" clean
make -C "${YANE_SOURCE}" -j"${JOBS}" TARGET_CPU="${TARGET_CPU}" YANEURAOU_EDITION="${EDITION}"

ENGINE_BIN="${YANE_SOURCE}/YaneuraOu-by-gcc"
if [[ ! -x "${ENGINE_BIN}" ]]; then
  echo "Engine binary not found after build: ${ENGINE_BIN}"
  exit 1
fi

echo "Built YaneuraOu binary: ${ENGINE_BIN}"
