#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <path/to/YaneuraOu-by-gcc> <path/to/nn.bin>"
  exit 1
fi

ENGINE_BIN="$(realpath "$1")"
NN_BIN="$(realpath "$2")"

if [[ ! -x "${ENGINE_BIN}" ]]; then
  echo "Engine binary not executable: ${ENGINE_BIN}"
  exit 1
fi
if [[ ! -f "${NN_BIN}" ]]; then
  echo "NNUE file not found: ${NN_BIN}"
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

EVAL_DIR="${TMP_DIR}/eval"
mkdir -p "${EVAL_DIR}"
cp "${NN_BIN}" "${EVAL_DIR}/nn.bin"

LOG_FILE="${TMP_DIR}/usi_test.log"
{
  echo "usi"
  echo "setoption name EvalDir value ${EVAL_DIR}"
  echo "isready"
  echo "usinewgame"
  echo "position startpos moves 7g7f 3c3d 2g2f"
  echo "go depth 4"
  echo "quit"
} | "${ENGINE_BIN}" | tee "${LOG_FILE}"

if ! rg -q "bestmove" "${LOG_FILE}"; then
  echo "bestmove was not returned. Check log: ${LOG_FILE}"
  exit 1
fi

if rg -qi "error|mismatch|failed" "${LOG_FILE}"; then
  echo "Potential loading/runtime issue found. Check log: ${LOG_FILE}"
  exit 1
fi

echo "YaneuraOu NNUE smoke test passed."
