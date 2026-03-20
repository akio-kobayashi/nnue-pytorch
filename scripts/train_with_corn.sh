#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_PATH="${ROOT_DIR}/config.template.yaml"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Common overrides:
#   TRAIN_BIN=... VAL_BIN=... RUN_NAME=... RUNS_ROOT=... USE_GPU=1
#   L1_SIZE=1024 L2_SIZE=8 L3_SIZE=96 FEATURES=HalfKP BATCH_SIZE=16384
#   SMART_FEN_SKIPPING=false RANDOM_FEN_SKIPPING=0 PY_DATA=false
#
# CORN-related overrides:
#   ENABLE_CORN=1
#   CORN_INPUT_BINS="/path/train.bin /path/train2.bin"
#   CORN_INPUT_CSVS="/path/a.csv /path/b.csv"
#   CORN_NUM_THRESHOLDS=7 CORN_WEIGHT=0.1
#   CORN_SCORE_SCALING=361 CORN_TEACHER_TEMPERATURE=1.0
#   CORN_SCORE_COLUMN=eval_score_cp
#   EXTRA_TRAIN_ARGS="--trainer.max_epochs=50"

TRAIN_BIN="${TRAIN_BIN:-../train.bin}"
VAL_BIN="${VAL_BIN:-../train.bin}"
RUN_NAME="${RUN_NAME:-corn_run_${TIMESTAMP}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/runs}"
RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
CONFIG_PATH="${RUN_DIR}/config.yaml"

USE_GPU="${USE_GPU:-1}"
L1_SIZE="${L1_SIZE:-1024}"
L2_SIZE="${L2_SIZE:-8}"
L3_SIZE="${L3_SIZE:-96}"
FEATURES="${FEATURES:-HalfKP}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
SMART_FEN_SKIPPING="${SMART_FEN_SKIPPING:-false}"
RANDOM_FEN_SKIPPING="${RANDOM_FEN_SKIPPING:-0}"
PY_DATA="${PY_DATA:-false}"
PY_DATA_SAMPLING_MODE="${PY_DATA_SAMPLING_MODE:-uniform}"

ENABLE_CORN="${ENABLE_CORN:-1}"
CORN_NUM_THRESHOLDS="${CORN_NUM_THRESHOLDS:-7}"
CORN_WEIGHT="${CORN_WEIGHT:-0.1}"
CORN_SCORE_SCALING="${CORN_SCORE_SCALING:-361}"
CORN_TEACHER_TEMPERATURE="${CORN_TEACHER_TEMPERATURE:-1.0}"
CORN_SCORE_COLUMN="${CORN_SCORE_COLUMN:-eval_score_cp}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

if [[ ! -f "${TEMPLATE_PATH}" ]]; then
  echo "template not found: ${TEMPLATE_PATH}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"

sed \
  -e "s|__L1_SIZE__|${L1_SIZE}|g" \
  -e "s|__L2_SIZE__|${L2_SIZE}|g" \
  -e "s|__L3_SIZE__|${L3_SIZE}|g" \
  -e "s|__RUN_NAME__|${RUN_NAME}|g" \
  -e "s|__RUN_DIR__|${RUN_DIR}|g" \
  -e "s|__RUNS_ROOT__|${RUNS_ROOT}|g" \
  "${TEMPLATE_PATH}" > "${CONFIG_PATH}"

sed \
  -e "s|^  train:.*$|  train: ${TRAIN_BIN}|g" \
  -e "s|^  val:.*$|  val: ${VAL_BIN}|g" \
  -e "s|^  features:.*$|  features: ${FEATURES}|g" \
  -e "s|^  batch_size:.*$|  batch_size: ${BATCH_SIZE}|g" \
  -e "s|^  smart_fen_skipping:.*$|  smart_fen_skipping: ${SMART_FEN_SKIPPING}|g" \
  -e "s|^  random_fen_skipping:.*$|  random_fen_skipping: ${RANDOM_FEN_SKIPPING}|g" \
  -e "s|^  py_data:.*$|  py_data: ${PY_DATA}|g" \
  -e "s|^  py_data_sampling_mode:.*$|  py_data_sampling_mode: ${PY_DATA_SAMPLING_MODE}|g" \
  "${CONFIG_PATH}" > "${CONFIG_PATH}.tmp"
mv "${CONFIG_PATH}.tmp" "${CONFIG_PATH}"

if [[ "${USE_GPU}" == "0" ]]; then
  sed \
    -e "s|^  accelerator:.*$|  accelerator: cpu|g" \
    -e "s|^  devices:.*$|  devices: 1|g" \
    "${CONFIG_PATH}" > "${CONFIG_PATH}.tmp"
  mv "${CONFIG_PATH}.tmp" "${CONFIG_PATH}"
fi

if [[ "${ENABLE_CORN}" == "1" ]]; then
  corn_args=(
    "--config" "${CONFIG_PATH}"
    "--num-thresholds" "${CORN_NUM_THRESHOLDS}"
    "--weight" "${CORN_WEIGHT}"
    "--score-scaling" "${CORN_SCORE_SCALING}"
    "--teacher-temperature" "${CORN_TEACHER_TEMPERATURE}"
    "--score-column" "${CORN_SCORE_COLUMN}"
  )

  if [[ -n "${CORN_INPUT_CSVS:-}" ]]; then
    read -r -a corn_input_csvs <<< "${CORN_INPUT_CSVS}"
    corn_args+=("--input-csv" "${corn_input_csvs[@]}")
  else
    read -r -a corn_input_bins <<< "${CORN_INPUT_BINS:-${TRAIN_BIN}}"
    corn_args+=("--input-bin" "${corn_input_bins[@]}")
  fi

  python "${ROOT_DIR}/corn_thresholds.py" "${corn_args[@]}" \
    2>&1 | tee "${RUN_DIR}/corn_thresholds.log"
fi

train_args=("fit" "--config" "${CONFIG_PATH}")
if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_TRAIN_ARGS} )
  train_args+=("${extra_args[@]}")
fi

echo "run=${RUN_NAME}"
echo "config=${CONFIG_PATH}"
python "${ROOT_DIR}/train.py" "${train_args[@]}" \
  2>&1 | tee "${RUN_DIR}/train.log"
