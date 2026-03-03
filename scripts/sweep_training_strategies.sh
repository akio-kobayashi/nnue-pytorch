#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_PATH="${ROOT_DIR}/config.template.yaml"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="${ROOT_DIR}/runs/strategy_sweep_${TIMESTAMP}"

# Optional overrides (export before run):
#   TRAIN_BIN=... VAL_BIN=... USE_GPU=1 L1_SIZE=1024 L2_SIZE=8 L3_SIZE=96
#   BATCH_SIZE=16384 LR_BASE=1.0 LR_BASE_BATCH=16384
TRAIN_BIN="${TRAIN_BIN:-../train.bin}"
VAL_BIN="${VAL_BIN:-../train.bin}"
USE_GPU="${USE_GPU:-1}"
L1_SIZE="${L1_SIZE:-1024}"
L2_SIZE="${L2_SIZE:-8}"
L3_SIZE="${L3_SIZE:-96}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
LR_BASE="${LR_BASE:-1.0}"
LR_BASE_BATCH="${LR_BASE_BATCH:-16384}"

# Linear scaling rule:
#   lr_scaled = LR_BASE * (BATCH_SIZE / LR_BASE_BATCH)
LR_SCALED="$(awk -v lr="${LR_BASE}" -v bs="${BATCH_SIZE}" -v b0="${LR_BASE_BATCH}" 'BEGIN { printf "%.8g", lr * (bs / b0) }')"

# name | ema_enabled | ema_decay | teacher_temperature | entropy_coef | outcome_pos_weight | py_data | sampling_mode
STRATEGIES=(
  "baseline 0 0.9995 1.0 1.0 1.0 0 uniform"
  "ema_only 1 0.9995 1.0 1.0 1.0 0 uniform"
  "loss_tuned 1 0.9995 1.2 0.8 1.1 0 uniform"
  "pydata_uniform 1 0.9995 1.0 1.0 1.0 1 uniform"
  "pydata_ply_balanced 1 0.9995 1.0 1.0 1.0 1 ply_balanced"
)

if [[ ! -f "${TEMPLATE_PATH}" ]]; then
  echo "template not found: ${TEMPLATE_PATH}" >&2
  exit 1
fi

mkdir -p "${RUNS_ROOT}"

for row in "${STRATEGIES[@]}"; do
  read -r name ema_enabled ema_decay teacher_temp entropy_coef outcome_pos_weight py_data sampling_mode <<< "${row}"

  run_name="strategy_${name}"
  run_dir="${RUNS_ROOT}/${run_name}"
  config_path="${run_dir}/config.yaml"

  mkdir -p "${run_dir}"

  # 1) Fill template placeholders.
  sed \
    -e "s|__L1_SIZE__|${L1_SIZE}|g" \
    -e "s|__L2_SIZE__|${L2_SIZE}|g" \
    -e "s|__L3_SIZE__|${L3_SIZE}|g" \
    -e "s|__RUN_NAME__|${run_name}|g" \
    -e "s|__RUN_DIR__|${run_dir}|g" \
    -e "s|__RUNS_ROOT__|${RUNS_ROOT}|g" \
    "${TEMPLATE_PATH}" > "${config_path}"

  # 2) Override strategy/data fields for this run.
  sed \
    -e "s|^  train:.*$|  train: ${TRAIN_BIN}|g" \
    -e "s|^  val:.*$|  val: ${VAL_BIN}|g" \
    -e "s|^  batch_size:.*$|  batch_size: ${BATCH_SIZE}|g" \
    -e "s|^  lr:.*$|  lr: [${LR_SCALED}]|g" \
    -e "s|^  ema_enabled:.*$|  ema_enabled: $( [[ \"${ema_enabled}\" == \"1\" ]] && echo true || echo false )|g" \
    -e "s|^  ema_decay:.*$|  ema_decay: ${ema_decay}|g" \
    -e "s|^  teacher_temperature:.*$|  teacher_temperature: ${teacher_temp}|g" \
    -e "s|^  entropy_coef:.*$|  entropy_coef: ${entropy_coef}|g" \
    -e "s|^  outcome_pos_weight:.*$|  outcome_pos_weight: ${outcome_pos_weight}|g" \
    -e "s|^  py_data:.*$|  py_data: $( [[ \"${py_data}\" == \"1\" ]] && echo true || echo false )|g" \
    -e "s|^  py_data_sampling_mode:.*$|  py_data_sampling_mode: ${sampling_mode}|g" \
    "${config_path}" > "${config_path}.tmp"
  mv "${config_path}.tmp" "${config_path}"

  # 3) Optional accelerator override.
  if [[ "${USE_GPU}" == "0" ]]; then
    sed -e "s|^  accelerator:.*$|  accelerator: cpu|g" \
        -e "s|^  devices:.*$|  devices: 1|g" \
        "${config_path}" > "${config_path}.tmp"
    mv "${config_path}.tmp" "${config_path}"
  fi

  echo "run=${run_name} batch_size=${BATCH_SIZE} lr=${LR_SCALED}"
  echo "===== start: ${run_name} ====="
  python "${ROOT_DIR}/train.py" --config "${config_path}" \
    2>&1 | tee "${run_dir}/train.log"
  echo "===== done: ${run_name} ====="
done

echo "all runs finished: ${RUNS_ROOT}"
