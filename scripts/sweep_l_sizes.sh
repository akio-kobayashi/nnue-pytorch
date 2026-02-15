#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_PATH="${ROOT_DIR}/config.template.yaml"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="${ROOT_DIR}/runs/layer_sweep_${TIMESTAMP}"

# Format: "l1 l2 l3"
COMBOS=(
  "768 8 64"
  "1024 8 96"
  "1280 16 128"
  "1536 16 192"
)

if [[ ! -f "${TEMPLATE_PATH}" ]]; then
  echo "template not found: ${TEMPLATE_PATH}" >&2
  exit 1
fi

mkdir -p "${RUNS_ROOT}"

for combo in "${COMBOS[@]}"; do
  read -r l1 l2 l3 <<< "${combo}"
  run_name="l1_${l1}_l2_${l2}_l3_${l3}"
  run_dir="${RUNS_ROOT}/${run_name}"
  config_path="${run_dir}/config.yaml"

  mkdir -p "${run_dir}"

  sed \
    -e "s|__L1_SIZE__|${l1}|g" \
    -e "s|__L2_SIZE__|${l2}|g" \
    -e "s|__L3_SIZE__|${l3}|g" \
    -e "s|__RUN_NAME__|${run_name}|g" \
    -e "s|__RUN_DIR__|${run_dir}|g" \
    -e "s|__RUNS_ROOT__|${RUNS_ROOT}|g" \
    "${TEMPLATE_PATH}" > "${config_path}"

  echo "===== start: ${run_name} ====="
  python "${ROOT_DIR}/train.py" --config "${config_path}" \
    2>&1 | tee "${run_dir}/train.log"
  echo "===== done: ${run_name} ====="
done

echo "all runs finished: ${RUNS_ROOT}"
