#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat <<'EOF'
usage: scripts/compare_ckpt_vs_yaneuraou.sh CHECKPOINT INPUT_BIN OUTPUT_DIR YANEURAOU_ENGINE [MAX_RECORDS]

Write two CSV files for the same PackedSfenValue .bin:
  OUTPUT_DIR/ckpt_eval.csv
  OUTPUT_DIR/yaneuraou_eval.csv

Arguments:
  CHECKPOINT       nnue-pytorch .ckpt path
  INPUT_BIN        PackedSfenValue .bin path
  OUTPUT_DIR       directory for CSV outputs
  YANEURAOU_ENGINE built YaneuraOu binary path
  MAX_RECORDS      optional, default: 0 (all records)

Environment variables:
  DEVICE           torch device for nnue-pytorch (default: cpu)
  BATCH_SIZE       batch size for nnue-pytorch CSV export (default: 1024)
  EVAL_DIR         YaneuraOu EvalDir passed before isready
                   default: <engine_dir>/eval

Example:
  DEVICE=mps BATCH_SIZE=4096 \
  scripts/compare_ckpt_vs_yaneuraou.sh \
    /path/to/model.ckpt \
    /path/to/sample.bin \
    /path/to/out \
    /Users/akio/Documents/GitHub/YaneuraOu/source/YaneuraOu-by-gcc-moe-HALFKP_256X2_32_32 \
    1000
EOF
  exit 1
fi

checkpoint=$1
input_bin=$2
output_dir=$3
yaneuraou_engine=$4
max_records=${5:-0}

device=${DEVICE:-cpu}
batch_size=${BATCH_SIZE:-1024}
eval_dir=${EVAL_DIR:-$(cd "$(dirname "$yaneuraou_engine")" && pwd)/eval}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

mkdir -p "$output_dir"

ckpt_csv="$output_dir/ckpt_eval.csv"
yaneuraou_csv="$output_dir/yaneuraou_eval.csv"

echo "checkpoint   : $checkpoint"
echo "input_bin    : $input_bin"
echo "output_dir   : $output_dir"
echo "engine       : $yaneuraou_engine"
echo "eval_dir     : $eval_dir"
echo "device       : $device"
echo "batch_size   : $batch_size"
echo "max_records  : $max_records"

python3 "$repo_root/psv_ckpt_eval_csv.py" \
  "$checkpoint" \
  "$input_bin" \
  "$ckpt_csv" \
  --device "$device" \
  --batch-size "$batch_size" \
  --max-records "$max_records"

printf 'usi\nsetoption name EvalDir value %s\nisready\ntest evalbin input %s output %s count %s\nquit\n' \
  "$eval_dir" \
  "$input_bin" \
  "$yaneuraou_csv" \
  "$max_records" | "$yaneuraou_engine"

echo "wrote:"
echo "  $ckpt_csv"
echo "  $yaneuraou_csv"
