#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
usage: scripts/relabel_bin_dir.sh CHECKPOINT INPUT_DIR OUTPUT_DIR [extra psv_re_eval args...]

Relabel all *.bin files under INPUT_DIR and write them as:
  OUTPUT_DIR/0000.bin
  OUTPUT_DIR/0001.bin
  ...

Examples:
  scripts/relabel_bin_dir.sh model.ckpt ./bins ./out --device cuda:0 --batch-size 4096
  scripts/relabel_bin_dir.sh model.ckpt ./bins ./out --use_ema
EOF
  exit 1
fi

checkpoint=$1
input_dir=$2
output_dir=$3
shift 3

mkdir -p "$output_dir"

mapfile -t input_files < <(find "$input_dir" -maxdepth 1 -type f -name '*.bin' | sort)

if [[ ${#input_files[@]} -eq 0 ]]; then
  echo "No .bin files found in $input_dir" >&2
  exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

for idx in "${!input_files[@]}"; do
  input_path=${input_files[$idx]}
  output_path=$(printf '%s/%04d.bin' "$output_dir" "$idx")

  echo "[$((idx + 1))/${#input_files[@]}] relabel $input_path -> $output_path"
  python3 "$repo_root/psv_re_eval.py" \
    "$checkpoint" \
    "$input_path" \
    "$output_path" \
    --device cuda:0 --batch-size 131056 \
    --input-format bin \
    "$@"
done
