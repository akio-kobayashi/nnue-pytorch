#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
usage: scripts/relabel_sfen_dir.sh CHECKPOINT INPUT_DIR OUTPUT_DIR [extra psv_re_eval args...]

Relabel all *.sfen files under INPUT_DIR and write them as:
  OUTPUT_DIR/0000.bin
  OUTPUT_DIR/0001.bin
  ...

Examples:
  scripts/relabel_sfen_dir.sh model.ckpt ./sfens ./out --device cuda:0 --batch-size 4096
  scripts/relabel_sfen_dir.sh model.ckpt ./sfens ./out --use_ema
EOF
  exit 1
fi

checkpoint=$1
input_dir=$2
output_dir=$3
shift 3

mkdir -p "$output_dir"

mapfile -t sfen_files < <(find "$input_dir" -maxdepth 1 -type f -name '*.sfen' | sort)

if [[ ${#sfen_files[@]} -eq 0 ]]; then
  echo "No .sfen files found in $input_dir" >&2
  exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

for idx in "${!sfen_files[@]}"; do
  input_path=${sfen_files[$idx]}
  output_path=$(printf '%s/%04d.bin' "$output_dir" "$idx")

  echo "[$((idx + 1))/${#sfen_files[@]}] relabel $input_path -> $output_path"
  python3 "$repo_root/psv_re_eval.py" \
    "$checkpoint" \
    "$input_path" \
    "$output_path" \
    --input-format sfen \
    "$@"
done
