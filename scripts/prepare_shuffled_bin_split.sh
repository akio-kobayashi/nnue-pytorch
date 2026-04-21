#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  prepare_shuffled_bin_split.sh --input-dir DIR --output-dir DIR [options]

Description:
  Shuffle 0101.bin .. 1014.bin into 0101_shuffled.bin .. 1014_shuffled.bin,
  write their absolute paths into train.txt, then shuffle 1015.bin and extract
  either the first N blocks or the first M samples into a validation binary and
  write its absolute path into val.txt.

Options:
  --input-dir DIR         Directory containing 0101.bin .. 1015.bin
  --output-dir DIR        Directory to place shuffled binaries and txt manifests
  --seed N                Base random seed (default: 0)
  --record-size BYTES     Packed record size in bytes (default: 40)
  --buffer-records N      Forwarded to shuffle_packedsfen.py (default: 8192)
  --num-buckets N         Forwarded to shuffle_packedsfen.py (default: 256)
  --val-samples N         Validation size in records
  --val-blocks N          Validation size in blocks
  --block-records N       Records per block when --val-blocks is used
  --train-manifest NAME   Train manifest filename (default: train.txt)
  --val-manifest NAME     Validation manifest filename (default: val.txt)

Examples:
  prepare_shuffled_bin_split.sh \
    --input-dir /data/bin \
    --output-dir /data/shuffled \
    --val-samples 1000000

  prepare_shuffled_bin_split.sh \
    --input-dir /data/bin \
    --output-dir /data/shuffled \
    --val-blocks 64 \
    --block-records 100000
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

INPUT_DIR=""
OUTPUT_DIR=""
SEED=0
RECORD_SIZE=40
BUFFER_RECORDS=8192
NUM_BUCKETS=256
VAL_SAMPLES=""
VAL_BLOCKS=""
BLOCK_RECORDS=""
TRAIN_MANIFEST="train.txt"
VAL_MANIFEST="val.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --record-size)
      RECORD_SIZE="$2"
      shift 2
      ;;
    --buffer-records)
      BUFFER_RECORDS="$2"
      shift 2
      ;;
    --num-buckets)
      NUM_BUCKETS="$2"
      shift 2
      ;;
    --val-samples)
      VAL_SAMPLES="$2"
      shift 2
      ;;
    --val-blocks)
      VAL_BLOCKS="$2"
      shift 2
      ;;
    --block-records)
      BLOCK_RECORDS="$2"
      shift 2
      ;;
    --train-manifest)
      TRAIN_MANIFEST="$2"
      shift 2
      ;;
    --val-manifest)
      VAL_MANIFEST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

[[ -n "$INPUT_DIR" ]] || fail "--input-dir is required"
[[ -n "$OUTPUT_DIR" ]] || fail "--output-dir is required"
[[ -d "$INPUT_DIR" ]] || fail "input directory not found: $INPUT_DIR"

if [[ -n "$VAL_SAMPLES" && -n "$VAL_BLOCKS" ]]; then
  fail "specify only one of --val-samples or --val-blocks"
fi
if [[ -z "$VAL_SAMPLES" && -z "$VAL_BLOCKS" ]]; then
  fail "either --val-samples or --val-blocks is required"
fi
if [[ -n "$VAL_BLOCKS" && -z "$BLOCK_RECORDS" ]]; then
  fail "--block-records is required when --val-blocks is used"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHUFFLER="$SCRIPT_DIR/shuffle_packedsfen.py"
[[ -x "$SHUFFLER" ]] || fail "shuffle helper not found or not executable: $SHUFFLER"

mkdir -p "$OUTPUT_DIR"
TRAIN_MANIFEST_PATH="$OUTPUT_DIR/$TRAIN_MANIFEST"
VAL_MANIFEST_PATH="$OUTPUT_DIR/$VAL_MANIFEST"
: > "$TRAIN_MANIFEST_PATH"

shuffle_one() {
  local input_path="$1"
  local output_path="$2"
  local seed="$3"

  python3 "$SHUFFLER" \
    "$input_path" \
    "$output_path" \
    --seed "$seed" \
    --buffer-records "$BUFFER_RECORDS" \
    --num-buckets "$NUM_BUCKETS"
}

for i in $(seq 101 1014); do
  stem=$(printf "%04d" "$i")
  input_path="$INPUT_DIR/$stem.bin"
  output_path="$OUTPUT_DIR/${stem}_shuffled.bin"
  [[ -f "$input_path" ]] || fail "missing training input: $input_path"
  shuffle_one "$input_path" "$output_path" $((SEED + i))
  python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$output_path" >> "$TRAIN_MANIFEST_PATH"
done

val_input="$INPUT_DIR/1015.bin"
val_shuffled="$OUTPUT_DIR/1015_shuffled.bin"
val_output="$OUTPUT_DIR/1015_val.bin"
[[ -f "$val_input" ]] || fail "missing validation source: $val_input"

shuffle_one "$val_input" "$val_shuffled" $((SEED + 1015))

if [[ -n "$VAL_SAMPLES" ]]; then
  count_bytes=$((VAL_SAMPLES * RECORD_SIZE))
  dd if="$val_shuffled" of="$val_output" bs="$count_bytes" count=1 status=none
else
  block_bytes=$((BLOCK_RECORDS * RECORD_SIZE))
  dd if="$val_shuffled" of="$val_output" bs="$block_bytes" count="$VAL_BLOCKS" status=none
fi

python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$val_output" > "$VAL_MANIFEST_PATH"

echo "created: $TRAIN_MANIFEST_PATH"
echo "created: $VAL_MANIFEST_PATH"
