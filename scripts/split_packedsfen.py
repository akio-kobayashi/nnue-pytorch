#!/usr/bin/env python3

import argparse
import os
import random
import sys


# Matches nnue_bin_dataset.PACKED_SFEN_VALUE_BYTES.
PACKED_SFEN_RECORD_SIZE = 40
DEFAULT_BUFFER_RECORDS = 8192


def parse_args():
  parser = argparse.ArgumentParser(
      description='Split a packed sfen dataset into train.bin and val.bin.')
  parser.add_argument('input', help='path to the input packed sfen binary')
  parser.add_argument(
      '--train-output',
      default='train.bin',
      help='path to the output training binary (default: train.bin)')
  parser.add_argument(
      '--val-output',
      default='val.bin',
      help='path to the output validation binary (default: val.bin)')
  parser.add_argument(
      '--val-ratio',
      type=float,
      default=0.01,
      help='fraction of records written to validation (default: 0.01)')
  parser.add_argument(
      '--seed',
      type=int,
      default=0,
      help='random seed used for the split (default: 0)')
  parser.add_argument(
      '--buffer-records',
      type=int,
      default=DEFAULT_BUFFER_RECORDS,
      help='records processed per read chunk (default: 8192)')
  return parser.parse_args()


def fail(message):
  print(f'error: {message}', file=sys.stderr)
  raise SystemExit(1)


def validate_args(args):
  if not 0.0 <= args.val_ratio <= 1.0:
    fail('--val-ratio must be between 0.0 and 1.0')
  if args.buffer_records <= 0:
    fail('--buffer-records must be a positive integer')
  if os.path.abspath(args.input) == os.path.abspath(args.train_output):
    fail('--train-output must differ from the input path')
  if os.path.abspath(args.input) == os.path.abspath(args.val_output):
    fail('--val-output must differ from the input path')
  if os.path.abspath(args.train_output) == os.path.abspath(args.val_output):
    fail('--train-output and --val-output must be different files')


def ensure_parent_dir(path):
  parent = os.path.dirname(os.path.abspath(path))
  if parent:
    os.makedirs(parent, exist_ok=True)


def main():
  args = parse_args()
  validate_args(args)

  input_size = os.path.getsize(args.input)
  if input_size % PACKED_SFEN_RECORD_SIZE != 0:
    fail(
        f'input size {input_size} is not a multiple of '
        f'{PACKED_SFEN_RECORD_SIZE} bytes')

  total_records = input_size // PACKED_SFEN_RECORD_SIZE
  val_records = round(total_records * args.val_ratio)
  rng = random.Random(args.seed)
  chunk_size = args.buffer_records * PACKED_SFEN_RECORD_SIZE

  ensure_parent_dir(args.train_output)
  ensure_parent_dir(args.val_output)

  remaining_records = total_records
  remaining_val_records = val_records
  written_train_records = 0
  written_val_records = 0

  with open(args.input, 'rb') as src, \
      open(args.train_output, 'wb') as train_out, \
      open(args.val_output, 'wb') as val_out:
    while True:
      chunk = src.read(chunk_size)
      if not chunk:
        break
      if len(chunk) % PACKED_SFEN_RECORD_SIZE != 0:
        fail('read a partial record from the input file')

      train_buffer = bytearray()
      val_buffer = bytearray()
      for offset in range(0, len(chunk), PACKED_SFEN_RECORD_SIZE):
        record = chunk[offset:offset + PACKED_SFEN_RECORD_SIZE]
        take_val = False
        if remaining_val_records > 0:
          take_val = rng.random() < (remaining_val_records / remaining_records)

        if take_val:
          val_buffer.extend(record)
          remaining_val_records -= 1
          written_val_records += 1
        else:
          train_buffer.extend(record)
          written_train_records += 1
        remaining_records -= 1

      if train_buffer:
        train_out.write(train_buffer)
      if val_buffer:
        val_out.write(val_buffer)

  print(
      'Split complete: '
      f'total={total_records} '
      f'train={written_train_records} '
      f'val={written_val_records} '
      f'seed={args.seed}',
      file=sys.stderr)


if __name__ == '__main__':
  main()
