#!/usr/bin/env python3

import argparse
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path


PACKED_SFEN_RECORD_SIZE = 40
DEFAULT_BUFFER_RECORDS = 8192
DEFAULT_NUM_BUCKETS = 256


def parse_args():
  parser = argparse.ArgumentParser(
      description=(
          'Shuffle a packed sfen binary by randomly distributing records into '
          'temporary buckets, shuffling each bucket, and concatenating them. '
          'This is an approximate external shuffle intended for large train.bin files.'
      ))
  parser.add_argument('input', help='path to the input packed sfen binary')
  parser.add_argument('output', help='path to the shuffled output binary')
  parser.add_argument(
      '--seed',
      type=int,
      default=0,
      help='random seed used for bucket assignment and intra-bucket shuffle (default: 0)')
  parser.add_argument(
      '--buffer-records',
      type=int,
      default=DEFAULT_BUFFER_RECORDS,
      help='records processed per read chunk (default: 8192)')
  parser.add_argument(
      '--num-buckets',
      type=int,
      default=DEFAULT_NUM_BUCKETS,
      help='number of temporary buckets used for shuffling (default: 256)')
  parser.add_argument(
      '--temp-dir',
      default=None,
      help='directory used for temporary bucket files (default: system temp dir)')
  parser.add_argument(
      '--keep-temp',
      action='store_true',
      help='keep temporary bucket files after completion for inspection')
  return parser.parse_args()


def fail(message):
  print(f'error: {message}', file=sys.stderr)
  raise SystemExit(1)


def validate_args(args):
  if args.buffer_records <= 0:
    fail('--buffer-records must be a positive integer')
  if args.num_buckets <= 1:
    fail('--num-buckets must be at least 2')
  if os.path.abspath(args.input) == os.path.abspath(args.output):
    fail('output path must differ from the input path')


def ensure_parent_dir(path):
  parent = os.path.dirname(os.path.abspath(path))
  if parent:
    os.makedirs(parent, exist_ok=True)


def iter_records(blob):
  for offset in range(0, len(blob), PACKED_SFEN_RECORD_SIZE):
    yield blob[offset:offset + PACKED_SFEN_RECORD_SIZE]


def bucket_paths(temp_root, num_buckets):
  return [os.path.join(temp_root, f'bucket_{i:04d}.bin') for i in range(num_buckets)]


def distribute_to_buckets(input_path, bucket_file_paths, rng, buffer_records):
  chunk_size = buffer_records * PACKED_SFEN_RECORD_SIZE
  bucket_counts = [0] * len(bucket_file_paths)
  total_records = 0

  with open(input_path, 'rb') as src:
    writers = [open(path, 'wb') for path in bucket_file_paths]
    try:
      while True:
        chunk = src.read(chunk_size)
        if not chunk:
          break
        if len(chunk) % PACKED_SFEN_RECORD_SIZE != 0:
          fail('read a partial record from the input file')

        bucket_buffers = [bytearray() for _ in bucket_file_paths]
        for record in iter_records(chunk):
          bucket_idx = rng.randrange(len(bucket_file_paths))
          bucket_buffers[bucket_idx].extend(record)
          bucket_counts[bucket_idx] += 1
          total_records += 1

        for bucket_idx, bucket_buffer in enumerate(bucket_buffers):
          if bucket_buffer:
            writers[bucket_idx].write(bucket_buffer)
    finally:
      for writer in writers:
        writer.close()

  return total_records, bucket_counts


def shuffle_bucket_file(path, rng):
  size = os.path.getsize(path)
  if size == 0:
    return 0
  if size % PACKED_SFEN_RECORD_SIZE != 0:
    fail(f'bucket file {path} has invalid size {size}')

  with open(path, 'rb') as f:
    blob = f.read()

  records = list(iter_records(blob))
  rng.shuffle(records)
  with open(path, 'wb') as f:
    for record in records:
      f.write(record)
  return len(records)


def write_shuffled_output(output_path, bucket_file_paths, rng):
  order = list(range(len(bucket_file_paths)))
  rng.shuffle(order)
  written_records = 0
  with open(output_path, 'wb') as out:
    for bucket_idx in order:
      path = bucket_file_paths[bucket_idx]
      with open(path, 'rb') as src:
        while True:
          chunk = src.read(1024 * 1024 * 8)
          if not chunk:
            break
          out.write(chunk)
      written_records += os.path.getsize(path) // PACKED_SFEN_RECORD_SIZE
  return written_records


def main():
  args = parse_args()
  validate_args(args)

  input_size = os.path.getsize(args.input)
  if input_size % PACKED_SFEN_RECORD_SIZE != 0:
    fail(
        f'input size {input_size} is not a multiple of '
        f'{PACKED_SFEN_RECORD_SIZE} bytes')

  ensure_parent_dir(args.output)
  rng = random.Random(args.seed)
  temp_root = tempfile.mkdtemp(prefix='shuffle_packedsfen_', dir=args.temp_dir)
  bucket_file_paths = bucket_paths(temp_root, args.num_buckets)

  try:
    total_records, bucket_counts = distribute_to_buckets(
        args.input, bucket_file_paths, rng, args.buffer_records)

    max_bucket_records = 0
    non_empty_buckets = 0
    for path in bucket_file_paths:
      bucket_records = shuffle_bucket_file(path, rng)
      if bucket_records:
        non_empty_buckets += 1
        max_bucket_records = max(max_bucket_records, bucket_records)

    written_records = write_shuffled_output(args.output, bucket_file_paths, rng)
    if written_records != total_records:
      fail(
          f'output record count mismatch: expected {total_records}, '
          f'wrote {written_records}')

    print(
        'Shuffle complete: '
        f'total={total_records} '
        f'buckets={args.num_buckets} '
        f'non_empty_buckets={non_empty_buckets} '
        f'max_bucket_records={max_bucket_records} '
        f'seed={args.seed}',
        file=sys.stderr)
  finally:
    if args.keep_temp:
      print(f'kept temporary bucket files in {temp_root}', file=sys.stderr)
    else:
      shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == '__main__':
  main()
