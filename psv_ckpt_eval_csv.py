import argparse
import csv
from pathlib import Path

import cshogi

from psv_re_eval import eval_model_batch, iter_bin_batches, load_model, records_to_sparse_batch
import nnue_dataset


def parse_args():
  parser = argparse.ArgumentParser(
      description=(
          "Evaluate PackedSfenValue .bin records with a nnue-pytorch checkpoint "
          "and write comparison-friendly CSV output."
      ))
  parser.add_argument("checkpoint", help="Path to a .ckpt checkpoint")
  parser.add_argument("input_bin", help="Input PackedSfenValue .bin")
  parser.add_argument("output_csv", help="Output CSV path")
  parser.add_argument("--batch-size", type=int, default=1024, help="Inference batch size")
  parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0")
  parser.add_argument("--features", dest="features_override", default=None, help="Optional feature override")
  parser.add_argument("--l1_size", type=int, default=None)
  parser.add_argument("--l2_size", type=int, default=None)
  parser.add_argument("--l3_size", type=int, default=None)
  parser.add_argument("--use_ema", action="store_true", help="Use EMA weights saved in the checkpoint")
  parser.add_argument(
      "--serialize-normalize",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Apply serialize-compatible FC weight clipping before inference",
  )
  parser.add_argument("--max-records", type=int, default=0, help="Maximum number of records to evaluate. 0 means all.")
  return parser.parse_args()


def main():
  args = parse_args()
  if args.batch_size <= 0:
    raise ValueError("--batch-size must be > 0")
  if args.max_records < 0:
    raise ValueError("--max-records must be >= 0")

  model, feature_set = load_model(args)
  device = args.device
  model.to(device)

  input_path = Path(args.input_bin)
  if not input_path.exists():
    raise FileNotFoundError(f"{input_path} does not exist")

  output_path = Path(args.output_csv)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  print(
      "csv eval input:",
      input_path,
      f"device={device}",
      f"batch_size={args.batch_size}",
      f"max_records={args.max_records if args.max_records else 'all'}",
  )
  print("csv eval output:", output_path)

  board = cshogi.Board()
  processed = 0
  remaining = args.max_records

  with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["index", "sfen", "orig_score", "ckpt_score", "ply", "result"],
    )
    writer.writeheader()

    for batch_records in iter_bin_batches(input_path, args.batch_size):
      if remaining > 0:
        batch_records = batch_records[:remaining]
      if not batch_records:
        break

      batch = records_to_sparse_batch(feature_set, batch_records)
      try:
        scores = eval_model_batch(model, batch, device)
      finally:
        nnue_dataset.destroy_sparse_batch(batch)

      for offset, (record, ckpt_score) in enumerate(zip(batch_records, scores)):
        board.set_psfen(record[:32])
        writer.writerow({
            "index": processed + offset,
            "sfen": board.sfen(),
            "orig_score": int.from_bytes(record[32:34], byteorder="little", signed=True),
            "ckpt_score": int(ckpt_score),
            "ply": int.from_bytes(record[36:38], byteorder="little", signed=False),
            "result": int.from_bytes(record[38:39], byteorder="little", signed=True),
        })

      processed += len(batch_records)
      if remaining > 0:
        remaining -= len(batch_records)
        if remaining <= 0:
          break

      if processed % max(args.batch_size * 32, 1) == 0:
        print(f"processed {processed}")

  print(f"saved {output_path}")


if __name__ == "__main__":
  main()
