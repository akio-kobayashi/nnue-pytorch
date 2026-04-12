import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

import cshogi
import features
import model as M
import nnue_dataset
import serialize


PACKED_SFEN_VALUE_BYTES = 40
PACKED_SFEN_BYTES = 32
SCORE_OFFSET = 32
MOVE_OFFSET = 34
PLY_OFFSET = 36
RESULT_OFFSET = 38
PADDING_OFFSET = 39


def parse_args():
  parser = argparse.ArgumentParser(
      description=(
          "Load a trained nnue-pytorch checkpoint, evaluate positions in batches, "
          "and write PackedSfenValue records with updated scores."
      ))
  parser.add_argument("checkpoint", help="Path to a .ckpt checkpoint")
  parser.add_argument("input_path", help="Input PackedSfenValue .bin or SFEN text file")
  parser.add_argument("output_bin", help="Output PackedSfenValue .bin")
  parser.add_argument("--input-format", choices=["auto", "bin", "sfen"], default="auto")
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
  parser.add_argument("--default-result", type=int, default=0, choices=[-1, 0, 1], help="game_result to use for SFEN text input")
  parser.add_argument("--default-move", type=int, default=0, help="move to use for SFEN text input")
  parser.add_argument("--default-ply", type=int, default=1, help="fallback ply for SFEN text input without a valid ply token")
  return parser.parse_args()


def resolve_model_args(args, checkpoint):
  hyper_parameters = checkpoint.get("hyper_parameters", {})
  inferred_args = serialize._infer_model_args_from_state_dict(checkpoint["state_dict"])
  resolved_features = args.features_override or hyper_parameters.get("features", inferred_args["features"])
  resolved_l1_size = args.l1_size or hyper_parameters.get("l1_size", inferred_args["l1_size"])
  resolved_l2_size = args.l2_size or hyper_parameters.get("l2_size", inferred_args["l2_size"])
  resolved_l3_size = args.l3_size or hyper_parameters.get("l3_size", inferred_args["l3_size"])
  return resolved_features, resolved_l1_size, resolved_l2_size, resolved_l3_size


def load_model(args):
  checkpoint = torch.load(args.checkpoint, map_location="cpu")
  resolved_features, resolved_l1_size, resolved_l2_size, resolved_l3_size = resolve_model_args(args, checkpoint)
  nnue = M.NNUE(
      features=resolved_features,
      l1_size=resolved_l1_size,
      l2_size=resolved_l2_size,
      l3_size=resolved_l3_size,
  )
  upgraded_state_dict = serialize._upgrade_legacy_state_dict(checkpoint["state_dict"], resolved_features)
  nnue.load_state_dict(upgraded_state_dict)
  serialize._load_checkpoint_extras(nnue, checkpoint)
  if args.use_ema and hasattr(nnue, "apply_ema_weights"):
    if not nnue.apply_ema_weights():
      raise RuntimeError("Requested --use_ema but no EMA weights were found in the checkpoint.")
  if args.serialize_normalize:
    normalize_model_for_serialize_inference(nnue)
  nnue.eval()
  return nnue, features.get_feature_set_from_name(resolved_features)


def normalize_model_for_serialize_inference(model):
  with torch.no_grad():
    if hasattr(model, "_clip_linear_weight"):
      for child in model.children():
        if not isinstance(child, torch.nn.Linear):
          continue
        if child == model.input:
          continue
        model._clip_linear_weight(child)


def eval_model_batch(model, batch, device):
  us, them, white, black, outcome, score, ply = batch.contents.get_tensors(device)
  with torch.inference_mode():
    evals = (model.forward(us, them, white, black) * float(model.NNUE_TO_SCORE)).reshape(-1)
  evals = evals.detach().cpu()
  them_mask = them.reshape(-1).detach().cpu() > 0.5
  evals[them_mask] *= -1.0
  evals = torch.clamp(torch.round(evals), -32768.0, 32767.0).to(torch.int16)
  return evals.tolist()


def detect_input_format(path: Path, requested: str) -> str:
  if requested != "auto":
    return requested
  if path.suffix.lower() == ".bin":
    return "bin"
  return "sfen"


def record_sfen_bytes(record: bytes) -> bytes:
  return record[:PACKED_SFEN_BYTES]


def record_score(record: bytes) -> int:
  return int.from_bytes(record[SCORE_OFFSET:SCORE_OFFSET + 2], byteorder="little", signed=True)


def record_move(record: bytes) -> int:
  return int.from_bytes(record[MOVE_OFFSET:MOVE_OFFSET + 2], byteorder="little", signed=False)


def record_ply(record: bytes) -> int:
  return int.from_bytes(record[PLY_OFFSET:PLY_OFFSET + 2], byteorder="little", signed=False)


def record_result(record: bytes) -> int:
  return int.from_bytes(record[RESULT_OFFSET:RESULT_OFFSET + 1], byteorder="little", signed=True)


def build_record(sfen_bytes: bytes, score: int, move: int, ply: int, result: int) -> bytes:
  if len(sfen_bytes) != PACKED_SFEN_BYTES:
    raise ValueError(f"Packed SFEN must be {PACKED_SFEN_BYTES} bytes, got {len(sfen_bytes)}")
  out = bytearray(PACKED_SFEN_VALUE_BYTES)
  out[:PACKED_SFEN_BYTES] = sfen_bytes
  out[SCORE_OFFSET:SCORE_OFFSET + 2] = int(score).to_bytes(2, byteorder="little", signed=True)
  out[MOVE_OFFSET:MOVE_OFFSET + 2] = int(move).to_bytes(2, byteorder="little", signed=False)
  out[PLY_OFFSET:PLY_OFFSET + 2] = int(ply).to_bytes(2, byteorder="little", signed=False)
  out[RESULT_OFFSET:RESULT_OFFSET + 1] = int(result).to_bytes(1, byteorder="little", signed=True)
  out[PADDING_OFFSET:PADDING_OFFSET + 1] = b"\x00"
  return bytes(out)


def validate_bin_size(path: Path) -> int:
  size = path.stat().st_size
  remainder = size % PACKED_SFEN_VALUE_BYTES
  if remainder != 0:
    raise ValueError(
        f"{path} size {size} is not a multiple of PackedSfenValue({PACKED_SFEN_VALUE_BYTES}) bytes."
    )
  return size // PACKED_SFEN_VALUE_BYTES


def extract_ply_from_sfen(sfen: str, default_ply: int) -> int:
  tokens = sfen.strip().split()
  if tokens and tokens[-1].isdigit():
    ply = int(tokens[-1])
    if ply > 0:
      return ply
  return default_ply


def iter_sfen_records(path: Path, default_ply: int, default_move: int, default_result: int) -> Iterator[bytes]:
  board = cshogi.Board()
  with open(path, "r", encoding="utf-8") as f:
    for line_no, raw_line in enumerate(f, start=1):
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      sfen = line[5:].strip() if line.startswith("sfen ") else line
      try:
        board.set_sfen(sfen)
      except Exception as exc:
        raise ValueError(f"Invalid SFEN at line {line_no}: {sfen}") from exc
      packed = bytes(board.to_psfen())
      ply = extract_ply_from_sfen(sfen, default_ply)
      yield build_record(packed, 0, default_move, ply, default_result)


def count_sfen_records(path: Path) -> int:
  total = 0
  with open(path, "r", encoding="utf-8") as f:
    for raw_line in f:
      line = raw_line.strip()
      if line and not line.startswith("#"):
        total += 1
  return total


def load_input_spec(args) -> tuple[str, Path, int | None]:
  input_path = Path(args.input_path)
  if not input_path.exists():
    raise FileNotFoundError(f"{input_path} does not exist")
  input_format = detect_input_format(input_path, args.input_format)
  if input_format == "bin":
    return input_format, input_path, validate_bin_size(input_path)
  return input_format, input_path, count_sfen_records(input_path)


def iter_bin_batches(path: Path, batch_size: int) -> Iterator[list[bytes]]:
  with open(path, "rb") as f:
    while True:
      chunk = f.read(batch_size * PACKED_SFEN_VALUE_BYTES)
      if not chunk:
        return
      if len(chunk) % PACKED_SFEN_VALUE_BYTES != 0:
        raise ValueError(
            f"Read a truncated chunk from {path}; size {len(chunk)} is not aligned to PackedSfenValue."
        )
      yield [
          chunk[offset:offset + PACKED_SFEN_VALUE_BYTES]
          for offset in range(0, len(chunk), PACKED_SFEN_VALUE_BYTES)
      ]


def iter_sfen_batches(
    path: Path,
    batch_size: int,
    default_ply: int,
    default_move: int,
    default_result: int,
) -> Iterator[list[bytes]]:
  batch = []
  for record in iter_sfen_records(path, default_ply, default_move, default_result):
    batch.append(record)
    if len(batch) >= batch_size:
      yield batch
      batch = []
  if batch:
    yield batch


def iter_input_batches(args, input_format: str, input_path: Path) -> Iterator[list[bytes]]:
  if input_format == "bin":
    yield from iter_bin_batches(input_path, args.batch_size)
    return
  yield from iter_sfen_batches(
      input_path,
      args.batch_size,
      default_ply=args.default_ply,
      default_move=args.default_move,
      default_result=args.default_result,
  )


def records_to_sparse_batch(feature_set, records: list[bytes]):
  board = cshogi.Board()
  fens = []
  scores = []
  plies = []
  results = []
  for record in records:
    board.set_psfen(np.frombuffer(record_sfen_bytes(record), dtype=np.uint8))
    fens.append(board.sfen())
    scores.append(record_score(record))
    plies.append(record_ply(record))
    results.append(record_result(record))
  return nnue_dataset.make_sparse_batch_from_fens(feature_set, fens, scores, plies, results)


def main():
  args = parse_args()
  if args.batch_size <= 0:
    raise ValueError("--batch-size must be > 0")

  model, feature_set = load_model(args)
  device = torch.device(args.device)
  model.to(device)

  input_format, input_path, total = load_input_spec(args)
  output_path = Path(args.output_bin)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  processed = 0
  with open(output_path, "wb") as out_file:
    for batch_records_ in iter_input_batches(args, input_format, input_path):
      batch = records_to_sparse_batch(feature_set, batch_records_)
      try:
        scores = eval_model_batch(model, batch, device)
      finally:
        nnue_dataset.destroy_sparse_batch(batch)

      for record, score in zip(batch_records_, scores):
        out_file.write(build_record(
            record_sfen_bytes(record),
            score,
            record_move(record),
            record_ply(record),
            record_result(record),
        ))

      processed += len(batch_records_)
      if processed % max(args.batch_size * 32, 1) == 0 or (total is not None and processed == total):
        if total is None:
          print(f"processed {processed}")
        else:
          print(f"processed {processed}/{total}")

  print(f"saved {output_path}")


if __name__ == "__main__":
  main()
