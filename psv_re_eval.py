import argparse
from pathlib import Path

import numpy as np
import torch

import cshogi
import features
import model as M
import nnue_dataset
import serialize


def parse_args():
  parser = argparse.ArgumentParser(
      description=(
          "Load a trained nnue-pytorch checkpoint, evaluate PackedSfenValue records in batches, "
          "replace only the score field, and write a new PackedSfenValue .bin file."
      ))
  parser.add_argument("checkpoint", help="Path to a .ckpt checkpoint")
  parser.add_argument("input_bin", help="Input PackedSfenValue .bin")
  parser.add_argument("output_bin", help="Output PackedSfenValue .bin")
  parser.add_argument("--batch-size", type=int, default=1024, help="Inference batch size")
  parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0")
  parser.add_argument("--features", dest="features_override", default=None, help="Optional feature override")
  parser.add_argument("--l1_size", type=int, default=None)
  parser.add_argument("--l2_size", type=int, default=None)
  parser.add_argument("--l3_size", type=int, default=None)
  parser.add_argument("--use_ema", action="store_true", help="Use EMA weights saved in the checkpoint")
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
  nnue.eval()
  return nnue, features.get_feature_set_from_name(resolved_features)


def eval_model_batch(model, batch, device):
  us, them, white, black, outcome, score, ply = batch.contents.get_tensors(device)
  with torch.inference_mode():
    evals = (model.forward(us, them, white, black) * 600.0).reshape(-1)
  evals = evals.detach().cpu().numpy()
  them_np = them.reshape(-1).detach().cpu().numpy()
  evals[them_np > 0.5] *= -1.0
  evals = np.rint(np.clip(evals, -32768.0, 32767.0)).astype(np.int16)
  return evals


def main():
  args = parse_args()
  if args.batch_size <= 0:
    raise ValueError("--batch-size must be > 0")

  model, feature_set = load_model(args)
  device = torch.device(args.device)
  model.to(device)

  input_psv = np.memmap(args.input_bin, dtype=cshogi.PackedSfenValue, mode="r")
  output_path = Path(args.output_bin)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_psv = np.memmap(output_path, dtype=cshogi.PackedSfenValue, mode="w+", shape=input_psv.shape)
  output_psv[:] = input_psv[:]

  board = cshogi.Board()
  total = len(input_psv)
  for start in range(0, total, args.batch_size):
    end = min(start + args.batch_size, total)
    fens = []
    scores = []
    plies = []
    results = []

    for record in input_psv[start:end]:
      board.set_psfen(record["sfen"])
      fens.append(board.sfen())
      scores.append(int(record["score"]))
      plies.append(int(record["gamePly"]))
      results.append(int(record["game_result"]))

    batch = nnue_dataset.make_sparse_batch_from_fens(feature_set, fens, scores, plies, results)
    try:
      output_psv[start:end]["score"] = eval_model_batch(model, batch, device)
    finally:
      nnue_dataset.destroy_sparse_batch(batch)

    processed = end
    if processed == total or processed % max(args.batch_size * 32, 1) == 0:
      print(f"processed {processed}/{total}")

  output_psv.flush()
  print(f"saved {output_path}")


if __name__ == "__main__":
  main()
