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
          "and write a new PackedSfenValue .bin file with updated scores."
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
  parser.add_argument(
      "--serialize-normalize",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Apply serialize-compatible FC weight clipping before inference",
  )
  parser.add_argument(
      "--fv-scale",
      type=float,
      default=16.0,
      help="YaneuraOu FV_SCALE used to convert the serialized NNUE output into the final evaluation value.",
  )
  return parser.parse_args()


def resolve_model_args(args, checkpoint):
  hyper_parameters = checkpoint.get("hyper_parameters", {})
  inferred_args = serialize._infer_model_args_from_state_dict(checkpoint["state_dict"])
  resolved_features = args.features_override or hyper_parameters.get("features", inferred_args["features"])
  resolved_l1_size = args.l1_size or hyper_parameters.get("l1_size", inferred_args["l1_size"])
  resolved_l2_size = args.l2_size or hyper_parameters.get("l2_size", inferred_args["l2_size"])
  resolved_l3_size = args.l3_size or hyper_parameters.get("l3_size", inferred_args["l3_size"])
  inferred_input_adapter = inferred_args.get("input_adapter", "none")
  resolved_input_adapter = hyper_parameters.get("input_adapter", inferred_input_adapter)
  if resolved_input_adapter == "none" and inferred_input_adapter != "none":
    resolved_input_adapter = inferred_input_adapter
  resolved_input_adapter_rank = hyper_parameters.get("input_adapter_rank", inferred_args.get("input_adapter_rank", 8))
  resolved_input_adapter_alpha = hyper_parameters.get("input_adapter_alpha", 1.0)
  resolved_input_adapter_init_std = hyper_parameters.get("input_adapter_init_std", 0.0)
  return (
      resolved_features,
      resolved_l1_size,
      resolved_l2_size,
      resolved_l3_size,
      resolved_input_adapter,
      resolved_input_adapter_rank,
      resolved_input_adapter_alpha,
      resolved_input_adapter_init_std,
  )


def normalize_model_for_serialize_inference(model):
  with torch.no_grad():
    if hasattr(model, "_clip_linear_weight"):
      for child in model.children():
        if not isinstance(child, torch.nn.Linear):
          continue
        if child == model.input:
          continue
        model._clip_linear_weight(child)


def load_model(args):
  checkpoint = torch.load(args.checkpoint, map_location="cpu")
  (
      resolved_features,
      resolved_l1_size,
      resolved_l2_size,
      resolved_l3_size,
      resolved_input_adapter,
      resolved_input_adapter_rank,
      resolved_input_adapter_alpha,
      resolved_input_adapter_init_std,
  ) = resolve_model_args(args, checkpoint)
  nnue = M.NNUE(
      features=resolved_features,
      l1_size=resolved_l1_size,
      l2_size=resolved_l2_size,
      l3_size=resolved_l3_size,
      input_adapter=resolved_input_adapter,
      input_adapter_rank=resolved_input_adapter_rank,
      input_adapter_alpha=resolved_input_adapter_alpha,
      input_adapter_init_std=resolved_input_adapter_init_std,
  )
  state_dict = checkpoint["state_dict"]
  if hasattr(serialize, "_upgrade_legacy_state_dict"):
    state_dict = serialize._upgrade_legacy_state_dict(state_dict, resolved_features)
  if hasattr(serialize, "_strip_non_serializable_state_dict_keys"):
    state_dict = serialize._strip_non_serializable_state_dict_keys(state_dict)
  nnue.load_state_dict(state_dict)
  serialize._load_checkpoint_extras(nnue, checkpoint)
  if args.use_ema and hasattr(nnue, "apply_ema_weights"):
    if not nnue.apply_ema_weights():
      raise RuntimeError("Requested --use_ema but no EMA weights were found in the checkpoint.")
  if args.serialize_normalize:
    normalize_model_for_serialize_inference(nnue)
  nnue.eval()
  feature_set = features.get_feature_set_from_name(resolved_features)
  print(
      "loaded checkpoint:",
      Path(args.checkpoint),
      f"features={resolved_features}",
      f"l1={resolved_l1_size}",
      f"l2={resolved_l2_size}",
      f"l3={resolved_l3_size}",
      f"input_adapter={resolved_input_adapter}",
      f"use_ema={args.use_ema}",
      f"serialize_normalize={args.serialize_normalize}",
      f"fv_scale={args.fv_scale}",
  )
  return nnue, feature_set


def eval_model_batch(model, batch, device, fv_scale: float):
  us, them, white, black, outcome, score, ply = batch.contents.get_tensors(device)
  with torch.inference_mode():
    # Approximate the final YaneuraOu evaluation value:
    # serialized_output ~= model.forward(...) * NNUE_TO_SCORE * model.FV_SCALE
    # final_score = serialized_output / FV_SCALE
    evals = (
        model.forward(us, them, white, black)
        * float(model.NNUE_TO_SCORE)
        * float(model.FV_SCALE)
        / float(fv_scale)
    ).reshape(-1)
  evals = evals.detach().cpu().numpy()
  them_np = them.reshape(-1).detach().cpu().numpy()
  evals[them_np > 0.5] *= -1.0
  evals = np.rint(np.clip(evals, -32768.0, 32767.0)).astype(np.int16)
  return evals


def log_score_samples(input_psv, output_scores, start, limit=3):
  count = min(limit, len(output_scores))
  for offset in range(count):
    record = input_psv[start + offset]
    print(
        f"sample[{offset + 1}]",
        f"old_score={int(record['score'])}",
        f"new_score={int(output_scores[offset])}",
        f"ply={int(record['gamePly'])}",
        f"result={int(record['game_result'])}",
    )


def main():
  args = parse_args()
  if args.batch_size <= 0:
    raise ValueError("--batch-size must be > 0")
  if args.fv_scale <= 0:
    raise ValueError("--fv-scale must be > 0")

  model, feature_set = load_model(args)
  device = torch.device(args.device)
  model.to(device)

  input_psv = np.memmap(args.input_bin, dtype=cshogi.PackedSfenValue, mode="r")
  output_path = Path(args.output_bin)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_psv = np.memmap(output_path, dtype=cshogi.PackedSfenValue, mode="w+", shape=input_psv.shape)
  output_psv[:] = input_psv[:]

  print(
      "relabel input:",
      Path(args.input_bin),
      f"records={len(input_psv)}",
      f"device={device}",
      f"batch_size={args.batch_size}",
  )
  print("relabel output:", output_path)

  board = cshogi.Board()
  total = len(input_psv)
  logged_samples = False
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
      output_scores = eval_model_batch(model, batch, device, args.fv_scale)
    finally:
      nnue_dataset.destroy_sparse_batch(batch)

    output_psv[start:end]["score"] = output_scores
    if not logged_samples:
      log_score_samples(input_psv, output_scores, start)
      logged_samples = True

    processed = end
    if processed == total or processed % max(args.batch_size * 32, 1) == 0:
      print(f"processed {processed}/{total}")

  output_psv.flush()
  print(f"saved {output_path}")


if __name__ == "__main__":
  main()
