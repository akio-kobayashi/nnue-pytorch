#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cshogi
import torch

import features as features_module
import nnue_dataset
import preference_dataset


def _load_dataset(path: str, context_type: str):
    data_path = Path(path)
    if data_path.suffix == ".bin":
        return preference_dataset.FixedRefBinaryDataset(path, context_type=context_type)
    return preference_dataset.FixedRefH5Dataset(path, context_type=context_type)


def _build_after_sfen(sfen: str, move: int) -> str:
    board = cshogi.Board(sfen)
    move_int = int(move)
    if not board.is_legal(move_int):
        raise ValueError(f"illegal move {move_int} for sfen: {sfen}")
    board.push(move_int)
    return board.sfen()


def _check_sparse_tensor(name: str, tensor: torch.Tensor, expected_batch: int, expected_features: int) -> None:
    if not tensor.is_sparse:
        raise TypeError(f"{name} is not sparse")
    coalesced = tensor.coalesce()
    if tuple(coalesced.shape) != (expected_batch, expected_features):
        raise ValueError(
            f"{name} shape mismatch: got {tuple(coalesced.shape)}, "
            f"expected {(expected_batch, expected_features)}"
        )
    indices = coalesced.indices()
    values = coalesced.values()
    if indices.numel() == 0:
        return
    row_min = int(indices[0].min())
    row_max = int(indices[0].max())
    col_min = int(indices[1].min())
    col_max = int(indices[1].max())
    if row_min < 0 or row_max >= expected_batch:
        raise ValueError(f"{name} row range invalid: [{row_min}, {row_max}] vs batch={expected_batch}")
    if col_min < 0 or col_max >= expected_features:
        raise ValueError(f"{name} col range invalid: [{col_min}, {col_max}] vs features={expected_features}")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check fixed-ref SFEN -> sparse tensor rebuild path.")
    parser.add_argument("--data", required=True, help="Preference dataset path (.bin or .h5)")
    parser.add_argument("--features", default="HalfKP", help="Feature set name")
    parser.add_argument("--context-type", default="bucket", help="Preference context type")
    parser.add_argument("--samples", type=int, default=32, help="Number of samples to check")
    parser.add_argument("--offset", type=int, default=0, help="Dataset start offset")
    parser.add_argument(
        "--mode",
        choices=("current", "after"),
        default="after",
        help="Check current SFENs or actual-move after SFENs",
    )
    parser.add_argument(
        "--builder",
        choices=("python", "dll"),
        default="dll",
        help="Sparse batch builder to use",
    )
    parser.add_argument("--device", default="cpu", help="Target torch device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    dataset = _load_dataset(args.data, args.context_type)
    feature_set = features_module.get_feature_set_from_name(args.features)
    end = min(len(dataset), args.offset + args.samples)
    if args.offset < 0 or args.offset >= len(dataset):
        raise IndexError(f"offset {args.offset} is outside dataset length {len(dataset)}")
    if end <= args.offset:
        raise ValueError("no samples selected")

    sfens: list[str] = []
    plies: list[int] = []
    for idx in range(args.offset, end):
        sample = dataset[idx]
        if args.mode == "current":
            sfens.append(sample.sfen)
            plies.append(int(sample.ply))
        else:
            sfens.append(_build_after_sfen(sample.sfen, int(sample.actual_move)))
            plies.append(int(sample.ply) + 1)

    batch = nnue_dataset.make_sparse_batch_from_fens(
        feature_set,
        sfens,
        [0] * len(sfens),
        plies,
        [0] * len(sfens),
        prefer_python_builder=(args.builder == "python"),
    )

    try:
        tensors = batch.contents.get_tensors(torch.device(args.device))
        us, them, white, black = tensors[:4]
    finally:
        nnue_dataset.destroy_sparse_batch(batch)

    if args.device == "cpu":
        white = white.clone()
        black = black.clone()

    expected_batch = len(sfens)
    expected_features = feature_set.num_features
    _check_sparse_tensor("white", white, expected_batch, expected_features)
    _check_sparse_tensor("black", black, expected_batch, expected_features)

    if us.shape[0] != expected_batch or them.shape[0] != expected_batch:
        raise ValueError(f"us/them batch mismatch: us={tuple(us.shape)} them={tuple(them.shape)}")

    print("OK")
    print(f"dataset={args.data}")
    print(f"builder={args.builder}")
    print(f"mode={args.mode}")
    print(f"samples={expected_batch}")
    print(f"features={feature_set.name} ({expected_features})")
    print(f"device={args.device}")


if __name__ == "__main__":
    main()
