"""
Export H5 preference data to a compact binary format that pairs
packed sfen values with per-position metadata for the C++ loader.

The binary file layout consists of two contiguous sections:
  Section 1 (PSV): N x 40 bytes  - PackedSfenValue records
  Section 2 (META): N x 9 bytes  - per-position metadata

META record layout (9 bytes, little-endian):
  Offset  Size  Field              Description
  0       1     game_result        1=win, 0=draw, 255=loss
  1       2     actual_move        u16
  3       2     ply                u16
  5       2     context_id         Elo bucket or player ID (u16)
  7       2     sample_weight_q12  weight x 1000, quantized (u16)

Usage:
  python export_preference_data.py <input.h5> [output_dir]
  python export_preference_data.py <input.h5> --context-type none
  python export_preference_data.py <input.h5> --context-type player
  python export_preference_data.py <input.h5> --context-type bucket \\
      --elo-buckets 1200 1600 2000 2400 --shuffle --seed 42

Output files:
  <output_dir>/train.bin          - PSV section (N x 40 bytes)
  <output_dir>/train_meta.bin     - META section (N x 9 bytes)
  <output_dir>/train.idx.pkl      - Index for Python fallback (pickle)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cshogi
import h5py
import numpy as np

# Constants
PACKED_SFN_SIZE = 40
META_SIZE = 9  # 1 (u8) + 4*2 (u16)

META_DTYPE = np.dtype([
    ("game_result", np.uint8),
    ("actual_move", "<u2"),
    ("ply", "<u2"),
    ("context_id", "<u2"),
    ("sample_weight_q12", "<u2"),
])  # game_result, move, ply, context_id, weight_q12

DEFAULT_ELO_BUCKETS = (1200, 1600, 2000, 2400)


# Helpers

def _attr_str(value, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def bucketize_elo(elo, edges: tuple = DEFAULT_ELO_BUCKETS) -> int:
    if elo is None:
        return 0
    value = float(elo)
    bucket_id = 0
    for edge in edges:
        if value < edge:
            return bucket_id
        bucket_id += 1
    return bucket_id


def compute_sample_weight(
    elo_value,
    slope: float = 0.001,
    intercept: float = 0.5,
    min_weight: float = 0.1,
) -> float:
    if elo_value is not None:
        return max(min_weight, slope * (float(elo_value) - 3000.0) + intercept)
    return intercept


def _extract_turn_ids(psv_array: np.ndarray) -> np.ndarray:
    """Decode side-to-move from PackedSfenValue records."""
    board = cshogi.Board()
    turns = np.empty(len(psv_array), dtype=np.uint8)
    for i, psv in enumerate(psv_array):
        board.set_psfen(np.asarray(psv, dtype=cshogi.PackedSfenValue).reshape(1))
        turns[i] = board.turn
    return turns


# Core export

def export(
    h5_path: str | Path,
    output_dir: str | Path,
    *,
    shuffle: bool = True,
    seed: int = 42,
    context_type: str = "elo",
    elo_bucket_edges: tuple[int, ...] = DEFAULT_ELO_BUCKETS,
    elo_weight_slope: float = 0.001,
    elo_weight_intercept: float = 0.5,
    elo_weight_min: float = 0.1,
) -> Path:
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = h5_path.stem
    psv_path = output_dir / f"{stem}.bin"
    meta_path = output_dir / f"{stem}_meta.bin"
    idx_path = output_dir / f"{stem}.idx.pkl"

    with h5py.File(h5_path, "r") as h5:
        # Pass 1: collect metadata and count positions
        game_names: list[str] = []
        game_positions: list[int] = []
        player_set: set[str] = set()
        game_meta_records: list[dict] = []

        for game_name in sorted(h5.keys()):
            grp = h5[game_name]
            positions = grp.get("positions")
            if positions is None:
                continue
            n = len(positions)
            if n <= 0:
                continue

            game_names.append(game_name)
            game_positions.append(n)

            attrs = grp.attrs
            black = _attr_str(attrs.get("black_player", attrs.get("player_b")))
            white = _attr_str(attrs.get("white_player", attrs.get("player_w")))
            player_set.add(black)
            player_set.add(white)

            game_meta_records.append({
                "black_player": black,
                "white_player": white,
                "game_result": int(attrs.get("game_result", 0)),
                "rating_b": float(attrs.get("rating_b", 0)) if attrs.get("rating_b") is not None else None,
                "rating_w": float(attrs.get("rating_w", 0)) if attrs.get("rating_w") is not None else None,
            })

        total = sum(game_positions)
        print(f"Pass 1: {len(game_names)} games, {total} positions")

        # Build player vocabulary (for player mode)
        if context_type == "player":
            player_to_id = {p: i for i, p in enumerate(sorted(player_set))}
        else:
            player_to_id = {}

    # Pass 2: extract PSV and META records
    psv_data: np.ndarray | None = None
    meta_data = np.empty(total, dtype=META_DTYPE)

    idx = 0
    with h5py.File(h5_path, "r") as h5:
        for gi, game_name in enumerate(game_names):
            grp = h5[game_name]
            positions_ds = grp["positions"]
            positions = positions_ds[:]
            attrs = grp.attrs

            n = len(positions)
            if n == 0:
                continue
            if psv_data is None:
                psv_data = np.empty(total, dtype=positions["psv"].dtype)

            game_result = int(attrs.get("game_result", 0))
            rating_b = game_meta_records[gi]["rating_b"]
            rating_w = game_meta_records[gi]["rating_w"]
            black_player = game_meta_records[gi]["black_player"]
            white_player = game_meta_records[gi]["white_player"]

            psv_chunk = positions["psv"]
            actual_moves = positions["actual_move"].astype(np.uint16, copy=False)
            plys_raw = positions["ply"]

            psv_data[idx:idx + n] = psv_chunk

            meta_slice = meta_data[idx:idx + n]
            meta_slice["game_result"] = game_result
            meta_slice["actual_move"] = actual_moves

            plys = np.asarray(plys_raw, dtype=np.int64)
            if np.any((plys < 0) | (plys > 0xFFFF)):
                bad = int(np.flatnonzero((plys < 0) | (plys > 0xFFFF))[0])
                raise ValueError(
                    f"ply={int(plys[bad])} is out of range for uint16 "
                    f"(game={game_name}, position_index={bad})"
                )
            meta_slice["ply"] = plys.astype(np.uint16, copy=False)

            if context_type == "player":
                turns = _extract_turn_ids(psv_chunk)
                context_ids = np.where(
                    turns == cshogi.BLACK,
                    player_to_id.get(black_player, player_to_id.get("unknown", 0)),
                    player_to_id.get(white_player, player_to_id.get("unknown", 0)),
                ).astype(np.uint16, copy=False)
            elif context_type == "bucket":
                turns = _extract_turn_ids(psv_chunk)
                black_bucket = bucketize_elo(rating_b, elo_bucket_edges)
                white_bucket = bucketize_elo(rating_w, elo_bucket_edges)
                context_ids = np.where(
                    turns == cshogi.BLACK,
                    black_bucket,
                    white_bucket,
                ).astype(np.uint16, copy=False)
            else:
                # elo or none: no bucketing, context_id = 0
                context_ids = np.zeros(n, dtype=np.uint16)

            max_context_id = int(context_ids.max())
            if max_context_id > 0xFFFF:
                raise ValueError(
                    f"context_id={max_context_id} is out of range for uint16 "
                    f"(game={game_name})"
                )
            meta_slice["context_id"] = context_ids

            if context_type == "player":
                elo_val = rating_b if rating_b is not None else 1500.0
            else:
                elo_val = rating_b if rating_b is not None else (
                    rating_w if rating_w is not None else 1500.0
                )
            sample_weight = compute_sample_weight(
                elo_val, elo_weight_slope, elo_weight_intercept, elo_weight_min
            )
            weight_q12 = round(sample_weight * 1000)
            if not (0 <= weight_q12 <= 0xFFFF):
                raise ValueError(
                    f"sample_weight_q12={weight_q12} is out of range for uint16 "
                    f"(game={game_name})"
                )
            meta_slice["sample_weight_q12"] = weight_q12

            idx += n

    print(f"Pass 2: extracted {idx} records")
    if psv_data is None:
        psv_data = np.empty(0, dtype=np.dtype(f"V{PACKED_SFN_SIZE}"))

    # Shuffle
    if shuffle:
        rng = np.random.default_rng(seed).permutation(total)
        psv_data = psv_data[rng]
        meta_data = meta_data[rng]

    # Write binary files
    with open(psv_path, "wb") as f:
        psv_data.tofile(f)

    with open(meta_path, "wb") as f:
        meta_data.tofile(f)

    # Write index file (for Python fallback / verification)
    game_end_offsets: list[int] = []
    cumulative = 0
    for n in game_positions:
        cumulative += n
        game_end_offsets.append(cumulative)

    idx_obj = {
        "game_names": game_names,
        "game_end_offsets": game_end_offsets,
        "player_to_id": player_to_id,
        "game_meta_records": game_meta_records,
        "total_positions": total,
        "shuffle_seed": seed if shuffle else None,
        "context_type": context_type,
    }
    with open(idx_path, "wb") as f:
        pickle.dump(idx_obj, f)

    print(f"Written: {psv_path} ({psv_data.nbytes} bytes)")
    print(f"Written: {meta_path} ({meta_data.nbytes} bytes)")
    print(f"Written: {idx_path} ({len(game_names)} games)")
    return psv_path


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Export H5 preference data to binary format for C++ loader"
    )
    parser.add_argument("h5_path", help="Input HDF5 file")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--context-type",
        choices=["elo", "player", "bucket", "none"],
        default="elo",
        help="Context type for context_id (default: elo). "
             "Use 'bucket' for Elo bucket, 'player' for player ID, "
             "'elo' or 'none' for fixed 0.",
    )
    parser.add_argument(
        "--elo-buckets",
        type=int,
        nargs="+",
        default=[1200, 1600, 2000, 2400],
        help="Elo bucket edges (default: 1200 1600 2000 2400)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle records",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed (default: 42)",
    )
    parser.add_argument(
        "--elo-weight-slope",
        type=float,
        default=0.001,
        help="Weight slope for Elo (default: 0.001)",
    )
    parser.add_argument(
        "--elo-weight-intercept",
        type=float,
        default=0.5,
        help="Weight intercept (default: 0.5)",
    )
    parser.add_argument(
        "--elo-weight-min",
        type=float,
        default=0.1,
        help="Minimum weight (default: 0.1)",
    )

    args = parser.parse_args()

    h5 = Path(args.h5_path)
    if not h5.exists():
        print(f"Error: {h5} does not exist", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else h5.parent

    export(
        h5_path=h5,
        output_dir=output_dir,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        context_type=args.context_type,
        elo_bucket_edges=tuple(args.elo_buckets),
        elo_weight_slope=args.elo_weight_slope,
        elo_weight_intercept=args.elo_weight_intercept,
        elo_weight_min=args.elo_weight_min,
    )


if __name__ == "__main__":
    main()
