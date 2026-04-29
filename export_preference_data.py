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
import os
import pickle
import struct
import sys
from pathlib import Path

import cshogi
import h5py
import numpy as np

# Constants
PACKED_SFN_SIZE = 40
META_SIZE = 9  # 1 (u8) + 4*2 (u16)

META_PACK = struct.Struct("<BHHHH")  # game_result, move, ply, context_id, weight_q12

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


def _require_u16(value: int, field: str, game_name: str, position_index: int) -> int:
    ivalue = int(value)
    if not (0 <= ivalue <= 0xFFFF):
        raise ValueError(
            f"{field}={ivalue} is out of range for uint16 "
            f"(game={game_name}, position_index={position_index})"
        )
    return ivalue


def _encode_move16(move: int) -> int:
    return int(move) & 0xFFFF


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
    psv_data = bytearray(PACKED_SFN_SIZE * total)
    meta_data = bytearray(META_SIZE * total)

    idx = 0
    with h5py.File(h5_path, "r") as h5:
        for gi, game_name in enumerate(game_names):
            grp = h5[game_name]
            positions = grp["positions"]
            attrs = grp.attrs

            game_result = int(attrs.get("game_result", 0))
            rating_b = game_meta_records[gi]["rating_b"]
            rating_w = game_meta_records[gi]["rating_w"]

            for pi in range(len(positions)):
                rec = positions[pi]
                psv_raw = np.asarray(rec["psv"], dtype=cshogi.PackedSfenValue).reshape(1)
                psv_bytes = psv_raw.tobytes()
                psv_data[idx * PACKED_SFN_SIZE:(idx + 1) * PACKED_SFN_SIZE] = psv_bytes

                move_q = _encode_move16(int(rec["actual_move"]))
                ply_q = _require_u16(int(rec["ply"]), "ply", game_name, pi)

                bd = cshogi.Board()
                bd.set_psfen(psv_raw)
                turn = bd.turn

                if context_type == "player":
                    pkey = "black_player" if turn == cshogi.BLACK else "white_player"
                    fkey = "player_b" if turn == cshogi.BLACK else "player_w"
                    player_name = _attr_str(attrs.get(pkey, attrs.get(fkey)))
                    context_id = player_to_id.get(player_name, player_to_id.get("unknown", 0))
                elif context_type == "bucket":
                    elo_key = "rating_b" if turn == cshogi.BLACK else "rating_w"
                    elo_value = rating_b if elo_key == "rating_b" else rating_w
                    context_id = bucketize_elo(elo_value, elo_bucket_edges)
                else:
                    # elo or none: no bucketing, context_id = 0
                    context_id = 0

                # Compute sample_weight
                if context_type == "player":
                    elo_val = rating_b if rating_b is not None else 1500.0
                else:
                    elo_val = rating_b if rating_b is not None else (
                        rating_w if rating_w is not None else 1500.0
                    )
                sample_weight = compute_sample_weight(
                    elo_val, elo_weight_slope, elo_weight_intercept, elo_weight_min
                )
                context_id = _require_u16(context_id, "context_id", game_name, pi)
                weight_q12 = _require_u16(
                    round(sample_weight * 1000),
                    "sample_weight_q12",
                    game_name,
                    pi,
                )

                META_PACK.pack_into(
                    meta_data,
                    idx * META_SIZE,
                    game_result, move_q, ply_q, context_id, weight_q12,
                )

                idx += 1

    print(f"Pass 2: extracted {idx} records")

    # Shuffle
    if shuffle:
        rng = list(range(total))
        import random as _random
        _random.Random(seed).shuffle(rng)

        shuffled_psv = bytearray(PACKED_SFN_SIZE * total)
        shuffled_meta = bytearray(META_SIZE * total)
        for new_idx, old_idx in enumerate(rng):
            so = old_idx * PACKED_SFN_SIZE
            sn = new_idx * PACKED_SFN_SIZE
            shuffled_psv[sn:sn + PACKED_SFN_SIZE] = psv_data[so:so + PACKED_SFN_SIZE]

            mo = old_idx * META_SIZE
            mn = new_idx * META_SIZE
            shuffled_meta[mn:mn + META_SIZE] = meta_data[mo:mo + META_SIZE]

        psv_data = shuffled_psv
        meta_data = shuffled_meta

    # Write binary files
    with open(psv_path, "wb") as f:
        f.write(psv_data)

    with open(meta_path, "wb") as f:
        f.write(meta_data)

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

    print(f"Written: {psv_path} ({len(psv_data)} bytes)")
    print(f"Written: {meta_path} ({len(meta_data)} bytes)")
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
