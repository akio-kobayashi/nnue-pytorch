#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cshogi

import preference_dataset


def _load_dataset(path: str, context_type: str):
    data_path = Path(path)
    if data_path.suffix == ".bin":
        return preference_dataset.FixedRefBinaryDataset(path, context_type=context_type)
    return preference_dataset.FixedRefH5Dataset(path, context_type=context_type)


def _move_to_usi(board: cshogi.Board, move: int) -> str:
    if hasattr(board, "move_to_usi"):
        return str(board.move_to_usi(int(move)))
    return str(int(move))


def _legal_moves_preview(board: cshogi.Board, limit: int) -> list[str]:
    moves: list[str] = []
    for i, move in enumerate(board.legal_moves):
        if i >= limit:
            break
        moves.append(_move_to_usi(board, int(move)))
    return moves


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect actual_move encoding in preference datasets.")
    parser.add_argument("--data", required=True, help="Preference dataset path (.bin or .h5)")
    parser.add_argument("--context-type", default="bucket", help="Preference context type")
    parser.add_argument("--offset", type=int, default=0, help="Dataset start offset")
    parser.add_argument("--samples", type=int, default=8, help="Number of samples to inspect")
    parser.add_argument("--legal-preview", type=int, default=12, help="How many legal moves to print")
    args = parser.parse_args()

    dataset = _load_dataset(args.data, args.context_type)
    end = min(len(dataset), args.offset + args.samples)
    if args.offset < 0 or args.offset >= len(dataset):
        raise IndexError(f"offset {args.offset} is outside dataset length {len(dataset)}")

    for idx in range(args.offset, end):
        sample = dataset[idx]
        board = cshogi.Board(sample.sfen)
        raw_move = int(sample.actual_move)

        direct_legal = False
        direct_usi = ""
        try:
            direct_legal = bool(board.is_legal(raw_move))
            if direct_legal:
                direct_usi = _move_to_usi(board, raw_move)
        except Exception as exc:
            direct_usi = f"<direct error: {exc}>"

        print(f"[{idx}]")
        print(f"sfen: {sample.sfen}")
        print(f"raw actual_move: {raw_move}")
        print(f"truncated u16: {raw_move & 0xFFFF}")
        print(f"direct legal: {direct_legal}")
        if direct_usi:
            print(f"direct usi: {direct_usi}")
        print(f"legal preview: {_legal_moves_preview(board, args.legal_preview)}")
        print("")


if __name__ == "__main__":
    main()
