from __future__ import annotations

import cshogi

MOVE_DROP = 1 << 14
MOVE_PROMOTE = 1 << 15


def _find_cshogi_attr(*names: str):
    for name in names:
        value = getattr(cshogi, name, None)
        if callable(value):
            return value
    raise AttributeError(f"cshogi is missing required helper; tried {names}")


_move_from = _find_cshogi_attr("move_from")
_move_to = _find_cshogi_attr("move_to")
_move_is_drop = _find_cshogi_attr("move_is_drop")
_move_is_promotion = _find_cshogi_attr("move_is_promotion", "move_is_promote")
_move_drop_hand_piece = _find_cshogi_attr("move_drop_hand_piece")


def decode_move16_to_cshogi_move(board: cshogi.Board, move16: int) -> int:
    raw = int(move16) & 0xFFFF
    to_sq = raw & 0x7F
    from_or_pt = (raw >> 7) & 0x7F
    is_drop = (raw & MOVE_DROP) != 0
    is_promote = (raw & MOVE_PROMOTE) != 0

    for move in board.legal_moves:
        move_int = int(move)
        if _move_to(move_int) != to_sq:
            continue
        if bool(_move_is_drop(move_int)) != is_drop:
            continue
        if bool(_move_is_promotion(move_int)) != is_promote:
            continue
        if is_drop:
            if _move_drop_hand_piece(move_int) != from_or_pt:
                continue
        else:
            if _move_from(move_int) != from_or_pt:
                continue
        return move_int

    raise ValueError(
        f"could not decode Move16={raw} for sfen={board.sfen()}"
    )
