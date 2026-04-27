from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cshogi
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_ELO_BUCKETS = (1200, 1600, 2000, 2400)


@dataclass(frozen=True)
class ContextValue:
  context_type: str
  context_id: int
  context_label: str


@dataclass(frozen=True)
class FixedRefSample:
  sfen: str
  actual_move: int
  ply: int
  game_result: int
  context: ContextValue
  sample_weight: float
  metadata: dict[str, Any]


def bucketize_elo(elo: float | int | None, bucket_edges: tuple[int, ...] = DEFAULT_ELO_BUCKETS) -> int:
  if elo is None:
    return 0
  value = float(elo)
  bucket_id = 0
  for edge in bucket_edges:
    if value < edge:
      return bucket_id
    bucket_id += 1
  return bucket_id


def packed_sfen_field_view(record: np.void) -> np.ndarray:
  return np.asarray(record["psv"], dtype=cshogi.PackedSfenValue).reshape(1)


def _decode_position_sfen(record: np.void) -> str:
  board = cshogi.Board()
  board.set_psfen(packed_sfen_field_view(record))
  return board.sfen()


def _attr_to_str(value: Any, default: str = "unknown") -> str:
  if value is None:
    return default
  if isinstance(value, bytes):
    return value.decode("utf-8", errors="replace")
  return str(value)


class FixedRefH5Dataset(Dataset):
  """
  HDF5 decision-point dataset for the fixed-reference preference route.

  This first slice intentionally returns only the current SFEN, actual move,
  and unified context information. Candidate generation with the fixed
  reference model V0 will be added later in the collate/batch-builder path.
  """

  def __init__(
      self,
      h5_path: str | Path,
      context_type: str = "elo",
      elo_bucket_edges: tuple[int, ...] = DEFAULT_ELO_BUCKETS,
      elo_weight_slope: float = 0.001,
      elo_weight_intercept: float = 0.5,
      elo_weight_min: float = 0.1,
  ) -> None:
    super().__init__()
    if context_type not in {"elo", "player"}:
      raise ValueError(f"Unsupported context_type: {context_type}")
    self.h5_path = str(h5_path)
    self.context_type = context_type
    self.elo_bucket_edges = tuple(int(v) for v in elo_bucket_edges)
    self.elo_weight_slope = float(elo_weight_slope)
    self.elo_weight_intercept = float(elo_weight_intercept)
    self.elo_weight_min = float(elo_weight_min)
    self._h5: h5py.File | None = None
    self._index = self._build_index()
    self._player_to_id = self._build_player_vocab()

  def _build_index(self) -> list[tuple[str, int]]:
    index: list[tuple[str, int]] = []
    with h5py.File(self.h5_path, "r") as h5_file:
      for game_name in sorted(h5_file.keys()):
        positions = h5_file[game_name].get("positions")
        if positions is None:
          continue
        for pos_idx in range(len(positions)):
          index.append((game_name, pos_idx))
    return index

  def _build_player_vocab(self) -> dict[str, int]:
    players: set[str] = set()
    with h5py.File(self.h5_path, "r") as h5_file:
      for game_name in sorted(h5_file.keys()):
        attrs = h5_file[game_name].attrs
        black = _attr_to_str(attrs.get("black_player", attrs.get("player_b")))
        white = _attr_to_str(attrs.get("white_player", attrs.get("player_w")))
        players.add(black)
        players.add(white)
    return {player: idx for idx, player in enumerate(sorted(players))}

  def _ensure_open(self) -> h5py.File:
    if self._h5 is None:
      self._h5 = h5py.File(self.h5_path, "r")
    return self._h5

  def __len__(self) -> int:
    return len(self._index)

  def _resolve_player_context(self, attrs: h5py.AttributeManager, turn: int) -> ContextValue:
    player_key = "black_player" if turn == cshogi.BLACK else "white_player"
    fallback_key = "player_b" if turn == cshogi.BLACK else "player_w"
    player_name = _attr_to_str(attrs.get(player_key, attrs.get(fallback_key)))
    context_id = self._player_to_id.get(player_name, self._player_to_id.get("unknown", 0))
    return ContextValue(
        context_type="player",
        context_id=context_id,
        context_label=player_name,
    )

  def _resolve_elo_context(self, attrs: h5py.AttributeManager, turn: int) -> ContextValue:
    elo_key = "rating_b" if turn == cshogi.BLACK else "rating_w"
    elo_value = attrs.get(elo_key)
    bucket_id = bucketize_elo(elo_value, self.elo_bucket_edges)
    bucket_label = f"elo_bucket:{bucket_id}"
    return ContextValue(
        context_type="elo",
        context_id=bucket_id,
        context_label=bucket_label,
    )

  def _resolve_context(self, attrs: h5py.AttributeManager, sfen: str) -> ContextValue:
    board = cshogi.Board(sfen)
    if self.context_type == "player":
      return self._resolve_player_context(attrs, board.turn)
    return self._resolve_elo_context(attrs, board.turn)

  def __getitem__(self, idx: int) -> FixedRefSample:
    h5_file = self._ensure_open()
    game_name, pos_idx = self._index[idx]
    group = h5_file[game_name]
    position = group["positions"][pos_idx]
    sfen = _decode_position_sfen(position)
    context = self._resolve_context(group.attrs, sfen)
    game_result = int(group.attrs.get("game_result", 0))

    # Calculate weight based on Elo.
    board = cshogi.Board(sfen)
    elo_key = "rating_b" if board.turn == cshogi.BLACK else "rating_w"
    elo_value = group.attrs.get(elo_key)
    if elo_value is not None:
      sample_weight = max(
          self.elo_weight_min,
          self.elo_weight_slope * (float(elo_value) - 3000.0) + self.elo_weight_intercept
      )
    else:
      sample_weight = self.elo_weight_intercept

    metadata = {
        "game_name": game_name,
        "file_path": _attr_to_str(group.attrs.get("file_path"), default=""),
        "kif_index": int(group.attrs.get("kif_index", 0)),
        "rating_b": group.attrs.get("rating_b"),
        "rating_w": group.attrs.get("rating_w"),
    }
    return FixedRefSample(
        sfen=sfen,
        actual_move=int(position["actual_move"]),
        ply=int(position["ply"]),
        game_result=game_result,
        context=context,
        sample_weight=float(sample_weight),
        metadata=metadata,
    )

  def __getstate__(self) -> dict[str, Any]:
    state = self.__dict__.copy()
    state["_h5"] = None
    return state


def fixed_ref_sample_to_dict(sample: FixedRefSample) -> dict[str, Any]:
  return {
      "sfen": sample.sfen,
      "actual_move": sample.actual_move,
      "ply": sample.ply,
      "game_result": sample.game_result,
      "context_type": sample.context.context_type,
      "context_id": sample.context.context_id,
      "context_label": sample.context.context_label,
      "metadata": sample.metadata,
  }


def collate_fixed_ref_samples(samples: list[FixedRefSample]) -> dict[str, Any]:
  return {
      "sfen": [sample.sfen for sample in samples],
      "actual_move": torch.tensor([sample.actual_move for sample in samples], dtype=torch.int64),
      "ply": torch.tensor([sample.ply for sample in samples], dtype=torch.float32).unsqueeze(1),
      "game_result": torch.tensor([sample.game_result for sample in samples], dtype=torch.int64),
      "context_id": torch.tensor([sample.context.context_id for sample in samples], dtype=torch.int64),
      "weight": torch.tensor([sample.sample_weight for sample in samples], dtype=torch.float32),
      "context_type": [sample.context.context_type for sample in samples],
      "context_label": [sample.context.context_label for sample in samples],
      "metadata": [sample.metadata for sample in samples],
  }
