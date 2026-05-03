from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cshogi
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_ELO_BUCKETS = (1200, 1600, 2000, 2400)
META_DTYPE = np.dtype([
  ("game_result", np.uint8),
  ("actual_move", "<u4"),
  ("ply", "<u2"),
  ("context_id", "<u2"),
  ("sample_weight_q12", "<u2"),
])


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


def _index_path_for_data(data_path: str | Path) -> Path:
  path = Path(data_path)
  if path.suffix == ".bin":
    return path.with_suffix(".idx.pkl")
  return path.with_suffix(".idx.pkl")


def _load_index_file(idx_path: str | Path) -> tuple[list[str], list[int], dict[str, int], list[dict]]:
  import pickle

  with open(idx_path, "rb") as f:
    idx = pickle.load(f)
  game_attrs = idx.get("game_attrs", idx.get("game_meta_records"))
  if game_attrs is None:
    raise KeyError(f"{idx_path} is missing game_attrs/game_meta_records")
  return (
      idx["game_names"],
      idx["game_end_offsets"],
      idx["player_to_id"],
      game_attrs,
  )


class FixedRefH5Dataset(Dataset):
  """
  HDF5 decision-point dataset for the fixed-reference preference route.

  This first slice intentionally returns only the current SFEN, actual move,
  and unified context information. Candidate generation with the fixed
  reference model V0 will be added later in the collate/batch-builder path.

  The dataset supports an optional index file (.idx) to avoid rescanning
  the HDF5 file at construction time.
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
    if context_type not in {"elo", "player", "none", "bucket"}:
      raise ValueError(f"Unsupported context_type: {context_type}")
    self.h5_path = str(h5_path)
    self.context_type = context_type
    self.elo_bucket_edges = tuple(int(v) for v in elo_bucket_edges)
    self.elo_weight_slope = float(elo_weight_slope)
    self.elo_weight_intercept = float(elo_weight_intercept)
    self.elo_weight_min = float(elo_weight_min)
    self._h5: h5py.File | None = None
    self._idx_path = str(_index_path_for_data(self.h5_path))
    self._game_names, self._game_end_offsets, self._player_to_id, self._game_attrs = self._load_index_or_build()
    self._length = self._game_end_offsets[-1] if self._game_end_offsets else 0

  def _load_index_or_build(self) -> tuple[list[str], list[int], dict[str, int], list[dict]]:
    """Load index from .idx file if available, else build by scanning H5."""
    idx_path = self._idx_path
    if Path(idx_path).exists():
      return _load_index_file(idx_path)
    return self._build_index_and_vocab()

  def _build_index_and_vocab(self) -> tuple[list[str], list[int], dict[str, int], list[dict]]:
    """Scan the HDF5 file once to build the game index, player vocab, and attrs."""
    game_names: list[str] = []
    game_end_offsets: list[int] = []
    player_set: set[str] = set()
    game_attrs_list: list[dict] = []
    total_positions = 0

    with h5py.File(self.h5_path, "r") as h5_file:
      for game_name in sorted(h5_file.keys()):
        positions = h5_file[game_name].get("positions")
        if positions is None:
          continue
        num_positions = len(positions)
        if num_positions <= 0:
          continue
        total_positions += num_positions
        game_names.append(game_name)
        game_end_offsets.append(total_positions)
        attrs = h5_file[game_name].attrs
        black = _attr_to_str(attrs.get("black_player", attrs.get("player_b")))
        white = _attr_to_str(attrs.get("white_player", attrs.get("player_w")))
        player_set.add(black)
        player_set.add(white)
        game_attrs_list.append({
          "black_player": black,
          "white_player": white,
          "game_result": int(attrs.get("game_result", 0)),
          "rating_b": float(attrs.get("rating_b", 0)) if attrs.get("rating_b") is not None else None,
          "rating_w": float(attrs.get("rating_w", 0)) if attrs.get("rating_w") is not None else None,
        })

    player_to_id = {player: idx for idx, player in enumerate(sorted(player_set))}
    return game_names, game_end_offsets, player_to_id, game_attrs_list

  def _ensure_open(self) -> h5py.File:
    if self._h5 is None:
      self._h5 = h5py.File(self.h5_path, "r")
    return self._h5

  def __len__(self) -> int:
    return self._length

  def _resolve_index(self, idx: int) -> tuple[str, int]:
    if idx < 0:
      idx += self._length
    if idx < 0 or idx >= self._length:
      raise IndexError(idx)
    game_idx = bisect.bisect_right(self._game_end_offsets, idx)
    game_start = 0 if game_idx == 0 else self._game_end_offsets[game_idx - 1]
    return self._game_names[game_idx], idx - game_start

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

  def _resolve_bucket_context(self, attrs: h5py.AttributeManager, turn: int) -> ContextValue:
    elo_key = "rating_b" if turn == cshogi.BLACK else "rating_w"
    elo_value = attrs.get(elo_key)
    bucket_id = bucketize_elo(elo_value, self.elo_bucket_edges)
    bucket_label = f"elo_bucket:{bucket_id}"
    return ContextValue(
        context_type="bucket",
        context_id=bucket_id,
        context_label=bucket_label,
    )

  def _resolve_context(self, attrs: h5py.AttributeManager, board: cshogi.Board) -> ContextValue:
    if self.context_type == "player":
      return self._resolve_player_context(attrs, board.turn)
    elif self.context_type == "bucket":
      return self._resolve_bucket_context(attrs, board.turn)
    else:
      # elo or none: context_id = 0
      return ContextValue(
          context_type=self.context_type,
          context_id=0,
          context_label="elo" if self.context_type == "elo" else "none",
      )

  def __getitem__(self, idx: int) -> FixedRefSample:
    h5_file = self._ensure_open()
    game_name, pos_idx = self._resolve_index(idx)
    group = h5_file[game_name]
    position = group["positions"][pos_idx]
    board = cshogi.Board()
    board.set_psfen(packed_sfen_field_view(position))
    sfen = board.sfen()
    context = self._resolve_context(group.attrs, board)
    game_result = int(group.attrs.get("game_result", 0))

    # Calculate weight based on Elo.
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


class FixedRefBinaryDataset(Dataset):
  """
  Binary decision-point dataset for the fixed-reference preference route.

  This reads records produced by export_preference_data.py:
  - <stem>.bin
  - <stem>_meta.bin
  - <stem>.idx.pkl
  """

  def __init__(
      self,
      bin_path: str | Path,
      context_type: str = "elo",
      elo_bucket_edges: tuple[int, ...] = DEFAULT_ELO_BUCKETS,
      elo_weight_slope: float = 0.001,
      elo_weight_intercept: float = 0.5,
      elo_weight_min: float = 0.1,
  ) -> None:
    super().__init__()
    if context_type not in {"elo", "player", "none", "bucket"}:
      raise ValueError(f"Unsupported context_type: {context_type}")
    self.bin_path = str(bin_path)
    self.context_type = context_type
    self.elo_bucket_edges = tuple(int(v) for v in elo_bucket_edges)
    self.elo_weight_slope = float(elo_weight_slope)
    self.elo_weight_intercept = float(elo_weight_intercept)
    self.elo_weight_min = float(elo_weight_min)
    self._meta_path = str(Path(self.bin_path).with_name(f"{Path(self.bin_path).stem}_meta.bin"))
    self._idx_path = str(_index_path_for_data(self.bin_path))
    self._game_names, self._game_end_offsets, self._player_to_id, self._game_attrs = _load_index_file(self._idx_path)
    self._id_to_player = {pid: player for player, pid in self._player_to_id.items()}
    self._psv = np.memmap(self.bin_path, dtype=cshogi.PackedSfenValue, mode="r")
    self._meta = np.memmap(self._meta_path, dtype=META_DTYPE, mode="r")
    if len(self._psv) != len(self._meta):
      raise ValueError(
          f"Binary/meta size mismatch: {self.bin_path} has {len(self._psv)} records, "
          f"{self._meta_path} has {len(self._meta)} records"
      )
    expected_length = self._game_end_offsets[-1] if self._game_end_offsets else 0
    if len(self._psv) != expected_length:
      raise ValueError(
          f"Index size mismatch: {self._idx_path} expects {expected_length} records, "
          f"but {self.bin_path} has {len(self._psv)}"
      )
    self._length = len(self._psv)

  def __len__(self) -> int:
    return self._length

  def _resolve_index(self, idx: int) -> tuple[int, int]:
    if idx < 0:
      idx += self._length
    if idx < 0 or idx >= self._length:
      raise IndexError(idx)
    game_idx = bisect.bisect_right(self._game_end_offsets, idx)
    game_start = 0 if game_idx == 0 else self._game_end_offsets[game_idx - 1]
    return game_idx, idx - game_start

  def _resolve_context_label(self, game_attrs: dict[str, Any], board: cshogi.Board, context_id: int) -> str:
    if self.context_type == "player":
      if board.turn == cshogi.BLACK:
        return _attr_to_str(game_attrs.get("black_player"))
      return _attr_to_str(game_attrs.get("white_player"))
    if self.context_type == "bucket":
      return f"elo_bucket:{context_id}"
    return "elo" if self.context_type == "elo" else "none"

  def __getitem__(self, idx: int) -> FixedRefSample:
    game_idx, _pos_idx = self._resolve_index(idx)
    game_name = self._game_names[game_idx]
    game_attrs = self._game_attrs[game_idx]
    psv = self._psv[idx]
    meta = self._meta[idx]

    board = cshogi.Board()
    board.set_psfen(psv["sfen"])
    sfen = board.sfen()

    context_id = int(meta["context_id"])
    context = ContextValue(
        context_type=self.context_type,
        context_id=context_id,
        context_label=self._resolve_context_label(game_attrs, board, context_id),
    )

    metadata = {
        "game_name": game_name,
        "file_path": _attr_to_str(game_attrs.get("file_path"), default=""),
        "kif_index": int(game_attrs.get("kif_index", 0)),
        "rating_b": game_attrs.get("rating_b"),
        "rating_w": game_attrs.get("rating_w"),
    }
    return FixedRefSample(
        sfen=sfen,
        actual_move=int(meta["actual_move"]),
        ply=int(meta["ply"]),
        game_result=int(meta["game_result"]),
        context=context,
        sample_weight=float(meta["sample_weight_q12"]) / 1000.0,
        metadata=metadata,
    )


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


def collate_fixed_ref_samples_for_training(samples: list[FixedRefSample]) -> dict[str, Any]:
  return {
      "sfen": [sample.sfen for sample in samples],
      "actual_move": torch.tensor([sample.actual_move for sample in samples], dtype=torch.int64),
      "ply": torch.tensor([sample.ply for sample in samples], dtype=torch.float32).unsqueeze(1),
      "context_id": torch.tensor([sample.context.context_id for sample in samples], dtype=torch.int64),
      "weight": torch.tensor([sample.sample_weight for sample in samples], dtype=torch.float32),
  }
