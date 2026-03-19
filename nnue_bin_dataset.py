import chess
import halfkp
import mmap
import random
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Sampler

PACKED_SFEN_VALUE_BYTES = 40

HUFFMAN_MAP = {0b000 : chess.PAWN, 0b001 : chess.KNIGHT, 0b010 : chess.BISHOP, 0b011 : chess.ROOK, 0b100: chess.QUEEN}


def _piece_type_constant(*names):
  for name in names:
    value = getattr(chess, name, None)
    if isinstance(value, int):
      return value
  return None


MAJOR_PIECE_TYPES = tuple(
  value for value in (
    _piece_type_constant('ROOK'),
    _piece_type_constant('BISHOP'),
    _piece_type_constant('DRAGON', 'PROM_ROOK'),
    _piece_type_constant('HORSE', 'PROM_BISHOP'),
  ) if value is not None
)

def twos(v, w):
  return v - int((v << 1) & 2**w)

class BitReader():
  def __init__(self, bytes, at):
    self.bytes = bytes
    self.seek(at)

  def readBits(self, n):
    r = self.bits & ((1 << n) - 1)
    self.bits >>= n
    self.position -= n
    return r

  def refill(self):
    while self.position <= 24:
      self.bits |= self.bytes[self.at] << self.position
      self.position += 8
      self.at += 1

  def seek(self, at):
    self.at = at
    self.bits = 0
    self.position = 0
    self.refill()

def is_quiet(board, from_, to_):
  for mv in board.legal_moves:
    if mv.from_square == from_ and mv.to_square == to_:
      return not board.is_capture(mv)
  return False

class ToTensor(object):
  def __init__(self, feature_set):
    self.features = feature_set

  def __call__(self, sample):
    bd, _, outcome, score, ply = sample
    us = torch.tensor([bd.turn])
    them = torch.tensor([not bd.turn])
    outcome = torch.tensor([outcome])
    score = torch.tensor([score])
    ply = torch.tensor([ply])
    aux = build_auxiliary_targets(bd)
    white, black = self.features.get_active_features(bd)
    return us.float(), them.float(), white.float(), black.float(), outcome.float(), score.float(), ply.float(), aux.float()

class RandomFlip(object):
  def __call__(self, sample):
    bd, move, outcome, score, ply = sample
    mirror = random.choice([False, True])
    if mirror:
      bd = bd.mirror()
    return bd, move, outcome, score, ply


def _attackers(board, color, sq):
  if not hasattr(board, 'attackers'):
    raise RuntimeError('py_data auxiliary labels require board.attackers().')
  return board.attackers(color, sq)


def _king_zone_squares(board, color):
  king_sq = board.king(color)
  if king_sq is None:
    return []
  if not hasattr(board, 'attacks'):
    raise RuntimeError('py_data auxiliary labels require board.attacks().')
  zone = {king_sq}
  zone.update(board.attacks(king_sq))
  return list(zone)


def _has_major_attack_into_zone(board, attacker_color, zone_squares):
  for sq in zone_squares:
    for attacker_sq in _attackers(board, attacker_color, sq):
      piece = board.piece_at(attacker_sq)
      if piece is not None and piece.piece_type in MAJOR_PIECE_TYPES:
        return 1.0
  return 0.0


def _has_hanging_major(board, color):
  for sq, piece in board.piece_map().items():
    if piece.color != color or piece.piece_type not in MAJOR_PIECE_TYPES:
      continue
    attacked = any(True for _ in _attackers(board, not color, sq))
    if not attacked:
      continue
    defended = any(True for _ in _attackers(board, color, sq))
    if not defended:
      return 1.0
  return 0.0


def build_auxiliary_targets(board):
  us = board.turn
  them = not us
  them_king_zone = _king_zone_squares(board, them)
  us_king_zone = _king_zone_squares(board, us)
  return torch.tensor([
    _has_major_attack_into_zone(board, us, them_king_zone),
    _has_major_attack_into_zone(board, them, us_king_zone),
    _has_hanging_major(board, us),
    _has_hanging_major(board, them),
  ])

class NNUEBinData(torch.utils.data.Dataset):
  def __init__(self, filename, feature_set):
    super(NNUEBinData, self).__init__()
    self.filename = filename
    self.len = os.path.getsize(filename) // PACKED_SFEN_VALUE_BYTES
    self.transform = ToTensor(feature_set)
    self.file = None

  def __len__(self):
    return self.len

  def _ensure_open(self):
    if self.file is None:
      self.file = open(self.filename, 'r+b')
      self.bytes = mmap.mmap(self.file.fileno(), 0)

  def get_ply_fast(self, idx):
    """
    Reads ply value directly from packed record without full board decode.
    ply is stored as little-endian uint16 at offset +36.
    """
    self._ensure_open()
    base = PACKED_SFEN_VALUE_BYTES * idx
    return int.from_bytes(self.bytes[base + 36:base + 38], byteorder='little', signed=False)

  def get_raw(self, idx):
    self._ensure_open()

    base = PACKED_SFEN_VALUE_BYTES * idx
    br = BitReader(self.bytes, base)

    bd = chess.Board(fen=None)
    bd.turn = not br.readBits(1)
    white_king_sq = br.readBits(6)
    black_king_sq = br.readBits(6)
    bd.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))
    bd.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))

    assert(black_king_sq != white_king_sq)

    for rank_ in range(8)[::-1]:
      br.refill()
      for file_ in range(8):
        i = chess.square(file_, rank_)
        if white_king_sq == i or black_king_sq == i:
          continue
        if br.readBits(1):
          assert(bd.piece_at(i) == None)
          piece_index = br.readBits(3)
          piece = HUFFMAN_MAP[piece_index]
          color = br.readBits(1)
          bd.set_piece_at(i, chess.Piece(piece, not color))
          br.refill()

    br.seek(base + 32)
    score = twos(br.readBits(16), 16)
    move = br.readBits(16)
    to_ = move & 63
    from_ = (move & (63 << 6)) >> 6

    br.refill()
    ply = br.readBits(16)
    bd.fullmove_number = ply // 2

    move = chess.Move(from_square=chess.SQUARES[from_], to_square=chess.SQUARES[to_])

    # 1, 0, -1
    game_result = br.readBits(8)
    outcome = {1: 1.0, 0: 0.5, 255: 0.0}[game_result]
    return bd, move, outcome, score, ply

  def __getitem__(self, idx):
    item = self.get_raw(idx)
    return self.transform(item)

  # Allows this class to be pickled (otherwise you will get file handle errors).
  def __getstate__(self):
    state = self.__dict__.copy()
    state['file'] = None
    state.pop('bytes', None)
    return state


class PlyBalancedSampler(Sampler):
  """
  Samples indices with replacement from ply buckets.
  Keeps the number of samples equal to target_size.
  """
  def __init__(self, buckets, target_size, seed=42):
    self.buckets = [bucket for bucket in buckets if bucket]
    if not self.buckets:
      raise ValueError('PlyBalancedSampler requires at least one non-empty bucket.')
    self.target_size = int(target_size)
    self.seed = int(seed)
    self.epoch = 0

  def __iter__(self):
    rng = random.Random(self.seed + self.epoch)
    for _ in range(self.target_size):
      bucket = rng.choice(self.buckets)
      yield rng.choice(bucket)

  def __len__(self):
    return self.target_size

  def set_epoch(self, epoch):
    self.epoch = int(epoch)


def build_ply_buckets(dataset, num_bins=8, max_positions=0):
  if num_bins <= 0:
    raise ValueError(f'num_bins must be > 0 (got {num_bins})')
  total = len(dataset)
  if total == 0:
    return []

  if max_positions and max_positions > 0 and max_positions < total:
    step = max(1, total // max_positions)
    indices = list(range(0, total, step))
    if len(indices) > max_positions:
      indices = indices[:max_positions]
  else:
    indices = list(range(total))

  plies = [dataset.get_ply_fast(i) for i in indices]
  if not plies:
    return []

  min_ply = min(plies)
  max_ply = max(plies)
  if max_ply == min_ply:
    return [indices]

  span = max_ply - min_ply + 1
  buckets = [[] for _ in range(num_bins)]
  for idx, ply in zip(indices, plies):
    bucket_index = min(num_bins - 1, (ply - min_ply) * num_bins // span)
    buckets[bucket_index].append(idx)
  return buckets


def create_sampling_strategy(dataset, mode='uniform', num_bins=8, max_positions=0, seed=42):
  mode = (mode or 'uniform').strip().lower()
  target_size = len(dataset) if not max_positions or max_positions <= 0 else min(len(dataset), int(max_positions))
  if mode == 'uniform':
    return None
  if mode == 'ply_balanced':
    buckets = build_ply_buckets(dataset, num_bins=num_bins, max_positions=max_positions)
    return PlyBalancedSampler(buckets, target_size=target_size, seed=seed)
  raise ValueError(f'Unsupported sampling mode: {mode}. Use uniform or ply_balanced.')
