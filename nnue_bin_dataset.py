import cshogi
import halfkp
import mmap
import random
import os
import torch
import torch.nn.functional as F
import numpy as np

PACKED_SFEN_VALUE_BYTES = 40
PACKED_SFEN_VALUE_DTYPE = cshogi.PackedSfenValue

PIECE_VALUES = {
    cshogi.LANCE: 430,
    cshogi.KNIGHT: 581,
    cshogi.SILVER: 716,
    cshogi.GOLD: 782,
    cshogi.BISHOP: 1008,
    cshogi.ROOK: 1193,
}

PROMOTED_TO_BASE = {
    cshogi.PROM_LANCE: cshogi.LANCE,
    cshogi.PROM_KNIGHT: cshogi.KNIGHT,
    cshogi.PROM_SILVER: cshogi.SILVER,
    cshogi.PROM_BISHOP: cshogi.BISHOP,
    cshogi.PROM_ROOK: cshogi.ROOK,
}

def is_quiet(board, from_, to_):
  for mv in board.legal_moves:
    if cshogi.move_from(mv) == from_ and cshogi.move_to(mv) == to_:
      return not board.is_capture(mv)
  return False


def _compute_npm(board):
  npm = 0
  for piece in board.pieces:
    if piece == cshogi.NONE:
      continue
    piece_type = cshogi.piece_to_piece_type(piece)
    if piece_type == cshogi.KING:
      continue
    base_type = PROMOTED_TO_BASE.get(piece_type, piece_type)
    npm += PIECE_VALUES.get(base_type, 0)
  return npm

class ToTensor(object):
  def __init__(self, feature_set):
    self.features = feature_set

  def __call__(self, sample):
    bd, _, outcome, score, ply, npm = sample
    us = torch.tensor([bd.turn])
    them = torch.tensor([not bd.turn])
    outcome = torch.tensor([outcome])
    score = torch.tensor([score])
    ply = torch.tensor([ply])
    npm = torch.tensor([npm])
    white, black = self.features.get_active_features(bd)
    return us.float(), them.float(), white.float(), black.float(), outcome.float(), score.float(), ply.float(), npm.float()

class RandomFlip(object):
  def __call__(self, sample):
    bd, move, outcome, score, ply, npm = sample
    mirror = random.choice([False, True])
    if mirror:
      bd = bd.mirror()
    return bd, move, outcome, score, ply, npm

class NNUEBinData(torch.utils.data.Dataset):
  def __init__(self, filename, feature_set):
    super(NNUEBinData, self).__init__()
    self.filename = filename
    self.len = os.path.getsize(filename) // PACKED_SFEN_VALUE_BYTES
    self.transform = ToTensor(feature_set)
    self.file = None
    self.records = None

  def __len__(self):
    return self.len

  def get_raw(self, idx):
    if self.file is None:
      self.file = open(self.filename, 'rb')
      self.bytes = mmap.mmap(self.file.fileno(), 0)
      self.records = np.frombuffer(self.bytes, dtype=PACKED_SFEN_VALUE_DTYPE)

    record = self.records[idx]
    bd = cshogi.Board()
    bd.set_psfen(np.asarray(record['sfen']))

    npm = _compute_npm(bd)
    score = int(record['score'])
    ply = int(record['gamePly'])
    bd.move_number = max(1, (ply + 1) // 2)
    move = bd.move_from_psv(int(record['move']))

    game_result = int(record['game_result'])
    if game_result == cshogi.DRAW:
      outcome = 0.5
    elif game_result == cshogi.BLACK_WIN:
      outcome = 1.0 if bd.turn == cshogi.BLACK else 0.0
    elif game_result == cshogi.WHITE_WIN:
      outcome = 1.0 if bd.turn == cshogi.WHITE else 0.0
    else:
      raise ValueError(f'Unexpected game_result: {game_result}')
    return bd, move, outcome, score, ply, npm


  def __getitem__(self, idx):
    item = self.get_raw(idx)
    return self.transform(item)

  # Allows this class to be pickled (otherwise you will get file handle errors).
  def __getstate__(self):
    state = self.__dict__.copy()
    state['file'] = None
    state['records'] = None
    state.pop('bytes', None)
    return state
