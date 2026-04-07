import cshogi
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 81
# NUM_PT = 10
NUM_PLANES = 1548

def orient(is_white_pov: bool, sq: int):
  return sq if is_white_pov == cshogi.BLACK else 80 - sq

def _piece_color(piece: int):
  return cshogi.BLACK if piece < cshogi.WPAWN else cshogi.WHITE

def halfkp_idx(is_white_pov: bool, king_sq: int, sq: int, piece: int):
  piece_type = cshogi.piece_to_piece_type(piece)
  p_idx = (piece_type - 1) * 2 + (_piece_color(piece) != is_white_pov)
  return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKP', 0x5d69d5b8, OrderedDict([('HalfKP', NUM_PLANES * NUM_SQ)]))

  def get_active_features(self, board: cshogi.Board):
    def piece_features(turn):
      indices = torch.zeros(NUM_PLANES * NUM_SQ)
      king_sq = orient(turn, board.king_square(turn))
      for sq, piece in enumerate(board.pieces):
        if piece == cshogi.NONE:
          continue
        if cshogi.piece_to_piece_type(piece) == cshogi.KING:
          continue
        indices[halfkp_idx(turn, king_sq, sq, piece)] = 1.0
      return indices
    return (piece_features(cshogi.BLACK), piece_features(cshogi.WHITE))

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKP^', 0x5d69d5b8, OrderedDict([('HalfKP', NUM_PLANES * NUM_SQ), ('HalfK', NUM_SQ), ('P', NUM_PLANES )]))
    self.base = Features()

  def get_active_features(self, board: cshogi.Board):
    white, black = self.base.get_active_features(board)
    def piece_features(base, color):
      indices = torch.zeros(NUM_SQ * 11)
      piece_count = 0
      for sq, piece in enumerate(board.pieces):
        if piece == cshogi.NONE:
          continue
        piece_type = cshogi.piece_to_piece_type(piece)
        if piece_type == cshogi.KING:
          continue
        piece_count += 1
        p_idx = (piece_type - 1) * 2 + (_piece_color(piece) != color)
        indices[(p_idx + 1) * NUM_SQ + orient(color, sq)] = 1.0
      indices[orient(color, board.king_square(color))] = piece_count
      return torch.cat((base, indices))
    return (piece_features(white, cshogi.BLACK), piece_features(black, cshogi.WHITE))

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    k_idx = idx // NUM_PLANES
    p_idx = idx % NUM_PLANES

    return [idx, self.get_factor_base_feature('HalfK') + k_idx, self.get_factor_base_feature('P') + p_idx]

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
