import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 81
# NUM_PT = 10
NUM_PLANES = 1548
NUM_PIECE_TYPES = (NUM_PLANES - 1 + NUM_SQ - 1) // NUM_SQ
NUM_PTC = NUM_PIECE_TYPES + 1

def orient(is_white_pov: bool, sq: int):
  return (63 * (not is_white_pov)) ^ sq

def halfkp_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
  p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
  return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES


def halfkp_piece_type_color_idx(is_white_pov: bool, p: chess.Piece):
  return (p.piece_type - 1) * 2 + (p.color != is_white_pov) + 1


def halfkp_piece_plane_idx(is_white_pov: bool, sq: int, p: chess.Piece):
  sq_idx = orient(is_white_pov, sq)
  ptc_idx = halfkp_piece_type_color_idx(is_white_pov, p)
  plane_idx = 1 + sq_idx + (ptc_idx - 1) * NUM_SQ
  return plane_idx, sq_idx, ptc_idx

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKP', 0x5d69d5b8, OrderedDict([('HalfKP', NUM_PLANES * NUM_SQ)]))

  def get_active_features(self, board: chess.Board):
    def piece_features(turn):
      indices = torch.zeros(NUM_PLANES * NUM_SQ)
      for sq, p in board.piece_map().items():
        if p.piece_type == chess.KING:
          continue
        indices[halfkp_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
      return indices
    return (piece_features(chess.WHITE), piece_features(chess.BLACK))

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKP^', 0x5d69d5b8, OrderedDict([('HalfKP', NUM_PLANES * NUM_SQ), ('HalfK', NUM_SQ), ('P', NUM_PLANES )]))
    self.base = Features()

  def get_active_features(self, board: chess.Board):
    white, black = self.base.get_active_features(board)
    def piece_features(base, color):
      indices = torch.zeros(NUM_SQ * 11)
      piece_count = 0
      # P feature
      for sq, p in board.piece_map().items():
        if p.piece_type == chess.KING:
          continue
        piece_count += 1
        p_idx = (p.piece_type - 1) * 2 + (p.color != color)
        indices[(p_idx + 1) * NUM_SQ + orient(color, sq)] = 1.0
      # HalfK feature
      indices[orient(color, board.king(color))] = piece_count
      return torch.cat((base, indices))
    return (piece_features(white, chess.WHITE), piece_features(black, chess.BLACK))

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    k_idx = idx // NUM_PLANES
    p_idx = idx % NUM_PLANES

    return [idx, self.get_factor_base_feature('HalfK') + k_idx, self.get_factor_base_feature('P') + p_idx]


class _BaseMultiFactorizedFeatures(FeatureBlock):
  def __init__(self, name, include_sq, include_ptc):
    factors = OrderedDict([
        ('HalfKP', NUM_PLANES * NUM_SQ),
        ('HalfK', NUM_SQ),
        ('P', NUM_PLANES),
    ])
    if include_sq:
      factors['SQ'] = NUM_SQ
    if include_ptc:
      factors['PTC'] = NUM_PTC
    super(_BaseMultiFactorizedFeatures, self).__init__(
        name,
        0x5d69d5b8,
        factors,
        main_factor_name='HalfKP')
    self.base = Features()
    self.include_sq = include_sq
    self.include_ptc = include_ptc

  @staticmethod
  def _plane_to_sq_ptc(plane_idx):
    if plane_idx <= 0:
      return 0, 0

    plane_offset = plane_idx - 1
    sq_idx = plane_offset % NUM_SQ
    ptc_idx = plane_offset // NUM_SQ + 1
    return sq_idx, ptc_idx

  def get_active_features(self, board: chess.Board):
    white, black = self.base.get_active_features(board)
    halfk_base = self.get_factor_base_feature('HalfK')
    p_base = self.get_factor_base_feature('P')
    sq_base = self.get_factor_base_feature('SQ') if self.include_sq else None
    ptc_base = self.get_factor_base_feature('PTC') if self.include_ptc else None

    def piece_features(base, color):
      indices = torch.zeros(self.num_features)
      indices[:self.num_real_features] = base

      piece_count = 0
      for sq, p in board.piece_map().items():
        if p.piece_type == chess.KING:
          continue

        piece_count += 1
        plane_idx, sq_idx, ptc_idx = halfkp_piece_plane_idx(color, sq, p)
        indices[p_base + plane_idx] = 1.0
        if self.include_sq:
          indices[sq_base + sq_idx] = 1.0
        if self.include_ptc:
          indices[ptc_base + ptc_idx] = 1.0

      indices[halfk_base + orient(color, board.king(color))] = piece_count
      return indices

    return (piece_features(white, chess.WHITE), piece_features(black, chess.BLACK))

  def _get_base_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    k_idx = idx // NUM_PLANES
    plane_idx = idx % NUM_PLANES
    return [
        idx,
        self.get_factor_base_feature('HalfK') + k_idx,
        self.get_factor_base_feature('P') + plane_idx,
    ]

  def get_feature_factors(self, idx):
    factors = self._get_base_feature_factors(idx)
    plane_idx = idx % NUM_PLANES
    sq_idx, ptc_idx = self._plane_to_sq_ptc(plane_idx)
    if self.include_sq:
      factors.append(self.get_factor_base_feature('SQ') + sq_idx)
    if self.include_ptc:
      factors.append(self.get_factor_base_feature('PTC') + ptc_idx)
    return factors


class HalfKPSQFeatures(_BaseMultiFactorizedFeatures):
  def __init__(self):
    super(HalfKPSQFeatures, self).__init__('HalfKPSQ', include_sq=True, include_ptc=False)


class HalfKPPTCFeatures(_BaseMultiFactorizedFeatures):
  def __init__(self):
    super(HalfKPPTCFeatures, self).__init__('HalfKPPTC', include_sq=False, include_ptc=True)


class MultiFactorizedFeatures(_BaseMultiFactorizedFeatures):
  def __init__(self):
    super(MultiFactorizedFeatures, self).__init__('HalfKPx4', include_sq=True, include_ptc=True)

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures, HalfKPSQFeatures, HalfKPPTCFeatures, MultiFactorizedFeatures]
