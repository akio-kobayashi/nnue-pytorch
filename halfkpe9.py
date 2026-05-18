import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 81
NUM_PLANES = 1548
FE_HAND_END = 90
EFFECT_STATES = 3 * 3
NUM_PIECE_KINDS = (NUM_PLANES - FE_HAND_END) // NUM_SQ
HALF_RELATIVE_KP_INPUTS = NUM_PIECE_KINDS * (9 * 2 - 1) * (9 * 2 - 1)


class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__(
      'HalfKPE9',
      0x5d69d5b9,
      OrderedDict([('HalfKPE9', NUM_PLANES * NUM_SQ * EFFECT_STATES)]))

  def get_active_features(self, board):
    raise NotImplementedError(
      'HalfKPE9 Python-side feature extraction is not implemented. '
      'Use the training_data_loader shared library path for training.')


class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__(
      'HalfKPE9^',
      0x5d69d5b9,
      OrderedDict([
        ('HalfKPE9', NUM_PLANES * NUM_SQ * EFFECT_STATES),
        ('HalfKP', NUM_PLANES * NUM_SQ),
        ('HalfK', NUM_SQ),
        ('PE9', NUM_PLANES * EFFECT_STATES),
        ('P', NUM_PLANES),
        ('HalfRelativeKP', HALF_RELATIVE_KP_INPUTS),
      ]))

  def get_active_features(self, board):
    raise NotImplementedError(
      'HalfKPE9^ Python-side feature extraction is not implemented. '
      'Use the training_data_loader shared library path for training.')

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    halfkp_inputs = NUM_PLANES * NUM_SQ
    effect_index = idx // halfkp_inputs
    halfkp_idx = idx % halfkp_inputs
    sq_k = halfkp_idx // NUM_PLANES
    p = halfkp_idx % NUM_PLANES

    factors = [
      idx,
      self.get_factor_base_feature('HalfKP') + halfkp_idx,
      self.get_factor_base_feature('HalfK') + sq_k,
      self.get_factor_base_feature('PE9') + effect_index * NUM_PLANES + p,
      self.get_factor_base_feature('P') + p,
    ]

    if p >= FE_HAND_END:
      piece_index = (p - FE_HAND_END) // NUM_SQ
      sq_p = (p - FE_HAND_END) % NUM_SQ
      file_k = sq_k % 9
      rank_k = sq_k // 9
      file_p = sq_p % 9
      rank_p = sq_p // 9
      relative_file = file_p - file_k + 8
      relative_rank = rank_p - rank_k + 8
      half_relative_idx = piece_index * 17 * 17 + relative_file * 17 + relative_rank
      factors.append(self.get_factor_base_feature('HalfRelativeKP') + half_relative_idx)

    return factors


def get_feature_block_clss():
  return [Features, FactorizedFeatures]
