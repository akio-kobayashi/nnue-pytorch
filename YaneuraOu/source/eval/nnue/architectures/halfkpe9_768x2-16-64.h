// Definition of input features and network structure used in NNUE evaluation function
// NNUE評価関数で用いる入力特徴量とネットワーク構造の定義

#include "../features/feature_set.h"
#include "../features/half_kpe9.h"

#include "../layers/input_slice.h"
#include "../layers/affine_transform.h"
#include "../layers/clipped_relu.h"

namespace Eval {

namespace NNUE {

// 評価関数で用いる入力特徴量
using RawFeatures = Features::FeatureSet<
    Features::HalfKPE9<Features::Side::kFriend>>;

// 変換後の入力特徴量の次元数
constexpr IndexType kTransformedFeatureDimensions = 768;

namespace Layers {

// ネットワーク構造の定義
using InputLayer = InputSlice<kTransformedFeatureDimensions * 2>;
using HiddenLayer1 = ClippedReLU<AffineTransform<InputLayer, 16>>;
using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, 64>>;
using OutputLayer = AffineTransform<HiddenLayer2, 1>;

}  // namespace Layers

using Network = Layers::OutputLayer;

}  // namespace NNUE

}  // namespace Eval
