#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <deque>
#include <random>

#include "YaneuraOu/source/config.h"
#include "YaneuraOu/source/usi.h"

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

#if defined (__x86_64__)
#define EXPORT
#define CDECL
#else
#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif
#endif

using namespace binpack;
//using namespace chess;

static Square orient(Color color, Square sq)
{
    if (color == Color::BLACK)
    {
        return sq;
    }
    else
    {
        // IMPORTANT: for now we use rotate180 instead of rank flip
        //            for compatibility with the stockfish master branch.
        //            Note that this is inconsistent with nodchip/master.
        return Inv(sq);
    }
}

//static Square orient_flip(Color color, Square sq)
//{
//    if (color == Color::BLACK)
//    {
//        return sq;
//    }
//    else
//    {
//        return sq.flippedVertically();
//    }
//}

struct HalfKP {
    static constexpr int NUM_SQ = 81;
    static constexpr int NUM_PLANES = 1548; // == fe_end
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 38;

    static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto& pos = *e.pos;
        Eval::BonaPiece* pieces = nullptr;
        if (color == Color::BLACK) {
            pieces = pos.eval_list()->piece_list_fb();
        }
        else {
            pieces = pos.eval_list()->piece_list_fw();
        }
        PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        auto sq_target_k = static_cast<Square>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int features_unordered[38];
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
            auto p = pieces[i];
            features_unordered[i] = static_cast<int>(Eval::fe_end) * static_cast<int>(sq_target_k) + p;
        }
        std::sort(features_unordered, features_unordered + PIECE_NUMBER_KING);
        for (int k = 0; k < PIECE_NUMBER_KING; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
        return INPUTS;
    }
};

struct HalfKPFactorized {
    // Factorized features
    static constexpr int K_INPUTS = HalfKP::NUM_SQ;
    static constexpr int PIECE_INPUTS = HalfKP::NUM_PLANES;
    static constexpr int INPUTS = HalfKP::INPUTS + K_INPUTS + PIECE_INPUTS;

    static constexpr int MAX_K_FEATURES = 1;
    static constexpr int MAX_PIECE_FEATURES = 38;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKP::MAX_ACTIVE_FEATURES + MAX_K_FEATURES + MAX_PIECE_FEATURES;

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto counter_before = counter;
        int offset = HalfKP::fill_features_sparse(i, e, features, values, counter, color);

        auto& pos = *e.pos;
        Eval::BonaPiece* pieces = nullptr;
        if (color == Color::BLACK) {
            pieces = pos.eval_list()->piece_list_fb();
        }
        else {
            pieces = pos.eval_list()->piece_list_fw();
        }

        {
            auto num_added_features = counter - counter_before;
            // king square factor
            PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
            auto sq_target_k = static_cast<Square>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = offset + static_cast<int>(sq_target_k);
            values[counter] = static_cast<float>(num_added_features);
            counter += 1;
        }
        offset += K_INPUTS;

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        int features_unordered[38];
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
            auto p = pieces[i];
            features_unordered[i] = offset + p;
        }
        std::sort(features_unordered, features_unordered + PIECE_NUMBER_KING);
        for (int k = 0; k < PIECE_NUMBER_KING; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
    }
};

// struct HalfKA {
//     static constexpr int NUM_SQ = 64;
//     static constexpr int NUM_PT = 12;
//     static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
//     static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

//     static constexpr int MAX_ACTIVE_FEATURES = 32;

//     static int feature_index(Color color, Square ksq, Square sq, Piece p)
//     {
//         auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
//         return 1 + static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
//     }

//     static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
//     {
//         auto& pos = e.pos;
//         auto pieces = pos.piecesBB();
//         auto ksq = pos.kingSquare(color);

//         // We order the features so that the resulting sparse
//         // tensor is coalesced.
//         int features_unordered[32];
//         int j = 0;
//         for(Square sq : pieces)
//         {
//             auto p = pos.pieceAt(sq);
//             features_unordered[j++] = feature_index(color, orient_flip(color, ksq), sq, p);
//         }
//         std::sort(features_unordered, features_unordered + j);
//         for (int k = 0; k < j; ++k)
//         {
//             int idx = counter * 2;
//             features[idx] = i;
//             features[idx + 1] = features_unordered[k];
//             values[counter] = 1.0f;
//             counter += 1;
//         }
//         return INPUTS;
//     }
// };

// struct HalfKAFactorized {
//     // Factorized features
//     static constexpr int PIECE_INPUTS = HalfKA::NUM_SQ * HalfKA::NUM_PT;
//     static constexpr int INPUTS = HalfKA::INPUTS + PIECE_INPUTS;

//     static constexpr int MAX_PIECE_FEATURES = 32;
//     static constexpr int MAX_ACTIVE_FEATURES = HalfKA::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

//     static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
//     {
//         auto counter_before = counter;
//         int offset = HalfKA::fill_features_sparse(i, e, features, values, counter, color);
//         auto& pos = e.pos;
//         auto pieces = pos.piecesBB();

//         // We order the features so that the resulting sparse
//         // tensor is coalesced. Note that we can just sort
//         // the parts where values are all 1.0f and leave the
//         // halfk feature where it was.
//         int features_unordered[32];
//         int j = 0;
//         for(Square sq : pieces)
//         {
//             auto p = pos.pieceAt(sq);
//             auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
//             features_unordered[j++] = offset + (p_idx * HalfKA::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
//         }
//         std::sort(features_unordered, features_unordered + j);
//         for (int k = 0; k < j; ++k)
//         {
//             int idx = counter * 2;
//             features[idx] = i;
//             features[idx + 1] = features_unordered[k];
//             values[counter] = 1.0f;
//             counter += 1;
//         }
//     }
// };

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        T::fill_features_sparse(i, e, features, values, counter, color);
    }
};

struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        size = entries.size();
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        ply = new float[size];

        num_active_white_features = 0;
        num_active_black_features = 0;

        std::memset(white, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));
        std::memset(black, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));

        for (int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    float* ply;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
        delete[] ply;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos->side_to_move() == Color::BLACK);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        ply[i] = e.ply;
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        FeatureSet<Ts...>::fill_features_sparse(i, e, white, white_values, num_active_white_features, Color::BLACK);
        FeatureSet<Ts...>::fill_features_sparse(i, e, black, black_values, num_active_black_features, Color::WHITE);
    }
};

namespace dlshogi {

    constexpr int MAX_HPAWN_NUM = 8;
    constexpr int MAX_HLANCE_NUM = 4;
    constexpr int MAX_HKNIGHT_NUM = 4;
    constexpr int MAX_HSILVER_NUM = 4;
    constexpr int MAX_HGOLD_NUM = 4;
    constexpr int MAX_HBISHOP_NUM = 2;
    constexpr int MAX_HROOK_NUM = 2;

    constexpr int MAX_PIECES_IN_HAND[] = {
        MAX_HPAWN_NUM,
        MAX_HLANCE_NUM,
        MAX_HKNIGHT_NUM,
        MAX_HSILVER_NUM,
        MAX_HGOLD_NUM,
        MAX_HBISHOP_NUM,
        MAX_HROOK_NUM,
    };
    constexpr int NUM_HAND_PIECE_TYPES = sizeof(MAX_PIECES_IN_HAND) / sizeof(MAX_PIECES_IN_HAND[0]);
    constexpr PieceType HAND_PIECE_TYPES[] = {
        PAWN,
        LANCE,
        KNIGHT,
        SILVER,
        GOLD,
        BISHOP,
        ROOK,
    };

    constexpr int PIECETYPE_NUM = 14;
    constexpr int MAX_ATTACK_NUM = 3;
    constexpr int MAX_PIECES_IN_HAND_SUM =
        MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM + MAX_HSILVER_NUM +
        MAX_HGOLD_NUM + MAX_HBISHOP_NUM + MAX_HROOK_NUM;
    constexpr int FEATURES1_PER_COLOR = PIECETYPE_NUM + PIECETYPE_NUM + MAX_ATTACK_NUM;
    constexpr int MAX_FEATURES1_NUM = 2 * FEATURES1_PER_COLOR;
    constexpr int MAX_FEATURES2_NUM = 2 * MAX_PIECES_IN_HAND_SUM + 1;
    constexpr int NUM_SQUARES = SQ_NB;

    inline int feature1_index(Color c, int f1idx, Square sq)
    {
        return (((int)c * FEATURES1_PER_COLOR) + f1idx) * NUM_SQUARES + (int)sq;
    }

    inline int feature2_index(int f2idx, Square sq)
    {
        return f2idx * NUM_SQUARES + (int)sq;
    }

    inline void set_features1(float* features1, Color c, int f1idx, Square sq)
    {
        features1[feature1_index(c, f1idx, sq)] = 1.0f;
    }

    inline void fill_features2_plane(float* features2, int f2idx)
    {
        auto* dst = features2 + f2idx * NUM_SQUARES;
        std::fill_n(dst, NUM_SQUARES, 1.0f);
    }

    inline void fill_features2_hand(float* features2, Color c, int offset, int num)
    {
        const int start = (int)c * MAX_PIECES_IN_HAND_SUM + offset;
        for (int i = 0; i < num; ++i)
        {
            fill_features2_plane(features2, start + i);
        }
    }

    template <Color SideToMove>
    void fill_input_features(const Position& position, float* features1, float* features2)
    {
        std::fill_n(features1, MAX_FEATURES1_NUM * NUM_SQUARES, 0.0f);
        std::fill_n(features2, MAX_FEATURES2_NUM * NUM_SQUARES, 0.0f);

        const Bitboard occupied_bb = position.pieces();
        const Bitboard pawns_bb = position.pieces(PAWN);
        const Bitboard without_pawns_bb = occupied_bb & ~pawns_bb;

        uint8_t attack_num[COLOR_NB][NUM_SQUARES] = {};

        for (auto sq : without_pawns_bb)
        {
            const Piece pc = position.piece_on(sq);
            const PieceType pt = type_of(pc);
            auto c = color_of(pc);
            auto sq_oriented = sq;
            auto attacks = effects_from(pc, sq, occupied_bb);

            if constexpr (SideToMove == WHITE)
            {
                c = ~c;
                sq_oriented = Inv(sq);
            }

            set_features1(features1, c, (int)pt - 1, sq_oriented);

            for (auto to : attacks)
            {
                auto to_oriented = to;
                if constexpr (SideToMove == WHITE)
                {
                    to_oriented = Inv(to);
                }

                set_features1(features1, c, PIECETYPE_NUM + (int)pt - 1, to_oriented);

                auto& num = attack_num[(int)c][(int)to_oriented];
                if (num < MAX_ATTACK_NUM)
                {
                    set_features1(features1, c, PIECETYPE_NUM + PIECETYPE_NUM + num, to_oriented);
                    ++num;
                }
            }
        }

        for (Color c = BLACK; c < COLOR_NB; ++c)
        {
            const Color board_color = SideToMove == BLACK ? c : ~c;
            auto pawns = pawns_bb & position.pieces(board_color);

            for (auto sq : pawns)
            {
                auto sq_oriented = sq;
                if constexpr (SideToMove == WHITE)
                {
                    sq_oriented = Inv(sq);
                }

                set_features1(features1, c, (int)PAWN - 1, sq_oriented);

                const Piece pawn = make_piece(board_color, PAWN);
                auto attacks = effects_from(pawn, sq, occupied_bb);
                for (auto to : attacks)
                {
                    auto to_oriented = to;
                    if constexpr (SideToMove == WHITE)
                    {
                        to_oriented = Inv(to);
                    }

                    set_features1(features1, c, PIECETYPE_NUM + (int)PAWN - 1, to_oriented);

                    auto& num = attack_num[(int)c][(int)to_oriented];
                    if (num < MAX_ATTACK_NUM)
                    {
                        set_features1(features1, c, PIECETYPE_NUM + PIECETYPE_NUM + num, to_oriented);
                        ++num;
                    }
                }
            }

            const Hand hand = position.hand_of(board_color);
            int offset = 0;
            for (int hp = 0; hp < NUM_HAND_PIECE_TYPES; ++hp)
            {
                const int num = std::min(hand_count(hand, HAND_PIECE_TYPES[hp]), MAX_PIECES_IN_HAND[hp]);
                fill_features2_hand(features2, c, offset, num);
                offset += MAX_PIECES_IN_HAND[hp];
            }
        }

        if (position.in_check())
        {
            fill_features2_plane(features2, 2 * MAX_PIECES_IN_HAND_SUM);
        }
    }

    struct Batch
    {
        static constexpr bool IS_BATCH = true;

        template <typename IgnoredFeatureSetT>
        Batch(IgnoredFeatureSetT, const std::vector<TrainingDataEntry>& entries)
        {
            size = entries.size();
            features1 = new float[size * MAX_FEATURES1_NUM * NUM_SQUARES];
            features2 = new float[size * MAX_FEATURES2_NUM * NUM_SQUARES];
            outcome = new float[size];
            score = new float[size];
            ply = new float[size];

            const int features1_stride = MAX_FEATURES1_NUM * NUM_SQUARES;
            const int features2_stride = MAX_FEATURES2_NUM * NUM_SQUARES;

            for (int i = 0; i < entries.size(); ++i)
            {
                const auto& e = entries[i];
                outcome[i] = (e.result + 1.0f) / 2.0f;
                score[i] = e.score;
                ply[i] = e.ply;

                auto* features1_dst = features1 + i * features1_stride;
                auto* features2_dst = features2 + i * features2_stride;
                if (e.pos->side_to_move() == BLACK)
                {
                    fill_input_features<BLACK>(*e.pos, features1_dst, features2_dst);
                }
                else
                {
                    fill_input_features<WHITE>(*e.pos, features1_dst, features2_dst);
                }
            }
        }

        int size;
        float* features1;
        float* features2;
        float* outcome;
        float* score;
        float* ply;

        ~Batch()
        {
            delete[] features1;
            delete[] features2;
            delete[] outcome;
            delete[] score;
            delete[] ply;
        }
    };
}

struct AnyStream
{
    virtual ~AnyStream() = default;
};

template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;

    Stream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filename, cyclic, skipPredicate))
    {
    }

    virtual StorageT* next() = 0;

protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filename, cyclic, skipPredicate)
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    std::future<StorageT*> m_next;
};

template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : Stream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int concurrency, const char* filename, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filename,
            cyclic,
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while (!m_stop_flag.load())
            {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);
                    BaseType::m_stream->fill(entries, m_batch_size);
                    if (entries.empty())
                    {
                        break;
                    }
                }

                auto batch = new StorageT(FeatureSet{}, entries);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    StorageT* next() override
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<StorageT*> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};

static bool initialized = false;

static void EnsureInitialize()
{
    if (initialized) {
        return;
    }
    initialized = true;

    USI::init(Options);
    //Bitboards::init();
    //Position::init();
    //Search::init();

    Threads.set(1);

    //Eval::init();

    is_ready();
}

extern "C" {

    EXPORT SparseBatch* get_sparse_batch_from_fens(
        const char* feature_set_c,
        int num_fens,
        const char* const* fens,
        int* scores,
        int* plies,
        int* results
    )
    {
        EnsureInitialize();

        std::vector<TrainingDataEntry> entries;
        entries.reserve(num_fens);
        for (int i = 0; i < num_fens; ++i)
        {
            auto& e = entries.emplace_back();
            e.pos->set(fens[i], &e.stateInfo, Threads.main());
            //movegen::forEachLegalMove(e.pos, [&](Move m){e.move = m;});
            e.move = MOVE_NONE;
            e.score = scores[i];
            e.ply = plies[i];
            e.result = results[i];
        }

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKP")
        {
            return new SparseBatch(FeatureSet<HalfKP>{}, entries);
        }
        else if (feature_set == "HalfKP^")
        {
            return new SparseBatch(FeatureSet<HalfKPFactorized>{}, entries);
        }
        // else if (feature_set == "HalfKA")
        // {
        //     return new SparseBatch(FeatureSet<HalfKA>{}, entries);
        // }
        // else if (feature_set == "HalfKA^")
        // {
        //     return new SparseBatch(FeatureSet<HalfKAFactorized>{}, entries);
        // }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set_c, int concurrency, const char* filename, int batch_size, bool cyclic, bool filtered, int random_fen_skipping)
    {
        EnsureInitialize();

        std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr;
        if (filtered || random_fen_skipping)
        {
            skipPredicate = [
                random_fen_skipping,
                    prob = double(random_fen_skipping) / (random_fen_skipping + 1),
                    filtered
            ](const TrainingDataEntry& e){

                    auto do_skip = [&]() {
                        std::bernoulli_distribution distrib(prob);
                        auto& prng = rng::get_thread_local_rng();
                        return distrib(prng);
                    };

                    auto do_filter = [&]() {
                        return (e.isCapturingMove() || e.isInCheck());
                    };

                    static thread_local std::mt19937 gen(std::random_device{}());
                    return (random_fen_skipping && do_skip()) || (filtered && do_filter());
                };
        }

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKP")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKP>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKP^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKPFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        // else if (feature_set == "HalfKA")
        // {
        //     return new FeaturedBatchStream<FeatureSet<HalfKA>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        // }
        // else if (feature_set == "HalfKA^")
        // {
        //     return new FeaturedBatchStream<FeatureSet<HalfKAFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        // }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        return stream->next();
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

    EXPORT Stream<dlshogi::Batch>* CDECL create_dlshogi_batch_stream(const char* filename, int concurrency, int batch_size, bool cyclic, bool filtered, int random_fen_skipping)
    {
        EnsureInitialize();

        std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr;
        if (filtered || random_fen_skipping)
        {
            skipPredicate = [
                random_fen_skipping,
                    prob = double(random_fen_skipping) / (random_fen_skipping + 1),
                    filtered
            ](const TrainingDataEntry& e){

                    auto do_skip = [&]() {
                        std::bernoulli_distribution distrib(prob);
                        auto& prng = rng::get_thread_local_rng();
                        return distrib(prng);
                    };

                    auto do_filter = [&]() {
                        return (e.isCapturingMove() || e.isInCheck());
                    };

                    return (random_fen_skipping && do_skip()) || (filtered && do_filter());
                };
        }

        return new FeaturedBatchStream<FeatureSet<HalfKP>, dlshogi::Batch>(
            concurrency,
            filename,
            batch_size,
            cyclic,
            skipPredicate
        );
    }

    EXPORT void CDECL destroy_dlshogi_batch_stream(Stream<dlshogi::Batch>* stream)
    {
        delete stream;
    }

    EXPORT dlshogi::Batch* CDECL fetch_next_dlshogi_batch(Stream<dlshogi::Batch>* stream)
    {
        return stream->next();
    }

    EXPORT void CDECL destroy_dlshogi_batch(dlshogi::Batch* batch)
    {
        delete batch;
    }

}

/* benches */ //*
#include <chrono>

int main()
{
    auto stream = create_sparse_batch_stream("HalfKP^", 4, R"(C:\shogi\training_data\suisho5.shuffled.qsearch\shuffled.bin)", 8192, true, false, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        if (i % 100 == 0) std::cout << i << '\n';
        destroy_sparse_batch(stream->next());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}
//*/
