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
#include <fstream>
#include <vector>
#include <cstring>

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
        context_id = new float[size];
        sample_weight = new float[size];

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
    float* context_id;
    float* sample_weight;

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
        delete[] context_id;
        delete[] sample_weight;
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

namespace {

    // Preference binary file reader: reads PSV + META from separate sections.
    struct PreferenceBinaryStream : training_data::BasicSfenInputStream
    {
        static constexpr std::size_t PACKED_SFN_SIZE = 40;
        static constexpr std::size_t META_SIZE = 11; // 1+4+2+2+2

        PreferenceBinaryStream(std::string psv_path, std::string meta_path, bool cyclic)
            : m_psv_stream(psv_path, std::ios::in | std::ios::binary),
              m_meta_stream(meta_path, std::ios::in | std::ios::binary),
              m_cyclic(cyclic), m_eof(false)
        {
            if (!m_psv_stream || !m_meta_stream)
            {
                m_eof = true;
                return;
            }
            // Get file sizes to compute record count
            m_psv_stream.seekg(0, std::ios::end);
            m_meta_stream.seekg(0, std::ios::end);
            std::size_t psv_size = m_psv_stream.tellg();
            std::size_t meta_size = m_meta_stream.tellg();
            m_num_records = psv_size / PACKED_SFN_SIZE;
            if (meta_size != m_num_records * META_SIZE)
            {
                std::cerr << "Meta file size mismatch. psv=" << psv_size
                          << " meta=" << meta_size << " records=" << m_num_records << std::endl;
                m_eof = true;
            }
        }

        std::optional<TrainingDataEntry> next() override
        {
            if (m_eof)
                return std::nullopt;

            Learner::PackedSfenValue psv;
            if (!m_psv_stream.read(reinterpret_cast<char*>(&psv), PACKED_SFN_SIZE))
            {
                if (m_cyclic)
                {
                    m_psv_stream.clear();
                    m_meta_stream.clear();
                    m_psv_stream.seekg(0);
                    m_meta_stream.seekg(0);
                    if (!m_psv_stream.read(reinterpret_cast<char*>(&psv), PACKED_SFN_SIZE))
                    {
                        m_eof = true;
                        return std::nullopt;
                    }
                }
                else
                {
                    m_eof = true;
                    return std::nullopt;
                }
            }

            unsigned char meta_raw[META_SIZE];
            if (!m_meta_stream.read(reinterpret_cast<char*>(meta_raw), META_SIZE))
            {
                m_eof = true;
                return std::nullopt;
            }

            auto entry = packedSfenValueToTrainingDataEntry(psv);
            // Parse meta: game_result(u8) + actual_move(u32) + ply(u16) + context_id(u16) + sample_weight_q12(u16)
            std::size_t offset = 0;
            entry.result = static_cast<int>(meta_raw[offset]) - (meta_raw[offset] == 255 ? 2 : (meta_raw[offset] == 0 ? -1 : 0));
            // Fix: game_result 0=-1, 1=0, 2=1 (for loss/draw/win)
            // Original format: 1=win, 0=draw, 255=loss
            // YaneuraOu TrainingDataEntry.result: 1=win, -1=loss, 0=draw
            if (meta_raw[offset] == 1) entry.result = 1;
            else if (meta_raw[offset] == 0) entry.result = 0;
            else if (meta_raw[offset] == 255) entry.result = -1;
            else entry.result = 0;
            offset += 1;

            // Preference binaries now store the source pipeline's uint32 move
            // encoding. The base TrainingDataEntry path here does not consume
            // entry.move, so keep it inert instead of mis-decoding it as Move16.
            offset += 4;
            entry.move = MOVE_NONE;
            entry.ply = static_cast<uint16_t>(meta_raw[offset]) | (static_cast<uint16_t>(meta_raw[offset + 1]) << 8);
            offset += 2;
            // context_id and sample_weight_q12 are stored but not used by base TrainingDataEntry
            // They are passed through SparseBatch directly
            m_stored_context_id = static_cast<uint16_t>(meta_raw[offset]) | (static_cast<uint16_t>(meta_raw[offset + 1]) << 8);
            offset += 2;
            m_stored_weight_q12 = static_cast<uint16_t>(meta_raw[offset]) | (static_cast<uint16_t>(meta_raw[offset + 1]) << 8);

            return entry;
        }

        bool eof() const override { return m_eof; }
        ~PreferenceBinaryStream() override {}

        uint16_t stored_context_id() const { return m_stored_context_id; }
        uint16_t stored_weight_q12() const { return m_stored_weight_q12; }

    private:
        std::fstream m_psv_stream;
        std::fstream m_meta_stream;
        bool m_cyclic;
        bool m_eof;
        std::size_t m_num_records;
        uint16_t m_stored_context_id = 0;
        uint16_t m_stored_weight_q12 = 0;
    };

    // Stream that wraps a base stream and provides context_id + weight per entry
    template <typename BaseStream>
    class ContextAwareBatchStream
    {
    public:
        ContextAwareBatchStream(int concurrency,
                                std::unique_ptr<BaseStream> base,
                                std::function<bool(const TrainingDataEntry&)> skipPredicate)
            : m_concurrency(concurrency),
              m_base(std::move(base)),
              m_skipPredicate(skipPredicate),
              m_cyclic(base->eof()) // will be set by fill
        {}

        std::vector<TrainingDataEntry> fill(std::size_t batch_size)
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(batch_size);
            m_cyclic = true;
            for (std::size_t i = 0; i < batch_size; ++i)
            {
                if (!m_base)
                    break;
                auto entry = m_base->next();
                if (!entry.has_value())
                {
                    m_cyclic = false;
                    break;
                }
                if (m_skipPredicate && m_skipPredicate(*entry))
                {
                    --i; // don't count skipped entries
                    continue;
                }
                entries.push_back(*entry);
            }
            return entries;
        }

        uint16_t last_context_id() const { return 0; }

    private:
        int m_concurrency;
        std::unique_ptr<BaseStream> m_base;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
        bool m_cyclic;
    };

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
