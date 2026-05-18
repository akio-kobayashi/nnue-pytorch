#ifndef _SFEN_STREAM_H_
#define _SFEN_STREAM_H_

#include "nnue_training_data_formats.h"
#include "../YaneuraOu/source/learn/learn.h"

#include <optional>
#include <fstream>
#include <string>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <ppl.h>
#endif

namespace training_data {

    using namespace binpack;

    static bool ends_with(const std::string& lhs, const std::string& end)
    {
        if (end.size() > lhs.size()) return false;

        return std::equal(end.rbegin(), end.rend(), lhs.rbegin());
    }

    static bool has_extension(const std::string& filename, const std::string& extension)
    {
        return ends_with(filename, "." + extension);
    }

    static std::string filename_with_extension(const std::string& filename, const std::string& ext)
    {
        if (ends_with(filename, ext))
        {
            return filename;
        }
        else
        {
            return filename + "." + ext;
        }
    }

    static std::string trim(const std::string& s)
    {
        const auto begin = s.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos) return "";
        const auto end = s.find_last_not_of(" \t\r\n");
        return s.substr(begin, end - begin + 1);
    }

    static std::string dirname(const std::string& path)
    {
        const auto pos = path.find_last_of("/\\");
        if (pos == std::string::npos) return "";
        return path.substr(0, pos);
    }

    static bool is_absolute_path(const std::string& path)
    {
        if (path.empty()) return false;
        if (path[0] == '/' || path[0] == '\\') return true;
        return path.size() > 1 && path[1] == ':';
    }

    static std::string join_path(const std::string& base, const std::string& child)
    {
        if (base.empty() || is_absolute_path(child)) return child;
        const char last = base.back();
        if (last == '/' || last == '\\') return base + child;
        return base + "/" + child;
    }

    struct BasicSfenInputStream
    {
        virtual std::optional<TrainingDataEntry> next() = 0;
        virtual void fill(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                auto v = this->next();
                if (!v.has_value())
                {
                    break;
                }
                vec.emplace_back(*v);
            }
        }

        virtual bool eof() const = 0;
        virtual ~BasicSfenInputStream() {}
    };

    struct BinSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "bin";

        BinSfenInputStream(std::string filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_stream(filename, openmode),
            m_filename(filename),
            m_eof(!m_stream),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate))
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            Learner::PackedSfenValue e;
            bool reopenedFileOnce = false;
            for(;;)
            {
                if(m_stream.read(reinterpret_cast<char*>(&e), sizeof(Learner::PackedSfenValue)))
                {
                    auto entry = packedSfenValueToTrainingDataEntry(e);
                    if (!m_skipPredicate || !m_skipPredicate(entry))
                        return entry;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return std::nullopt;

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return std::nullopt;

                        continue;
                    }

                    m_eof = true;
                    return std::nullopt;
                }
            }
        }

        void fill(std::vector<TrainingDataEntry>& vec, std::size_t n) override
        {
            std::vector<Learner::PackedSfenValue> packedSfenValues(n);
            bool reopenedFileOnce = false;
            for (;;)
            {
                if (m_stream.read(reinterpret_cast<char*>(&packedSfenValues[0]), sizeof(Learner::PackedSfenValue) * n))
                {
                    vec.resize(n);
#ifdef _WIN32
                    concurrency::parallel_for(size_t(0), n, [&vec, &packedSfenValues](size_t i)
                        {
                            vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
                        });
#else
                    for (size_t i = 0; i < n; ++i)
                    {
                        vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
                    }
#endif
                    return;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return;

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return;

                        continue;
                    }

                    m_eof = true;
                    return;
                }
            }
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinSfenInputStream() override {}

    private:
        std::fstream m_stream;
        std::string m_filename;
        bool m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
    };

    inline std::vector<std::string> load_manifest_filenames(const std::string& manifestPath)
    {
        std::ifstream manifest(manifestPath);
        if (!manifest)
            throw std::runtime_error("Failed to open manifest: " + manifestPath);

        std::vector<std::string> filenames;
        const auto baseDir = dirname(manifestPath);
        std::string line;
        while (std::getline(manifest, line))
        {
            const auto commentPos = line.find('#');
            if (commentPos != std::string::npos)
                line = line.substr(0, commentPos);
            line = trim(line);
            if (line.empty())
                continue;
            filenames.push_back(join_path(baseDir, line));
        }

        if (filenames.empty())
            throw std::runtime_error("Manifest is empty: " + manifestPath);

        return filenames;
    }

    struct ManifestBinSfenInputStream : BasicSfenInputStream
    {
        ManifestBinSfenInputStream(
            std::vector<std::string> filenames,
            bool cyclic,
            std::function<bool(const TrainingDataEntry&)> skipPredicate
        ) :
            m_filenames(std::move(filenames)),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate)),
            m_eof(m_filenames.empty())
        {
            if (!m_eof)
                open_current_file();
        }

        std::optional<TrainingDataEntry> next() override
        {
            while (!m_eof)
            {
                auto v = m_currentStream->next();
                if (v.has_value())
                    return v;
                if (!advance_stream())
                    return std::nullopt;
            }
            return std::nullopt;
        }

        bool eof() const override
        {
            return m_eof;
        }

    private:
        void open_current_file()
        {
            m_currentStream = std::make_unique<BinSfenInputStream>(
                m_filenames[m_currentIndex],
                false,
                m_skipPredicate
            );
            if (m_currentStream->eof())
                throw std::runtime_error("Failed to open training data file: " + m_filenames[m_currentIndex]);
        }

        bool advance_stream()
        {
            if (m_filenames.empty())
            {
                m_eof = true;
                return false;
            }

            ++m_currentIndex;
            if (m_currentIndex >= m_filenames.size())
            {
                if (!m_cyclic)
                {
                    m_eof = true;
                    return false;
                }
                m_currentIndex = 0;
            }

            open_current_file();
            return true;
        }

        std::vector<std::string> m_filenames;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
        std::size_t m_currentIndex = 0;
        bool m_eof = false;
        std::unique_ptr<BinSfenInputStream> m_currentStream;
    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file(const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));
        if (has_extension(filename, "txt"))
            return std::make_unique<ManifestBinSfenInputStream>(load_manifest_filenames(filename), cyclic, std::move(skipPredicate));

        return nullptr;
    }

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file_parallel(int concurrency, const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        // TODO (low priority): optimize and parallelize .bin reading.
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));
        if (has_extension(filename, "txt"))
            return std::make_unique<ManifestBinSfenInputStream>(load_manifest_filenames(filename), cyclic, std::move(skipPredicate));

        return nullptr;
    }
}

#endif
