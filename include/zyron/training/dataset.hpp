#pragma once

#include "zyron/tokenizer/tokenizer.hpp"

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace zyron::training {

struct SequencePair {
    std::vector<std::size_t> input;
    std::vector<std::size_t> target;
};

struct DatasetConfig {
    std::vector<std::string> paths;
    const tokenizer::Tokenizer* tokenizer{nullptr};
    std::size_t sequence_length{128};
    std::size_t chunk_bytes{1u << 20};
    bool add_bos{false};
    bool add_eos{true};
    bool validation_only{false};
    std::size_t validation_period{0};
    std::size_t shard_count{1};
    std::size_t shard_index{0};
};

class StreamingTextDataset {
public:
    explicit StreamingTextDataset(DatasetConfig config);

    [[nodiscard]] bool next(SequencePair& sample);
    void reset();

    [[nodiscard]] std::size_t sequence_index() const noexcept { return next_sequence_index_; }
    [[nodiscard]] bool exhausted() const noexcept { return exhausted_; }

private:
    bool fill_tokens();
    bool open_next_file();
    void finish_current_file();
    bool accepts_sequence(std::size_t index) const noexcept;

    DatasetConfig config_;
    std::size_t file_index_{0};
    std::ifstream input_;
    bool current_file_finished_{true};
    bool exhausted_{false};
    bool bos_inserted_for_current_file_{false};
    std::size_t next_sequence_index_{0};
    std::vector<std::size_t> token_buffer_;
};

} // namespace zyron::training
