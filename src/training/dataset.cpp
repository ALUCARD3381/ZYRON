#include "zyron/training/dataset.hpp"

#include <stdexcept>
#include <utility>

namespace zyron::training {

StreamingTextDataset::StreamingTextDataset(DatasetConfig config)
    : config_(std::move(config)) {
    if (config_.paths.empty()) throw std::invalid_argument("ZYRON Dataset: no input files");
    if (!config_.tokenizer) throw std::invalid_argument("ZYRON Dataset: tokenizer is required");
    if (config_.sequence_length == 0) throw std::invalid_argument("ZYRON Dataset: sequence_length must be > 0");
    if (config_.chunk_bytes == 0) throw std::invalid_argument("ZYRON Dataset: chunk_bytes must be > 0");

    if (config_.shard_count == 0 || config_.shard_index >= config_.shard_count) throw std::invalid_argument("ZYRON Dataset: invalid shard");
    if (config_.add_bos && !config_.tokenizer->has_bos()) throw std::invalid_argument("ZYRON Dataset: tokenizer has no BOS token");
    if (config_.add_eos && !config_.tokenizer->has_eos()) throw std::invalid_argument("ZYRON Dataset: tokenizer has no EOS token");
    reset();
}

void StreamingTextDataset::reset() {
    if (input_.is_open()) input_.close();
    file_index_ = 0;
    current_file_finished_ = true;
    exhausted_ = false;
    bos_inserted_for_current_file_ = false;
    next_sequence_index_ = 0;
    token_buffer_.clear();
    token_buffer_.reserve(config_.chunk_bytes + config_.sequence_length + 2);
}

bool StreamingTextDataset::accepts_sequence(std::size_t index) const noexcept {
    if (config_.validation_period == 0) return !config_.validation_only;
    const bool is_validation = (index % config_.validation_period) == 0;
    if (is_validation != config_.validation_only) return false;
    return (index % config_.shard_count) == config_.shard_index;
}

bool StreamingTextDataset::open_next_file() {
    while (file_index_ < config_.paths.size()) {
        input_.open(config_.paths[file_index_], std::ios::binary);
        if (!input_) throw std::runtime_error("ZYRON Dataset: failed to open file: " + config_.paths[file_index_]);
        ++file_index_;
        current_file_finished_ = false;
        bos_inserted_for_current_file_ = false;
        if (config_.add_bos) {
            token_buffer_.push_back(config_.tokenizer->bos_id());
            bos_inserted_for_current_file_ = true;
        }
        return true;
    }
    return false;
}

void StreamingTextDataset::finish_current_file() {
    if (current_file_finished_) return;
    if (input_.is_open()) input_.close();
    current_file_finished_ = true;
    if (config_.add_eos) token_buffer_.push_back(config_.tokenizer->eos_id());
}

bool StreamingTextDataset::fill_tokens() {
    const std::size_t required = config_.sequence_length + 1;
    while (token_buffer_.size() < required && !exhausted_) {
        if (current_file_finished_) {
            if (!open_next_file()) { exhausted_ = true; break; }
        }

        std::string chunk(config_.chunk_bytes, '\0');
        input_.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const auto count = input_.gcount();
        if (count > 0) {
            chunk.resize(static_cast<std::size_t>(count));
            const auto ids = config_.tokenizer->encode(chunk, false, false);
            token_buffer_.insert(token_buffer_.end(), ids.begin(), ids.end());
        }

        if (input_.eof() || count == 0) {
            finish_current_file();
        }
    }
    return token_buffer_.size() >= required;
}

bool StreamingTextDataset::next(SequencePair& sample) {
    sample.input.clear();
    sample.target.clear();
    while (true) {
        if (!fill_tokens()) return false;
        const std::size_t index = next_sequence_index_++;
        sample.input.assign(token_buffer_.begin(), token_buffer_.begin() + static_cast<std::ptrdiff_t>(config_.sequence_length));
        sample.target.assign(token_buffer_.begin() + 1, token_buffer_.begin() + static_cast<std::ptrdiff_t>(config_.sequence_length + 1));
        token_buffer_.erase(token_buffer_.begin(), token_buffer_.begin() + static_cast<std::ptrdiff_t>(config_.sequence_length));
        if (accepts_sequence(index)) return true;
    }
}

} // namespace zyron::training
