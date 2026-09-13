#pragma once

#include "zyron/training/dataset.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace zyron::training {

struct Batch {
    std::vector<std::vector<std::size_t>> input_ids;
    std::vector<std::vector<std::size_t>> target_ids;
    [[nodiscard]] std::size_t size() const noexcept { return input_ids.size(); }
    [[nodiscard]] bool empty() const noexcept { return input_ids.empty(); }
};

struct DataLoaderConfig {
    DatasetConfig dataset;
    std::size_t batch_size{4};
    bool shuffle{true};
    std::size_t shuffle_buffer_size{256};
    std::size_t workers{1};
    std::size_t prefetch_batches{2};
    bool drop_last{false};
    std::uint32_t seed{0};
};

class DataLoader {
public:
    explicit DataLoader(DataLoaderConfig config);
    ~DataLoader();
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    [[nodiscard]] bool next(Batch& batch);
    void reset();
    [[nodiscard]] const DataLoaderConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::size_t epoch() const noexcept { return epoch_; }

private:
    void start_prefetch();
    void stop_prefetch();
    void worker_loop(std::size_t worker_id);
    bool next_sample(StreamingTextDataset& dataset, SequencePair& sample, std::mt19937& rng, std::vector<SequencePair>& shuffle_buffer);
    bool fill_batch(StreamingTextDataset& dataset, Batch& batch, std::mt19937& rng, std::vector<SequencePair>& shuffle_buffer);
    void push_batch(Batch batch);
    bool pop_batch(Batch& batch);

    DataLoaderConfig config_;
    std::size_t epoch_{0};
    std::unique_ptr<StreamingTextDataset> local_dataset_;
    std::mt19937 local_rng_;
    std::vector<SequencePair> local_shuffle_buffer_;

    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<Batch> queue_;
    std::vector<std::thread> workers_;
    bool stopping_{false};
    std::size_t active_workers_{0};
    std::size_t finished_workers_{0};
};

} // namespace zyron::training
