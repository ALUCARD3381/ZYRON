#include "zyron/training/dataloader.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace zyron::training {

DataLoader::DataLoader(DataLoaderConfig config)
    : config_(std::move(config)),
      local_rng_(config_.seed) {
    if (config_.batch_size == 0) throw std::invalid_argument("ZYRON DataLoader: batch_size must be > 0");
    if (config_.shuffle && config_.shuffle_buffer_size == 0) throw std::invalid_argument("ZYRON DataLoader: shuffle_buffer_size must be > 0");
    if (config_.workers == 0) config_.workers = 1;
    reset();
}

DataLoader::~DataLoader() { stop_prefetch(); }

void DataLoader::stop_prefetch() {
    if (workers_.empty()) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
    for (auto& worker : workers_) if (worker.joinable()) worker.join();
    workers_.clear();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
        stopping_ = false;
        active_workers_ = 0;
        finished_workers_ = 0;
    }
}

void DataLoader::start_prefetch() {
    const auto count = std::max<std::size_t>(1, config_.workers);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
        stopping_ = false;
        active_workers_ = count;
        finished_workers_ = 0;
    }
    workers_.reserve(count);
    for (std::size_t i=0;i<count;++i) workers_.emplace_back(&DataLoader::worker_loop,this,i);
}

void DataLoader::reset() {
    stop_prefetch();
    ++epoch_;
    local_dataset_.reset();
    local_shuffle_buffer_.clear();
    local_rng_.seed(config_.seed + static_cast<std::uint32_t>(epoch_ * 7919u));
    if (config_.prefetch_batches > 0) start_prefetch();
    else local_dataset_=std::make_unique<StreamingTextDataset>(config_.dataset);
}

bool DataLoader::next_sample(
    StreamingTextDataset& dataset,
    SequencePair& sample,
    std::mt19937& rng,
    std::vector<SequencePair>& shuffle_buffer) {

    if (!config_.shuffle) return dataset.next(sample);
    while (shuffle_buffer.size() < config_.shuffle_buffer_size) {
        SequencePair candidate;
        if (!dataset.next(candidate)) break;
        shuffle_buffer.push_back(std::move(candidate));
    }
    if (shuffle_buffer.empty()) return false;
    std::uniform_int_distribution<std::size_t> pick(0,shuffle_buffer.size()-1);
    const auto index=pick(rng);
    sample=std::move(shuffle_buffer[index]);
    SequencePair replacement;
    if (dataset.next(replacement)) shuffle_buffer[index]=std::move(replacement);
    else { shuffle_buffer[index]=std::move(shuffle_buffer.back()); shuffle_buffer.pop_back(); }
    return true;
}

bool DataLoader::fill_batch(StreamingTextDataset& dataset, Batch& batch, std::mt19937& rng, std::vector<SequencePair>& shuffle_buffer) {
    batch.input_ids.clear(); batch.target_ids.clear();
    batch.input_ids.reserve(config_.batch_size); batch.target_ids.reserve(config_.batch_size);
    for (std::size_t i=0;i<config_.batch_size;++i) {
        SequencePair sample;
        if (!next_sample(dataset,sample,rng,shuffle_buffer)) break;
        batch.input_ids.push_back(std::move(sample.input));
        batch.target_ids.push_back(std::move(sample.target));
    }
    if (batch.empty()) return false;
    if (config_.drop_last && batch.size()!=config_.batch_size) { batch.input_ids.clear(); batch.target_ids.clear(); return false; }
    return true;
}

void DataLoader::push_batch(Batch batch) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock,[&]{ return stopping_ || queue_.size()<std::max<std::size_t>(1,config_.prefetch_batches); });
    if (stopping_) return;
    queue_.push_back(std::move(batch));
    lock.unlock(); not_empty_.notify_one();
}

void DataLoader::worker_loop(std::size_t worker_id) {
    auto dataset_config=config_.dataset;
    dataset_config.shard_count=std::max<std::size_t>(1,config_.workers);
    dataset_config.shard_index=worker_id;
    StreamingTextDataset dataset(dataset_config);
    std::mt19937 rng(config_.seed + static_cast<std::uint32_t>(worker_id*7919u + epoch_));
    std::vector<SequencePair> shuffle_buffer;
    if (config_.shuffle) shuffle_buffer.reserve(config_.shuffle_buffer_size);

    while (true) {
        Batch batch;
        if (!fill_batch(dataset,batch,rng,shuffle_buffer)) break;
        push_batch(std::move(batch));
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) break;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++finished_workers_;
    }
    not_empty_.notify_all();
}

bool DataLoader::pop_batch(Batch& batch) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock,[&]{ return stopping_ || !queue_.empty() || finished_workers_==active_workers_; });
    if (stopping_ || queue_.empty()) return false;
    batch=std::move(queue_.front()); queue_.pop_front();
    lock.unlock(); not_full_.notify_one();
    return true;
}

bool DataLoader::next(Batch& batch) {
    if (config_.prefetch_batches > 0) return pop_batch(batch);
    if (!local_dataset_) local_dataset_=std::make_unique<StreamingTextDataset>(config_.dataset);
    return fill_batch(*local_dataset_,batch,local_rng_,local_shuffle_buffer_);
}

} // namespace zyron::training
