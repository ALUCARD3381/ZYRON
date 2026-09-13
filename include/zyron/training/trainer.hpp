#pragma once

#include "zyron/model/checkpoint.hpp"
#include "zyron/model/language_model.hpp"
#include "zyron/tokenizer/bpe.hpp"
#include "zyron/training/adamw.hpp"
#include "zyron/training/dataloader.hpp"
#include "zyron/training/scheduler.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace zyron::training {

struct TrainerConfig {
    std::size_t max_steps{1000};
    std::size_t log_interval{10};
    std::size_t eval_interval{100};
    std::size_t validation_steps{10};
    std::size_t checkpoint_interval{500};
    float gradient_clip_norm{1.0f};
    std::size_t gradient_accumulation_steps{1};
    std::string checkpoint_path{"models/checkpoint.zyron"};
    std::string log_path{};
};

struct TrainState {
    std::size_t step{0};
    double elapsed_seconds{0.0};
    double tokens_per_second{0.0};
    float train_loss{0.0f};
    float validation_loss{0.0f};
    float perplexity{0.0f};
    float learning_rate{0.0f};
    float grad_norm{0.0f};
    std::size_t rss_bytes{0};
};

class Trainer final {
public:
    Trainer(
        model::LanguageModel& model,
        tokenizer::BPE& tokenizer,
        Optimizer& optimizer,
        WarmupCosineScheduler& scheduler,
        DataLoader& train_loader,
        DataLoader* validation_loader,
        TrainerConfig config = {});

    [[nodiscard]] TrainState train(std::size_t initial_step = 0);
    [[nodiscard]] float evaluate(DataLoader& loader, std::size_t steps);

private:
    static std::size_t current_rss_bytes() noexcept;
    void log(const TrainState& state);
    void save_checkpoint(std::size_t step);

    model::LanguageModel& model_;
    tokenizer::BPE& tokenizer_;
    Optimizer& optimizer_;
    WarmupCosineScheduler& scheduler_;
    DataLoader& train_loader_;
    DataLoader* validation_loader_;
    TrainerConfig config_;
    std::string log_file_path_;
};

} // namespace zyron::training
