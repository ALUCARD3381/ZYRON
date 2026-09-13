#pragma once

#include "zyron/model/language_model.hpp"
#include "zyron/tokenizer/bpe.hpp"
#include "zyron/training/adamw.hpp"
#include "zyron/training/scheduler.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace zyron::model {

struct CheckpointState {
    transformer::Config config;
    std::size_t training_step{0};
    std::size_t optimizer_step{0};
    float optimizer_learning_rate{0.0f};
    float optimizer_beta1{0.9f};
    float optimizer_beta2{0.999f};
    float optimizer_epsilon{1e-8f};
    float optimizer_weight_decay{0.01f};
    std::vector<Tensor> model_parameters;
    std::vector<Tensor> first_moments;
    std::vector<Tensor> second_moments;
    std::size_t scheduler_step{0};
    float scheduler_base_learning_rate{0.0f};
    std::size_t scheduler_warmup_steps{0};
    std::size_t scheduler_total_steps{0};
    float scheduler_min_learning_rate{0.0f};
    std::vector<std::byte> tokenizer_data;
};

class Checkpoint final {
public:
    static void save(
        const std::string& path,
        const LanguageModel& model,
        const training::AdamW& optimizer,
        const training::WarmupCosineScheduler& scheduler,
        const tokenizer::BPE& tokenizer,
        std::size_t training_step);

    [[nodiscard]] static CheckpointState load(const std::string& path);

    static void restore(
        const CheckpointState& state,
        LanguageModel& model,
        training::AdamW& optimizer,
        training::WarmupCosineScheduler& scheduler,
        tokenizer::BPE& tokenizer);
};

} // namespace zyron::model
