#include "zyron/training/trainer.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>
#include <unistd.h>

namespace zyron::training {

namespace {

void scale_gradients(
    const std::vector<autograd::Variable*>& parameters,
    float scale) {
    for (auto* parameter : parameters) {
        if (!parameter || !parameter->has_grad()) continue;
        Tensor grad = parameter->grad().contiguous();
        float* data = grad.data();
        for (std::size_t i = 0; i < grad.size(); ++i) data[i] *= scale;
        parameter->grad() = std::move(grad);
    }
}

} // namespace

Trainer::Trainer(
    model::LanguageModel& model,
    tokenizer::BPE& tokenizer,
    Optimizer& optimizer,
    WarmupCosineScheduler& scheduler,
    DataLoader& train_loader,
    DataLoader* validation_loader,
    TrainerConfig config)
    : model_(model), tokenizer_(tokenizer), optimizer_(optimizer), scheduler_(scheduler),
      train_loader_(train_loader), validation_loader_(validation_loader), config_(std::move(config)),
      log_file_path_(config_.log_path) {

    if (config_.max_steps == 0) throw std::invalid_argument("ZYRON Trainer: max_steps must be > 0");
    if (config_.log_interval == 0) config_.log_interval = 1;
    if (config_.eval_interval == 0) config_.eval_interval = config_.max_steps;
    if (config_.validation_steps == 0) config_.validation_steps = 1;
    if (config_.checkpoint_interval == 0) config_.checkpoint_interval = config_.max_steps + 1;
    if (config_.gradient_clip_norm <= 0.0f || !std::isfinite(config_.gradient_clip_norm)) {
        throw std::invalid_argument("ZYRON Trainer: gradient_clip_norm must be finite and > 0");
    }
    if (config_.gradient_accumulation_steps == 0) {
        config_.gradient_accumulation_steps = 1;
    }
}

std::size_t Trainer::current_rss_bytes() noexcept {
#if defined(__linux__)
    std::ifstream in("/proc/self/status");
    std::string key;
    std::size_t value = 0;
    std::string unit;
    while (in >> key >> value >> unit) {
        if (key == "VmRSS:") {
            return value * 1024u;
        }
    }
#endif
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
        return static_cast<std::size_t>(usage.ru_maxrss);
#else
        return static_cast<std::size_t>(usage.ru_maxrss) * 1024u;
#endif
    }
    return 0;
}

float Trainer::evaluate(DataLoader& loader, std::size_t steps) {
    const bool was_training = model_.is_training();
    model_.eval();
    autograd::NoGradGuard no_grad;

    double total = 0.0;
    std::size_t count = 0;
    Batch batch;
    loader.reset();

    while (count < steps && loader.next(batch)) {
        const auto loss = model_.loss(batch.input_ids, batch.target_ids);
        total += static_cast<double>(loss.value()[0]);
        ++count;
    }

    model_.train(was_training);
    if (count == 0) return 0.0f;
    return static_cast<float>(total / static_cast<double>(count));
}

void Trainer::log(const TrainState& state) {
    std::cout << "ZYRON Training\n"
              << "----------------------------\n"
              << "Step:       " << state.step << '\n'
              << "Loss:       " << std::fixed << std::setprecision(6) << state.train_loss << '\n'
              << "Val Loss:   " << state.validation_loss << '\n'
              << "Perplexity: " << state.perplexity << '\n'
              << "LR:         " << state.learning_rate << '\n'
              << "Grad Norm:  " << state.grad_norm << '\n'
              << "Tokens/sec: " << state.tokens_per_second << '\n'
              << "RAM:        " << (state.rss_bytes / (1024.0 * 1024.0)) << " MiB\n"
              << "Time:       " << state.elapsed_seconds << " s\n"
              << "----------------------------\n";

    if (!log_file_path_.empty()) {
        std::ofstream out(log_file_path_, std::ios::app);
        if (out) {
            out << state.step << ','
                << state.train_loss << ','
                << state.validation_loss << ','
                << state.perplexity << ','
                << state.learning_rate << ','
                << state.grad_norm << ','
                << state.tokens_per_second << ','
                << state.rss_bytes << ','
                << state.elapsed_seconds << '\n';
        }
    }
}

void Trainer::save_checkpoint(std::size_t step) {
    if (config_.checkpoint_path.empty()) return;
    model::Checkpoint::save(
        config_.checkpoint_path,
        model_,
        dynamic_cast<AdamW&>(optimizer_),
        scheduler_,
        tokenizer_,
        step);
}

TrainState Trainer::train(std::size_t initial_step) {
    if (initial_step > config_.max_steps) throw std::invalid_argument("ZYRON Trainer: initial_step exceeds max_steps");

    model_.train(true);
    TrainState state;
    state.step = initial_step;
    const auto start = std::chrono::steady_clock::now();
    std::size_t tokens_seen = 0;
    std::size_t last_tokens = 0;
    auto last_log_time = start;

    std::size_t empty_loader_resets = 0;
    std::size_t accumulation_count = 0;
    double accumulated_loss = 0.0;
    std::size_t accumulated_tokens = 0;
    optimizer_.zero_grad();

    while (state.step < config_.max_steps) {
        Batch batch;
        if (!train_loader_.next(batch)) {
            if (accumulation_count > 0) {
                const float scale = 1.0f / static_cast<float>(accumulation_count);
                scale_gradients(optimizer_.parameters(), scale);
                state.train_loss = static_cast<float>(accumulated_loss / static_cast<double>(accumulation_count));
                state.grad_norm = clip_grad_norm(optimizer_.parameters(), config_.gradient_clip_norm);
                scheduler_.step(optimizer_);
                optimizer_.step();
                ++state.step;
                tokens_seen += accumulated_tokens;
                accumulation_count = 0;
                accumulated_loss = 0.0;
                accumulated_tokens = 0;
                optimizer_.zero_grad();
                continue;
            }
            ++empty_loader_resets;
            if (empty_loader_resets >= 2) {
                throw std::runtime_error("ZYRON Trainer: training DataLoader produced no batches");
            }
            train_loader_.reset();
            continue;
        }
        empty_loader_resets = 0;

        auto loss = model_.loss(batch.input_ids, batch.target_ids);
        accumulated_loss += static_cast<double>(loss.value()[0]);
        ++accumulation_count;
        const std::size_t batch_tokens =
            batch.size() * (batch.input_ids.empty() ? 0u : batch.input_ids.front().size());
        accumulated_tokens += batch_tokens;
        loss.backward();

        const bool should_step =
            accumulation_count >= config_.gradient_accumulation_steps ||
            state.step + 1 >= config_.max_steps;
        if (!should_step) continue;

        const float scale = 1.0f / static_cast<float>(accumulation_count);
        scale_gradients(optimizer_.parameters(), scale);
        state.train_loss = static_cast<float>(accumulated_loss / static_cast<double>(accumulation_count));
        state.grad_norm = clip_grad_norm(optimizer_.parameters(), config_.gradient_clip_norm);

        scheduler_.step(optimizer_);
        optimizer_.step();
        ++state.step;

        tokens_seen += accumulated_tokens;
        accumulation_count = 0;
        accumulated_loss = 0.0;
        accumulated_tokens = 0;
        optimizer_.zero_grad();

        if (state.step % config_.eval_interval == 0 && validation_loader_) {
            state.validation_loss = evaluate(*validation_loader_, config_.validation_steps);
            state.perplexity = std::exp(std::min(20.0f, state.validation_loss));
            model_.train(true);
        }

        if (state.step % config_.checkpoint_interval == 0 || state.step == config_.max_steps) {
            save_checkpoint(state.step);
        }

        if (state.step % config_.log_interval == 0 || state.step == config_.max_steps) {
            const auto now = std::chrono::steady_clock::now();
            state.elapsed_seconds = std::chrono::duration<double>(now - start).count();
            const double interval_seconds = std::chrono::duration<double>(now - last_log_time).count();
            state.tokens_per_second = interval_seconds > 0.0
                ? static_cast<double>(tokens_seen - last_tokens) / interval_seconds
                : 0.0;
            last_tokens = tokens_seen;
            last_log_time = now;
            state.learning_rate = optimizer_.learning_rate();
            state.rss_bytes = current_rss_bytes();
            log(state);
        }
    }

    return state;
}

} // namespace zyron::training
