#include "zyron/model/language_model.hpp"
#include "zyron/tokenizer/bpe.hpp"
#include "zyron/training/adamw.hpp"
#include "zyron/training/dataloader.hpp"
#include "zyron/training/scheduler.hpp"

#include <chrono>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <memory>

int main() {
    using namespace zyron;
    using namespace zyron::training;

    tokenizer::BPE tokenizer;
    tokenizer.train_text("o gato dorme. o gato corre. o cao dorme. ", 280);

    auto config = transformer::tiny(tokenizer.vocab_size());
    config.hidden_size = 32;
    config.num_layers = 1;
    config.num_heads = 2;
    config.intermediate_size = 64;
    config.max_sequence_length = 16;

    const std::string data_path = "phase4_benchmark_dataset.txt";
    {
        std::ofstream out(data_path);
        for (int i = 0; i < 40; ++i) out << "o gato dorme. o gato corre. o cao dorme. o gato come.\n";
    }
    DatasetConfig dataset;
    dataset.paths = {data_path};
    dataset.tokenizer = &tokenizer;
    dataset.sequence_length = 16;
    dataset.chunk_bytes = 128;
    dataset.validation_period = 0;

    DataLoaderConfig dl;
    dl.dataset = dataset;
    dl.batch_size = 2;
    dl.shuffle = false;
    dl.prefetch_batches = 0;
    dl.drop_last = false;

    DataLoader loader(dl);
    model::LanguageModel model(config, 42);
    AdamW optimizer(model.parameters(), 3e-4f);
    WarmupCosineScheduler scheduler(3e-4f, 2, 10, 3e-5f);

    double total_ms = 0.0;
    std::size_t iterations = 0;
    float sink = 0.0f;
    Batch batch;
    while (iterations < 5 && loader.next(batch)) {
        optimizer.zero_grad();
        const auto start = std::chrono::steady_clock::now();
        auto loss = model.loss(batch.input_ids, batch.target_ids);
        loss.backward();
        clip_grad_norm(optimizer.parameters(), 1.0f);
        scheduler.step(optimizer);
        optimizer.step();
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end-start).count();
        sink += loss.value()[0];
        ++iterations;
    }

    std::remove(data_path.c_str());
    std::cout << "ZYRON Phase 4 training benchmark\n"
              << "================================\n"
              << "Batch: 2\n"
              << "Sequence: 16\n"
              << "Iterations: " << iterations << "\n"
              << "Average step: " << (iterations ? total_ms / iterations : 0.0) << " ms\n"
              << "sink=" << sink << "\n";
    return 0;
}
