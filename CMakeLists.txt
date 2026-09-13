cmake_minimum_required(VERSION 3.16)

project(ZYRON
    VERSION 0.8.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

option(ZYRON_ENABLE_NEON "Build ARMv7 NEON kernels when targeting ARM" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(zyron_core
    src/core/memory.cpp
    src/core/runtime.cpp
    src/math/kernels.cpp
    src/quantization/quantization.cpp
    src/quantization/quantized_linear.cpp
    src/core/tensor.cpp
    src/math/ops.cpp
    src/autograd/autograd.cpp
    src/nn/linear.cpp
    src/nn/dropout.cpp
    src/nn/embedding.cpp
    src/nn/normalization.cpp
    src/nn/sequential.cpp
    src/tokenizer/bpe.cpp
    src/transformer/attention.cpp
    src/transformer/feed_forward.cpp
    src/transformer/transformer_block.cpp
    src/transformer/transformer.cpp
    src/model/language_model.cpp
    src/model/quantized_language_model.cpp
    src/model/generation.cpp
    src/model/checkpoint.cpp
    src/training/dataset.cpp
    src/training/dataloader.cpp
    src/training/adamw.cpp
    src/training/scheduler.cpp
    src/training/trainer.cpp
)

target_link_libraries(zyron_core PUBLIC Threads::Threads)

target_include_directories(zyron_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(zyron_core PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(ZYRON_ENABLE_NEON AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|armv7|armv7l|armhf)")
    target_compile_definitions(zyron_core PRIVATE ZYRON_ENABLE_NEON=1)
    target_compile_options(zyron_core PRIVATE -mfpu=neon)
endif()

enable_testing()

add_executable(zyron_tests
    tests/test_tensor.cpp
)
target_link_libraries(zyron_tests PRIVATE zyron_core)
add_test(NAME zyron_tests COMMAND zyron_tests)

add_executable(zyron_autograd_tests
    tests/test_autograd.cpp
)
target_link_libraries(zyron_autograd_tests PRIVATE zyron_core)
add_test(NAME zyron_autograd_tests COMMAND zyron_autograd_tests)

add_executable(zyron_nn_tests
    tests/test_nn.cpp
)
target_link_libraries(zyron_nn_tests PRIVATE zyron_core)
add_test(NAME zyron_nn_tests COMMAND zyron_nn_tests)

add_executable(zyron_tokenizer_tests
    tests/test_tokenizer.cpp
)
target_link_libraries(zyron_tokenizer_tests PRIVATE zyron_core)
add_test(NAME zyron_tokenizer_tests COMMAND zyron_tokenizer_tests)

add_executable(zyron_transformer_tests
    tests/test_transformer.cpp
)
target_link_libraries(zyron_transformer_tests PRIVATE zyron_core)
add_test(NAME zyron_transformer_tests COMMAND zyron_transformer_tests)

add_executable(zyron_benchmark
    benchmarks/benchmark_tensor.cpp
)
target_link_libraries(zyron_benchmark PRIVATE zyron_core)

add_executable(zyron_autograd_benchmark
    benchmarks/benchmark_autograd.cpp
)
target_link_libraries(zyron_autograd_benchmark PRIVATE zyron_core)

add_executable(zyron_nn_benchmark
    benchmarks/benchmark_nn.cpp
)
target_link_libraries(zyron_nn_benchmark PRIVATE zyron_core)

add_executable(zyron_transformer_benchmark
    benchmarks/benchmark_transformer.cpp
)
target_link_libraries(zyron_transformer_benchmark PRIVATE zyron_core)

add_executable(zyron_phase2_example
    examples/phase2_mlp.cpp
)
target_link_libraries(zyron_phase2_example PRIVATE zyron_core)

add_executable(zyron_phase3_example
    examples/phase3_language_model.cpp
)
target_link_libraries(zyron_phase3_example PRIVATE zyron_core)

add_executable(zyron_tokenizer
    tools/zyron_tokenizer.cpp
)
target_link_libraries(zyron_tokenizer PRIVATE zyron_core)

add_executable(zyron_training_tests
    tests/test_training.cpp
)
target_link_libraries(zyron_training_tests PRIVATE zyron_core)
add_test(NAME zyron_training_tests COMMAND zyron_training_tests)

add_executable(zyron_training_benchmark
    benchmarks/benchmark_training.cpp
)
target_link_libraries(zyron_training_benchmark PRIVATE zyron_core)

add_executable(zyron_phase5_tests
    tests/test_phase5.cpp
)
target_link_libraries(zyron_phase5_tests PRIVATE zyron_core)
add_test(NAME zyron_phase5_tests COMMAND zyron_phase5_tests)

add_executable(zyron_final_tests tests/test_final.cpp)
target_link_libraries(zyron_final_tests PRIVATE zyron_core)
add_test(NAME zyron_final_tests COMMAND zyron_final_tests)

add_executable(zyron_phase5_benchmark
    benchmarks/benchmark_phase5.cpp
)
target_link_libraries(zyron_phase5_benchmark PRIVATE zyron_core)

add_executable(zyron_train
    tools/zyron_train.cpp
)
target_link_libraries(zyron_train PRIVATE zyron_core)

add_executable(zyron_generate tools/zyron_generate.cpp)
target_link_libraries(zyron_generate PRIVATE zyron_core)

add_executable(zyron_quantize tools/zyron_quantize.cpp)
target_link_libraries(zyron_quantize PRIVATE zyron_core)
