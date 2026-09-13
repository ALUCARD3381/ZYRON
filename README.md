# ZYRON — Zero-level Yield, Reasoning & Optimization Network

ZYRON é uma engine de IA construída em C++20, evoluída de baixo para cima para treinamento e inferência de modelos Transformer, com prioridade em correção, memória previsível, portabilidade e execução em hardware limitado.

### Modern Transformer features

The current Transformer stack supports rotary positional embeddings (RoPE), grouped-query attention (GQA) through `num_kv_heads`, weight tying between token embeddings and the language-model head, and incremental KV-cache inference. Legacy checkpoints use the original sinusoidal/untied/full-MHA configuration when loaded.

## Estado

```text
FASE 1 — Core + Tensor Engine                 ✅
FASE 2 — Autograd + Neural Network Engine     ✅
FASE 3 — BPE + Transformer + Language Model   ✅
FASE 4 — Training + Evaluation                ✅
FASE 5 — Performance + Mobile + Quantization  ✅
```

O núcleo não depende de Python para execução, tokenização, treinamento ou inferência.

## Plataformas

O projeto é portátil entre Linux/x86_64 e Linux ARM. A implementação da Fase 5 inclui caminho ARMv7/NEON e foi desenhada para `armeabi-v7a`/`armhf` e dispositivos com pouca RAM.

No seu ambiente ARMv7, o alvo esperado é:

```text
uname -m              -> armv7l
dpkg architecture     -> armhf
compiler target       -> arm-linux-gnueabihf
```

## Estrutura

```text
ZYRON/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── include/zyron/
│   ├── core/
│   │   ├── dtype.hpp
│   │   ├── memory.hpp
│   │   ├── runtime.hpp
│   │   ├── shape.hpp
│   │   └── tensor.hpp
│   ├── math/
│   │   ├── kernels.hpp
│   │   └── ops.hpp
│   ├── autograd/
│   ├── nn/
│   ├── tokenizer/
│   ├── transformer/
│   ├── model/
│   │   ├── checkpoint.hpp
│   │   ├── generation.hpp
│   │   ├── language_model.hpp
│   │   └── quantized_language_model.hpp
│   ├── training/
│   └── quantization/
│       ├── quantization.hpp
│       └── quantized_linear.hpp
├── src/
├── tests/
├── benchmarks/
├── examples/
├── tools/
├── data/
└── models/
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Para uma máquina ARM sem NEON, pode desabilitar o caminho NEON:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DZYRON_ENABLE_NEON=OFF
```

Quando o target é ARMv7 e `ZYRON_ENABLE_NEON=ON`, o CMake adiciona `-mfpu=neon`. Em runtime, o kernel verifica a capacidade NEON antes de executar a rota SIMD.

## Fase 1 — Core + Tensor Engine

Inclui `Shape`, `DType`, `Memory`, `Tensor<float>` e operações matemáticas básicas. O Tensor suporta strides, views, reshape, transpose, contiguous, broadcasting, reductions e MatMul.

A memória do Tensor é gerenciada por RAII e, na Fase 5, passou a usar alinhamento de 64 bytes para facilitar kernels SIMD. Também existem estatísticas de bytes vivos, pico, alocações vivas e alocações totais.

## Fase 2 — Autograd + Neural Network Engine

Inclui reverse-mode autodiff, acumulação de gradientes, gradient checking e layers como `Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, ativações e losses.

## Fase 3 — Tokenizer + Transformer + Language Model

Inclui:

- byte-level BPE;
- encode/decode e save/load `.zytok`;
- embedding;
- positional encoding sinusoidal legado + RoPE configurável;
- multi-head self-attention com GQA/MHA configurável e KV cache na geração;
- causal mask;
- residual connections;
- RMSNorm/LayerNorm;
- feed-forward + GELU;
- Transformer configurável;
- Language Model autoregressivo com weight tying opcional;
- generation com temperature, top-k, top-p, repetition penalty e no-repeat n-gram blocking.

## Fase 4 — Training + Evaluation

Inclui:

- leitura de `.txt` em chunks;
- `StreamingTextDataset`;
- DataLoader com batching, shuffle buffer, workers e prefetch;
- AdamW;
- warmup + cosine decay;
- gradient clipping;
- gradient accumulation configurável;
- dropout configurável durante o treino;
- validation loss e perplexity;
- logs e métricas;
- checkpoint `.zyron` com pesos, configuração, optimizer, scheduler, step e tokenizer.

`zyron_train --threads N` controla o paralelismo dos kernels matemáticos. `--workers N` controla os produtores do DataLoader; são recursos independentes.

Opções relevantes de treino: `--dropout F` e `--gradient-accumulation N`. Na geração, `--no-repeat-ngram-size N` bloqueia n-gramas repetidos. Checkpoints atuais usam formato v3 e continuam aceitando formatos v1/v2.

## Fase 5 — Performance + Mobile + Quantization

### Runtime e threading

`zyron::runtime` fornece:

```cpp
zyron::runtime::set_num_threads(N);
zyron::runtime::num_threads();
zyron::runtime::recommended_threads();
zyron::runtime::cpu_features();
zyron::runtime::process_rss_bytes();
```

O padrão é uma thread. Isso evita oversubscription em celulares. A recomendação automática é conservadora e limitada a quatro threads.

`parallel_for()` é usado pelos kernels de compute para dividir trabalho independente em faixas sem criar dependências entre threads.

### MatMul otimizado

A rota rápida de `zyron::math::matmul` para tensors contíguos implementa:

- fast path 2D;
- fast path batched;
- tiling por colunas;
- SIMD x86 SSE2;
- SIMD ARMv7 NEON quando disponível;
- paralelismo por linhas;
- fallback escalar;
- suporte preservado para MatMul arbitrariamente strided por uma rota de referência.

O benchmark dedicado mostra o backend usado e permite medir o efeito do número de threads.

### Quantização de inferência

`zyron::quantization` implementa quantização simétrica de pesos por coluna:

```text
FP32
  ↓
INT8 / INT4
  ↓
scale por coluna
  ↓
inference matmul
```

Características:

- INT8;
- INT4 assinado, dois pesos por byte;
- armazenamento column-major para dot products contíguos;
- quantização dinâmica das ativações por linha;
- `QuantizedMatrix::dequantize()`;
- `quantization::matmul()`;
- `QuantizedLinear` sem autograd;
- save/load `.zyq`;
- bytes e fator de compressão.

No caminho ARMv7 com NEON, o dot product INT8 utiliza intrinsics NEON quando a CPU anuncia NEON via `getauxval(AT_HWCAP)`.

### Language Model quantizado

`zyron::model::QuantizedLanguageModel` cria um caminho de inferência separado a partir de um `LanguageModel` treinado. Os pesos das projeções e do embedding são quantizados para INT8 ou INT4; normalizações e bias permanecem em FP32.

A classe oferece:

```cpp
forward(ids);
generate(ids, config);
generate_text(tokenizer, prompt, config);
```

O caminho é deliberadamente separado do autograd e do treinamento.

## Benchmarks

### Benchmark histórico das fases anteriores

```bash
./build/zyron_benchmark
./build/zyron_autograd_benchmark
./build/zyron_nn_benchmark
./build/zyron_transformer_benchmark
./build/zyron_training_benchmark
```

### Benchmark da Fase 5

```bash
./build/zyron_phase5_benchmark
./build/zyron_phase5_benchmark 2
./build/zyron_phase5_benchmark 4
```

O benchmark registra:

- largura de ponteiro;
- threads configuradas;
- threads recomendadas;
- backend SIMD;
- presença de NEON;
- MatMul FP32;
- armazenamento e compressão INT8/INT4;
- MatMul quantizado;
- inferência FP32/INT8/INT4 de um LM pequeno;
- estatísticas de memória e RSS.

## Testes

```bash
ctest --test-dir build --output-on-failure
```

A suíte atual cobre as cinco fases, incluindo:

- Tensor e Shape;
- operações elementwise e broadcasting;
- views/strides;
- MatMul 2D e batched;
- autograd e gradient checking;
- Neural Network Engine;
- BPE;
- Transformer e causal mask;
- dataset/training/checkpoint;
- alinhamento e estatísticas de memória;
- runtime/threading;
- kernels otimizados;
- INT8/INT4 quantization;
- serialização `.zyq`;
- `QuantizedLinear`;
- `QuantizedLanguageModel` e geração determinística.

## Sanitizers

No ambiente de desenvolvimento, a suíte completa também foi executada com ASan + UBSan:

```bash
cmake -S . -B build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer'
cmake --build build-sanitize -j2
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=print_stacktrace=1 \
ctest --test-dir build-sanitize --output-on-failure
```

## Treinamento depois da Fase 5

O projeto continua pronto para o fluxo:

```text
Dataset TXT
    ↓
Tokenizer BPE
    ↓
Streaming DataLoader
    ↓
Transformer
    ↓
Cross Entropy
    ↓
Backward
    ↓
Gradient Clipping
    ↓
AdamW
    ↓
Warmup + Cosine
    ↓
Checkpoint
```

A Fase 5 não altera o formato do treinamento em FP32; as otimizações e quantizações adicionadas aqui são voltadas principalmente para execução/inferência e para acelerar kernels sem mudar a API de autograd.

## Parte 3 — inferência e quantização

A versão atual inclui:

- checkpoint de inferência quantizado `.zyqmodel` para INT8/INT4 via `QuantizedLanguageModel::save/load`;
- ferramenta `zyron_quantize` para converter checkpoints FP32 em modelos quantizados;
- geração em lote com KV cache compartilhado para prompts de mesmo comprimento;
- QAT com fake-quantization e straight-through estimator (STE), configurável com `Config::use_qat`/`qat_bits` e pela CLI `--qat --qat-bits 4|8`;
- caminho de atenção fused no `forward_cached`, evitando materializar a matriz completa de scores/probabilidades durante inferência incremental.

O checkpoint de treino continua sendo FP32 para permitir retomada de treinamento. O `.zyqmodel` é um artefato separado e somente de inferência.
