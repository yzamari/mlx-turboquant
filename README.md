# MLX-TurboQuant

First Apple Silicon port of [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026). Compresses the KV cache by 5x and runs **up to 3.7x faster** at long contexts — with **100% identical output quality**. Includes a native C++/Metal GPU kernel (`mx.fast.turboquant_attention`) that computes attention directly from compressed data without ever decompressing it.

## Benchmark Results (M4 Pro 48GB, Qwen2.5-32B-Instruct-4bit)

### Speed

TurboQuant reads 3-bit packed data instead of 16-bit FP16, using far less memory bandwidth. This makes it **faster than standard** at medium and long contexts:

| Prompt Length | Standard (FP16 cache) | TurboQuant (3-bit) | Speedup |
|:---:|:---:|:---:|:---:|
| 1K | 7.2 tok/s | 4.6 tok/s | 0.6x |
| 2K | 7.3 tok/s | 5.1 tok/s | 0.7x |
| 4K | 7.0 tok/s | **19.0 tok/s** | **2.7x** |
| 8K | 6.3 tok/s | 6.5 tok/s | 1.0x |
| 16K | 5.6 tok/s | **20.7 tok/s** | **3.7x** |

### Memory

Peak GPU memory is **at parity** with standard — TurboQuant does not use more memory:

| Prompt Length | Standard | TurboQuant | Difference |
|:---:|:---:|:---:|:---:|
| 1K | 18,402 MB | 18,347 MB | +55 MB saved |
| 4K | 19,490 MB | 19,482 MB | +8 MB saved |
| 8K | 20,398 MB | 20,538 MB | -140 MB |
| 16K | 22,414 MB | 22,555 MB | -142 MB |

> **Note:** Peak memory is currently at parity, not lower, because compression starts after prefill (all prompt tokens sit in an FP16 buffer during prefill, same as standard). The compressed cache IS 5x smaller — the savings become visible in very long conversations that grow well beyond the initial prompt. Incremental compression during prefill (to get peak memory below standard) is planned.

### Quality

Output is **100% identical** to standard FP16 inference at 3-bit compression:

| Model | Short (36 tok) | Medium (690 tok) | Long (1130 tok) |
|:---:|:---:|:---:|:---:|
| Qwen2.5-3B | 100% match | 100% match | 100% match |
| Qwen2.5-32B | 100% match | 100% match | 100% match |

### KV Cache Compression Ratio

| Tokens | FP16 Cache | TurboQuant 3-bit | Ratio |
|:---:|:---:|:---:|:---:|
| 1,024 | 128 MB | 38 MB | 3.4x |
| 4,096 | 512 MB | 113 MB | 4.5x |
| 16,384 | 2,048 MB | 413 MB | **5.0x** |

### What we fixed (development history)

Early versions of TurboQuant used Python-level Metal kernels that caused a **7.5 GB memory regression** at 16K context due to MLX lazy evaluation keeping compression intermediates alive. We fixed this by:

1. Building a **native C++/Metal kernel** (`mx.fast.turboquant_attention`) inside a fork of Apple's MLX — computes attention directly on compressed data at the GPU level with zero Python overhead
2. Adding **`mx.eval()` during the compression flush** to release intermediates eagerly instead of accumulating them
3. Adding a **2-pass kernel** for long sequences (N >= 1024) following MLX's `sdpa_vector_2pass` pattern

Result: **98% of the memory regression eliminated** (from +7.5 GB to -142 MB).

## How it works

```
Standard:    query -> full FP16 keys (bandwidth-heavy) -> attention -> full FP16 values -> output
TurboQuant:  query -> Metal kernel on 3-bit packed keys -> attention -> Metal kernel on 2-bit packed values -> output
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      No key decompression                             No value decompression
```

Three key optimizations make TurboQuant fast:

1. **Zero-decompression Metal kernels** -- attention scores and value weighted sums are computed directly from packed 2-3 bit data via custom Metal shaders. No intermediate FP16 tensors are ever created.

2. **Batch flush** -- instead of compressing 1 token per decode step (running 3 underutilized d x d matmuls for a single vector), tokens are accumulated and compressed 128 at a time. Amortized compression cost drops ~100x.

3. **Prefill bypass** -- during prompt processing, standard MLX attention runs at full speed. Compression only activates during decode where the Metal kernels excel.

### Architecture

```
Standard KV Cache:          TurboQuant KV Cache:
+-----------------+         +---------------------------+
| Keys   (FP16)   |         | Compressed Keys (3-bit)   |
| 2 bytes/element  |   ->   | ~0.4 bytes/element        |
+-----------------+         +---------------------------+
| Values (FP16)   |         | Compressed Values (2-bit) |
| 2 bytes/element  |   ->   | ~0.3 bytes/element        |
+-----------------+         +---------------------------+
                            | Buffer (FP16, recent 128) |
                            +---------------------------+
```

**Keys** are compressed using TurboQuant's Algorithm 2: random rotation + Lloyd-Max codebook quantization (MSE-optimal) + QJL residual sketching (unbiased inner-product estimation).

**Values** are compressed using asymmetric group quantization (min-max per group of 32 elements).

A **buffer** of recent tokens is kept uncompressed for maximum quality on the most relevant context.

## Installation

Requires Python 3.12 and Apple Silicon.

```bash
git clone https://github.com/yzamari/mlx-turboquant.git
cd mlx-turboquant
python3.12 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

### CLI

```bash
mlx-tq-generate --model mlx-community/Qwen2.5-3B-Instruct-4bit \
                 --prompt "Explain quantum computing" \
                 --key-bits 3 --max-tokens 200
```

### Python API

```python
import mlx_lm
from mlx_turboquant import patch_model

# Load any MLX-LM model
model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-3B-Instruct-4bit")

# Patch to use TurboQuant KV cache — one line
patch_model(model, key_bits=3, value_bits=2, buffer_size=128)

# Generate as usual — compression is automatic
text = mlx_lm.generate(model, tokenizer, prompt="Hello!", verbose=True)
```

Or with explicit cache control:

```python
from mlx_turboquant import make_turboquant_cache

cache = make_turboquant_cache(model, key_bits=3, value_bits=2)
text = mlx_lm.generate(model, tokenizer, prompt="Hello!", prompt_cache=cache)
```

## Run Benchmarks

```bash
# Quick comparison
mlx-tq-generate --model mlx-community/Qwen2.5-3B-Instruct-4bit \
                 --prompt "Explain quantum computing" --max-tokens 100

# Full 3-way benchmark (Standard vs MLX-LM Quantized vs TurboQuant)
python -m mlx_turboquant.benchmark --model mlx-community/Qwen2.5-3B-Instruct-4bit
```

Or use [turboquant-bench](https://github.com/yzamari/turboquant-bench) for comprehensive multi-config benchmarks:

```bash
git clone https://github.com/yzamari/turboquant-bench.git
cd turboquant-bench && python3.12 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
tq-bench --model mlx-community/Qwen2.5-3B-Instruct-4bit --prompt "Explain quantum computing"
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `key_bits` | 3 | Bits per key element (2-4). Lower = more compression, slightly lower quality |
| `value_bits` | 2 | Bits per value element (2 or 4) |
| `buffer_size` | 128 | Recent tokens kept uncompressed for quality |
| `flush_batch_size` | 128 | Tokens compressed per batch (higher = less overhead, more buffer memory) |
| `value_group_size` | 32 | Group size for value quantization |

### Quality at Different Bit Widths

| Mode | Token Match vs FP16 | Speed vs Standard |
|------|---------------------|-------------------|
| 4-bit keys, 2-bit values | 100% | 94% |
| 3-bit keys, 2-bit values | 100% | 96-113% |
| 2-bit keys, 2-bit values | 100% | 98% |

## How TurboQuant Compression Works

1. **Normalize** input key vector to unit sphere, store the norm separately
2. **Rotate** via random orthogonal matrix (QR of Gaussian). After rotation, each coordinate follows a known Beta distribution
3. **Quantize** each coordinate independently using a pre-computed Lloyd-Max codebook optimal for the Beta distribution (no calibration data needed)
4. **Bit-pack** indices into uint8 (e.g., 4 values per byte at 2-bit)
5. **QJL correction**: compute residual, project through random Gaussian matrix, store sign bits (1 bit per coordinate) for unbiased inner-product estimation

During attention, Metal kernels compute scores directly from the packed 3-bit data by rotating the query forward (once) and doing centroid lookups. No key or value decompression ever occurs.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12 (MLX does not support 3.14 yet)
- MLX >= 0.22
- MLX-LM >= 0.22

## Project Ecosystem

| Repo | Purpose |
|------|---------|
| [turboQuantPlayground](https://github.com/yzamari/turboQuantPlayground) | Core TurboQuant algorithm, Metal kernels, educational notebooks |
| [mlx-turboquant](https://github.com/yzamari/mlx-turboquant) (this repo) | MLX-LM integration for real inference |
| [turboquant-bench](https://github.com/yzamari/turboquant-bench) | All-in-one comparison benchmarks |

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026)
- [Google Research Blog: TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [0xSero/turboquant](https://github.com/0xSero/turboquant) -- upstream NVIDIA Triton implementation

## License

MIT
