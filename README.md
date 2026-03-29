# MLX-TurboQuant

TurboQuant KV cache compression for MLX-LM inference on Apple Silicon. **Faster than standard inference on medium and long contexts, with 100% output quality.**

## Benchmark Results (M4 Pro 48GB)

### Peak Memory — 32B Model Long Context

The main challenge of KV cache compression is reducing **peak GPU memory** at long contexts. This table compares three configurations:

- **Standard** — vanilla MLX-LM with FP16 KV cache (no compression)
- **TQ v1 (Python Metal)** — TurboQuant with Python-level Metal kernels (before native kernel)
- **TQ v2 (Native C++ Metal)** — TurboQuant with native `mx.fast.turboquant_attention()` kernel + memory-optimized flush

| Prompt Length | Standard | TQ v1 (Python Metal) | TQ v2 (Native C++ Metal) |
|:---:|:---:|:---:|:---:|
| 1K | 18,402 MB | 18,368 MB (-34) | **18,347 MB (+55)** |
| 2K | 18,930 MB | 19,138 MB (-209) | **18,938 MB (-8)** |
| 4K | 19,490 MB | 20,662 MB (-1,173) | **19,482 MB (+8)** |
| 8K | 20,398 MB | **23,752 MB (-3,355)** | **20,538 MB (-140)** |
| 16K | 22,414 MB | **29,891 MB (-7,477)** | **22,555 MB (-142)** |

TQ v1 used up to **7.5 GB more** memory than standard at 16K because Python-level Metal kernel intermediates and lazy evaluation during compression accumulated without being freed. TQ v2 fixes this with a native C++/Metal kernel (`mx.fast.turboquant_attention`) built into a fork of Apple's MLX framework, plus `mx.eval()` calls during the compression flush to release intermediates eagerly. Result: **98% of the memory overhead eliminated**.

### Speed — 32B Model Long Context

| Prompt Length | Standard | TQ v1 (Python Metal) | TQ v2 (Native C++ Metal) |
|:---:|:---:|:---:|:---:|
| 1K | 7.2 tok/s | 8.3 tok/s (115%) | 4.6 tok/s (64%) |
| 2K | 7.3 tok/s | 8.6 tok/s (118%) | 5.1 tok/s (70%) |
| 4K | 7.0 tok/s | 19.0 tok/s (271%) | **19.0 tok/s (271%)** |
| 8K | 6.3 tok/s | 6.2 tok/s (98%) | 6.5 tok/s (103%) |
| 16K | 5.6 tok/s | 26.3 tok/s (469%) | **20.7 tok/s (369%)** |

TurboQuant is **up to 3.7x faster** at long contexts because reading 3-bit packed data requires far less memory bandwidth than 16-bit FP16 keys.

### Speed & Quality — 3B Model

| Context | Standard | TQ v2 (Native) | Speed | Quality |
|---------|:---:|:---:|:---:|:---:|
| Short (36 tok) | 81.3 tok/s | 63.8 tok/s | 78% | **100% match** |
| Medium (690 tok) | 64.8 tok/s | **73.5 tok/s** | **113%** | **100% match** |
| Long (1130 tok) | 63.1 tok/s | **70.5 tok/s** | **112%** | **100% match** |

Output quality is **100% identical** to standard FP16 at 3-bit compression across all prompt lengths tested.

### KV Cache Compression Ratio

| Compressed Tokens | FP16 Cache | TurboQuant 3-bit | Compression |
|:---:|:---:|:---:|:---:|
| 1,024 | 128 MB | 38 MB | 3.4x |
| 4,096 | 512 MB | 113 MB | 4.5x |
| 16,384 | 2,048 MB | 413 MB | **5.0x** |

Memory savings grow with context length. At 16K tokens, TurboQuant saves 1.6 GB of KV cache per model.

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
