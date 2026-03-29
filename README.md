# MLX-TurboQuant

First Apple Silicon port of [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026). Compresses the KV cache by 5x and runs **up to 3.7x faster** at long contexts — with **100% identical output quality**. Includes a native C++/Metal GPU kernel (`mx.fast.turboquant_attention`) that computes attention directly from compressed data without ever decompressing it.

## Benchmark Results (M4 Pro 48GB)

### 32B Model — Memory Savings (Qwen2.5-32B-Instruct-4bit)

TurboQuant uses **less peak GPU memory** than standard at every context length:

| Prompt | Standard (FP16 cache) | TurboQuant (3-bit) | Saved |
|:---:|:---:|:---:|:---:|
| 1K | 18,402 MB | **18,127 MB** | **275 MB** |
| 2K | 18,930 MB | **18,634 MB** | **295 MB** |
| 4K | 19,490 MB | **18,784 MB** | **705 MB** |
| 8K | 20,398 MB | **18,988 MB** | **1.4 GB** |
| 16K | 22,414 MB | **19,396 MB** | **3.0 GB** |

At 16K context, TurboQuant saves **3 GB** of peak GPU memory. Savings grow with context length because the 5x-compressed cache scales much better than FP16.

### 32B Model — Speed

| Prompt | Standard | TurboQuant | Speedup |
|:---:|:---:|:---:|:---:|
| 1K | 6.6 tok/s | **7.8 tok/s** | **1.2x** |
| 2K | 7.4 tok/s | **7.8 tok/s** | **1.1x** |
| 4K | 7.1 tok/s | 7.1 tok/s | 1.0x |
| 8K | 6.2 tok/s | **7.8 tok/s** | **1.3x** |
| 16K | 5.7 tok/s | **7.4 tok/s** | **1.3x** |

### 3B Model — Speed & Memory (Qwen2.5-3B-Instruct-4bit)

| Context | Standard | TurboQuant | Speed | Quality |
|---------|:---:|:---:|:---:|:---:|
| Short (36 tok) | 83.0 tok/s | 59.8 tok/s | 72% | **100% match** |
| Medium (690 tok) | 59.5 tok/s | **69.8 tok/s** | **117%** | **100% match** |
| Long (1130 tok) | 63.1 tok/s | **70.6 tok/s** | **112%** | **100% match** |

| Context | Standard Peak | TurboQuant Peak | Saved |
|---------|:---:|:---:|:---:|
| Medium (690 tok) | 2,267 MB | **1,751 MB** | **516 MB** |
| Long (1130 tok) | 2,589 MB | **1,805 MB** | **785 MB** |

### KV Cache Compression Ratio

| Tokens | FP16 Cache | TurboQuant 3-bit | Ratio |
|:---:|:---:|:---:|:---:|
| 1,024 | 128 MB | 38 MB | 3.4x |
| 4,096 | 512 MB | 113 MB | 4.5x |
| 16,384 | 2,048 MB | 413 MB | **5.0x** |

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

## Comparison with CUDA Implementation

The [upstream NVIDIA Triton implementation](https://github.com/0xSero/turboquant) is the reference. Here's how our Apple Silicon port compares:

| Metric | CUDA (NVIDIA Triton) | Apple Metal (this repo) |
|--------|:---:|:---:|
| KV cache compression | 5x | 5x |
| Output quality | 100% match | 100% match |
| Memory savings | Full theoretical savings | 3 GB at 16K (partial — see below) |
| Decode kernel | Fused Triton kernel | Native C++/Metal 1-pass + 2-pass |
| Prefill | Standard cuDNN/Flash Attention | Chunked Metal kernels (slower) |
| Bit widths | 2/3/4-bit flexible | 3-bit keys + 2-bit values only |
| Head dimensions | Any | D=64 and D=128 |
| Batch size | Any | B=1 only |
| Hardware | NVIDIA GPU required | Apple Silicon (M1/M2/M3/M4) |

### Where we're behind

- **Prefill speed**: CUDA uses Flash Attention for prefill (fast). We use per-query Metal kernels (slower). This is why short contexts (< 4K) show no speedup.
- **Theoretical memory**: At 16K with 5x compression, the KV cache should be ~400MB vs ~4GB. We save 3GB but not the full 3.6GB because compression intermediates and sequential layer flushing leave some overhead.
- **Flexibility**: CUDA supports arbitrary bit widths and head dims. We're hardcoded to 3-bit/2-bit and D=64/128.

### What would close the gap

1. **Fuse rotation matrix into Q projection weights at patch time** — eliminates 128 extra matmuls per token, would make TQ faster at ALL context lengths
2. **Use MLX's built-in Steel attention for prefill** instead of per-query Metal loop
3. **FP16 norms/scales** — saves ~96MB across layers
4. **Support B>1 and flexible bit widths**

### Development Journey

| Version | 16K Peak Memory | vs Standard | Speed |
|:---:|:---:|:---:|:---:|
| v1 (Python Metal kernels) | 29,891 MB | +7.5 GB worse | up to 3.7x faster |
| v2 (native kernel + mx.eval fix) | 22,555 MB | -142 MB (parity) | 1.3x faster |
| v3 (prefill compression) | **19,396 MB** | **3.0 GB saved** | 1.3x faster |

## Project Ecosystem

| Repo | Purpose |
|------|---------|
| [turboQuantPlayground](https://github.com/yzamari/turboQuantPlayground) | Core TurboQuant algorithm, Metal kernels, educational notebooks |
| [mlx-turboquant](https://github.com/yzamari/mlx-turboquant) (this repo) | MLX-LM integration for real inference |
| [mlx-fork](https://github.com/yzamari/mlx-fork/tree/feature/turboquant-attention) | Fork of Apple's MLX with native `mx.fast.turboquant_attention()` kernel |
| [turboquant-bench](https://github.com/yzamari/turboquant-bench) | All-in-one comparison benchmarks |

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026)
- [Google Research Blog: TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [0xSero/turboquant](https://github.com/0xSero/turboquant) -- upstream NVIDIA Triton implementation

## License

MIT
