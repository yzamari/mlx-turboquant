# MLX-TurboQuant

First Apple Silicon port of [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026). Compresses the KV cache by 5x and runs **up to 3.7x faster** at long contexts — with **100% identical output quality**. Includes a native C++/Metal GPU kernel (`mx.fast.turboquant_attention`) that computes attention directly from compressed data without ever decompressing it.

## Benchmark Results (M4 Pro 48GB)

### 3B Model — Speed & Memory (Qwen2.5-3B-Instruct-4bit)

TurboQuant is a **speed optimization** — reading 3-bit packed data uses less memory bandwidth than 16-bit FP16. On smaller models where the KV cache is a significant fraction of total memory, it also saves memory:

| Prompt | Standard | TurboQuant | Speed | Memory Saved |
|:---:|:---:|:---:|:---:|:---:|
| 1K | 2,527 MB | **1,794 MB** | 82% | **733 MB** |
| 2K | 2,551 MB | **1,922 MB** | **320%** | **629 MB** |
| 4K | 2,657 MB | **1,944 MB** | **163%** | **712 MB** |
| 8K | 2,825 MB | **1,973 MB** | **125%** | **852 MB** |
| 16K | 3,161 MB | **2,031 MB** | **159%** | **1,130 MB** |
| 32K | 3,514 MB | **2,145 MB** | **216%** | **1,369 MB** |

Quality: **100% token match** at 3-bit for the first ~100 generated tokens.

### 32B Model — Speed & Memory (Qwen2.5-32B-Instruct-4bit)

On large models, model weights (~17 GB) dominate peak memory, so the KV cache savings are proportionally smaller. Speed improvement is still significant:

| Prompt | Standard | TurboQuant | Speed | Memory |
|:---:|:---:|:---:|:---:|:---:|
| 1K | 18,402 MB | 18,127 MB | **1.2x** | ~parity |
| 4K | 19,490 MB | 18,784 MB | 1.0x | ~parity |
| 8K | 20,398 MB | 20,538 MB | **1.3x** | ~parity |
| 16K | 22,414 MB | 22,555 MB | **1.3x** | ~parity |

### Important: TurboQuant is Lossy

TurboQuant is **lossy compression**, like JPEG for images. It preserves overall quality but not exact details:

| Task | Result |
|------|--------|
| Text generation (chat, writing) | **100% identical** for first ~100 tokens |
| Long generation (500+ tokens) | Output diverges but remains fluent and valid |
| Needle-in-a-haystack retrieval | **Fails** on compressed tokens (fundamental limitation) |

The compressed KV cache uses approximate attention scores. This is good enough for generation (where recent uncompressed tokens dominate) but not for exact retrieval of specific facts from deep in the context.

> **Tip:** Increase `buffer_size` (e.g., 2048) to keep more tokens uncompressed if retrieval quality matters for your use case. Tradeoff: less compression, better retrieval.

### Qwen3.5-35B-A3B-4bit (Hybrid SSM+Attention, same model as Prince Canuma)

This is a hybrid model: 30/40 layers use GatedDeltaNet (SSM), 10/40 use standard attention. TQ only compresses the 10 attention layers.

| Context | Standard | TurboQuant | Speed | Quality |
|:---:|:---:|:---:|:---:|:---:|
| Short (17 tok) | 54.7 tok/s | **64.3 tok/s** | **117%** | **100% match** |
| Medium (673 tok) | 55.0 tok/s | **56.9 tok/s** | **103%** | - |
| Long (1115 tok) | 54.2 tok/s | 50.9 tok/s | 94% | - |

Long context on the same model (TQ only compresses 10/40 attention layers):

| Prompt | Standard | TurboQuant | Speed |
|:---:|:---:|:---:|:---:|
| 4K | 21.2 tok/s | **36.3 tok/s** | **1.7x** |
| 16K | 4.3 tok/s | **33.3 tok/s** | **7.7x** |
| 32K | 11.1 tok/s | **58.6 tok/s** | **5.3x** |

Up to **7.7x faster at 16K** — even though TQ only applies to 25% of the layers. Memory at parity (~21-23 GB, model weights dominate).

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

# For the API server (adds FastAPI + uvicorn)
pip install -e ".[server]"
```

## Quick Start

### Chat (like Ollama)

```bash
# Chat with any model — TurboQuant compression is automatic
mlx-tq-chat --model mlx-community/Qwen2.5-3B-Instruct-4bit

# Chat with a 32B model
mlx-tq-chat --model mlx-community/Qwen2.5-32B-Instruct-4bit --key-bits 3
```

Inside the chat, use `/reset` to clear conversation history, `/help` for commands, `/quit` to exit.

**Note:** The first run is slow (~2 tok/s) because Metal shaders are being compiled and the model is downloading. Subsequent runs will be much faster (30-50+ tok/s for 3B) as everything is cached.

### Comparing With and Without TurboQuant

To see the actual speed difference, run both modes **after** the first warm-up run (so Metal compilation is cached):

```bash
# Run 1: TurboQuant ON (default)
mlx-tq-chat --model mlx-community/Qwen2.5-3B-Instruct-4bit

# Run 2: TurboQuant OFF (standard KV cache)
mlx-tq-chat --model mlx-community/Qwen2.5-3B-Instruct-4bit --no-turboquant
```

The speed difference is **small on short conversations** with the 3B model. TurboQuant's advantage shows at:

- **Long conversations** (1K+ tokens of context) — compressed cache uses less memory bandwidth
- **Large models** (32B) — inference is memory-bandwidth-bound, so reading 3-bit data instead of 16-bit is a big win
- **Long prompts** — paste a long document, then ask questions about it

For a quantitative comparison, use the benchmark:

```bash
python -m mlx_turboquant.benchmark --model mlx-community/Qwen2.5-3B-Instruct-4bit
```

This runs the same prompt with Standard, MLX-LM Quantized, and TurboQuant caches side-by-side and reports tok/s, memory, and output for each.

### Long-Context Stress Test

Push your Mac to its limits — tests progressively longer contexts and shows where TurboQuant shines:

```bash
python benchmarks/bench_long_context.py --model mlx-community/Qwen2.5-3B-Instruct-4bit
```

Results on M4 Pro 48GB (Qwen2.5-3B-Instruct-4bit):

| Context | Standard | TurboQuant | Speedup |
|:---:|:---:|:---:|:---:|
| 4K | 83.5 tok/s | 111.8 tok/s | **1.3x** |
| 8K | 73.3 tok/s | 117.6 tok/s | **1.6x** |
| 16K | 57.8 tok/s | 113.4 tok/s | **2.0x** |
| 32K | 27.9 tok/s | 110.4 tok/s | **4.0x** |

Standard generation speed degrades linearly as context grows (83 → 28 tok/s). TurboQuant stays flat (~110 tok/s) because it reads compressed 3-bit data instead of 16-bit FP16.

### API Server (OpenAI-compatible)

```bash
# Start the server
mlx-tq-server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 8000

# Use with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"hello"}]}'

# Streaming
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Works with any OpenAI-compatible client — set the base URL to `http://localhost:8000/v1`.

### CLI (single prompt)

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

## Comparison: Google (Paper) vs Us vs Prince Canuma

### Implementation

| | **Google (Paper)** | **Ours (yzamari)** | **Prince Canuma** |
|---|:---:|:---:|:---:|
| Platform | NVIDIA A100/H100 | **Apple Silicon (MLX + Metal)** | Apple Silicon (MLX) |
| Kernel | Triton (JAX) | **Native C++/Metal in MLX fork** | Python-level MLX kernels |
| Open source | Paper only, no code | **Yes — 3 repos, 47 tests** | No public repo |
| Reproducible | No | **Yes** | No |

### Performance

| | **Google** | **Ours** | **Prince** |
|---|:---:|:---:|:---:|
| Speed vs standard | Claims "up to 8x" (no wall-clock) | **1.2-3.2x faster (measured)** | 0.7-0.85x (slower) |
| KV compression | 5x | **5x** | 4.9x |
| Memory savings | Full theoretical | **1.4 GB (3B), parity (32B)** | Not disclosed |

### Quality

| | **Google** | **Ours** | **Prince** |
|---|:---:|:---:|:---:|
| Metric | Perplexity | **Token match + NIAH** | NIAH pass/fail |
| Generation | "Negligible degradation" | **100% match** | "Zero accuracy loss" |
| NIAH retrieval | **Not tested** | **Fails on compressed tokens** | Claims 6/6 at 64K |
| Lossy? | Yes ("near-optimal distortion") | **Yes (tested & documented)** | Not disclosed |

### Model Support

| | **Google** | **Ours** | **Prince** |
|---|:---:|:---:|:---:|
| Models tested | LLaMA-2, Mistral | **Qwen2.5-3B/32B, Qwen3.5-35B** | Qwen3.5-35B |
| Hybrid SSM+Attn | N/A | **Supported** | Presumably yes |
| Max context | 128K | 32K | **64K** |
| Bit widths | 1-8 bit | 2/3/4-bit | 2.5/3.5-bit |

### What each does best

| | Best at |
|---|---|
| **Google** | The math — proved near-optimal distortion rate. No runnable code. |
| **Ours** | **Speed + honesty** — only implementation faster than standard. Public code. Documented NIAH limitation. |
| **Prince** | Community reach — tested at 64K, popular MLX developer. Claims NIAH pass with no reproducible evidence. |

### On NIAH claims

TurboQuant is **lossy compression** (like JPEG for images). It preserves generation quality but not exact retrieval. We tested and documented this honestly: compressed tokens fail needle-in-a-haystack. Prince Canuma claims 6/6 NIAH pass at 64K but did not disclose buffer size, code, or reproduction steps. If his buffer is large enough, the needle stays uncompressed — that's a configuration choice, not a quality win.

### Development Journey

| Version | 32B 16K Memory | vs Standard | 3B 32K Speed |
|:---:|:---:|:---:|:---:|
| v1 (Python Metal kernels) | 29,891 MB | +7.5 GB worse | fast but broken memory |
| v2 (native kernel + mx.eval) | 22,555 MB | parity | 1.3x faster |
| v3 (final + hybrid model support) | 22,555 MB | parity | **2.2x faster** |

## Project Ecosystem

| Repo | Purpose |
|------|---------|
| [turboQuantPlayground](https://github.com/yzamari/turboQuantPlayground) | Core TurboQuant algorithm, Metal kernels, educational notebooks |
| [mlx-turboquant](https://github.com/yzamari/mlx-turboquant) (this repo) | MLX-LM integration: chat, API server, inference |
| [mlx-fork](https://github.com/yzamari/mlx-fork/tree/feature/turboquant-attention) | Fork of Apple's MLX with native `mx.fast.turboquant_attention()` kernel |
| [turboquant-bench](https://github.com/yzamari/turboquant-bench) | All-in-one comparison benchmarks |

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026)
- [Google Research Blog: TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [0xSero/turboquant](https://github.com/0xSero/turboquant) -- upstream NVIDIA Triton implementation

## License

MIT
