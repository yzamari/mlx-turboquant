# MLX-TurboQuant

TurboQuant KV cache compression for MLX-LM inference on Apple Silicon.

Drop-in replacement for MLX-LM's standard KVCache that compresses keys via TurboQuant product quantization and values via group quantization, achieving ~5x memory reduction in the KV cache with minimal quality loss.

## How it works

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

**Keys** are compressed using TurboQuant's Algorithm 2 (MSE quantization + QJL residual sketching), which provides unbiased inner-product estimation — critical for attention score accuracy.

**Values** are compressed using asymmetric group quantization (min-max per group).

A **buffer** of recent tokens is kept uncompressed for maximum quality on the most relevant context.

## Installation

```bash
pip install git+https://github.com/yzamari/mlx-turboquant.git
```

Or from source:

```bash
git clone https://github.com/yzamari/mlx-turboquant.git
cd mlx-turboquant
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

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

# Patch to use TurboQuant KV cache
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

## Benchmark

```bash
python -m mlx_turboquant.benchmark --model mlx-community/Qwen2.5-3B-Instruct-4bit
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `key_bits` | 3 | Bits per key element (2-4). Lower = more compression |
| `value_bits` | 2 | Bits per value element (2 or 4) |
| `buffer_size` | 128 | Recent tokens kept uncompressed |
| `value_group_size` | 32 | Group size for value quantization |

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python >= 3.11
- MLX >= 0.22
- MLX-LM >= 0.22

## Based on

- **TurboQuant** (Zandieh et al., ICLR 2026): Near-optimal KV cache compression
- **turboquant-mac**: Apple Silicon port ([repo](https://github.com/yzamari/turboQuantPlayground))
- **MLX-LM**: Apple's LLM inference framework for MLX
