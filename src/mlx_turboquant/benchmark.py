"""
Benchmark: compare TurboQuant vs standard vs quantized KV cache.

Compares three modes:
  1. Standard KVCache (FP16)
  2. MLX-LM QuantizedKVCache (4-bit or 8-bit)
  3. TurboQuantCache (3-bit keys + 2-bit values)

Measures: cache memory, tokens/sec, and output quality.

Usage:
    python -m mlx_turboquant.benchmark --model mlx-community/Qwen2.5-3B-Instruct-4bit
"""

import argparse
import time
import sys
from typing import Optional

import mlx.core as mx


def _generate_tokens(
    model,
    tokenizer,
    prompt_text: str,
    max_tokens: int,
    prompt_cache=None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
) -> dict:
    """Run generation and collect metrics."""
    import mlx_lm
    from mlx_lm.generate import generate_step

    # Tokenize
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt_text
    else:
        formatted = prompt_text

    tokens = mx.array(tokenizer.encode(formatted))

    # Generate
    gen_kwargs = dict(max_tokens=max_tokens)
    if prompt_cache is not None:
        gen_kwargs["prompt_cache"] = prompt_cache
    if kv_bits is not None:
        gen_kwargs["kv_bits"] = kv_bits
        gen_kwargs["kv_group_size"] = kv_group_size

    mx.reset_peak_memory()
    start = time.perf_counter()
    output_tokens = []
    text = ""

    for response in mlx_lm.stream_generate(
        model, tokenizer, formatted, **gen_kwargs
    ):
        text += response.text

    elapsed = time.perf_counter() - start
    peak_mem = mx.get_peak_memory() / 1e6  # MB

    # Get cache size
    cache_bytes = 0
    if prompt_cache is not None:
        cache_bytes = sum(c.nbytes for c in prompt_cache)

    return {
        "text": text,
        "tokens": response.generation_tokens,
        "elapsed": elapsed,
        "prompt_tps": response.prompt_tps,
        "gen_tps": response.generation_tps,
        "peak_memory_mb": peak_mem,
        "cache_bytes": cache_bytes,
    }


def run_benchmark(
    model_path: str,
    prompt: str = "Explain the theory of relativity in detail. Start from the basics and build up to the key equations.",
    max_tokens: int = 100,
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
):
    """Run benchmark comparing three cache modes."""
    import mlx_lm
    from mlx_lm.models.cache import KVCache, QuantizedKVCache
    from mlx_turboquant.patch import make_turboquant_cache

    print(f"Loading model: {model_path}")
    model, tokenizer = mlx_lm.load(model_path)
    print(f"Max tokens: {max_tokens}")
    print(f"Prompt: {prompt[:80]}...")
    print()

    results = {}

    # 1. Standard KVCache
    print("=" * 60)
    print("Mode 1: Standard KVCache (FP16)")
    print("=" * 60)
    r1 = _generate_tokens(model, tokenizer, prompt, max_tokens)
    results["standard"] = r1
    print(f"  Output: {r1['text'][:100]}...")
    print(f"  Tokens: {r1['tokens']}, Gen: {r1['gen_tps']:.1f} tok/s")
    print(f"  Peak memory: {r1['peak_memory_mb']:.0f} MB")
    print()

    # 2. MLX-LM QuantizedKVCache (4-bit)
    print("=" * 60)
    print("Mode 2: QuantizedKVCache (4-bit)")
    print("=" * 60)
    r2 = _generate_tokens(
        model, tokenizer, prompt, max_tokens, kv_bits=4, kv_group_size=64
    )
    results["quantized_4bit"] = r2
    print(f"  Output: {r2['text'][:100]}...")
    print(f"  Tokens: {r2['tokens']}, Gen: {r2['gen_tps']:.1f} tok/s")
    print(f"  Peak memory: {r2['peak_memory_mb']:.0f} MB")
    print()

    # 3. TurboQuantCache
    print("=" * 60)
    print(f"Mode 3: TurboQuantCache (key={key_bits}b, value={value_bits}b)")
    print("=" * 60)
    tq_cache = make_turboquant_cache(
        model,
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
    )
    r3 = _generate_tokens(
        model, tokenizer, prompt, max_tokens, prompt_cache=tq_cache
    )
    results["turboquant"] = r3
    print(f"  Output: {r3['text'][:100]}...")
    print(f"  Tokens: {r3['tokens']}, Gen: {r3['gen_tps']:.1f} tok/s")
    print(f"  Peak memory: {r3['peak_memory_mb']:.0f} MB")
    print(f"  Cache memory: {r3['cache_bytes'] / 1e3:.1f} KB")
    print()

    # Summary
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<25} {'Gen tok/s':>10} {'Peak MB':>10}")
    print("-" * 47)
    for name, r in results.items():
        print(f"{name:<25} {r['gen_tps']:>10.1f} {r['peak_memory_mb']:>10.0f}")
    print()

    # Output comparison (first N chars)
    print("Output comparison (first 100 chars):")
    for name, r in results.items():
        print(f"  {name}: {r['text'][:100]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache modes")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="Model path",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens"
    )
    parser.add_argument("--key-bits", type=int, default=3, help="Key bits")
    parser.add_argument("--value-bits", type=int, default=2, help="Value bits")
    parser.add_argument(
        "--buffer-size", type=int, default=128, help="Buffer size"
    )

    args = parser.parse_args()
    prompt = args.prompt or (
        "Explain the theory of relativity in detail. "
        "Start from the basics and build up to the key equations."
    )

    run_benchmark(
        model_path=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    main()
