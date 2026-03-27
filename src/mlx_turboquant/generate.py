"""
High-level generate function that wraps mlx_lm with TurboQuant KV compression.
"""

import time
from typing import Optional

import mlx.core as mx

from mlx_turboquant.patch import patch_model, make_turboquant_cache


def generate(
    model_path: str,
    prompt: str,
    max_tokens: int = 100,
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
    value_group_size: int = 32,
    temp: float = 0.0,
    verbose: bool = True,
    use_turboquant: bool = True,
) -> dict:
    """
    Load a model, optionally patch with TurboQuant, and generate text.

    Args:
        model_path: HuggingFace model path or local path.
        prompt: Input prompt string.
        max_tokens: Maximum tokens to generate.
        key_bits: Key compression bits (2-4).
        value_bits: Value compression bits (2 or 4).
        buffer_size: Uncompressed buffer tokens.
        value_group_size: Value quantization group size.
        temp: Sampling temperature (0.0 = greedy).
        verbose: Print generation output.
        use_turboquant: If True, use TurboQuant cache. If False, standard cache.

    Returns:
        Dict with text, tokens, timing, and memory info.
    """
    import mlx_lm

    if verbose:
        print(f"Loading model: {model_path}")

    model, tokenizer = mlx_lm.load(model_path)

    # Build cache
    prompt_cache = None
    if use_turboquant:
        if verbose:
            print(
                f"TurboQuant: key_bits={key_bits}, value_bits={value_bits}, "
                f"buffer={buffer_size}"
            )
        prompt_cache = make_turboquant_cache(
            model,
            key_bits=key_bits,
            value_bits=value_bits,
            value_group_size=value_group_size,
            buffer_size=buffer_size,
        )

    # Tokenize
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    tokens = mx.array(tokenizer.encode(formatted))

    if verbose:
        print(f"Prompt tokens: {tokens.size}")
        print("=" * 60)

    # Generate using mlx_lm's stream_generate
    from mlx_lm.sample_utils import make_sampler

    text = ""
    sampler = make_sampler(temp=temp)
    gen_kwargs = dict(
        max_tokens=max_tokens,
        sampler=sampler,
    )
    if prompt_cache is not None:
        gen_kwargs["prompt_cache"] = prompt_cache

    start = time.perf_counter()
    for response in mlx_lm.stream_generate(
        model, tokenizer, formatted, **gen_kwargs
    ):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text

    elapsed = time.perf_counter() - start

    if verbose:
        print()
        print("=" * 60)
        print(f"Generated {response.generation_tokens} tokens in {elapsed:.2f}s")
        print(f"  Prompt:     {response.prompt_tps:.1f} tok/s")
        print(f"  Generation: {response.generation_tps:.1f} tok/s")
        print(f"  Peak memory: {response.peak_memory:.2f} GB")

    # Collect cache memory info
    cache_bytes = 0
    if prompt_cache is not None:
        cache_bytes = sum(c.nbytes for c in prompt_cache)
        if verbose:
            print(f"  Cache memory: {cache_bytes / 1e6:.1f} MB")
            # Show per-layer sample
            sample = prompt_cache[0]
            report = sample.memory_report()
            print(
                f"  Layer 0: {report['compressed_tokens']} compressed + "
                f"{report['buffer_tokens']} buffer tokens"
            )

    return {
        "text": text,
        "generation_tokens": response.generation_tokens,
        "prompt_tokens": response.prompt_tokens,
        "prompt_tps": response.prompt_tps,
        "generation_tps": response.generation_tps,
        "elapsed_seconds": elapsed,
        "peak_memory_gb": response.peak_memory,
        "cache_bytes": cache_bytes,
    }
