#!/usr/bin/env python3
"""
Long-run memory profiler: generates enough tokens to trigger TurboQuant compression.

With buffer_size=128, compression kicks in at token 129+prompt_tokens.
We generate 200 tokens to see compressed + buffer behavior.
"""

import time
import mlx.core as mx


def mb(bytes_val):
    return bytes_val / (1024 * 1024)


def profile_standard(model, tokenizer, formatted, max_tokens):
    """Profile standard KV cache."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    print("\n" + "=" * 80)
    print(f"STANDARD KV Cache — {max_tokens} tokens")
    print("=" * 80)

    mx.clear_cache()
    mx.reset_peak_memory()

    sampler = make_sampler(temp=0.0)
    step = 0
    mem_log = []

    for response in mlx_lm.stream_generate(
        model, tokenizer, formatted,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        mx.synchronize()
        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
        cache_mem = mx.get_cache_memory()

        mem_log.append({
            "step": step,
            "active_mb": mb(active),
            "peak_mb": mb(peak),
        })

        if step < 3 or step % 20 == 0 or step == max_tokens - 1:
            print(f"  Step {step:4d}: active={mb(active):8.1f} MB  peak={mb(peak):8.1f} MB")

        step += 1

    return mem_log


def profile_turboquant(model, tokenizer, formatted, max_tokens):
    """Profile TurboQuant KV cache with detailed per-step breakdown."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    from mlx_turboquant.patch import make_turboquant_cache

    print("\n" + "=" * 80)
    print(f"TURBOQUANT KV Cache — {max_tokens} tokens")
    print("=" * 80)

    mx.clear_cache()
    mx.reset_peak_memory()

    tq_cache = make_turboquant_cache(
        model, key_bits=3, value_bits=2, buffer_size=128,
    )

    # Measure quantizer matrix cost
    mx.synchronize()
    quantizer_mem = mx.get_active_memory()
    print(f"  Quantizer matrices: {mb(quantizer_mem):8.1f} MB (36 layers x Pi + S)")

    mx.reset_peak_memory()

    sampler = make_sampler(temp=0.0)
    step = 0
    mem_log = []
    flush_steps = []

    for response in mlx_lm.stream_generate(
        model, tokenizer, formatted,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=tq_cache,
    ):
        mx.synchronize()
        active = mx.get_active_memory()
        peak = mx.get_peak_memory()

        l0 = tq_cache[0]
        compressed = l0.compressed_tokens
        buffered = l0.buffer_tokens

        mem_log.append({
            "step": step,
            "active_mb": mb(active),
            "peak_mb": mb(peak),
            "compressed": compressed,
            "buffered": buffered,
        })

        if step < 3 or step % 20 == 0 or step == max_tokens - 1 or compressed > 0 and mem_log[-2]["compressed"] == 0 if len(mem_log) > 1 else False:
            print(f"  Step {step:4d}: active={mb(active):8.1f} MB  peak={mb(peak):8.1f} MB  "
                  f"[comp={compressed} buf={buffered}]")

        step += 1

    # Final detailed breakdown
    print(f"\n  === Final Cache State ===")
    total_cache = sum(c.nbytes for c in tq_cache)
    print(f"  Total cache nbytes: {mb(total_cache):8.3f} MB")
    l0 = tq_cache[0]
    report = l0.memory_report()
    print(f"  Layer 0:")
    print(f"    Compressed keys:   {mb(report['compressed_keys_bytes']):8.3f} MB  ({report['compressed_tokens']} tokens)")
    print(f"    Compressed values: {mb(report['compressed_values_bytes']):8.3f} MB")
    print(f"    Buffer:            {mb(report['buffer_bytes']):8.3f} MB  ({report['buffer_tokens']} tokens)")

    return mem_log


def profile_turboquant_step_detail(model, tokenizer, formatted, max_tokens):
    """
    Deeply instrument a TurboQuant run to measure memory at each sub-operation.

    This profiles what happens INSIDE a single decode step when compressed data exists.
    """
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    from mlx_turboquant.patch import make_turboquant_cache
    from mlx_turboquant.cache import TurboQuantCache

    print("\n" + "=" * 80)
    print("DETAILED: Memory cost breakdown per decode step operations")
    print("=" * 80)

    # We need to manually simulate what happens during a decode step
    # to measure each sub-operation's memory cost.

    # First, do a real generation to get the cache filled
    mx.clear_cache()

    tq_cache = make_turboquant_cache(
        model, key_bits=3, value_bits=2, buffer_size=128,
    )
    sampler = make_sampler(temp=0.0)

    # Generate enough to have compressed data
    for response in mlx_lm.stream_generate(
        model, tokenizer, formatted,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=tq_cache,
    ):
        pass

    mx.synchronize()

    # Now measure individual operations with the filled cache
    l0 = tq_cache[0]
    print(f"  Cache state: {l0.compressed_tokens} compressed + {l0.buffer_tokens} buffer")

    if l0.compressed_tokens == 0:
        print("  WARNING: No compressed data! Need more tokens or smaller buffer_size.")
        return

    # --- 1. GQA expansion of compressed data ---
    n_heads = 16
    n_kv_heads = 4
    repeats = n_heads // n_kv_heads
    kc = l0._keys_compressed

    mx.clear_cache()
    mx.reset_peak_memory()
    before = mx.get_active_memory()

    mse_expanded = mx.repeat(kc.mse_indices, repeats, axis=1)
    qjl_expanded = mx.repeat(kc.qjl_signs, repeats, axis=1)
    norms_expanded = mx.repeat(kc.norms, repeats, axis=-1)
    res_norms_expanded = mx.repeat(kc.residual_norms, repeats, axis=-1)
    mx.eval(mse_expanded, qjl_expanded, norms_expanded, res_norms_expanded)

    gqa_peak = mx.get_peak_memory()
    gqa_active = mx.get_active_memory()
    orig_bytes = kc.mse_indices.nbytes + kc.qjl_signs.nbytes + kc.norms.nbytes + kc.residual_norms.nbytes
    expanded_bytes = mse_expanded.nbytes + qjl_expanded.nbytes + norms_expanded.nbytes + res_norms_expanded.nbytes

    print(f"\n  1. GQA expand compressed keys ({l0.compressed_tokens} tokens, {repeats}x):")
    print(f"     Original:  {mb(orig_bytes):8.3f} MB")
    print(f"     Expanded:  {mb(expanded_bytes):8.3f} MB")
    print(f"     Peak:      {mb(gqa_peak):8.3f} MB")
    print(f"     Overhead:  {mb(gqa_peak) - mb(expanded_bytes):+8.3f} MB")

    # --- 2. Value decompression ---
    from turboquant_mac.kv_cache import dequantize_values

    mx.clear_cache()
    mx.reset_peak_memory()

    v_decompressed = dequantize_values(l0._values_compressed, group_size=32, backend="mlx")
    mx.eval(v_decompressed)

    val_decomp_peak = mx.get_peak_memory()
    compressed_val_bytes = l0._values_compressed.data.nbytes + l0._values_compressed.scales.nbytes + l0._values_compressed.zeros.nbytes

    print(f"\n  2. Value decompression ({l0.compressed_tokens} tokens):")
    print(f"     Compressed: {mb(compressed_val_bytes):8.3f} MB")
    print(f"     Decompressed: {mb(v_decompressed.nbytes):8.3f} MB")
    print(f"     Peak:         {mb(val_decomp_peak):8.3f} MB")

    # --- 3. GQA expand decompressed values ---
    mx.clear_cache()
    mx.reset_peak_memory()

    v_expanded = mx.repeat(v_decompressed, repeats, axis=1)
    mx.eval(v_expanded)

    val_gqa_peak = mx.get_peak_memory()
    print(f"\n  3. GQA expand decompressed values:")
    print(f"     Before expand: {mb(v_decompressed.nbytes):8.3f} MB")
    print(f"     After expand:  {mb(v_expanded.nbytes):8.3f} MB")
    print(f"     Peak:          {mb(val_gqa_peak):8.3f} MB")

    # --- 4. Buffer GQA expansion ---
    mx.clear_cache()
    mx.reset_peak_memory()

    buf_keys_expanded = mx.repeat(l0._key_buffer, repeats, axis=1)
    buf_vals_expanded = mx.repeat(l0._value_buffer, repeats, axis=1)
    mx.eval(buf_keys_expanded, buf_vals_expanded)

    buf_gqa_peak = mx.get_peak_memory()
    buf_orig = l0._key_buffer.nbytes + l0._value_buffer.nbytes
    buf_expanded = buf_keys_expanded.nbytes + buf_vals_expanded.nbytes

    print(f"\n  4. Buffer GQA expansion ({l0.buffer_tokens} tokens):")
    print(f"     Before: {mb(buf_orig):8.3f} MB")
    print(f"     After:  {mb(buf_expanded):8.3f} MB")
    print(f"     Peak:   {mb(buf_gqa_peak):8.3f} MB")

    # --- Total per-step overhead ---
    total_overhead = mb(gqa_peak) + mb(val_decomp_peak) + mb(val_gqa_peak) + mb(buf_gqa_peak)
    print(f"\n  === TOTAL per-decode-step transient memory ===")
    print(f"     Key GQA expand:     {mb(gqa_peak):8.3f} MB")
    print(f"     Value decompress:   {mb(val_decomp_peak):8.3f} MB")
    print(f"     Value GQA expand:   {mb(val_gqa_peak):8.3f} MB")
    print(f"     Buffer GQA expand:  {mb(buf_gqa_peak):8.3f} MB")
    print(f"     SUM (peak transient): {total_overhead:8.3f} MB")
    print(f"     NOTE: These don't all overlap, but show the magnitude of temporary allocations")


def main():
    import mlx_lm

    model_path = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    prompt = "Explain the theory of relativity in great detail. Start from the basics of special relativity, cover time dilation, length contraction, mass-energy equivalence, and then move to general relativity including curved spacetime and gravitational waves. Provide mathematical formulations where relevant."
    max_tokens = 200

    print("=" * 80)
    print(f"MLX Memory Profiler - Long Run")
    print(f"Model: {model_path}")
    print(f"Max tokens: {max_tokens}")
    print(f"Buffer size: 128 (compression kicks in at ~prompt_len + 128 tokens)")
    print("=" * 80)

    model, tokenizer = mlx_lm.load(model_path)
    mx.synchronize()

    # Format prompt
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = len(tokenizer.encode(formatted))
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Total tokens will be: {prompt_tokens + max_tokens}")
    print(f"Compression should start around step: {max(0, 128 - prompt_tokens)}")

    # Run standard
    std_log = profile_standard(model, tokenizer, formatted, max_tokens)

    # Run TurboQuant
    tq_log = profile_turboquant(model, tokenizer, formatted, max_tokens)

    # Detailed step breakdown (with small buffer to force early compression)
    print("\n\n" + "=" * 80)
    print("DETAILED PROFILING: Using buffer_size=32 to force early compression")
    print("=" * 80)

    from mlx_turboquant.patch import make_turboquant_cache
    from mlx_lm.sample_utils import make_sampler

    mx.clear_cache()
    small_cache = make_turboquant_cache(
        model, key_bits=3, value_bits=2, buffer_size=32,
    )
    sampler = make_sampler(temp=0.0)
    for response in mlx_lm.stream_generate(
        model, tokenizer, formatted,
        max_tokens=100,
        sampler=sampler,
        prompt_cache=small_cache,
    ):
        pass
    mx.synchronize()

    l0 = small_cache[0]
    print(f"Cache state: {l0.compressed_tokens} compressed + {l0.buffer_tokens} buffer")

    # Now do the detailed sub-operation profiling
    profile_turboquant_step_detail(model, tokenizer, formatted, max_tokens=100)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    std_peak_active = max(m["active_mb"] for m in std_log)
    tq_peak_active = max(m["active_mb"] for m in tq_log)
    std_peak = max(m["peak_mb"] for m in std_log)
    tq_peak = max(m["peak_mb"] for m in tq_log)

    print(f"  Standard peak active: {std_peak_active:8.1f} MB")
    print(f"  TurboQuant peak active: {tq_peak_active:8.1f} MB")
    print(f"  Active delta: {tq_peak_active - std_peak_active:+8.1f} MB")
    print()
    print(f"  Standard peak memory: {std_peak:8.1f} MB")
    print(f"  TurboQuant peak memory: {tq_peak:8.1f} MB")
    print(f"  Peak delta: {tq_peak - std_peak:+8.1f} MB")

    print("\n  Step-by-step active memory:")
    print(f"  {'Step':>5s}  {'Standard':>10s}  {'TurboQ':>10s}  {'Delta':>10s}  {'TQ Compressed':>14s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")
    for i in range(0, min(len(std_log), len(tq_log)), 20):
        s = std_log[i]
        t = tq_log[i]
        comp = t.get("compressed", "?")
        delta = t["active_mb"] - s["active_mb"]
        print(f"  {i:5d}  {s['active_mb']:10.1f}  {t['active_mb']:10.1f}  {delta:+10.1f}  {comp:>14}")
    # Last step
    i = min(len(std_log), len(tq_log)) - 1
    s = std_log[i]
    t = tq_log[i]
    comp = t.get("compressed", "?")
    delta = t["active_mb"] - s["active_mb"]
    print(f"  {i:5d}  {s['active_mb']:10.1f}  {t['active_mb']:10.1f}  {delta:+10.1f}  {comp:>14}")


if __name__ == "__main__":
    main()
