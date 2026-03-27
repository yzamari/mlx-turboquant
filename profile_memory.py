#!/usr/bin/env python3
"""
Comprehensive MLX Memory Profiler for TurboQuant vs Standard KV Cache.

Profiles memory at each decode step to pinpoint exactly where TurboQuant's
memory overhead comes from.

Key questions answered:
1. Where does peak memory occur (prefill? decode? flush?)
2. How much memory does mx.repeat (GQA expansion) cost?
3. How much do the quantizer matrices (Pi, S, centroids) cost?
4. How much does value decompression cost per step?
5. How much do Metal kernel compilation/caching contribute?
6. Does lazy evaluation cause memory spikes when mx.eval() is called?
"""

import time
import sys
import mlx.core as mx


def mb(bytes_val):
    return bytes_val / (1024 * 1024)


def print_mem(label, reset_peak=False):
    """Print current memory state with label."""
    active = mx.get_active_memory()
    peak = mx.get_peak_memory()
    cache = mx.get_cache_memory()
    print(f"  [{label:40s}]  active={mb(active):8.1f} MB  peak={mb(peak):8.1f} MB  cache={mb(cache):8.1f} MB")
    if reset_peak:
        mx.reset_peak_memory()
    return active, peak, cache


def profile_standard_cache(model, tokenizer, prompt_text, max_tokens=20):
    """Profile standard KV cache generation step by step."""
    import mlx_lm
    from mlx_lm.generate import generate_step

    print("\n" + "=" * 80)
    print("PROFILING: Standard KV Cache (baseline)")
    print("=" * 80)

    # Format prompt
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
    print(f"  Prompt tokens: {tokens.size}")

    # Clear everything
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.eval(tokens)

    print_mem("Before generation", reset_peak=True)

    # Use stream_generate to match actual usage
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)

    step = 0
    mem_log = []

    mx.reset_peak_memory()

    for response in mlx_lm.stream_generate(
        model, tokenizer, formatted,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        # Force evaluation
        mx.synchronize()

        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
        cache_mem = mx.get_cache_memory()

        mem_log.append({
            "step": step,
            "active_mb": mb(active),
            "peak_mb": mb(peak),
            "cache_mb": mb(cache_mem),
        })

        if step < 5 or step % 5 == 0 or step == max_tokens - 1:
            print(f"  Step {step:3d}: active={mb(active):8.1f} MB  peak={mb(peak):8.1f} MB  cache={mb(cache_mem):8.1f} MB")

        step += 1

    final_active = mx.get_active_memory()
    final_peak = mx.get_peak_memory()
    print(f"\n  FINAL: active={mb(final_active):8.1f} MB  peak={mb(final_peak):8.1f} MB")

    return mem_log, final_peak


def profile_turboquant_cache(model, tokenizer, prompt_text, max_tokens=20):
    """Profile TurboQuant KV cache generation step by step."""
    import mlx_lm
    from mlx_turboquant.patch import make_turboquant_cache, _attention_patched

    print("\n" + "=" * 80)
    print("PROFILING: TurboQuant KV Cache")
    print("=" * 80)

    # Format prompt
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
    print(f"  Prompt tokens: {tokens.size}")

    # Clear everything
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.eval(tokens)

    print_mem("Before TQ cache creation", reset_peak=True)

    # Create TurboQuant cache
    tq_cache = make_turboquant_cache(
        model,
        key_bits=3,
        value_bits=2,
        buffer_size=128,
    )
    mx.synchronize()
    print_mem("After TQ cache creation (quantizer matrices)")

    # Measure quantizer matrix sizes
    layer0 = tq_cache[0]
    kq = layer0.key_quantizer
    pi_bytes = kq.mse_quantizer.Pi.nbytes
    s_bytes = kq.S.nbytes
    centroids_bytes = kq.mse_quantizer.centroids.nbytes
    boundaries_bytes = kq.mse_quantizer.boundaries.nbytes
    n_layers = len(tq_cache)
    print(f"\n  === Quantizer Matrix Sizes (per layer) ===")
    print(f"    Pi (rotation):       {mb(pi_bytes):8.3f} MB  shape={kq.mse_quantizer.Pi.shape}")
    print(f"    S (QJL):             {mb(s_bytes):8.3f} MB  shape={kq.S.shape}")
    print(f"    Centroids:           {mb(centroids_bytes):8.3f} MB  shape={kq.mse_quantizer.centroids.shape}")
    print(f"    Boundaries:          {mb(boundaries_bytes):8.3f} MB  shape={kq.mse_quantizer.boundaries.shape}")
    per_layer_total = pi_bytes + s_bytes + centroids_bytes + boundaries_bytes
    print(f"    Per-layer total:     {mb(per_layer_total):8.3f} MB")
    print(f"    All {n_layers} layers total: {mb(per_layer_total * n_layers):8.3f} MB")
    print()

    mx.clear_cache()
    mx.reset_peak_memory()

    # Generate with TurboQuant
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)

    step = 0
    mem_log = []

    for response in mlx_lm.stream_generate(
        model, tokenizer, formatted,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=tq_cache,
    ):
        mx.synchronize()

        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
        cache_mem = mx.get_cache_memory()

        mem_log.append({
            "step": step,
            "active_mb": mb(active),
            "peak_mb": mb(peak),
            "cache_mb": mb(cache_mem),
        })

        if step < 5 or step % 5 == 0 or step == max_tokens - 1:
            # Also get cache state
            l0 = tq_cache[0]
            compressed = l0.compressed_tokens
            buffered = l0.buffer_tokens

            print(f"  Step {step:3d}: active={mb(active):8.1f} MB  peak={mb(peak):8.1f} MB  "
                  f"cache={mb(cache_mem):8.1f} MB  [comp={compressed} buf={buffered}]")

        step += 1

    final_active = mx.get_active_memory()
    final_peak = mx.get_peak_memory()
    print(f"\n  FINAL: active={mb(final_active):8.1f} MB  peak={mb(final_peak):8.1f} MB")

    # Final cache memory breakdown
    print(f"\n  === TQ Cache Memory Breakdown (all layers) ===")
    total_cache_bytes = sum(c.nbytes for c in tq_cache)
    print(f"    Total cache nbytes: {mb(total_cache_bytes):8.3f} MB")
    l0 = tq_cache[0]
    report = l0.memory_report()
    print(f"    Layer 0 detail:")
    print(f"      Compressed keys:   {mb(report['compressed_keys_bytes']):8.3f} MB  ({report['compressed_tokens']} tokens)")
    print(f"      Compressed values: {mb(report['compressed_values_bytes']):8.3f} MB")
    print(f"      Buffer:            {mb(report['buffer_bytes']):8.3f} MB  ({report['buffer_tokens']} tokens)")

    return mem_log, final_peak


def profile_gqa_expansion():
    """Isolate the memory cost of mx.repeat for GQA expansion."""
    print("\n" + "=" * 80)
    print("PROFILING: GQA Expansion (mx.repeat) Memory Cost")
    print("=" * 80)

    # Qwen2.5-3B: n_kv_heads=4, n_heads=16, head_dim=128
    # So repeats = 16/4 = 4
    B = 1
    n_kv_heads = 4
    n_heads = 16
    head_dim = 128
    repeats = n_heads // n_kv_heads

    for seq_len in [1, 10, 50, 128, 256]:
        mx.clear_cache()
        mx.reset_peak_memory()

        kv = mx.random.normal((B, n_kv_heads, seq_len, head_dim))
        mx.eval(kv)

        mx.reset_peak_memory()
        kv_expanded = mx.repeat(kv, repeats, axis=1)
        mx.eval(kv_expanded)

        peak = mx.get_peak_memory()
        active = mx.get_active_memory()
        input_mb = kv.nbytes / 1e6
        output_mb = kv_expanded.nbytes / 1e6

        print(f"  seq_len={seq_len:4d}: input={input_mb:8.3f} MB  output={output_mb:8.3f} MB  "
              f"peak={mb(peak):8.3f} MB  overhead={mb(peak) - input_mb - output_mb:+8.3f} MB")

        del kv_expanded
        mx.clear_cache()


def profile_concatenate_cost():
    """Isolate the memory cost of mx.concatenate for cache growing."""
    print("\n" + "=" * 80)
    print("PROFILING: mx.concatenate Memory Cost (cache append)")
    print("=" * 80)

    B = 1
    n_kv_heads = 4
    head_dim = 128

    for existing_len in [64, 128, 256, 512]:
        mx.clear_cache()
        mx.reset_peak_memory()

        existing = mx.random.normal((B, n_kv_heads, existing_len, head_dim))
        new_token = mx.random.normal((B, n_kv_heads, 1, head_dim))
        mx.eval(existing, new_token)

        mx.reset_peak_memory()
        result = mx.concatenate([existing, new_token], axis=2)
        mx.eval(result)

        peak = mx.get_peak_memory()
        input_bytes = existing.nbytes + new_token.nbytes
        output_bytes = result.nbytes

        print(f"  existing={existing_len:4d}: input_total={mb(input_bytes):8.3f} MB  "
              f"output={mb(output_bytes):8.3f} MB  peak={mb(peak):8.3f} MB  "
              f"overhead={mb(peak) - mb(input_bytes):+8.3f} MB")

        del result
        mx.clear_cache()


def profile_quantization_cost():
    """Profile the memory cost of TurboQuant quantization (flush operation)."""
    print("\n" + "=" * 80)
    print("PROFILING: TurboQuant Quantization (flush) Memory Cost")
    print("=" * 80)

    from turboquant_mac.quantizer import TurboQuantProd
    from turboquant_mac.kv_cache import quantize_values

    B = 1
    n_kv_heads = 4
    head_dim = 128

    quantizer = TurboQuantProd(dim=head_dim, bits=3, seed=42, backend="mlx")

    for n_tokens in [1, 4, 16, 64]:
        mx.clear_cache()

        keys = mx.random.normal((B, n_kv_heads, n_tokens, head_dim))
        values = mx.random.normal((B, n_kv_heads, n_tokens, head_dim))
        mx.eval(keys, values)

        input_bytes = keys.nbytes + values.nbytes
        mx.reset_peak_memory()

        key_q = quantizer.quantize(keys)
        mx.eval(key_q.mse_indices, key_q.qjl_signs, key_q.residual_norms, key_q.norms)

        quant_peak = mx.get_peak_memory()
        quant_active = mx.get_active_memory()

        mx.reset_peak_memory()
        val_q = quantize_values(values, bits=2, group_size=32, backend="mlx")
        mx.eval(val_q.data, val_q.scales, val_q.zeros)

        val_peak = mx.get_peak_memory()

        # Compressed sizes
        key_compressed_bytes = key_q.mse_indices.nbytes + key_q.qjl_signs.nbytes + key_q.residual_norms.nbytes + key_q.norms.nbytes
        val_compressed_bytes = val_q.data.nbytes + val_q.scales.nbytes + val_q.zeros.nbytes

        print(f"  n_tokens={n_tokens:3d}: input={mb(input_bytes):8.3f} MB  "
              f"key_quant_peak={mb(quant_peak):8.3f} MB  val_quant_peak={mb(val_peak):8.3f} MB  "
              f"compressed_keys={mb(key_compressed_bytes):8.3f} MB  compressed_vals={mb(val_compressed_bytes):8.3f} MB")

        del key_q, val_q
        mx.clear_cache()


def profile_value_decompression_cost():
    """Profile the memory cost of value decompression (per decode step)."""
    print("\n" + "=" * 80)
    print("PROFILING: Value Decompression Memory Cost (per decode step)")
    print("=" * 80)

    from turboquant_mac.kv_cache import quantize_values, dequantize_values

    B = 1
    n_kv_heads = 4
    head_dim = 128
    n_heads = 16
    repeats = n_heads // n_kv_heads

    for n_compressed in [10, 50, 100, 200, 500]:
        mx.clear_cache()

        values = mx.random.normal((B, n_kv_heads, n_compressed, head_dim))
        mx.eval(values)
        val_q = quantize_values(values, bits=2, group_size=32, backend="mlx")
        mx.eval(val_q.data, val_q.scales, val_q.zeros)
        del values
        mx.clear_cache()

        mx.reset_peak_memory()

        # Decompress
        v_decompressed = dequantize_values(val_q, group_size=32, backend="mlx")
        mx.eval(v_decompressed)
        decomp_peak = mx.get_peak_memory()
        decomp_active = mx.get_active_memory()

        # GQA expand
        mx.reset_peak_memory()
        v_expanded = mx.repeat(v_decompressed, repeats, axis=1)
        mx.eval(v_expanded)
        expand_peak = mx.get_peak_memory()

        print(f"  n_compressed={n_compressed:4d}: "
              f"decompress_peak={mb(decomp_peak):8.3f} MB  "
              f"gqa_expand_peak={mb(expand_peak):8.3f} MB  "
              f"decompressed={mb(v_decompressed.nbytes):8.3f} MB  "
              f"expanded={mb(v_expanded.nbytes):8.3f} MB")

        del v_decompressed, v_expanded
        mx.clear_cache()


def profile_compressed_score_cost():
    """Profile the memory cost of computing attention scores from compressed keys."""
    print("\n" + "=" * 80)
    print("PROFILING: Compressed Attention Score Memory Cost")
    print("=" * 80)

    from turboquant_mac.quantizer import TurboQuantProd

    B = 1
    n_kv_heads = 4
    n_heads = 16
    head_dim = 128
    repeats = n_heads // n_kv_heads

    quantizer = TurboQuantProd(dim=head_dim, bits=3, seed=42, backend="mlx")

    for n_compressed in [10, 50, 100, 200]:
        mx.clear_cache()

        keys = mx.random.normal((B, n_kv_heads, n_compressed, head_dim))
        mx.eval(keys)
        key_q = quantizer.quantize(keys)
        mx.eval(key_q.mse_indices, key_q.qjl_signs, key_q.residual_norms, key_q.norms)
        del keys
        mx.clear_cache()

        # Single query token (decode)
        query = mx.random.normal((B, n_heads, 1, head_dim))
        mx.eval(query)

        # The attention path in patch.py does:
        # 1. GQA expand compressed data
        # 2. Reshape for Metal kernel
        # 3. Call Metal kernels

        mx.reset_peak_memory()

        # Simulate what compute_compressed_scores does
        mse_packed = mx.repeat(key_q.mse_indices, repeats, axis=1)
        qjl_signs = mx.repeat(key_q.qjl_signs, repeats, axis=1)
        norms = mx.repeat(key_q.norms, repeats, axis=-1)
        res_norms = mx.repeat(key_q.residual_norms, repeats, axis=-1)
        mx.eval(mse_packed, qjl_signs, norms, res_norms)

        expand_peak = mx.get_peak_memory()
        expand_active = mx.get_active_memory()

        # Size of expanded data
        expanded_bytes = mse_packed.nbytes + qjl_signs.nbytes + norms.nbytes + res_norms.nbytes
        original_bytes = key_q.mse_indices.nbytes + key_q.qjl_signs.nbytes + key_q.norms.nbytes + key_q.residual_norms.nbytes

        print(f"  n_compressed={n_compressed:4d}: "
              f"original_compressed={mb(original_bytes):8.3f} MB  "
              f"after_gqa_expand={mb(expanded_bytes):8.3f} MB  "
              f"expand_peak={mb(expand_peak):8.3f} MB  "
              f"overhead_ratio={expanded_bytes/original_bytes:.1f}x")

        del mse_packed, qjl_signs, norms, res_norms
        mx.clear_cache()


def profile_metal_kernel_compilation():
    """Profile memory cost of Metal kernel compilation/caching."""
    print("\n" + "=" * 80)
    print("PROFILING: Metal Kernel Compilation Memory Cost")
    print("=" * 80)

    from turboquant_mac.backends import metal_kernels
    # Clear kernel caches
    metal_kernels._mse_kernel_cache.clear()
    metal_kernels._qjl_kernel_cache.clear()
    metal_kernels._value_ws_kernel_cache.clear()

    from turboquant_mac.quantizer import TurboQuantProd
    from turboquant_mac.kv_cache import quantize_values, dequantize_values

    B = 1
    n_kv_heads = 4
    head_dim = 128

    quantizer = TurboQuantProd(dim=head_dim, bits=3, seed=42, backend="mlx")

    keys = mx.random.normal((B, n_kv_heads, 10, head_dim))
    values = mx.random.normal((B, n_kv_heads, 10, head_dim))
    query = mx.random.normal((B * 16, head_dim))
    mx.eval(keys, values, query)

    mx.clear_cache()
    mx.reset_peak_memory()
    mem_before = mx.get_active_memory()

    # Trigger key quantization
    key_q = quantizer.quantize(keys)
    mx.eval(key_q.mse_indices, key_q.qjl_signs, key_q.residual_norms, key_q.norms)
    mem_after_encode = mx.get_active_memory()
    peak_after_encode = mx.get_peak_memory()
    print(f"  After key quantize: active_delta={mb(mem_after_encode - mem_before):+8.3f} MB  peak={mb(peak_after_encode):8.3f} MB")

    # Trigger value quantize + dequantize
    val_q = quantize_values(values, bits=2, group_size=32, backend="mlx")
    mx.eval(val_q.data, val_q.scales, val_q.zeros)
    mx.reset_peak_memory()
    mem_before_dequant = mx.get_active_memory()
    v_dec = dequantize_values(val_q, group_size=32, backend="mlx")
    mx.eval(v_dec)
    mem_after_dequant = mx.get_active_memory()
    peak_dequant = mx.get_peak_memory()
    print(f"  After value dequant: active_delta={mb(mem_after_dequant - mem_before_dequant):+8.3f} MB  peak={mb(peak_dequant):8.3f} MB")

    # Trigger MSE score + QJL score kernels
    from turboquant_mac.backends.metal_kernels import turboquant_attention_score_metal

    mse_packed = mx.repeat(key_q.mse_indices, 4, axis=1)
    qjl_signs = mx.repeat(key_q.qjl_signs, 4, axis=1)
    norms = mx.repeat(key_q.norms, 4, axis=-1)
    res_norms = mx.repeat(key_q.residual_norms, 4, axis=-1)
    mx.eval(mse_packed, qjl_signs, norms, res_norms)

    flat_mse = mse_packed.reshape(B * 16, 10, -1)
    flat_signs = qjl_signs.reshape(B * 16, 10, -1)
    flat_norms = norms.reshape(B * 16, 10)
    flat_res_norms = res_norms.reshape(B * 16, 10)
    mx.eval(flat_mse, flat_signs, flat_norms, flat_res_norms)

    mx.reset_peak_memory()
    mem_before_score = mx.get_active_memory()
    scores = turboquant_attention_score_metal(
        query=query,
        mse_packed=flat_mse,
        qjl_signs=flat_signs,
        norms=flat_norms,
        residual_norms=flat_res_norms,
        Pi=quantizer.mse_quantizer.Pi,
        S=quantizer.S,
        centroids=quantizer.mse_quantizer.centroids,
        mse_bits=quantizer.mse_quantizer.bits,
        qjl_scale=quantizer.qjl_scale,
    )
    mx.eval(scores)
    mem_after_score = mx.get_active_memory()
    peak_score = mx.get_peak_memory()
    print(f"  After attention score kernels: active_delta={mb(mem_after_score - mem_before_score):+8.3f} MB  peak={mb(peak_score):8.3f} MB")

    print(f"\n  Kernel cache sizes: "
          f"mse_score={len(metal_kernels._mse_kernel_cache)}, "
          f"qjl_score={len(metal_kernels._qjl_kernel_cache)}, "
          f"value_ws={len(metal_kernels._value_ws_kernel_cache)}")


def profile_lazy_eval_spikes():
    """Test whether lazy evaluation causes memory spikes."""
    print("\n" + "=" * 80)
    print("PROFILING: Lazy Evaluation Memory Spike Test")
    print("=" * 80)

    mx.clear_cache()
    mx.reset_peak_memory()

    # Build a chain of operations WITHOUT eval
    a = mx.random.normal((1000, 128))
    b = mx.random.normal((128, 128))
    c = a
    for i in range(10):
        c = mx.matmul(c, b)  # 10 chained matmuls, unevaluated

    active_before = mx.get_active_memory()
    print(f"  Before eval (10 chained matmuls): active={mb(active_before):8.3f} MB")

    mx.reset_peak_memory()
    mx.eval(c)  # Force all at once
    active_after = mx.get_active_memory()
    peak_after = mx.get_peak_memory()
    print(f"  After eval:                        active={mb(active_after):8.3f} MB  peak={mb(peak_after):8.3f} MB")
    print(f"  -> Lazy eval spike: {mb(peak_after) - mb(active_after):+8.3f} MB above final active")

    # Compare: eval after each step
    mx.clear_cache()
    mx.reset_peak_memory()
    c = a
    for i in range(10):
        c = mx.matmul(c, b)
        mx.eval(c)

    peak_incremental = mx.get_peak_memory()
    active_incremental = mx.get_active_memory()
    print(f"  Incremental eval (eval each step): active={mb(active_incremental):8.3f} MB  peak={mb(peak_incremental):8.3f} MB")
    print(f"  -> Batch eval used {mb(peak_after - peak_incremental):+8.3f} MB more peak than incremental")


def main():
    import mlx_lm

    model_path = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    prompt = "Explain the theory of relativity in detail. Start from the basics and build up to the key equations and their implications for our understanding of space, time, and gravity."
    max_tokens = 40

    print("=" * 80)
    print(f"MLX Memory Profiler for TurboQuant")
    print(f"Model: {model_path}")
    print(f"Max tokens: {max_tokens}")
    print(f"MLX version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print(f"System memory: {mx.device_info()['memory_size'] / 1e9:.1f} GB")
    print("=" * 80)

    # ---- Micro-benchmarks first (no model needed) ----
    profile_lazy_eval_spikes()
    profile_gqa_expansion()
    profile_concatenate_cost()
    profile_quantization_cost()
    profile_value_decompression_cost()
    profile_compressed_score_cost()

    # ---- Load model ----
    print("\n" + "=" * 80)
    print("Loading model...")
    print("=" * 80)
    mx.clear_cache()
    mx.reset_peak_memory()

    model, tokenizer = mlx_lm.load(model_path)
    mx.synchronize()

    model_active = mx.get_active_memory()
    model_peak = mx.get_peak_memory()
    print(f"  Model loaded: active={mb(model_active):8.1f} MB  peak={mb(model_peak):8.1f} MB")

    # ---- Profile Metal kernel compilation ----
    profile_metal_kernel_compilation()

    # ---- Profile standard cache ----
    mx.clear_cache()
    mx.reset_peak_memory()
    std_log, std_peak = profile_standard_cache(model, tokenizer, prompt, max_tokens)

    # ---- Profile TurboQuant cache ----
    mx.clear_cache()
    mx.reset_peak_memory()
    tq_log, tq_peak = profile_turboquant_cache(model, tokenizer, prompt, max_tokens)

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY: Peak Memory Comparison")
    print("=" * 80)
    print(f"  Standard KV Cache peak:   {mb(std_peak):8.1f} MB")
    print(f"  TurboQuant KV Cache peak: {mb(tq_peak):8.1f} MB")
    print(f"  Difference:               {mb(tq_peak - std_peak):+8.1f} MB")

    print("\n  Step-by-step comparison (active memory):")
    print(f"  {'Step':>5s}  {'Standard MB':>12s}  {'TurboQuant MB':>14s}  {'Delta MB':>10s}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*14}  {'-'*10}")
    for i in range(min(len(std_log), len(tq_log))):
        s = std_log[i]
        t = tq_log[i]
        delta = t["active_mb"] - s["active_mb"]
        if i < 5 or i % 5 == 0 or i == len(std_log) - 1:
            print(f"  {i:5d}  {s['active_mb']:12.1f}  {t['active_mb']:14.1f}  {delta:+10.1f}")


if __name__ == "__main__":
    main()
