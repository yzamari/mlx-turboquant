"""
Tests for the fused decode Metal kernel and the fused attention path.

Verifies that the single-pass kernel (scores + online softmax + value dequant +
weighted sum) produces correct results matching the legacy multi-kernel path.
"""

import math

import pytest
import mlx.core as mx

from mlx_turboquant.cache import TurboQuantCache


# Common test parameters
HEAD_DIM = 128
BATCH = 1
N_HEADS = 4
KEY_BITS = 3
VALUE_BITS = 2
BUFFER_SIZE = 16
VALUE_GROUP_SIZE = 32


def _make_cache(**kwargs):
    """Create a TurboQuantCache with test defaults."""
    defaults = dict(
        head_dim=HEAD_DIM,
        key_bits=KEY_BITS,
        value_bits=VALUE_BITS,
        value_group_size=VALUE_GROUP_SIZE,
        buffer_size=BUFFER_SIZE,
        layer_idx=0,
    )
    defaults.update(kwargs)
    return TurboQuantCache(**defaults)


def _random_kv(seq_len, batch=BATCH, n_heads=N_HEADS, head_dim=HEAD_DIM):
    """Generate random key/value tensors."""
    keys = mx.random.normal((batch, n_heads, seq_len, head_dim))
    values = mx.random.normal((batch, n_heads, seq_len, head_dim))
    mx.eval(keys, values)
    return keys, values


def _fill_cache_decode(cache, n_total, n_heads=N_HEADS):
    """Fill cache by feeding tokens one-at-a-time to trigger decode flush."""
    for _ in range(n_total):
        k, v = _random_kv(1, n_heads=n_heads)
        buf_k, buf_v = cache.update_and_fetch(k, v)
        mx.eval(buf_k, buf_v)
    return buf_k, buf_v


class TestFusedKernelDirect:
    """Test the fused Metal kernel directly via compute_fused_attention."""

    def test_fused_kernel_output_shape(self):
        """Verify fused kernel returns (BH, D), (BH,), (BH,) shapes."""
        from mlx_turboquant.metal.fused_decode import turboquant_fused_decode_metal

        BH = 4
        N = 16
        D = HEAD_DIM

        # Create fake data matching expected shapes
        q_rot = mx.random.normal((BH, D))
        q_sketch = mx.random.normal((BH, D))

        # mse_bits=3 -> effective 2 bits -> 4 vals/byte -> packed_d = D/4 = 32
        packed_d = D // 4
        mse = mx.zeros((BH, N, packed_d), dtype=mx.uint8)
        # signs: D/8 = 16 bytes
        packed_d_signs = D // 8
        signs = mx.zeros((BH, N, packed_d_signs), dtype=mx.uint8)
        norms = mx.ones((BH, N))
        res_norms = mx.ones((BH, N))
        centroids = mx.array([0.0, 0.5, -0.5, 1.0])  # 4 centroids for 2-bit

        # value_bits=2 -> 4 vals/byte -> packed_v = D/4 = 32
        packed_v = D // 4
        v_data = mx.zeros((BH, N, packed_v), dtype=mx.uint8)
        n_groups = D // VALUE_GROUP_SIZE
        v_scales = mx.ones((BH, N, n_groups))
        v_zeros = mx.zeros((BH, N, n_groups))

        mx.eval(q_rot, q_sketch, mse, signs, norms, res_norms, centroids,
                v_data, v_scales, v_zeros)

        qjl_scale = math.sqrt(math.pi / 2.0) / D
        sm_scale = 1.0 / math.sqrt(D)

        out, out_m, out_l = turboquant_fused_decode_metal(
            q_rot=q_rot, q_sketch=q_sketch,
            mse=mse, signs=signs,
            norms=norms, res_norms=res_norms,
            centroids=centroids,
            v_data=v_data, v_scales=v_scales, v_zeros=v_zeros,
            mse_bits=KEY_BITS, v_bits=VALUE_BITS,
            D=D, group_size=VALUE_GROUP_SIZE,
            qjl_scale=qjl_scale, sm_scale=sm_scale,
        )
        mx.eval(out, out_m, out_l)

        assert out.shape == (BH, D), f"Expected ({BH}, {D}), got {out.shape}"
        assert out_m.shape == (BH,), f"Expected ({BH},), got {out_m.shape}"
        assert out_l.shape == (BH,), f"Expected ({BH},), got {out_l.shape}"

    def test_fused_vs_legacy_no_gqa(self):
        """Fused output should approximate the legacy decompressed matmul path (no GQA)."""
        from mlx_turboquant.patch import _turboquant_attention, _turboquant_attention_fused

        cache = _make_cache(buffer_size=16)
        buf_k, buf_v = _fill_cache_decode(cache, n_total=64)

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        scale = 1.0 / math.sqrt(HEAD_DIM)

        # Fused path
        output_fused = _turboquant_attention_fused(
            queries, buf_k, buf_v, cache, scale, mask=None
        )
        mx.eval(output_fused)

        # Legacy (Phase 3 two-kernel) path
        output_legacy = _turboquant_attention(
            queries, buf_k, buf_v, cache, scale, mask=None
        )
        mx.eval(output_legacy)

        assert output_fused.shape == output_legacy.shape
        diff = float(mx.mean(mx.abs(output_fused - output_legacy)))
        assert diff < 0.5, (
            f"Fused output should approximate legacy, got mean diff={diff}"
        )

    def test_fused_vs_legacy_gqa(self):
        """Fused output should approximate legacy with GQA (n_kv_heads=2, n_heads=8)."""
        from mlx_turboquant.patch import _turboquant_attention, _turboquant_attention_fused

        N_KV_HEADS = 2
        N_QUERY_HEADS = 8

        cache = _make_cache(buffer_size=16)

        # Fill with KV heads = 2
        for _ in range(64):
            k, v = _random_kv(1, n_heads=N_KV_HEADS)
            buf_k, buf_v = cache.update_and_fetch(k, v)
            mx.eval(buf_k, buf_v)

        assert cache._keys_compressed is not None

        queries = mx.random.normal((BATCH, N_QUERY_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        scale = 1.0 / math.sqrt(HEAD_DIM)

        # Fused path
        output_fused = _turboquant_attention_fused(
            queries, buf_k, buf_v, cache, scale, mask=None
        )
        mx.eval(output_fused)

        # Legacy path
        output_legacy = _turboquant_attention(
            queries, buf_k, buf_v, cache, scale, mask=None
        )
        mx.eval(output_legacy)

        assert output_fused.shape == (BATCH, N_QUERY_HEADS, 1, HEAD_DIM)
        assert output_legacy.shape == (BATCH, N_QUERY_HEADS, 1, HEAD_DIM)

        diff = float(mx.mean(mx.abs(output_fused - output_legacy)))
        assert diff < 0.5, (
            f"GQA fused output should approximate legacy, got mean diff={diff}"
        )

    def test_fused_edge_n_zero(self):
        """N=0 compressed tokens: compute_fused_attention returns None gracefully."""
        cache = _make_cache(buffer_size=32)
        keys, values = _random_kv(10)
        cache.update_and_fetch(keys, values)

        assert cache._keys_compressed is None

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        result = cache.compute_fused_attention(queries, scale=1.0 / math.sqrt(HEAD_DIM))
        assert result is None

    def test_fused_edge_n_one(self):
        """Small number of compressed tokens should work."""
        # Use small buffer+flush to ensure compression triggers.
        # flush_batch_size defaults to buffer_size, so flush_threshold = 2*buffer_size.
        # Feed decode tokens (n_new=1) one at a time past the threshold.
        cache = _make_cache(buffer_size=8, flush_batch_size=1)
        # Feed 10 decode tokens: buffer_size=8, flush_batch=1, threshold=9
        # At token 10: buffer has 10 > 9, flush 2 -> 2 compressed + 8 buffer
        for _ in range(10):
            k, v = _random_kv(1)
            cache.update_and_fetch(k, v)
            mx.eval(cache._key_buffer, cache._value_buffer)

        assert cache.compressed_tokens > 0, (
            f"Expected compressed tokens, got {cache.compressed_tokens}"
        )

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        result = cache.compute_fused_attention(queries, scale=1.0 / math.sqrt(HEAD_DIM))
        assert result is not None

        acc, m, l = result
        mx.eval(acc, m, l)

        assert acc.shape == (BATCH, N_HEADS, 1, HEAD_DIM)
        assert m.shape == (BATCH, N_HEADS, 1)
        assert l.shape == (BATCH, N_HEADS, 1)
        assert not mx.any(mx.isnan(acc)).item()

    def test_fused_large_n(self):
        """Large N stress test: fused kernel handles many compressed tokens."""
        cache = _make_cache(buffer_size=16)
        # Prefill triggers flush only during decode (n_new=1).
        # Feed many decode tokens to accumulate compressed data.
        # With buffer_size=16 and flush_batch_size=16, threshold=32.
        # After 64 decode tokens: ~32 compressed + 16 in buffer.
        buf_k, buf_v = _fill_cache_decode(cache, n_total=200)

        n_compressed = cache.compressed_tokens
        assert n_compressed > 100, (
            f"Expected >100 compressed tokens, got {n_compressed}"
        )

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        result = cache.compute_fused_attention(queries, scale=1.0 / math.sqrt(HEAD_DIM))
        assert result is not None

        acc, m, l = result
        mx.eval(acc, m, l)

        assert acc.shape == (BATCH, N_HEADS, 1, HEAD_DIM)
        assert not mx.any(mx.isnan(acc)).item()
        assert not mx.any(mx.isnan(m)).item()
        # l should be positive (sum of exp terms)
        assert mx.all(l > 0).item()

    def test_fused_attention_integration(self):
        """Full _turboquant_attention_fused with buffer merge produces valid output."""
        from mlx_turboquant.patch import _turboquant_attention_fused

        cache = _make_cache(buffer_size=16)
        buf_k, buf_v = _fill_cache_decode(cache, n_total=64)

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        scale = 1.0 / math.sqrt(HEAD_DIM)
        output = _turboquant_attention_fused(
            queries, buf_k, buf_v, cache, scale, mask=None
        )
        mx.eval(output)

        assert output.shape == (BATCH, N_HEADS, 1, HEAD_DIM)
        assert not mx.any(mx.isnan(output)).item()
        assert not mx.any(mx.isinf(output)).item()

        # Output should have reasonable magnitude (not degenerate)
        max_val = float(mx.max(mx.abs(output)))
        assert max_val > 1e-6, f"Output too close to zero: max abs = {max_val}"
        assert max_val < 100.0, f"Output too large: max abs = {max_val}"

    def test_fused_numerical_stability(self):
        """Large score values should not cause overflow in online softmax."""
        from mlx_turboquant.patch import _turboquant_attention_fused

        cache = _make_cache(buffer_size=16)

        # Use large-magnitude keys to create large scores
        for _ in range(64):
            k = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM)) * 10.0
            v = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
            mx.eval(k, v)
            buf_k, buf_v = cache.update_and_fetch(k, v)
            mx.eval(buf_k, buf_v)

        assert cache._keys_compressed is not None

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM)) * 10.0
        mx.eval(queries)

        scale = 1.0 / math.sqrt(HEAD_DIM)
        output = _turboquant_attention_fused(
            queries, buf_k, buf_v, cache, scale, mask=None
        )
        mx.eval(output)

        assert output.shape == (BATCH, N_HEADS, 1, HEAD_DIM)
        assert not mx.any(mx.isnan(output)).item(), "NaN detected in output with large scores"
        assert not mx.any(mx.isinf(output)).item(), "Inf detected in output with large scores"


class TestFusedVsFullDecompression:
    """Compare fused path against full decompression + standard attention."""

    def test_vs_full_decompression(self):
        """Fused attention should approximate full decompression -> matmul path."""
        from mlx_turboquant.patch import _turboquant_attention_fused

        cache = _make_cache(buffer_size=16)
        keys, values = _random_kv(64)
        buf_k, buf_v = cache.update_and_fetch(keys, values)
        mx.eval(buf_k, buf_v)

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        scale = 1.0 / math.sqrt(HEAD_DIM)

        # Fused path
        output_fused = _turboquant_attention_fused(
            queries, buf_k, buf_v, cache, scale, mask=None
        )
        mx.eval(output_fused)

        # Full decompression + standard attention
        all_keys = cache._get_all_keys()
        all_values = cache._get_all_values()
        mx.eval(all_keys, all_values)

        # Handle GQA for legacy path
        n_kv_heads = all_keys.shape[1]
        repeats = N_HEADS // n_kv_heads
        if repeats > 1:
            all_keys = mx.repeat(all_keys, repeats, axis=1)
            all_values = mx.repeat(all_values, repeats, axis=1)

        legacy_scores = (queries @ mx.transpose(all_keys, (0, 1, 3, 2))) * scale
        legacy_weights = mx.softmax(legacy_scores, axis=-1, precise=True)
        output_legacy = legacy_weights @ all_values
        mx.eval(output_legacy)

        diff = float(mx.mean(mx.abs(output_fused - output_legacy)))
        assert diff < 0.5, (
            f"Fused output should approximate full decompression, got mean diff={diff}"
        )
