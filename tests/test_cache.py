"""
Tests for TurboQuantCache — verifies MLX-LM compatibility and correctness.

Phase 2: Tests updated for buffer-only update_and_fetch() and direct
attention from compressed data via Metal kernels.
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
BUFFER_SIZE = 32  # Small for testing
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


def _random_kv(seq_len: int, batch=BATCH, n_heads=N_HEADS, head_dim=HEAD_DIM):
    """Generate random key/value tensors."""
    keys = mx.random.normal((batch, n_heads, seq_len, head_dim))
    values = mx.random.normal((batch, n_heads, seq_len, head_dim))
    mx.eval(keys, values)
    return keys, values


class TestBasicInterface:
    """Test that TurboQuantCache conforms to MLX-LM's _BaseCache interface."""

    def test_empty_on_init(self):
        cache = _make_cache()
        assert cache.empty()
        assert cache.offset == 0
        assert cache.size() == 0

    def test_update_and_fetch_returns_buffer_only(self):
        """update_and_fetch returns buffer-only (keys, values) with correct shapes."""
        cache = _make_cache()
        keys, values = _random_kv(10)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        # No compression triggered, buffer == all tokens
        assert out_k.shape == (BATCH, N_HEADS, 10, HEAD_DIM)
        assert out_v.shape == (BATCH, N_HEADS, 10, HEAD_DIM)
        assert cache.offset == 10
        assert not cache.empty()

    def test_update_and_fetch_buffer_size_after_flush(self):
        """After flush, update_and_fetch returns buffer_size tokens only."""
        cache = _make_cache(buffer_size=16)
        keys, values = _random_kv(64)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        # Buffer should be capped at buffer_size
        assert out_k.shape == (BATCH, N_HEADS, 16, HEAD_DIM)
        assert out_v.shape == (BATCH, N_HEADS, 16, HEAD_DIM)
        assert cache.offset == 64
        assert cache.compressed_tokens == 48
        assert cache.buffer_tokens == 16

    def test_sequential_append(self):
        """Appending tokens one at a time accumulates correctly in buffer."""
        cache = _make_cache()

        for i in range(5):
            keys, values = _random_kv(1)
            out_k, out_v = cache.update_and_fetch(keys, values)
            mx.eval(out_k, out_v)

        assert cache.offset == 5
        # No flush triggered (5 < 32), buffer has all 5 tokens
        assert out_k.shape == (BATCH, N_HEADS, 5, HEAD_DIM)
        assert out_v.shape == (BATCH, N_HEADS, 5, HEAD_DIM)

    def test_prefill_then_decode(self):
        """Simulate real inference: prefill many tokens, then decode one at a time."""
        cache = _make_cache(buffer_size=16)

        # Prefill
        keys, values = _random_kv(20)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)
        assert cache.offset == 20
        # 20 > 16, so 4 compressed + 16 in buffer
        assert cache.compressed_tokens == 4
        assert out_k.shape == (BATCH, N_HEADS, 16, HEAD_DIM)

        # Decode 5 tokens
        for i in range(5):
            keys, values = _random_kv(1)
            out_k, out_v = cache.update_and_fetch(keys, values)
            mx.eval(out_k, out_v)

        assert cache.offset == 25
        # Buffer may have flushed more; check it is <= buffer_size
        assert out_k.shape[2] <= cache.buffer_size + 5

    def test_nbytes_property(self):
        cache = _make_cache()
        keys, values = _random_kv(10)
        cache.update_and_fetch(keys, values)
        mx.eval(cache._key_buffer, cache._value_buffer)
        assert cache.nbytes > 0

    def test_is_trimmable(self):
        cache = _make_cache()
        assert cache.is_trimmable()

    def test_trim(self):
        cache = _make_cache()
        keys, values = _random_kv(10)
        cache.update_and_fetch(keys, values)

        trimmed = cache.trim(3)
        assert trimmed == 3
        assert cache.offset == 7

    def test_trim_more_than_offset(self):
        cache = _make_cache()
        keys, values = _random_kv(5)
        cache.update_and_fetch(keys, values)

        trimmed = cache.trim(100)
        assert trimmed == 5
        assert cache.offset == 0

    def test_make_mask(self):
        """make_mask should return valid attention masks."""
        cache = _make_cache()
        keys, values = _random_kv(10)
        cache.update_and_fetch(keys, values)

        # Single token decode mask
        mask = cache.make_mask(N=1, return_array=True, window_size=None)
        assert mask is None  # N=1 with no window returns None

    def test_meta_state(self):
        cache = _make_cache()
        assert cache.meta_state == ""
        cache.meta_state = ""  # Should not raise


class TestCompression:
    """Test that compression actually happens and saves memory."""

    def test_buffer_flush_triggers(self):
        """When buffer exceeds buffer_size, tokens get compressed."""
        cache = _make_cache(buffer_size=16)

        # Add more than buffer_size tokens
        keys, values = _random_kv(20)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        assert cache._keys_compressed is not None
        assert cache.compressed_tokens > 0
        assert cache.buffer_tokens <= 16

    def test_memory_savings(self):
        """Compressed cache uses less memory than raw FP32 storage."""
        cache = _make_cache(buffer_size=16)

        # Add enough tokens to trigger compression
        keys, values = _random_kv(64)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        # Raw FP32 cost: 64 tokens * n_heads * head_dim * 4 bytes * 2 (k+v)
        raw_bytes = 64 * N_HEADS * HEAD_DIM * 4 * 2
        actual_bytes = cache.nbytes

        # TurboQuant should use significantly less
        assert actual_bytes < raw_bytes, (
            f"Cache bytes {actual_bytes} should be less than raw {raw_bytes}"
        )

    def test_memory_report(self):
        cache = _make_cache(buffer_size=16)
        keys, values = _random_kv(32)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        report = cache.memory_report()
        assert report["total_tokens"] == 32
        assert report["compressed_tokens"] > 0
        assert report["buffer_tokens"] <= 16
        assert report["total_bytes"] > 0
        assert report["compressed_keys_bytes"] > 0
        assert report["compressed_values_bytes"] > 0

    def test_buffer_portion_exact(self):
        """Buffer portion returned by update_and_fetch should match originals."""
        cache = _make_cache(buffer_size=16)

        keys, values = _random_kv(64)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        # Buffer should be last 16 tokens from the original
        orig_k = keys[:, :, -16:, :]
        mx.eval(orig_k)

        diff = float(mx.max(mx.abs(out_k - orig_k)))
        assert diff < 1e-5, f"Buffer keys should be exact, got diff={diff}"

    def test_decompressed_values_reasonable(self):
        """Legacy _get_all_values should still produce reasonable output."""
        cache = _make_cache(buffer_size=16)

        keys, values = _random_kv(64)
        cache.update_and_fetch(keys, values)

        # Use legacy method to get full decompressed values
        all_v = cache._get_all_values()
        mx.eval(all_v)

        assert all_v.shape == (BATCH, N_HEADS, 64, HEAD_DIM)

        # Buffer portion should be exact
        buf_v = all_v[:, :, -16:, :]
        orig_v = values[:, :, -16:, :]
        mx.eval(buf_v, orig_v)
        diff = float(mx.max(mx.abs(buf_v - orig_v)))
        assert diff < 1e-5, f"Buffer values should be exact, got diff={diff}"

    def test_multiple_flush_cycles(self):
        """Multiple buffer flushes should accumulate compressed tokens correctly."""
        cache = _make_cache(buffer_size=8)

        # Three rounds of adding tokens that trigger flushes
        for i in range(3):
            keys, values = _random_kv(12)
            out_k, out_v = cache.update_and_fetch(keys, values)
            mx.eval(out_k, out_v)

        assert cache.offset == 36
        # Buffer should be <= buffer_size
        assert out_k.shape[2] <= 8
        assert cache.compressed_tokens > 0


class TestPerLayerSeeds:
    """Test that different layers get different quantization seeds."""

    def test_different_layer_indices(self):
        """Caches with different layer_idx produce different compressed representations."""
        cache0 = _make_cache(layer_idx=0, buffer_size=8)
        cache1 = _make_cache(layer_idx=1, buffer_size=8)

        keys, values = _random_kv(16)

        cache0.update_and_fetch(keys, values)
        cache1.update_and_fetch(keys, values)

        # Use legacy method to compare full decompressed keys
        all_k0 = cache0._get_all_keys()
        all_k1 = cache1._get_all_keys()
        mx.eval(all_k0, all_k1)

        # Compressed (decompressed) portions should differ due to different seeds
        compressed_k0 = all_k0[:, :, :8, :]
        compressed_k1 = all_k1[:, :, :8, :]
        mx.eval(compressed_k0, compressed_k1)

        diff = float(mx.max(mx.abs(compressed_k0 - compressed_k1)))
        assert diff > 0, "Different layer seeds should produce different outputs"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token(self):
        """Single token should work (no compression triggered)."""
        cache = _make_cache()
        keys, values = _random_kv(1)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        assert cache.offset == 1
        assert cache._keys_compressed is None

    def test_exact_buffer_size(self):
        """Exactly buffer_size tokens: no flush should happen."""
        cache = _make_cache(buffer_size=32)
        keys, values = _random_kv(32)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        assert cache._keys_compressed is None
        assert cache.buffer_tokens == 32
        assert out_k.shape[2] == 32

    def test_buffer_size_plus_one(self):
        """buffer_size + 1 tokens should trigger a flush."""
        cache = _make_cache(buffer_size=32)
        keys, values = _random_kv(33)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        assert cache._keys_compressed is not None
        assert cache.compressed_tokens == 1
        assert cache.buffer_tokens == 32
        # update_and_fetch returns buffer only
        assert out_k.shape[2] == 32

    def test_large_prefill(self):
        """Large prefill triggers compression correctly."""
        cache = _make_cache(buffer_size=16)
        keys, values = _random_kv(256)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        assert cache.offset == 256
        # update_and_fetch returns buffer only
        assert out_k.shape[2] == 16
        assert cache.compressed_tokens == 240
        assert cache.buffer_tokens == 16


class TestCompressedScores:
    """Test Phase 2 compute_compressed_scores and get_decompressed_values."""

    def test_compute_compressed_scores_none_when_no_compressed(self):
        """Returns None when no tokens have been compressed."""
        cache = _make_cache()
        keys, values = _random_kv(10)
        cache.update_and_fetch(keys, values)

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        result = cache.compute_compressed_scores(queries)
        assert result is None

    def test_compute_compressed_scores_shape_decode(self):
        """Compressed scores have correct shape during decode (n_q=1)."""
        cache = _make_cache(buffer_size=16)
        keys, values = _random_kv(64)
        cache.update_and_fetch(keys, values)
        mx.eval(cache._key_buffer)

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        scores = cache.compute_compressed_scores(queries, scale=1.0)
        mx.eval(scores)

        assert scores is not None
        n_compressed = cache.compressed_tokens
        assert scores.shape == (BATCH, N_HEADS, 1, n_compressed)

    def test_compute_compressed_scores_shape_prefill(self):
        """Compressed scores have correct shape during prefill (n_q > 1)."""
        cache = _make_cache(buffer_size=8)
        # First, force some compressed data
        keys, values = _random_kv(16)
        cache.update_and_fetch(keys, values)
        mx.eval(cache._key_buffer)

        # Now query with multiple tokens
        n_q = 4
        queries = mx.random.normal((BATCH, N_HEADS, n_q, HEAD_DIM))
        mx.eval(queries)

        scores = cache.compute_compressed_scores(queries, scale=1.0)
        mx.eval(scores)

        assert scores is not None
        n_compressed = cache.compressed_tokens
        assert scores.shape == (BATCH, N_HEADS, n_q, n_compressed)

    def test_compressed_scores_vs_decompressed_matmul(self):
        """
        Compressed scores should approximate scores from decompressed keys.

        This validates that the Metal kernel path produces reasonable results.
        """
        cache = _make_cache(buffer_size=16)
        keys, values = _random_kv(64)
        cache.update_and_fetch(keys, values)

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        # Method 1: Compressed scores via Metal kernels
        compressed_scores = cache.compute_compressed_scores(queries, scale=1.0)
        mx.eval(compressed_scores)

        # Method 2: Decompressed keys matmul (legacy)
        decompressed_keys = cache.key_quantizer.dequantize(cache._keys_compressed)
        mx.eval(decompressed_keys)
        reference_scores = queries @ mx.transpose(decompressed_keys, (0, 1, 3, 2))
        mx.eval(reference_scores)

        # They should be close (same quantized data, different computation path)
        diff = float(mx.mean(mx.abs(compressed_scores - reference_scores)))
        assert diff < 0.5, (
            f"Metal kernel scores should match decompressed matmul, "
            f"got mean diff={diff}"
        )

    def test_get_decompressed_values_none_when_no_compressed(self):
        """Returns None when no values have been compressed."""
        cache = _make_cache()
        keys, values = _random_kv(10)
        cache.update_and_fetch(keys, values)

        result = cache.get_decompressed_values()
        assert result is None

    def test_get_decompressed_values_shape(self):
        """Decompressed values have correct shape."""
        cache = _make_cache(buffer_size=16)
        keys, values = _random_kv(64)
        cache.update_and_fetch(keys, values)

        decompressed = cache.get_decompressed_values()
        mx.eval(decompressed)

        assert decompressed is not None
        n_compressed = cache.compressed_tokens
        assert decompressed.shape == (BATCH, N_HEADS, n_compressed, HEAD_DIM)


class TestDirectAttention:
    """Test the full Phase 2 attention path (patched_sdpa)."""

    def test_full_attention_decode(self):
        """
        Full attention computation with compressed + buffer during decode.

        Verifies that the _turboquant_attention function produces
        reasonable output (not NaN, correct shape).
        """
        from mlx_turboquant.patch import _turboquant_attention

        cache = _make_cache(buffer_size=16)
        keys, values = _random_kv(64)
        buf_k, buf_v = cache.update_and_fetch(keys, values)
        mx.eval(buf_k, buf_v)

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        scale = 1.0 / math.sqrt(HEAD_DIM)
        output = _turboquant_attention(
            queries, buf_k, buf_v, cache, scale, mask=None
        )
        mx.eval(output)

        assert output.shape == (BATCH, N_HEADS, 1, HEAD_DIM)
        assert not mx.any(mx.isnan(output)).item()

    def test_full_attention_no_compressed(self):
        """Attention works when no compressed data exists (buffer only)."""
        from mlx_turboquant.patch import _turboquant_attention

        cache = _make_cache(buffer_size=32)
        keys, values = _random_kv(10)
        buf_k, buf_v = cache.update_and_fetch(keys, values)
        mx.eval(buf_k, buf_v)

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        scale = 1.0 / math.sqrt(HEAD_DIM)

        # No compressed data -- should still work via buffer-only path
        assert cache._keys_compressed is None
        output = _turboquant_attention(
            queries, buf_k, buf_v, cache, scale, mask=None
        )
        mx.eval(output)

        assert output.shape == (BATCH, N_HEADS, 1, HEAD_DIM)
        assert not mx.any(mx.isnan(output)).item()

    def test_attention_output_reasonable(self):
        """
        Compare Phase 2 attention output to legacy (full decompression) path.

        They should produce similar results since both operate on the same
        quantized data, just via different computation paths.
        """
        from mlx_turboquant.patch import _turboquant_attention

        cache_ph2 = _make_cache(buffer_size=16)
        keys, values = _random_kv(64)
        buf_k, buf_v = cache_ph2.update_and_fetch(keys, values)
        mx.eval(buf_k, buf_v)

        queries = mx.random.normal((BATCH, N_HEADS, 1, HEAD_DIM))
        mx.eval(queries)

        scale = 1.0 / math.sqrt(HEAD_DIM)

        # Phase 2: direct attention from compressed data
        output_ph2 = _turboquant_attention(
            queries, buf_k, buf_v, cache_ph2, scale, mask=None
        )
        mx.eval(output_ph2)

        # Legacy: full decompression then standard attention
        all_keys = cache_ph2._get_all_keys()
        all_values = cache_ph2._get_all_values()
        mx.eval(all_keys, all_values)
        legacy_scores = (queries @ mx.transpose(all_keys, (0, 1, 3, 2))) * scale
        legacy_weights = mx.softmax(legacy_scores, axis=-1, precise=True)
        output_legacy = legacy_weights @ all_values
        mx.eval(output_legacy)

        # Should be close (same quantized data, different paths)
        diff = float(mx.mean(mx.abs(output_ph2 - output_legacy)))
        assert diff < 0.5, (
            f"Phase 2 output should approximate legacy, got mean diff={diff}"
        )
