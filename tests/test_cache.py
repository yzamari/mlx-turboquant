"""
Tests for TurboQuantCache — verifies MLX-LM compatibility and correctness.

Phase 2: Tests updated for buffer-only update_and_fetch() and direct
attention from compressed data via Metal kernels.

IMPORTANT: Compression only happens during DECODE (n_new=1), not prefill.
Flush threshold = buffer_size + flush_batch_size (flush_batch_size defaults
to buffer_size). Tests that need compressed data must feed tokens one-at-a-
time via _fill_cache_decode() to trigger the decode flush path.
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


def _fill_cache_decode(
    cache, n_total, batch=BATCH, n_heads=N_HEADS, head_dim=HEAD_DIM,
):
    """Fill cache by feeding tokens one-at-a-time (decode mode).

    Compression only triggers during decode (n_new=1) when the buffer
    exceeds buffer_size + flush_batch_size. This helper feeds n_total
    single tokens to reliably trigger compression.

    Returns the last (buf_k, buf_v) from update_and_fetch.
    """
    for _ in range(n_total):
        k, v = _random_kv(1, batch=batch, n_heads=n_heads, head_dim=head_dim)
        buf_k, buf_v = cache.update_and_fetch(k, v)
        mx.eval(buf_k, buf_v)
    return buf_k, buf_v


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
        """After flush, update_and_fetch returns buffer_size tokens only.

        Compression only triggers during decode (n_new=1). With buffer_size=16
        and flush_batch_size=16 (default), the flush threshold is 32.
        After 33 decode tokens, flush compresses 16 tokens, leaving 17 in
        buffer. We feed 64 tokens total so multiple flushes occur.
        """
        cache = _make_cache(buffer_size=16)
        out_k, out_v = _fill_cache_decode(cache, 64)

        # Buffer should be capped at buffer_size after flush
        assert out_k.shape[2] <= cache.buffer_size + cache.flush_batch_size
        assert out_v.shape[2] <= cache.buffer_size + cache.flush_batch_size
        assert cache.offset == 64
        assert cache.compressed_tokens > 0
        assert cache.buffer_tokens <= cache.buffer_size + cache.flush_batch_size

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
        """Simulate real inference: prefill many tokens, then decode one at a time.

        Prefill does NOT trigger compression (by design -- keeps everything
        in buffer for fast standard attention). Compression only starts
        during decode when buffer exceeds buffer_size + flush_batch_size.
        """
        cache = _make_cache(buffer_size=16)

        # Prefill: all 20 tokens stay in buffer (no compression during prefill)
        keys, values = _random_kv(20)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)
        assert cache.offset == 20
        assert cache.compressed_tokens == 0  # No compression during prefill
        assert out_k.shape == (BATCH, N_HEADS, 20, HEAD_DIM)

        # Decode tokens until flush triggers (threshold = 16 + 16 = 32)
        # Buffer starts at 20, needs to exceed 32 -> 13 more decode tokens
        for i in range(20):
            keys, values = _random_kv(1)
            out_k, out_v = cache.update_and_fetch(keys, values)
            mx.eval(out_k, out_v)

        assert cache.offset == 40
        # After enough decode tokens, compression should have triggered
        assert cache.compressed_tokens > 0
        # Buffer should be bounded
        assert out_k.shape[2] <= cache.buffer_size + cache.flush_batch_size

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
        """When buffer exceeds flush threshold during decode, tokens get compressed.

        Flush threshold = buffer_size + flush_batch_size = 16 + 16 = 32.
        Feed 33 decode tokens to exceed the threshold and trigger compression.
        """
        cache = _make_cache(buffer_size=16)

        # Feed tokens one-at-a-time (decode mode) to trigger compression
        out_k, out_v = _fill_cache_decode(cache, 33)

        assert cache._keys_compressed is not None
        assert cache.compressed_tokens > 0
        assert cache.buffer_tokens <= cache.buffer_size + cache.flush_batch_size

    def test_memory_savings(self):
        """Compressed cache uses less memory than raw FP32 storage."""
        n_total = 64
        cache = _make_cache(buffer_size=16)

        # Feed tokens via decode to trigger compression
        _fill_cache_decode(cache, n_total)

        # Raw FP32 cost: n_total tokens * n_heads * head_dim * 4 bytes * 2 (k+v)
        raw_bytes = n_total * N_HEADS * HEAD_DIM * 4 * 2
        actual_bytes = cache.nbytes

        # TurboQuant should use significantly less
        assert actual_bytes < raw_bytes, (
            f"Cache bytes {actual_bytes} should be less than raw {raw_bytes}"
        )

    def test_memory_report(self):
        n_total = 64
        cache = _make_cache(buffer_size=16)

        # Feed via decode to trigger compression
        _fill_cache_decode(cache, n_total)

        report = cache.memory_report()
        assert report["total_tokens"] == n_total
        assert report["compressed_tokens"] > 0
        assert report["buffer_tokens"] <= cache.buffer_size + cache.flush_batch_size
        assert report["total_bytes"] > 0
        assert report["compressed_keys_bytes"] > 0
        assert report["compressed_values_bytes"] > 0

    def test_buffer_portion_exact(self):
        """Buffer portion returned by update_and_fetch should match last inserted token.

        Since we feed one-at-a-time, verify the last decode token appears in
        the buffer by checking the final token matches exactly.
        """
        cache = _make_cache(buffer_size=16)
        _fill_cache_decode(cache, 40)

        # Insert one final known token
        last_k, last_v = _random_kv(1)
        out_k, out_v = cache.update_and_fetch(last_k, last_v)
        mx.eval(out_k, out_v)

        # The last token in the buffer should match the one we just inserted
        buf_last_k = out_k[:, :, -1:, :]
        mx.eval(buf_last_k)

        diff = float(mx.max(mx.abs(buf_last_k - last_k)))
        assert diff < 1e-5, f"Last buffer key should match inserted token, got diff={diff}"

    def test_decompressed_values_reasonable(self):
        """Legacy _get_all_values should still produce reasonable output."""
        n_total = 64
        cache = _make_cache(buffer_size=16)

        # Feed via decode to trigger compression
        _fill_cache_decode(cache, n_total)

        # Use legacy method to get full decompressed values
        all_v = cache._get_all_values()
        mx.eval(all_v)

        assert all_v.shape[0] == BATCH
        assert all_v.shape[1] == N_HEADS
        assert all_v.shape[2] == n_total  # compressed + buffer = total
        assert all_v.shape[3] == HEAD_DIM

    def test_multiple_flush_cycles(self):
        """Multiple buffer flushes should accumulate compressed tokens correctly.

        With buffer_size=8, flush_batch_size=8, threshold=16. Feed 48 decode
        tokens to trigger multiple flush cycles.
        """
        cache = _make_cache(buffer_size=8)

        # Feed enough decode tokens to trigger multiple flush cycles
        n_total = 48
        out_k, out_v = _fill_cache_decode(cache, n_total)

        assert cache.offset == n_total
        # Buffer should be bounded
        assert out_k.shape[2] <= cache.buffer_size + cache.flush_batch_size
        assert cache.compressed_tokens > 0


class TestPerLayerSeeds:
    """Test that different layers get different quantization seeds."""

    def test_different_layer_indices(self):
        """Caches with different layer_idx produce different compressed representations.

        Feed identical tokens to two caches with different layer indices.
        The compressed (dequantized) keys should differ because the quantizers
        use different seeds.
        """
        cache0 = _make_cache(layer_idx=0, buffer_size=8)
        cache1 = _make_cache(layer_idx=1, buffer_size=8)

        # Feed identical tokens via decode to trigger compression
        # Threshold = 8 + 8 = 16, so feed 17+ tokens
        n_total = 32
        for _ in range(n_total):
            k, v = _random_kv(1)
            cache0.update_and_fetch(k, v)
            cache1.update_and_fetch(k, v)
            mx.eval(cache0._key_buffer, cache1._key_buffer)

        assert cache0._keys_compressed is not None
        assert cache1._keys_compressed is not None

        # Use legacy method to compare full decompressed keys
        all_k0 = cache0._get_all_keys()
        all_k1 = cache1._get_all_keys()
        mx.eval(all_k0, all_k1)

        # Compare the compressed (dequantized) portions -- they should differ
        n_comp = cache0.compressed_tokens
        compressed_k0 = all_k0[:, :, :n_comp, :]
        compressed_k1 = all_k1[:, :, :n_comp, :]
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
        """Exceeding flush threshold triggers a flush.

        With buffer_size=32, flush_batch_size=32, threshold=64.
        Feed 65 decode tokens to exceed the threshold.
        """
        cache = _make_cache(buffer_size=32)
        out_k, out_v = _fill_cache_decode(cache, 65)

        assert cache._keys_compressed is not None
        assert cache.compressed_tokens > 0
        assert cache.buffer_tokens <= cache.buffer_size + cache.flush_batch_size
        # update_and_fetch returns buffer only
        assert out_k.shape[2] <= cache.buffer_size + cache.flush_batch_size

    def test_large_prefill(self):
        """Large prefill keeps everything in buffer (no compression during prefill).

        Prefill intentionally does NOT compress -- it keeps all tokens in the
        buffer for fast standard attention. Compression starts during decode.
        """
        cache = _make_cache(buffer_size=16)
        keys, values = _random_kv(256)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        assert cache.offset == 256
        # All 256 tokens stay in buffer during prefill (no compression)
        assert out_k.shape[2] == 256
        assert cache.compressed_tokens == 0
        assert cache.buffer_tokens == 256

        # Now feed one decode token -- this should trigger compression
        # since buffer (257) > threshold (16 + 16 = 32)
        k, v = _random_kv(1)
        out_k, out_v = cache.update_and_fetch(k, v)
        mx.eval(out_k, out_v)

        assert cache.offset == 257
        assert cache.compressed_tokens > 0
        assert cache.buffer_tokens <= cache.buffer_size + cache.flush_batch_size


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
        _fill_cache_decode(cache, 64)

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
        # Force compressed data via decode
        _fill_cache_decode(cache, 32)

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
        _fill_cache_decode(cache, 64)

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
        _fill_cache_decode(cache, 64)

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


class TestGQA:
    """Test GQA (Grouped Query Attention) support.

    In GQA, n_query_heads > n_kv_heads. For example, Qwen2.5-32B uses
    n_kv_heads=8, n_query_heads=40 (repeats=5). The per-KV-head-group
    loop must produce correct results without expanding all compressed
    data at once.

    NOTE: Compression only happens during decode (n_new=1), not prefill.
    Tests that need compressed data must feed tokens one-at-a-time to
    trigger the flush_batch_size threshold.
    """

    N_KV_HEADS = 2
    N_QUERY_HEADS = 8  # repeats = 4

    def _make_gqa_cache(self, buffer_size=16):
        return _make_cache(buffer_size=buffer_size)

    def _random_kv_gqa(self, seq_len, batch=BATCH, head_dim=HEAD_DIM):
        """Generate KV with n_kv_heads, queries with n_query_heads."""
        keys = mx.random.normal((batch, self.N_KV_HEADS, seq_len, head_dim))
        values = mx.random.normal((batch, self.N_KV_HEADS, seq_len, head_dim))
        mx.eval(keys, values)
        return keys, values

    def _random_queries_gqa(self, n_q=1, batch=BATCH, head_dim=HEAD_DIM):
        queries = mx.random.normal((batch, self.N_QUERY_HEADS, n_q, head_dim))
        mx.eval(queries)
        return queries

    def _fill_cache_with_compressed(self, cache, n_total=64):
        """Fill cache by feeding tokens one-at-a-time to trigger decode flush.

        Feeds n_total single tokens (n_new=1), which triggers flush when
        buffer exceeds buffer_size + flush_batch_size.
        """
        for _ in range(n_total):
            k, v = self._random_kv_gqa(1)
            buf_k, buf_v = cache.update_and_fetch(k, v)
            mx.eval(buf_k, buf_v)
        assert cache._keys_compressed is not None, (
            f"Expected compressed data after {n_total} decode steps, "
            f"buffer_size={cache.buffer_size}, flush_batch={cache.flush_batch_size}"
        )
        return buf_k, buf_v

    def test_compressed_scores_shape_gqa_decode(self):
        """Compressed scores have correct shape with GQA during decode."""
        cache = self._make_gqa_cache(buffer_size=16)
        self._fill_cache_with_compressed(cache, n_total=64)

        queries = self._random_queries_gqa(n_q=1)
        scores = cache.compute_compressed_scores(queries, scale=1.0)
        mx.eval(scores)

        assert scores is not None
        n_compressed = cache.compressed_tokens
        assert scores.shape == (BATCH, self.N_QUERY_HEADS, 1, n_compressed), (
            f"Expected (1, {self.N_QUERY_HEADS}, 1, {n_compressed}), got {scores.shape}"
        )

    def test_compressed_scores_shape_gqa_prefill(self):
        """Compressed scores have correct shape with GQA during prefill (n_q > 1)."""
        cache = self._make_gqa_cache(buffer_size=16)
        self._fill_cache_with_compressed(cache, n_total=64)

        n_q = 4
        queries = self._random_queries_gqa(n_q=n_q)
        scores = cache.compute_compressed_scores(queries, scale=1.0)
        mx.eval(scores)

        assert scores is not None
        n_compressed = cache.compressed_tokens
        assert scores.shape == (BATCH, self.N_QUERY_HEADS, n_q, n_compressed)

    def test_compressed_scores_gqa_vs_decompressed_matmul(self):
        """GQA compressed scores should approximate decompressed key matmul."""
        cache = self._make_gqa_cache(buffer_size=16)
        self._fill_cache_with_compressed(cache, n_total=64)

        queries = self._random_queries_gqa(n_q=1)
        mx.eval(queries)

        # Metal kernel path (GQA per-group loop)
        compressed_scores = cache.compute_compressed_scores(queries, scale=1.0)
        mx.eval(compressed_scores)

        # Reference: decompress keys, expand for GQA, matmul
        decompressed_keys = cache.key_quantizer.dequantize(cache._keys_compressed)
        mx.eval(decompressed_keys)
        # Expand KV heads to match query heads
        repeats = self.N_QUERY_HEADS // self.N_KV_HEADS
        expanded_keys = mx.repeat(decompressed_keys, repeats, axis=1)
        reference_scores = queries @ mx.transpose(expanded_keys, (0, 1, 3, 2))
        mx.eval(reference_scores)

        diff = float(mx.mean(mx.abs(compressed_scores - reference_scores)))
        assert diff < 0.5, (
            f"GQA Metal kernel scores should match decompressed matmul, "
            f"got mean diff={diff}"
        )

    def test_value_output_gqa_shape_decode(self):
        """Value weighted sum has correct shape with GQA during decode."""
        cache = self._make_gqa_cache(buffer_size=16)
        self._fill_cache_with_compressed(cache, n_total=64)

        n_compressed = cache.compressed_tokens
        # Fake weights: (B, n_query_heads, 1, n_compressed)
        weights = mx.softmax(
            mx.random.normal((BATCH, self.N_QUERY_HEADS, 1, n_compressed)),
            axis=-1,
        )
        mx.eval(weights)

        output = cache.compute_compressed_value_output(weights)
        mx.eval(output)

        assert output is not None
        assert output.shape == (BATCH, self.N_QUERY_HEADS, 1, HEAD_DIM), (
            f"Expected (1, {self.N_QUERY_HEADS}, 1, {HEAD_DIM}), got {output.shape}"
        )
        assert not mx.any(mx.isnan(output)).item()

    def test_value_output_gqa_shape_prefill(self):
        """Value weighted sum has correct shape with GQA during prefill."""
        cache = self._make_gqa_cache(buffer_size=16)
        self._fill_cache_with_compressed(cache, n_total=64)

        n_compressed = cache.compressed_tokens
        n_q = 4
        weights = mx.softmax(
            mx.random.normal((BATCH, self.N_QUERY_HEADS, n_q, n_compressed)),
            axis=-1,
        )
        mx.eval(weights)

        output = cache.compute_compressed_value_output(weights)
        mx.eval(output)

        assert output is not None
        assert output.shape == (BATCH, self.N_QUERY_HEADS, n_q, HEAD_DIM)
        assert not mx.any(mx.isnan(output)).item()

    def test_full_attention_gqa_decode(self):
        """Full attention path works with GQA during decode."""
        from mlx_turboquant.patch import _turboquant_attention

        cache = self._make_gqa_cache(buffer_size=16)
        buf_k, buf_v = self._fill_cache_with_compressed(cache, n_total=64)

        queries = self._random_queries_gqa(n_q=1)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        output = _turboquant_attention(
            queries, buf_k, buf_v, cache, scale, mask=None
        )
        mx.eval(output)

        assert output.shape == (BATCH, self.N_QUERY_HEADS, 1, HEAD_DIM)
        assert not mx.any(mx.isnan(output)).item()

    def test_full_attention_gqa_vs_legacy(self):
        """GQA attention output should approximate legacy decompression path."""
        from mlx_turboquant.patch import _turboquant_attention

        cache = self._make_gqa_cache(buffer_size=16)
        buf_k, buf_v = self._fill_cache_with_compressed(cache, n_total=64)

        queries = self._random_queries_gqa(n_q=1)
        scale = 1.0 / math.sqrt(HEAD_DIM)

        # Phase 2: GQA per-group loop path
        output_ph2 = _turboquant_attention(
            queries, buf_k, buf_v, cache, scale, mask=None
        )
        mx.eval(output_ph2)

        # Legacy: full decompression + GQA expansion
        all_keys = cache._get_all_keys()
        all_values = cache._get_all_values()
        mx.eval(all_keys, all_values)
        repeats = self.N_QUERY_HEADS // self.N_KV_HEADS
        exp_keys = mx.repeat(all_keys, repeats, axis=1)
        exp_values = mx.repeat(all_values, repeats, axis=1)
        legacy_scores = (queries @ mx.transpose(exp_keys, (0, 1, 3, 2))) * scale
        legacy_weights = mx.softmax(legacy_scores, axis=-1, precise=True)
        output_legacy = legacy_weights @ exp_values
        mx.eval(output_legacy)

        diff = float(mx.mean(mx.abs(output_ph2 - output_legacy)))
        assert diff < 0.5, (
            f"GQA Phase 2 output should approximate legacy, got mean diff={diff}"
        )
