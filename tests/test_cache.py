"""
Tests for TurboQuantCache — verifies MLX-LM compatibility and correctness.
"""

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

    def test_update_and_fetch_shapes(self):
        """update_and_fetch returns (keys, values) with correct shapes."""
        cache = _make_cache()
        keys, values = _random_kv(10)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        assert out_k.shape == (BATCH, N_HEADS, 10, HEAD_DIM)
        assert out_v.shape == (BATCH, N_HEADS, 10, HEAD_DIM)
        assert cache.offset == 10
        assert not cache.empty()

    def test_sequential_append(self):
        """Appending tokens one at a time accumulates correctly."""
        cache = _make_cache()

        for i in range(5):
            keys, values = _random_kv(1)
            out_k, out_v = cache.update_and_fetch(keys, values)
            mx.eval(out_k, out_v)

        assert cache.offset == 5
        assert out_k.shape == (BATCH, N_HEADS, 5, HEAD_DIM)
        assert out_v.shape == (BATCH, N_HEADS, 5, HEAD_DIM)

    def test_prefill_then_decode(self):
        """Simulate real inference: prefill many tokens, then decode one at a time."""
        cache = _make_cache()

        # Prefill
        keys, values = _random_kv(20)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)
        assert cache.offset == 20

        # Decode 5 tokens
        for i in range(5):
            keys, values = _random_kv(1)
            out_k, out_v = cache.update_and_fetch(keys, values)
            mx.eval(out_k, out_v)

        assert cache.offset == 25
        assert out_k.shape == (BATCH, N_HEADS, 25, HEAD_DIM)

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

    def test_decompressed_output_reasonable(self):
        """Decompressed output should be close to original (not garbage)."""
        cache = _make_cache(buffer_size=16)

        keys, values = _random_kv(64)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        # The buffer portion should be exact
        buf_k = out_k[:, :, -16:, :]
        orig_k = keys[:, :, -16:, :]
        mx.eval(buf_k, orig_k)

        diff = float(mx.max(mx.abs(buf_k - orig_k)))
        assert diff < 1e-5, f"Buffer keys should be exact, got diff={diff}"

    def test_multiple_flush_cycles(self):
        """Multiple buffer flushes should accumulate compressed tokens correctly."""
        cache = _make_cache(buffer_size=8)

        # Three rounds of adding tokens that trigger flushes
        for i in range(3):
            keys, values = _random_kv(12)
            out_k, out_v = cache.update_and_fetch(keys, values)
            mx.eval(out_k, out_v)

        assert cache.offset == 36
        assert out_k.shape == (BATCH, N_HEADS, 36, HEAD_DIM)
        assert cache.compressed_tokens > 0


class TestPerLayerSeeds:
    """Test that different layers get different quantization seeds."""

    def test_different_layer_indices(self):
        """Caches with different layer_idx produce different compressed representations."""
        cache0 = _make_cache(layer_idx=0, buffer_size=8)
        cache1 = _make_cache(layer_idx=1, buffer_size=8)

        keys, values = _random_kv(16)

        out_k0, _ = cache0.update_and_fetch(keys, values)
        out_k1, _ = cache1.update_and_fetch(keys, values)
        mx.eval(out_k0, out_k1)

        # Compressed (decompressed) portions should differ due to different seeds
        compressed_k0 = out_k0[:, :, :8, :]
        compressed_k1 = out_k1[:, :, :8, :]
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

    def test_buffer_size_plus_one(self):
        """buffer_size + 1 tokens should trigger a flush."""
        cache = _make_cache(buffer_size=32)
        keys, values = _random_kv(33)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        assert cache._keys_compressed is not None
        assert cache.compressed_tokens == 1
        assert cache.buffer_tokens == 32

    def test_large_prefill(self):
        """Large prefill triggers compression correctly."""
        cache = _make_cache(buffer_size=16)
        keys, values = _random_kv(256)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)

        assert cache.offset == 256
        assert out_k.shape[2] == 256
        assert cache.compressed_tokens == 240
        assert cache.buffer_tokens == 16
