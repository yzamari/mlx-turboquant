"""
TurboQuantCache: MLX-LM compatible KV cache with TurboQuant compression.

Subclasses mlx_lm's _BaseCache so it can be used as a drop-in replacement.
Keys are compressed via TurboQuant product quantization (Algorithm 2),
values via asymmetric group quantization. A recent-token buffer is kept
uncompressed for quality.

The model sees plain mx.array from update_and_fetch(), so attention
proceeds via the standard mx.fast.scaled_dot_product_attention path
(no special quantized matmul needed).
"""

import mlx.core as mx
from mlx_lm.models.cache import _BaseCache, create_attention_mask

from turboquant_mac.quantizer import TurboQuantProd, ProdQuantized
from turboquant_mac.kv_cache import (
    quantize_values,
    dequantize_values,
    ValueQuantized,
)


class TurboQuantCache(_BaseCache):
    """
    KV cache that transparently compresses via TurboQuant.

    Keys: TurboQuant product quantization (MSE + QJL residual).
    Values: asymmetric group quantization.
    Buffer: most recent tokens kept in FP16/FP32 for quality.

    Compatible with MLX-LM's cache interface:
        keys, values = cache.update_and_fetch(keys, values)
    returns decompressed mx.array that the model uses directly.
    """

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.buffer_size = buffer_size
        self.layer_idx = layer_idx

        # Key quantizer (per-layer seed for diversity)
        self.key_quantizer = TurboQuantProd(
            dim=head_dim,
            bits=key_bits,
            seed=42 + layer_idx * 7,
            backend="mlx",
        )

        # State
        self.offset = 0
        self._keys_compressed: ProdQuantized | None = None
        self._values_compressed: ValueQuantized | None = None
        self._key_buffer: mx.array | None = None
        self._value_buffer: mx.array | None = None

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Store new KV pairs (compressing old ones) and return
        full accumulated KV (decompressed) for attention.

        Args:
            keys: (B, n_kv_heads, n_new, head_dim)
            values: (B, n_kv_heads, n_new, head_dim)

        Returns:
            (all_keys, all_values) as plain mx.array for attention.
        """
        n_new = keys.shape[2]
        self.offset += n_new

        # Append to buffer
        if self._key_buffer is not None:
            self._key_buffer = mx.concatenate(
                [self._key_buffer, keys], axis=2
            )
            self._value_buffer = mx.concatenate(
                [self._value_buffer, values], axis=2
            )
        else:
            self._key_buffer = keys
            self._value_buffer = values

        # Flush oldest buffer tokens to compressed storage when buffer overflows
        if self._key_buffer.shape[2] > self.buffer_size:
            self._flush()

        # Return decompressed full cache
        all_keys = self._get_all_keys()
        all_values = self._get_all_values()
        return all_keys, all_values

    def _flush(self):
        """Compress oldest buffer tokens into quantized storage."""
        n_flush = self._key_buffer.shape[2] - self.buffer_size
        if n_flush <= 0:
            return

        keys_to_compress = self._key_buffer[:, :, :n_flush, :]
        values_to_compress = self._value_buffer[:, :, :n_flush, :]
        self._key_buffer = self._key_buffer[:, :, n_flush:, :]
        self._value_buffer = self._value_buffer[:, :, n_flush:, :]

        # Compress keys with TurboQuant product quantization
        new_key_q = self.key_quantizer.quantize(keys_to_compress)

        # Compress values with group quantization
        new_val_q = quantize_values(
            values_to_compress,
            bits=self.value_bits,
            group_size=self.value_group_size,
            backend="mlx",
        )

        if self._keys_compressed is None:
            self._keys_compressed = new_key_q
            self._values_compressed = new_val_q
        else:
            # Concatenate compressed storage along sequence dimension
            self._keys_compressed = ProdQuantized(
                mse_indices=mx.concatenate(
                    [self._keys_compressed.mse_indices, new_key_q.mse_indices],
                    axis=-2,
                ),
                qjl_signs=mx.concatenate(
                    [self._keys_compressed.qjl_signs, new_key_q.qjl_signs],
                    axis=-2,
                ),
                residual_norms=mx.concatenate(
                    [
                        self._keys_compressed.residual_norms,
                        new_key_q.residual_norms,
                    ],
                    axis=-1,
                ),
                norms=mx.concatenate(
                    [self._keys_compressed.norms, new_key_q.norms],
                    axis=-1,
                ),
                mse_bits=new_key_q.mse_bits,
            )
            self._values_compressed = ValueQuantized(
                data=mx.concatenate(
                    [self._values_compressed.data, new_val_q.data], axis=-2
                ),
                scales=mx.concatenate(
                    [self._values_compressed.scales, new_val_q.scales],
                    axis=-2,
                ),
                zeros=mx.concatenate(
                    [self._values_compressed.zeros, new_val_q.zeros],
                    axis=-2,
                ),
                bits=self.value_bits,
            )

    def _get_all_keys(self) -> mx.array:
        """Return decompressed keys: compressed + buffer."""
        parts = []
        if self._keys_compressed is not None:
            k_decompressed = self.key_quantizer.dequantize(
                self._keys_compressed
            )
            parts.append(k_decompressed)
        if self._key_buffer is not None:
            parts.append(self._key_buffer)
        if len(parts) == 0:
            raise RuntimeError("Cache is empty, cannot get keys")
        return mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0]

    def _get_all_values(self) -> mx.array:
        """Return decompressed values: compressed + buffer."""
        parts = []
        if self._values_compressed is not None:
            v_decompressed = dequantize_values(
                self._values_compressed,
                self.value_group_size,
                backend="mlx",
            )
            parts.append(v_decompressed)
        if self._value_buffer is not None:
            parts.append(self._value_buffer)
        if len(parts) == 0:
            raise RuntimeError("Cache is empty, cannot get values")
        return mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0]

    # ---- _BaseCache interface ----

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @property
    def state(self):
        """Return serializable state (buffer only; compressed not easily serializable)."""
        return self._key_buffer, self._value_buffer

    @state.setter
    def state(self, v):
        if v is not None and v:
            self._key_buffer, self._value_buffer = v
            if self._key_buffer is not None:
                self.offset = self._key_buffer.shape[2]

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass

    def empty(self):
        return self._key_buffer is None and self._keys_compressed is None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def size(self):
        return self.offset

    @property
    def nbytes(self) -> int:
        """Total memory used by compressed + buffer storage."""
        total = 0
        if self._keys_compressed is not None:
            kc = self._keys_compressed
            total += kc.mse_indices.nbytes + kc.qjl_signs.nbytes
            total += kc.residual_norms.nbytes + kc.norms.nbytes
        if self._values_compressed is not None:
            vc = self._values_compressed
            total += vc.data.nbytes + vc.scales.nbytes + vc.zeros.nbytes
        if self._key_buffer is not None:
            total += self._key_buffer.nbytes
        if self._value_buffer is not None:
            total += self._value_buffer.nbytes
        return total

    @property
    def compressed_tokens(self) -> int:
        """Number of tokens in compressed storage."""
        if self._keys_compressed is None:
            return 0
        return self._keys_compressed.mse_indices.shape[-2]

    @property
    def buffer_tokens(self) -> int:
        """Number of tokens in the uncompressed buffer."""
        if self._key_buffer is None:
            return 0
        return self._key_buffer.shape[2]

    def memory_report(self) -> dict:
        """Detailed memory usage breakdown."""
        report = {
            "compressed_keys_bytes": 0,
            "compressed_values_bytes": 0,
            "buffer_bytes": 0,
            "total_bytes": 0,
            "compressed_tokens": self.compressed_tokens,
            "buffer_tokens": self.buffer_tokens,
            "total_tokens": self.offset,
        }
        if self._keys_compressed is not None:
            kc = self._keys_compressed
            report["compressed_keys_bytes"] = (
                kc.mse_indices.nbytes
                + kc.qjl_signs.nbytes
                + kc.residual_norms.nbytes
                + kc.norms.nbytes
            )
        if self._values_compressed is not None:
            vc = self._values_compressed
            report["compressed_values_bytes"] = (
                vc.data.nbytes + vc.scales.nbytes + vc.zeros.nbytes
            )
        if self._key_buffer is not None:
            report["buffer_bytes"] += self._key_buffer.nbytes
        if self._value_buffer is not None:
            report["buffer_bytes"] += self._value_buffer.nbytes
        report["total_bytes"] = (
            report["compressed_keys_bytes"]
            + report["compressed_values_bytes"]
            + report["buffer_bytes"]
        )
        return report
