"""
TurboQuantCache: MLX-LM compatible KV cache with TurboQuant compression.

Subclasses mlx_lm's _BaseCache so it can be used as a drop-in replacement.
Keys are compressed via TurboQuant product quantization (Algorithm 2),
values via asymmetric group quantization. A recent-token buffer is kept
uncompressed for quality.

Phase 2: update_and_fetch() returns ONLY the buffer (uncompressed recent
tokens). A patched scaled_dot_product_attention computes attention scores
for compressed keys directly from packed data via Metal kernels, avoiding
the decompress-every-step bottleneck.
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

    Phase 2 architecture:
        keys, values = cache.update_and_fetch(keys, values)
    returns ONLY the buffer (uncompressed recent tokens). The patched
    scaled_dot_product_attention handles compressed data via Metal kernels.
    """

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        flush_batch_size: int = 0,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.buffer_size = buffer_size
        # Batch flush: accumulate this many tokens before compressing all at once.
        # Default: same as buffer_size (128). This means flush happens every 128
        # decode steps, compressing 128 tokens in one GPU-saturated batch, instead
        # of compressing 1 token per step with 3 underutilized d×d matmuls.
        self.flush_batch_size = flush_batch_size or buffer_size
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
        ONLY the buffer (uncompressed recent tokens).

        The patched scaled_dot_product_attention will handle compressed
        data separately via Metal kernels.

        Args:
            keys: (B, n_kv_heads, n_new, head_dim)
            values: (B, n_kv_heads, n_new, head_dim)

        Returns:
            (buffer_keys, buffer_values) -- only the uncompressed buffer.
            The patched attention function accesses compressed data via
            cache._keys_compressed / cache._values_compressed.
        """
        n_new = keys.shape[2]
        self.offset += n_new
        is_prefill = n_new > 1

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

        # During prefill: DON'T compress. Keep everything in buffer for fast
        # standard attention. Only start compressing during decode (n_new=1)
        # when buffer overflows past the batch threshold.
        #
        # CRITICAL OPTIMIZATION: flush in BATCHES, not one token at a time.
        # With buffer_size=128 and flush_batch_size=128, the effective buffer
        # is 256 tokens. When it hits 257, we flush 128 tokens in one batch.
        # This means the 3 dense d×d matmuls (rotation, inverse rotation, QJL)
        # operate on (128, d) instead of (1, d) — GPU-saturated instead of
        # catastrophically underutilized. Amortized cost drops ~100x.
        flush_threshold = self.buffer_size + self.flush_batch_size
        if not is_prefill and self._key_buffer.shape[2] > flush_threshold:
            self._flush()

        # Return ONLY the buffer -- compressed data is handled by patched attention
        return self._key_buffer, self._value_buffer

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
        """Return decompressed keys: compressed + buffer (legacy, used by tests)."""
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
        """Return decompressed values: compressed + buffer (legacy, used by tests)."""
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

    # ---- Phase 2: Direct attention from compressed data ----

    def compute_compressed_scores(
        self, queries: mx.array, scale: float = 1.0
    ) -> mx.array | None:
        """
        Compute attention scores for compressed keys using Metal kernels.

        Avoids decompressing keys entirely -- scores are computed directly
        from packed 3-bit data on the GPU.

        Args:
            queries: (B, n_heads, n_q, head_dim) query vectors.
            scale: attention scale factor (typically 1/sqrt(d)).

        Returns:
            scores: (B, n_heads, n_q, n_compressed) attention logits,
            or None if no compressed data exists.
        """
        if self._keys_compressed is None:
            return None

        from turboquant_mac.backends.metal_kernels import (
            turboquant_attention_score_metal,
        )

        B, n_heads, n_q, D = queries.shape
        kc = self._keys_compressed

        # Retrieve quantizer matrices (precomputed, shared across tokens)
        Pi = self.key_quantizer.mse_quantizer.Pi
        S = self.key_quantizer.S
        centroids = self.key_quantizer.mse_quantizer.centroids
        mse_bits = self.key_quantizer.mse_quantizer.bits
        qjl_scale = self.key_quantizer.qjl_scale

        # Handle GQA: n_heads (query) may differ from n_kv_heads (key)
        n_kv_heads = kc.mse_indices.shape[1]
        repeats = n_heads // n_kv_heads

        def _expand_kv_head(t, axis=1):
            """Repeat KV head dimension to match query heads."""
            if repeats == 1:
                return t
            return mx.repeat(t, repeats, axis=axis)

        # Expand compressed data to match query head count
        mse_packed = _expand_kv_head(kc.mse_indices)
        qjl_signs = _expand_kv_head(kc.qjl_signs)
        norms = _expand_kv_head(kc.norms, axis=-1)
        res_norms = _expand_kv_head(kc.residual_norms, axis=-1)

        # Metal kernels operate on flat (BH, ...) tensors
        n_compressed = mse_packed.shape[2]
        mse_flat = mse_packed.reshape(B * n_heads, n_compressed, -1)
        qjl_flat = qjl_signs.reshape(B * n_heads, n_compressed, -1)
        norms_flat = norms.reshape(B * n_heads, n_compressed)
        res_norms_flat = res_norms.reshape(B * n_heads, n_compressed)

        if n_q == 1:
            # Decode path: single query token, optimal for Metal kernels
            q_flat = queries.reshape(B * n_heads, D)

            scores_flat = turboquant_attention_score_metal(
                query=q_flat,
                mse_packed=mse_flat,
                qjl_signs=qjl_flat,
                norms=norms_flat,
                residual_norms=res_norms_flat,
                Pi=Pi,
                S=S,
                centroids=centroids,
                mse_bits=mse_bits,
                qjl_scale=qjl_scale,
            )
            # (BH, N) -> (B, n_heads, 1, N)
            scores = scores_flat.reshape(B, n_heads, 1, n_compressed) * scale
        else:
            # Prefill path: multiple query tokens -- loop over each
            all_scores = []
            for qi in range(n_q):
                q_flat = queries[:, :, qi, :].reshape(B * n_heads, D)
                s = turboquant_attention_score_metal(
                    query=q_flat,
                    mse_packed=mse_flat,
                    qjl_signs=qjl_flat,
                    norms=norms_flat,
                    residual_norms=res_norms_flat,
                    Pi=Pi,
                    S=S,
                    centroids=centroids,
                    mse_bits=mse_bits,
                    qjl_scale=qjl_scale,
                )
                all_scores.append(s)
            # Stack: list of (BH, N) -> (BH, n_q, N) -> (B, H, n_q, N)
            scores_flat = mx.stack(all_scores, axis=1)
            scores = scores_flat.reshape(B, n_heads, n_q, n_compressed) * scale

        return scores

    def get_decompressed_values(self) -> mx.array | None:
        """
        Decompress values from quantized storage (called once per step).
        LEGACY: prefer compute_compressed_value_output() for zero-decompression path.

        Returns:
            (B, n_kv_heads, n_compressed, head_dim) or None.
        """
        if self._values_compressed is None:
            return None
        return dequantize_values(
            self._values_compressed,
            self.value_group_size,
            backend="mlx",
        )

    def compute_compressed_value_output(
        self, weights: mx.array
    ) -> mx.array | None:
        """
        Compute weighted sum of compressed values WITHOUT decompressing them.

        Uses a fused Metal kernel that dequantizes values on-the-fly in GPU
        registers. The full (B, H, N, D) decompressed value tensor is NEVER
        materialized in memory.

        Args:
            weights: (B, n_heads, n_q, n_compressed) attention weights
                     (already softmaxed, sliced to compressed portion only).

        Returns:
            (B, n_heads, n_q, head_dim) weighted sum output,
            or None if no compressed values exist.
        """
        if self._values_compressed is None:
            return None

        from turboquant_mac.backends.metal_kernels import (
            turboquant_value_weighted_sum_metal,
        )

        vc = self._values_compressed
        B, n_heads, n_q, n_compressed = weights.shape
        D = self.head_dim

        # Handle GQA: expand compressed values heads to match query heads
        n_kv_heads = vc.data.shape[1]
        repeats = n_heads // n_kv_heads

        def _expand_kv_head(t, axis=1):
            if repeats == 1:
                return t
            return mx.repeat(t, repeats, axis=axis)

        val_data = _expand_kv_head(vc.data)       # (B, n_heads, N, packed_d)
        val_scales = _expand_kv_head(vc.scales)    # (B, n_heads, N, n_groups)
        val_zeros = _expand_kv_head(vc.zeros)      # (B, n_heads, N, n_groups)

        if n_q == 1:
            # Decode path: single query, optimal
            # Flatten batch and heads: (B*H, N_compressed)
            w_flat = weights.reshape(B * n_heads, n_compressed)
            vd_flat = val_data.reshape(B * n_heads, n_compressed, -1)
            vs_flat = val_scales.reshape(B * n_heads, n_compressed, -1)
            vz_flat = val_zeros.reshape(B * n_heads, n_compressed, -1)

            # Fused Metal kernel: (B*H, D)
            out_flat = turboquant_value_weighted_sum_metal(
                weights=w_flat,
                packed_values=vd_flat,
                scales=vs_flat,
                zeros=vz_flat,
                value_bits=self.value_bits,
                group_size=self.value_group_size,
                head_dim=D,
            )
            # (B*H, D) -> (B, n_heads, 1, D)
            return out_flat.reshape(B, n_heads, 1, D)
        else:
            # Prefill path: loop over query positions
            all_outputs = []
            for qi in range(n_q):
                w_qi = weights[:, :, qi, :]  # (B, n_heads, n_compressed)
                w_flat = w_qi.reshape(B * n_heads, n_compressed)
                vd_flat = val_data.reshape(B * n_heads, n_compressed, -1)
                vs_flat = val_scales.reshape(B * n_heads, n_compressed, -1)
                vz_flat = val_zeros.reshape(B * n_heads, n_compressed, -1)

                out_flat = turboquant_value_weighted_sum_metal(
                    weights=w_flat,
                    packed_values=vd_flat,
                    scales=vs_flat,
                    zeros=vz_flat,
                    value_bits=self.value_bits,
                    group_size=self.value_group_size,
                    head_dim=D,
                )
                all_outputs.append(out_flat)

            # Stack: list of (B*H, D) -> (B*H, n_q, D) -> (B, H, n_q, D)
            stacked = mx.stack(all_outputs, axis=1)
            return stacked.reshape(B, n_heads, n_q, D)

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
