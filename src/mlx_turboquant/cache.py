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
        """Compress oldest buffer tokens into quantized storage.

        When n_flush is large (e.g., 16K tokens after prefill), compresses in
        chunks of flush_batch_size to avoid a memory spike from creating massive
        temporary arrays (rotation matmul, QJL projection) all at once.
        """
        n_flush = self._key_buffer.shape[2] - self.buffer_size
        if n_flush <= 0:
            return

        # Chunked flush: compress flush_batch_size tokens at a time to keep
        # peak memory low. Without this, compressing 16K tokens at once creates
        # (16K, 128) @ (128, 128) temporaries that spike peak memory by ~7 GB.
        chunk_size = self.flush_batch_size
        if n_flush > chunk_size:
            keys_to_flush = self._key_buffer[:, :, :n_flush, :]
            values_to_flush = self._value_buffer[:, :, :n_flush, :]
            self._key_buffer = self._key_buffer[:, :, n_flush:, :]
            self._value_buffer = self._value_buffer[:, :, n_flush:, :]
            # Force evaluation of ALL slices so the original buffer can be freed.
            # Without this, keys_to_flush holds a lazy view keeping the full
            # pre-flush buffer alive for the entire loop.
            mx.eval(
                keys_to_flush, values_to_flush,
                self._key_buffer, self._value_buffer,
            )

            for start in range(0, n_flush, chunk_size):
                end = min(start + chunk_size, n_flush)
                k_chunk = keys_to_flush[:, :, start:end, :]
                v_chunk = values_to_flush[:, :, start:end, :]
                self._compress_and_store(k_chunk, v_chunk)
                # Force evaluation to release this chunk's compression
                # intermediates before the next chunk starts. Without this,
                # lazy eval keeps ALL chunks' intermediates alive at once.
                self._eval_compressed()
            return

        keys_to_compress = self._key_buffer[:, :, :n_flush, :]
        values_to_compress = self._value_buffer[:, :, :n_flush, :]
        self._key_buffer = self._key_buffer[:, :, n_flush:, :]
        self._value_buffer = self._value_buffer[:, :, n_flush:, :]

        self._compress_and_store(keys_to_compress, values_to_compress)
        self._eval_compressed()

    def _eval_compressed(self):
        """Force evaluation of compressed arrays to release intermediates."""
        if self._keys_compressed is not None:
            mx.eval(
                self._keys_compressed.mse_indices,
                self._keys_compressed.qjl_signs,
                self._keys_compressed.norms,
                self._keys_compressed.residual_norms,
                self._values_compressed.data,
                self._values_compressed.scales,
                self._values_compressed.zeros,
            )

    def _compress_and_store(self, keys, values):
        """Quantize keys+values and append to compressed storage."""
        new_key_q = self.key_quantizer.quantize(keys)
        new_val_q = quantize_values(
            values,
            bits=self.value_bits,
            group_size=self.value_group_size,
            backend="mlx",
        )

        if self._keys_compressed is None:
            self._keys_compressed = new_key_q
            self._values_compressed = new_val_q
        else:
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
                    [self._keys_compressed.residual_norms, new_key_q.residual_norms],
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
                    [self._values_compressed.scales, new_val_q.scales], axis=-2
                ),
                zeros=mx.concatenate(
                    [self._values_compressed.zeros, new_val_q.zeros], axis=-2
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

    # ---- Fused decode: scores + softmax + value dequant in one GPU pass ----

    def compute_fused_attention(
        self, queries: mx.array, scale: float = 1.0
    ) -> tuple[mx.array, mx.array, mx.array] | None:
        """
        Fused decode: compute scores + online softmax + value dequant + weighted
        sum in a single Metal GPU pass. Zero intermediate tensors.

        Args:
            queries: (B, n_heads, 1, head_dim) query vectors (decode only).
            scale: attention scale factor (typically 1/sqrt(d)).

        Returns:
            (acc, m, l) where:
                acc: (B, n_heads, 1, D) unnormalized weighted accumulator
                m: (B, n_heads, 1) running max for log-sum-exp merge
                l: (B, n_heads, 1) running sum for log-sum-exp merge
            or None if no compressed data exists.
        """
        if self._keys_compressed is None or self._values_compressed is None:
            return None

        B, n_heads, n_q, D = queries.shape
        kc = self._keys_compressed
        vc = self._values_compressed

        # Retrieve quantizer matrices
        Pi = self.key_quantizer.mse_quantizer.Pi
        S = self.key_quantizer.S
        centroids = self.key_quantizer.mse_quantizer.centroids
        mse_eff_bits = self.key_bits - 1  # TurboQuantProd uses bits-1 for MSE
        qjl_scale = self.key_quantizer.qjl_scale

        # Use native C++/Metal kernel — handles GQA and query rotation internally
        # Input shapes: queries (B, H_q, 1, D), kc.mse_indices (B, H_kv, N, packed_d)
        acc, m, l = mx.fast.turboquant_attention(
            queries,
            kc.mse_indices,       # (B, H_kv, N, packed_d_mse) uint8
            kc.qjl_signs,         # (B, H_kv, N, packed_d_signs) uint8
            kc.norms,             # (B, H_kv, N)
            kc.residual_norms,    # (B, H_kv, N)
            centroids,            # (n_centroids,)
            vc.data,              # (B, H_kv, N, packed_d_v) uint8
            vc.scales,            # (B, H_kv, N, n_groups)
            vc.zeros,             # (B, H_kv, N, n_groups)
            Pi,                   # (D, D) rotation matrix
            S,                    # (D, D) sketch matrix
            scale=scale,
            qjl_scale=qjl_scale,
            mse_bits=mse_eff_bits,
            v_bits=self.value_bits,
            group_size=self.value_group_size,
        )
        # acc: (B, H_q, 1, D), m: (B, H_q, 1), l: (B, H_q, 1)
        return acc, m, l

    def _fused_flat(
        self, q_rot, q_sketch, kc, vc, n_heads, n_compressed, B, D,
        centroids, mse_bits, qjl_scale, sm_scale, kernel_fn,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Fused decode when n_heads == n_kv_heads (no GQA)."""
        BH = B * n_heads
        mse_flat = kc.mse_indices.reshape(BH, n_compressed, -1)
        signs_flat = kc.qjl_signs.reshape(BH, n_compressed, -1)
        norms_flat = kc.norms.reshape(BH, n_compressed)
        res_norms_flat = kc.residual_norms.reshape(BH, n_compressed)

        vd_flat = vc.data.reshape(BH, n_compressed, -1)
        vs_flat = vc.scales.reshape(BH, n_compressed, -1)
        vz_flat = vc.zeros.reshape(BH, n_compressed, -1)

        acc, m, l = kernel_fn(
            q_rot=q_rot, q_sketch=q_sketch,
            mse=mse_flat, signs=signs_flat,
            norms=norms_flat, res_norms=res_norms_flat,
            centroids=centroids,
            v_data=vd_flat, v_scales=vs_flat, v_zeros=vz_flat,
            mse_bits=mse_bits, v_bits=self.value_bits,
            D=D, group_size=self.value_group_size,
            qjl_scale=qjl_scale, sm_scale=sm_scale,
        )
        return acc, m, l

    def _fused_gqa(
        self, q_rot, q_sketch, queries, kc, vc,
        n_kv_heads, repeats, n_compressed, B, D,
        Pi, S, centroids, mse_bits, qjl_scale, sm_scale, kernel_fn,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Fused decode with GQA via per-KV-head-group loop."""
        all_acc = []
        all_m = []
        all_l = []

        for kv_h in range(n_kv_heads):
            # Slice this KV head's compressed key data
            mse_h = kc.mse_indices[:, kv_h, :, :]       # (B, N, packed_d)
            signs_h = kc.qjl_signs[:, kv_h, :, :]       # (B, N, packed_d_signs)
            norms_h = kc.norms[:, kv_h, :]               # (B, N)
            res_h = kc.residual_norms[:, kv_h, :]        # (B, N)

            # Slice this KV head's compressed value data
            vd_h = vc.data[:, kv_h, :, :]    # (B, N, packed_v)
            vs_h = vc.scales[:, kv_h, :, :]   # (B, N, n_groups)
            vz_h = vc.zeros[:, kv_h, :, :]    # (B, N, n_groups)

            # Tile along batch dim for `repeats` query heads
            mse_tiled = mx.repeat(mse_h, repeats, axis=0)
            signs_tiled = mx.repeat(signs_h, repeats, axis=0)
            norms_tiled = mx.repeat(norms_h, repeats, axis=0)
            res_tiled = mx.repeat(res_h, repeats, axis=0)
            vd_tiled = mx.repeat(vd_h, repeats, axis=0)
            vs_tiled = mx.repeat(vs_h, repeats, axis=0)
            vz_tiled = mx.repeat(vz_h, repeats, axis=0)

            # Query heads for this KV group
            q_start = kv_h * repeats
            q_end = q_start + repeats
            q_group = queries[:, q_start:q_end, 0, :]  # (B, repeats, D)
            q_group_flat = q_group.reshape(B * repeats, D)

            # Compute rotated/sketched for this query group
            qr = mx.matmul(q_group_flat.astype(mx.float32), mx.transpose(Pi))
            qs = mx.matmul(q_group_flat.astype(mx.float32), mx.transpose(S))

            acc_h, m_h, l_h = kernel_fn(
                q_rot=qr, q_sketch=qs,
                mse=mse_tiled, signs=signs_tiled,
                norms=norms_tiled, res_norms=res_tiled,
                centroids=centroids,
                v_data=vd_tiled, v_scales=vs_tiled, v_zeros=vz_tiled,
                mse_bits=mse_bits, v_bits=self.value_bits,
                D=D, group_size=self.value_group_size,
                qjl_scale=qjl_scale, sm_scale=sm_scale,
            )
            # (B*repeats, D) -> (B, repeats, D)
            all_acc.append(acc_h.reshape(B, repeats, D))
            all_m.append(m_h.reshape(B, repeats))
            all_l.append(l_h.reshape(B, repeats))

        # Concatenate across KV head groups -> (B, n_heads, D), (B, n_heads), (B, n_heads)
        acc = mx.concatenate(all_acc, axis=1).reshape(B * (n_kv_heads * repeats), D)
        m = mx.concatenate(all_m, axis=1).reshape(B * (n_kv_heads * repeats))
        l = mx.concatenate(all_l, axis=1).reshape(B * (n_kv_heads * repeats))
        return acc, m, l

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
        n_compressed = kc.mse_indices.shape[2]
        repeats = n_heads // n_kv_heads

        if repeats == 1:
            # No GQA: flatten directly, no expansion needed
            scores = self._score_keys_flat(
                queries, kc, n_heads, n_compressed, n_q, B, D,
                Pi, S, centroids, mse_bits, qjl_scale,
                turboquant_attention_score_metal,
            )
        else:
            # GQA: loop per KV head group to avoid repeats-x memory expansion.
            # For each KV head, slice its compressed data, tile it for the
            # `repeats` query heads via batch-dim repeat (one head at a time),
            # run the Metal kernel, then discard the tiled copy.
            # Peak memory: repeats * (1 head's data) instead of repeats * (all heads' data).
            scores = self._score_keys_gqa(
                queries, kc, n_kv_heads, repeats, n_compressed, n_q, B, D,
                Pi, S, centroids, mse_bits, qjl_scale,
                turboquant_attention_score_metal,
            )

        return scores * scale

    def _score_keys_flat(
        self, queries, kc, n_heads, n_compressed, n_q, B, D,
        Pi, S, centroids, mse_bits, qjl_scale, kernel_fn,
    ) -> mx.array:
        """Score compressed keys when n_heads == n_kv_heads (no GQA)."""
        mse_flat = kc.mse_indices.reshape(B * n_heads, n_compressed, -1)
        qjl_flat = kc.qjl_signs.reshape(B * n_heads, n_compressed, -1)
        norms_flat = kc.norms.reshape(B * n_heads, n_compressed)
        res_norms_flat = kc.residual_norms.reshape(B * n_heads, n_compressed)

        if n_q == 1:
            q_flat = queries.reshape(B * n_heads, D)
            scores_flat = kernel_fn(
                query=q_flat, mse_packed=mse_flat, qjl_signs=qjl_flat,
                norms=norms_flat, residual_norms=res_norms_flat,
                Pi=Pi, S=S, centroids=centroids,
                mse_bits=mse_bits, qjl_scale=qjl_scale,
            )
            return scores_flat.reshape(B, n_heads, 1, n_compressed)
        else:
            all_scores = []
            for qi in range(n_q):
                q_flat = queries[:, :, qi, :].reshape(B * n_heads, D)
                s = kernel_fn(
                    query=q_flat, mse_packed=mse_flat, qjl_signs=qjl_flat,
                    norms=norms_flat, residual_norms=res_norms_flat,
                    Pi=Pi, S=S, centroids=centroids,
                    mse_bits=mse_bits, qjl_scale=qjl_scale,
                )
                all_scores.append(s)
            scores_flat = mx.stack(all_scores, axis=1)
            return scores_flat.reshape(B, n_heads, n_q, n_compressed)

    def _score_keys_gqa(
        self, queries, kc, n_kv_heads, repeats, n_compressed, n_q, B, D,
        Pi, S, centroids, mse_bits, qjl_scale, kernel_fn,
    ) -> mx.array:
        """Score compressed keys with GQA via per-KV-head-group loop.

        For each KV head, slices its compressed data and tiles it along the
        batch dimension for the `repeats` query heads sharing that KV head.
        Peak memory is repeats * (one head's data) instead of repeats * (all heads' data).
        """
        n_heads = n_kv_heads * repeats
        all_group_scores = []

        for kv_h in range(n_kv_heads):
            # Slice this KV head's compressed data: (B, N, packed_d)
            mse_h = kc.mse_indices[:, kv_h, :, :]       # (B, N, packed_d)
            signs_h = kc.qjl_signs[:, kv_h, :, :]       # (B, N, packed_d_signs)
            norms_h = kc.norms[:, kv_h, :]               # (B, N)
            res_h = kc.residual_norms[:, kv_h, :]        # (B, N)

            # Tile along batch dim for `repeats` query heads:
            # (B, N, packed_d) -> (B*repeats, N, packed_d)
            mse_tiled = mx.repeat(mse_h, repeats, axis=0)
            signs_tiled = mx.repeat(signs_h, repeats, axis=0)
            norms_tiled = mx.repeat(norms_h, repeats, axis=0)
            res_tiled = mx.repeat(res_h, repeats, axis=0)

            # Query heads for this KV group: (B, repeats, n_q, D)
            q_start = kv_h * repeats
            q_end = q_start + repeats
            q_group = queries[:, q_start:q_end, :, :]

            if n_q == 1:
                # (B, repeats, 1, D) -> (B*repeats, D)
                q_flat = q_group.reshape(B * repeats, D)
                scores_h = kernel_fn(
                    query=q_flat, mse_packed=mse_tiled, qjl_signs=signs_tiled,
                    norms=norms_tiled, residual_norms=res_tiled,
                    Pi=Pi, S=S, centroids=centroids,
                    mse_bits=mse_bits, qjl_scale=qjl_scale,
                )
                # (B*repeats, N) -> (B, repeats, N)
                all_group_scores.append(scores_h.reshape(B, repeats, n_compressed))
            else:
                # Prefill: loop over query positions within this group
                qi_scores = []
                for qi in range(n_q):
                    q_flat = q_group[:, :, qi, :].reshape(B * repeats, D)
                    s = kernel_fn(
                        query=q_flat, mse_packed=mse_tiled, qjl_signs=signs_tiled,
                        norms=norms_tiled, residual_norms=res_tiled,
                        Pi=Pi, S=S, centroids=centroids,
                        mse_bits=mse_bits, qjl_scale=qjl_scale,
                    )
                    qi_scores.append(s)
                # list of (B*repeats, N) -> (B*repeats, n_q, N) -> (B, repeats, n_q, N)
                stacked = mx.stack(qi_scores, axis=1)
                all_group_scores.append(stacked.reshape(B, repeats, n_q, n_compressed))

        if n_q == 1:
            # list of (B, repeats, N) -> (B, n_heads, N) -> (B, n_heads, 1, N)
            scores = mx.concatenate(all_group_scores, axis=1)
            return scores.reshape(B, n_heads, 1, n_compressed)
        else:
            # list of (B, repeats, n_q, N) -> (B, n_heads, n_q, N)
            return mx.concatenate(all_group_scores, axis=1)

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

        # Handle GQA: n_heads (query/weight) may differ from n_kv_heads (value)
        n_kv_heads = vc.data.shape[1]
        repeats = n_heads // n_kv_heads

        if repeats == 1:
            # No GQA: flatten directly, no expansion needed
            return self._value_output_flat(
                weights, vc, n_heads, n_q, n_compressed, B, D,
                turboquant_value_weighted_sum_metal,
            )
        else:
            # GQA: loop per KV head group to avoid repeats-x memory expansion
            return self._value_output_gqa(
                weights, vc, n_kv_heads, repeats, n_q, n_compressed, B, D,
                turboquant_value_weighted_sum_metal,
            )

    def _value_output_flat(
        self, weights, vc, n_heads, n_q, n_compressed, B, D, kernel_fn,
    ) -> mx.array:
        """Compute value weighted sum when n_heads == n_kv_heads (no GQA)."""
        vd_flat = vc.data.reshape(B * n_heads, n_compressed, -1)
        vs_flat = vc.scales.reshape(B * n_heads, n_compressed, -1)
        vz_flat = vc.zeros.reshape(B * n_heads, n_compressed, -1)

        if n_q == 1:
            w_flat = weights.reshape(B * n_heads, n_compressed)
            out_flat = kernel_fn(
                weights=w_flat, packed_values=vd_flat,
                scales=vs_flat, zeros=vz_flat,
                value_bits=self.value_bits,
                group_size=self.value_group_size,
                head_dim=D,
            )
            return out_flat.reshape(B, n_heads, 1, D)
        else:
            all_outputs = []
            for qi in range(n_q):
                w_flat = weights[:, :, qi, :].reshape(B * n_heads, n_compressed)
                out_flat = kernel_fn(
                    weights=w_flat, packed_values=vd_flat,
                    scales=vs_flat, zeros=vz_flat,
                    value_bits=self.value_bits,
                    group_size=self.value_group_size,
                    head_dim=D,
                )
                all_outputs.append(out_flat)
            stacked = mx.stack(all_outputs, axis=1)
            return stacked.reshape(B, n_heads, n_q, D)

    def _value_output_gqa(
        self, weights, vc, n_kv_heads, repeats, n_q, n_compressed, B, D,
        kernel_fn,
    ) -> mx.array:
        """Compute value weighted sum with GQA via per-KV-head-group loop.

        For each KV head, slices its compressed value data and tiles it along
        the batch dimension for the `repeats` query heads sharing that KV head.
        Peak memory: repeats * (one head's data) instead of repeats * (all heads' data).
        """
        n_heads = n_kv_heads * repeats
        all_group_outputs = []

        for kv_h in range(n_kv_heads):
            # Slice this KV head's value data: (B, N, packed_d) etc.
            vd_h = vc.data[:, kv_h, :, :]    # (B, N, packed_d)
            vs_h = vc.scales[:, kv_h, :, :]   # (B, N, n_groups)
            vz_h = vc.zeros[:, kv_h, :, :]    # (B, N, n_groups)

            # Tile along batch dim for `repeats` query heads:
            # (B, N, packed_d) -> (B*repeats, N, packed_d)
            vd_tiled = mx.repeat(vd_h, repeats, axis=0)
            vs_tiled = mx.repeat(vs_h, repeats, axis=0)
            vz_tiled = mx.repeat(vz_h, repeats, axis=0)

            # Weights for this group's query heads: (B, repeats, n_q, N)
            q_start = kv_h * repeats
            q_end = q_start + repeats
            w_group = weights[:, q_start:q_end, :, :]

            if n_q == 1:
                # (B, repeats, 1, N) -> (B*repeats, N)
                w_flat = w_group.reshape(B * repeats, n_compressed)
                out_h = kernel_fn(
                    weights=w_flat, packed_values=vd_tiled,
                    scales=vs_tiled, zeros=vz_tiled,
                    value_bits=self.value_bits,
                    group_size=self.value_group_size,
                    head_dim=D,
                )
                # (B*repeats, D) -> (B, repeats, D)
                all_group_outputs.append(out_h.reshape(B, repeats, D))
            else:
                qi_outputs = []
                for qi in range(n_q):
                    w_flat = w_group[:, :, qi, :].reshape(B * repeats, n_compressed)
                    out_qi = kernel_fn(
                        weights=w_flat, packed_values=vd_tiled,
                        scales=vs_tiled, zeros=vz_tiled,
                        value_bits=self.value_bits,
                        group_size=self.value_group_size,
                        head_dim=D,
                    )
                    qi_outputs.append(out_qi)
                # list of (B*repeats, D) -> (B*repeats, n_q, D) -> (B, repeats, n_q, D)
                stacked = mx.stack(qi_outputs, axis=1)
                all_group_outputs.append(stacked.reshape(B, repeats, n_q, D))

        if n_q == 1:
            # list of (B, repeats, D) -> (B, n_heads, D) -> (B, n_heads, 1, D)
            output = mx.concatenate(all_group_outputs, axis=1)
            return output.reshape(B, n_heads, 1, D)
        else:
            # list of (B, repeats, n_q, D) -> (B, n_heads, n_q, D)
            return mx.concatenate(all_group_outputs, axis=1)

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
