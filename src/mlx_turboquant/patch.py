"""
Model patching utility: inject TurboQuantCache into any MLX-LM model.

Two approaches:
1. patch_model() — monkey-patches model.make_cache() so the standard
   mlx_lm generate pipeline uses TurboQuantCache automatically.
2. make_turboquant_cache() — returns a cache list you can pass directly
   as prompt_cache to generate_step().

Phase 2: Monkey-patches scaled_dot_product_attention in mlx_lm.models.base
to compute attention scores directly from compressed keys via Metal kernels.

Phase 3: ZERO DECOMPRESSION path — values are also never decompressed.
A fused Metal kernel computes the weighted sum of quantized values on-the-fly,
avoiding the (B, H, N_compressed, D) intermediate tensor entirely.
Neither keys nor values are ever materialized in full-precision during decode.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_turboquant.cache import TurboQuantCache

# Global flag to ensure we only patch attention once
_attention_patched = False


def _detect_head_dim(model: nn.Module) -> int:
    """Detect head_dim by inspecting the first transformer layer."""
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Model has no .layers attribute; cannot detect head_dim")

    layer0 = layers[0]

    # Try common attribute paths for the attention module
    attn = None
    for attr_name in ("self_attn", "attention", "attn"):
        attn = getattr(layer0, attr_name, None)
        if attn is not None:
            break

    if attn is None:
        raise ValueError(
            f"Cannot find attention module in layer 0. "
            f"Available attrs: {[a for a in dir(layer0) if not a.startswith('_')]}"
        )

    # Try to get head_dim from the attention module
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is not None:
        return int(head_dim)

    # Fallback: infer from k_proj weight shape
    k_proj = getattr(attn, "k_proj", None)
    if k_proj is not None:
        weight = getattr(k_proj, "weight", None)
        if weight is not None:
            n_kv_heads = getattr(attn, "n_kv_heads", None) or getattr(
                attn, "num_key_value_heads", None
            )
            if n_kv_heads is not None:
                return weight.shape[0] // n_kv_heads

    raise ValueError(
        "Cannot determine head_dim from the model architecture. "
        "Please specify it explicitly."
    )


def _detect_num_layers(model: nn.Module) -> int:
    """Detect the number of transformer layers."""
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("Model has no .layers attribute")
    return len(layers)


def _apply_mask(all_scores, mask, compressed_scores, buffer_scores, cache, n_q):
    """Apply attention mask to concatenated scores [compressed | buffer]."""
    if mask is None:
        return all_scores

    if isinstance(mask, str) and mask == "causal":
        if n_q > 1:
            total_kv = all_scores.shape[-1]
            offset = cache.offset - n_q
            q_indices = mx.arange(offset, offset + n_q)[:, None]
            k_indices = mx.arange(total_kv)[None, :]
            causal = q_indices >= k_indices
            all_scores = mx.where(
                causal, all_scores, mx.finfo(all_scores.dtype).min
            )
        return all_scores

    if not isinstance(mask, mx.array):
        return all_scores

    n_compressed = (
        compressed_scores.shape[-1] if compressed_scores is not None else 0
    )
    n_buf = buffer_scores.shape[-1]
    total = n_compressed + n_buf

    if mask.dtype == mx.bool_:
        if mask.shape[-1] == total:
            return mx.where(mask, all_scores, mx.finfo(all_scores.dtype).min)
        elif mask.shape[-1] == n_buf and n_compressed > 0:
            compressed_mask = mx.ones(
                (*mask.shape[:-1], n_compressed), dtype=mx.bool_
            )
            full_mask = mx.concatenate([compressed_mask, mask], axis=-1)
            return mx.where(full_mask, all_scores, mx.finfo(all_scores.dtype).min)
        return mx.where(mask, all_scores, mx.finfo(all_scores.dtype).min)
    else:
        if mask.shape[-1] == total:
            return all_scores + mask
        elif mask.shape[-1] == n_buf and n_compressed > 0:
            pad = mx.zeros((*mask.shape[:-1], n_compressed))
            full_mask = mx.concatenate([pad, mask], axis=-1)
            return all_scores + full_mask
        return all_scores + mask


def _turboquant_attention_fused(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: TurboQuantCache,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    """
    Fully fused attention: scores + online softmax + value dequant in ONE GPU pass.

    The compressed portion is handled by a single Metal kernel that loops over
    all N compressed tokens, computing online softmax and accumulating dequantized
    value contributions in GPU registers. Zero intermediate tensors.

    The buffer portion is computed via standard matmul (small, fast), then the
    two partial results are merged using log-sum-exp arithmetic.

    Only supports decode (n_q=1).
    """
    B, n_heads, n_q, D = queries.shape
    assert n_q == 1, "Fused decode only supports n_q=1"

    # 1. Fused kernel on compressed data: returns unnormalized (acc_c, m_c, l_c)
    fused_result = cache.compute_fused_attention(queries, scale=scale)

    # 2. Buffer scores via standard matmul
    n_kv_heads = keys.shape[1]
    n_repeats = n_heads // n_kv_heads
    if n_repeats > 1:
        buf_keys = mx.repeat(keys, n_repeats, axis=1)
        buf_values = mx.repeat(values, n_repeats, axis=1)
    else:
        buf_keys = keys
        buf_values = values

    # (B, H, 1, n_buf)
    buffer_scores = (queries @ mx.transpose(buf_keys, (0, 1, 3, 2))) * scale

    # Apply mask to buffer scores only (compressed tokens are always visible)
    if mask is not None and not (isinstance(mask, str) and mask == "causal"):
        if isinstance(mask, mx.array):
            n_buf = buffer_scores.shape[-1]
            if mask.dtype == mx.bool_:
                if mask.shape[-1] >= n_buf:
                    buf_mask = mask[..., -n_buf:]
                    buffer_scores = mx.where(
                        buf_mask, buffer_scores, mx.finfo(buffer_scores.dtype).min
                    )
            else:
                if mask.shape[-1] >= n_buf:
                    buf_mask = mask[..., -n_buf:]
                    buffer_scores = buffer_scores + buf_mask

    if fused_result is None:
        # No compressed data: standard buffer-only path
        attn_weights = mx.softmax(buffer_scores, axis=-1, precise=True)
        return attn_weights @ buf_values

    acc_c, m_c, l_c = fused_result  # (B, H, 1, D), (B, H, 1), (B, H, 1)

    # 3. Compute buffer softmax stats
    # m_b = max(buffer_scores), p_b = exp(scores - m_b), l_b = sum(p_b)
    m_b = mx.max(buffer_scores, axis=-1)          # (B, H, 1)
    p_b = mx.exp(buffer_scores - m_b[..., None])  # (B, H, 1, n_buf)
    l_b = mx.sum(p_b, axis=-1)                    # (B, H, 1)
    acc_b = p_b @ buf_values                       # (B, H, 1, D)

    # 4. Merge using log-sum-exp
    m_both = mx.maximum(m_c, m_b)                        # (B, H, 1)
    scale_c = mx.exp(m_c - m_both)                       # (B, H, 1)
    scale_b = mx.exp(m_b - m_both)                       # (B, H, 1)
    l_both = l_c * scale_c + l_b * scale_b               # (B, H, 1)

    # Normalize
    acc_merged = (acc_c * scale_c[..., None] + acc_b * scale_b[..., None])  # (B, H, 1, D)
    output = acc_merged / (l_both[..., None] + 1e-10)

    return output


def _turboquant_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: TurboQuantCache,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    """
    Compute attention with compressed + buffer data -- ZERO decompression path.

    keys/values are the BUFFER-ONLY portion from update_and_fetch().
    Compressed portion is accessed via cache._keys_compressed / _values_compressed.

    Phase 3 flow (NO intermediate decompression):
      1. Metal kernel scores on compressed keys (no key decompression)
      2. Standard matmul scores on buffer keys (small, fast)
      3. Concatenate [compressed | buffer], apply mask, softmax
      4. Split weights into compressed vs buffer portions
      5. Fused Metal kernel: weights @ quantized_values (no value decompression!)
      6. Standard matmul: buffer_weights @ buffer_values (small, fast)
      7. Sum the two partial outputs
    """
    B, n_heads, n_q, D = queries.shape

    # 1. Compressed scores via Metal kernels (no key decompression!)
    compressed_scores = cache.compute_compressed_scores(queries, scale=scale)

    # 2. Buffer scores via standard matmul (small, fast)
    n_kv_heads = keys.shape[1]
    n_repeats = n_heads // n_kv_heads
    if n_repeats > 1:
        buf_keys = mx.repeat(keys, n_repeats, axis=1)
        buf_values = mx.repeat(values, n_repeats, axis=1)
    else:
        buf_keys = keys
        buf_values = values

    buffer_scores = (
        queries @ mx.transpose(buf_keys, (0, 1, 3, 2))
    ) * scale  # (B, H, n_q, n_buf)

    # 3. Concatenate scores: [compressed | buffer]
    if compressed_scores is not None:
        n_compressed = compressed_scores.shape[-1]
        all_scores = mx.concatenate([compressed_scores, buffer_scores], axis=-1)
    else:
        n_compressed = 0
        all_scores = buffer_scores

    # 4. Apply mask
    all_scores = _apply_mask(
        all_scores, mask, compressed_scores, buffer_scores, cache, n_q
    )

    # 5. Softmax over all positions
    attn_weights = mx.softmax(all_scores, axis=-1, precise=True)

    # 6. Compute output WITHOUT decompressing values
    if n_compressed > 0:
        # Split weights: compressed portion vs buffer portion
        compressed_weights = attn_weights[..., :n_compressed]
        buffer_weights = attn_weights[..., n_compressed:]

        # Fused Metal kernel: weighted sum directly from quantized values
        # No (B, H, N, D) decompressed tensor is ever created!
        compressed_output = cache.compute_compressed_value_output(
            compressed_weights
        )

        # Buffer portion: standard matmul (small, fast)
        buffer_output = buffer_weights @ buf_values

        # Sum partial outputs
        output = compressed_output + buffer_output
    else:
        # No compressed data, standard buffer-only attention
        output = attn_weights @ buf_values

    return output


_PREFILL_CHUNK_SIZE = 256


def _turboquant_attention_chunked_prefill(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: TurboQuantCache,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    """
    Handle prefill with compressed data by chunking queries.

    Without chunking, the two-kernel path materializes a (B, H, qL, N_total)
    score matrix which is huge for long prefills (e.g. 16K × 16K × 40 heads).
    Chunking limits this to (B, H, chunk_size, N_total) per iteration.
    """
    n_q = queries.shape[2]
    if n_q <= _PREFILL_CHUNK_SIZE:
        return _turboquant_attention(queries, keys, values, cache, scale, mask)

    outputs = []
    original_offset = cache.offset
    for q_start in range(0, n_q, _PREFILL_CHUNK_SIZE):
        q_end = min(q_start + _PREFILL_CHUNK_SIZE, n_q)
        q_chunk = queries[:, :, q_start:q_end, :]

        # Temporarily adjust cache.offset so _apply_mask computes correct
        # causal mask positions for this chunk. Without this, all chunks
        # would use the same (wrong) offset.
        cache.offset = original_offset - n_q + q_end

        # Slice mask for this query chunk if applicable
        mask_chunk = mask
        if isinstance(mask, mx.array) and mask.ndim >= 3:
            if mask.shape[-2] > 1:
                mask_chunk = mask[..., q_start:q_end, :]

        out_chunk = _turboquant_attention(
            q_chunk, keys, values, cache, scale, mask_chunk
        )
        outputs.append(out_chunk)
        mx.eval(out_chunk)  # Release score matrix intermediates

    cache.offset = original_offset  # Restore
    return mx.concatenate(outputs, axis=2)


def _patch_attention():
    """
    Monkey-patch mlx_lm's scaled_dot_product_attention to detect
    TurboQuantCache and route through Metal kernels for compressed data.
    """
    import mlx_lm.models.base as base_module

    original_sdpa = base_module.scaled_dot_product_attention

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask=None,
        sinks=None,
    ):
        n_q = queries.shape[2]

        # Skip TQ for short contexts where Metal kernel dispatch overhead
        # exceeds the bandwidth savings from compressed data. Standard SDPA
        # is faster when there are few compressed tokens.
        _MIN_COMPRESSED_FOR_TQ = 256
        if (isinstance(cache, TurboQuantCache)
            and cache._keys_compressed is not None
            and cache.compressed_tokens < _MIN_COMPRESSED_FOR_TQ
            and n_q == 1):
            # Few compressed tokens: decompress and use standard (faster)
            all_keys = cache._get_all_keys()
            all_values = cache._get_all_values()
            return original_sdpa(
                queries, all_keys, all_values, cache=None,
                scale=scale, mask=mask, sinks=sinks,
            )

        # Route through Metal kernel attention for decode (n_q=1)
        # with enough compressed data for TQ to be worthwhile.
        if (isinstance(cache, TurboQuantCache)
            and cache._keys_compressed is not None
            and n_q == 1):
            # Prefer fully fused path (single GPU pass), fall back to
            # Phase 3 two-kernel path if fused kernel fails
            try:
                return _turboquant_attention_fused(
                    queries, keys, values, cache, scale, mask
                )
            except Exception:
                return _turboquant_attention(
                    queries, keys, values, cache, scale, mask
                )

        # Prefill with compressed data: use two-kernel path (zero decompression)
        # Metal kernels already support qL > 1 by looping over query positions.
        # Chunk queries to limit score matrix size (qL × N_total per head).
        if isinstance(cache, TurboQuantCache) and cache._keys_compressed is not None and n_q > 1:
            return _turboquant_attention_chunked_prefill(
                queries, keys, values, cache, scale, mask
            )

        return original_sdpa(
            queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks,
        )

    base_module.scaled_dot_product_attention = patched_sdpa


def make_turboquant_cache(
    model: nn.Module,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 128,
    head_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
) -> list[TurboQuantCache]:
    """
    Create a list of TurboQuantCache objects for use as prompt_cache.

    Also patches scaled_dot_product_attention (once) to handle
    compressed data via Metal kernels.

    Args:
        model: An MLX-LM model.
        key_bits: Bits for key compression (2-4). Default: 3.
        value_bits: Bits for value compression (2 or 4). Default: 2.
        value_group_size: Group size for value quantization. Default: 32.
        buffer_size: Number of recent tokens kept uncompressed. Default: 128.
        head_dim: Override auto-detected head dimension.
        num_layers: Override auto-detected layer count.

    Returns:
        List of TurboQuantCache, one per layer.
    """
    global _attention_patched
    if not _attention_patched:
        _patch_attention()
        _attention_patched = True

    if head_dim is None:
        head_dim = _detect_head_dim(model)
    if num_layers is None:
        num_layers = _detect_num_layers(model)

    return [
        TurboQuantCache(
            head_dim=head_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            value_group_size=value_group_size,
            buffer_size=buffer_size,
            layer_idx=i,
        )
        for i in range(num_layers)
    ]


def patch_model(
    model: nn.Module,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 128,
    head_dim: Optional[int] = None,
) -> nn.Module:
    """
    Monkey-patch an MLX-LM model to use TurboQuantCache.

    After patching, mlx_lm.generate() and stream_generate() will
    automatically use TurboQuant KV compression with Phase 2 direct
    attention (Metal kernels on compressed data).

    Args:
        model: An MLX-LM model to patch.
        key_bits: Bits for key compression. Default: 3.
        value_bits: Bits for value compression. Default: 2.
        value_group_size: Group size for value quantization. Default: 32.
        buffer_size: Recent tokens kept uncompressed. Default: 128.
        head_dim: Override auto-detected head dimension.

    Returns:
        The same model (patched in-place).
    """
    global _attention_patched
    if not _attention_patched:
        _patch_attention()
        _attention_patched = True

    if head_dim is None:
        head_dim = _detect_head_dim(model)

    num_layers = _detect_num_layers(model)

    def _make_cache():
        return make_turboquant_cache(
            model,
            key_bits=key_bits,
            value_bits=value_bits,
            value_group_size=value_group_size,
            buffer_size=buffer_size,
            head_dim=head_dim,
            num_layers=num_layers,
        )

    model.make_cache = _make_cache
    return model
