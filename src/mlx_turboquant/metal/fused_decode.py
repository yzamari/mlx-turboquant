"""
Fused decode Metal kernel for TurboQuant.

One Metal shader that does scores + online softmax + value dequant + weighted sum
in a single GPU pass. Zero intermediate tensors.

Grid: (BH, 1, 1) -- one thread per batch_head.
Loops over all N compressed tokens sequentially.
Outputs UNNORMALIZED: acc[D], m, l (for merging with buffer).
"""

import math
import mlx.core as mx

# Kernel cache: config tuple -> compiled kernel
_fused_decode_kernel_cache: dict = {}


# ---------------------------------------------------------------------------
# Metal shader template
# ---------------------------------------------------------------------------
FUSED_DECODE_SOURCE = """
    uint bh = thread_position_in_grid.x;
    uint BH = q_rot_shape[0];
    if (bh >= BH) return;

    uint N = mse_shape[1];

    // Online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[{D}];
    for (uint d = 0; d < {D}; d++) acc[d] = 0.0f;

    // Preload queries (read once, reuse N times)
    float qr[{D}];
    float qs[{D}];
    for (uint d = 0; d < {D}; d++) {{
        qr[d] = q_rot[bh * {D} + d];
        qs[d] = q_sketch[bh * {D} + d];
    }}

    for (uint n = 0; n < N; n++) {{
        // MSE score
        float mse_score = 0.0f;
        for (uint byte_idx = 0; byte_idx < {PACKED_D}; byte_idx++) {{
            uint8_t packed = mse[bh * N * {PACKED_D} + n * {PACKED_D} + byte_idx];
            uint pi = (uint)packed;
            for (uint sub = 0; sub < {MSE_VALS_PER_BYTE}; sub++) {{
                uint coord = byte_idx * {MSE_VALS_PER_BYTE} + sub;
                if (coord < {D}) {{
                    uint idx = (pi >> (sub * {MSE_BITS})) & {MSE_BIT_MASK};
                    mse_score += qr[coord] * centroids[idx];
                }}
            }}
        }}
        mse_score *= norms[bh * N + n];

        // QJL correction
        float qjl_dot = 0.0f;
        for (uint byte_idx = 0; byte_idx < {PACKED_D_SIGNS}; byte_idx++) {{
            uint8_t packed = signs[bh * N * {PACKED_D_SIGNS} + n * {PACKED_D_SIGNS} + byte_idx];
            uint pi = (uint)packed;
            for (uint bit = 0; bit < 8; bit++) {{
                uint coord = byte_idx * 8 + bit;
                if (coord < {D}) {{
                    float sv = ((pi >> bit) & 1) ? 1.0f : -1.0f;
                    qjl_dot += qs[coord] * sv;
                }}
            }}
        }}
        float score = (mse_score + qjl_dot * res_norms[bh * N + n] * {QJL_SCALE}) * {SM_SCALE};

        // Online softmax
        float m_new = (score > m_i) ? score : m_i;
        float alpha = exp(m_i - m_new);
        float p = exp(score - m_new);
        l_i = l_i * alpha + p;
        for (uint d = 0; d < {D}; d++) acc[d] *= alpha;

        // Value dequant + accumulate
        for (uint byte_idx = 0; byte_idx < {PACKED_V}; byte_idx++) {{
            uint8_t vb = v_data[bh * N * {PACKED_V} + n * {PACKED_V} + byte_idx];
            uint vi = (uint)vb;
            for (uint sub = 0; sub < {V_VALS_PER_BYTE}; sub++) {{
                uint coord = byte_idx * {V_VALS_PER_BYTE} + sub;
                if (coord < {D}) {{
                    uint qval = (vi >> (sub * {V_BITS})) & {V_BIT_MASK};
                    uint gi = coord / {V_GROUP_SIZE};
                    float s = v_scales[bh * N * {V_N_GROUPS} + n * {V_N_GROUPS} + gi];
                    float z = v_zeros[bh * N * {V_N_GROUPS} + n * {V_N_GROUPS} + gi];
                    acc[coord] += p * ((float)qval * s + z);
                }}
            }}
        }}
        m_i = m_new;
    }}

    // Write unnormalized output + softmax state
    for (uint d = 0; d < {D}; d++) out[bh * {D} + d] = acc[d];
    out_m[bh] = m_i;
    out_l[bh] = l_i;
"""


def _get_mse_packing_params(mse_bits: int) -> tuple[int, int, int]:
    """Return (effective_bits, vals_per_byte, bit_mask) for MSE packing.

    mse_bits is the key_bits parameter (e.g. 3). The MSE quantizer uses
    bits-1 effective bits internally, so mse_bits=3 -> 2-bit MSE indices.
    """
    eff = mse_bits - 1  # TurboQuantProd uses bits-1 for MSE stage
    if eff == 1:
        return 1, 8, 0x01
    elif eff == 2:
        return 2, 4, 0x03
    elif eff <= 4:
        return 4, 2, 0x0F
    else:
        return 8, 1, 0xFF


def _get_value_packing_params(v_bits: int) -> tuple[int, int, int]:
    """Return (effective_bits, vals_per_byte, bit_mask) for value packing."""
    if v_bits == 2:
        return 2, 4, 0x03
    elif v_bits == 4:
        return 4, 2, 0x0F
    else:
        return 8, 1, 0xFF


def _build_kernel_source(
    D: int,
    mse_bits: int,
    v_bits: int,
    group_size: int,
    qjl_scale: float,
    sm_scale: float,
) -> str:
    """Build the Metal shader source with all template parameters substituted."""
    mse_eff_bits, mse_vpb, mse_mask = _get_mse_packing_params(mse_bits)
    v_eff_bits, v_vpb, v_mask = _get_value_packing_params(v_bits)

    packed_d = (D + mse_vpb - 1) // mse_vpb
    packed_d_signs = (D + 7) // 8
    packed_v = (D + v_vpb - 1) // v_vpb
    n_groups = D // group_size

    return FUSED_DECODE_SOURCE.format(
        D=D,
        PACKED_D=packed_d,
        MSE_BITS=mse_eff_bits,
        MSE_VALS_PER_BYTE=mse_vpb,
        MSE_BIT_MASK=mse_mask,
        PACKED_D_SIGNS=packed_d_signs,
        PACKED_V=packed_v,
        V_BITS=v_eff_bits,
        V_VALS_PER_BYTE=v_vpb,
        V_BIT_MASK=v_mask,
        V_GROUP_SIZE=group_size,
        V_N_GROUPS=n_groups,
        QJL_SCALE=f"{qjl_scale:.10f}f",
        SM_SCALE=f"{sm_scale:.10f}f",
    )


def turboquant_fused_decode_metal(
    q_rot: mx.array,       # (BH, D) rotated query
    q_sketch: mx.array,    # (BH, D) sketched query
    mse: mx.array,         # (BH, N, packed_d) uint8
    signs: mx.array,       # (BH, N, packed_d_signs) uint8
    norms: mx.array,       # (BH, N)
    res_norms: mx.array,   # (BH, N)
    centroids: mx.array,   # (n_clusters,)
    v_data: mx.array,      # (BH, N, packed_v) uint8
    v_scales: mx.array,    # (BH, N, n_groups)
    v_zeros: mx.array,     # (BH, N, n_groups)
    mse_bits: int,
    v_bits: int,
    D: int,
    group_size: int,
    qjl_scale: float,
    sm_scale: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Fused decode: scores + online softmax + value dequant + weighted sum.

    Returns:
        (out, out_m, out_l) where:
            out: (BH, D) unnormalized accumulator
            out_m: (BH,) running max for log-sum-exp merge
            out_l: (BH,) running sum for log-sum-exp merge
    """
    BH = q_rot.shape[0]

    cache_key = (D, mse_bits, v_bits, group_size, round(qjl_scale, 10), round(sm_scale, 10))
    if cache_key not in _fused_decode_kernel_cache:
        source = _build_kernel_source(D, mse_bits, v_bits, group_size, qjl_scale, sm_scale)
        _fused_decode_kernel_cache[cache_key] = mx.fast.metal_kernel(
            name=f"turboquant_fused_decode_d{D}_mb{mse_bits}_vb{v_bits}_g{group_size}",
            input_names=["q_rot", "q_sketch", "mse", "signs", "norms", "res_norms",
                         "centroids", "v_data", "v_scales", "v_zeros"],
            output_names=["out", "out_m", "out_l"],
            source=source,
        )

    kernel = _fused_decode_kernel_cache[cache_key]

    # Ensure correct dtypes
    q_rot = q_rot.astype(mx.float32)
    q_sketch = q_sketch.astype(mx.float32)
    mse = mse.astype(mx.uint8)
    signs = signs.astype(mx.uint8)
    norms = norms.astype(mx.float32)
    res_norms = res_norms.astype(mx.float32)
    centroids = centroids.astype(mx.float32)
    v_data = v_data.astype(mx.uint8)
    v_scales = v_scales.astype(mx.float32)
    v_zeros = v_zeros.astype(mx.float32)

    outputs = kernel(
        inputs=[q_rot, q_sketch, mse, signs, norms, res_norms,
                centroids, v_data, v_scales, v_zeros],
        output_shapes=[(BH, D), (BH,), (BH,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(BH, 1, 1),
        threadgroup=(1, 1, 1),
    )

    return outputs[0], outputs[1], outputs[2]
