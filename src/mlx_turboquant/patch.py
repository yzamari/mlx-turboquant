"""
Model patching utility: inject TurboQuantCache into any MLX-LM model.

Two approaches:
1. patch_model() — monkey-patches model.make_cache() so the standard
   mlx_lm generate pipeline uses TurboQuantCache automatically.
2. make_turboquant_cache() — returns a cache list you can pass directly
   as prompt_cache to generate_step().
"""

from typing import Optional

import mlx.nn as nn

from mlx_turboquant.cache import TurboQuantCache


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
    automatically use TurboQuant KV compression.

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
