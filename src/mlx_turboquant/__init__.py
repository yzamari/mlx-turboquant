"""
MLX-TurboQuant: TurboQuant KV cache compression for MLX-LM inference.

Drop-in replacement for MLX-LM's KVCache that compresses keys via
TurboQuant (product quantization) and values via group quantization,
achieving ~5x memory reduction with minimal quality loss.
"""

__version__ = "0.1.0"

from mlx_turboquant.cache import TurboQuantCache
from mlx_turboquant.patch import patch_model, make_turboquant_cache
from mlx_turboquant.generate import generate

__all__ = [
    "TurboQuantCache",
    "patch_model",
    "make_turboquant_cache",
    "generate",
]
