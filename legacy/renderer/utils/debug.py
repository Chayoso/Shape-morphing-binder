"""Debug utilities."""

from __future__ import annotations
from typing import Tuple
import os

DEBUG_ENV_VAR = "GS_DEBUG"


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled via environment variable."""
    return os.environ.get(DEBUG_ENV_VAR, "0").lower() not in ("0", "", "false")


def debug_print(*args, **kwargs):
    """Print debug message if debug mode is enabled."""
    if is_debug_enabled():
        print(*args, **kwargs)


def get_tensor_stats(tensor) -> Tuple[float, float, float]:
    """
    Get min, max, mean statistics of a tensor.
    
    Args:
        tensor: PyTorch tensor
    
    Returns:
        (min, max, mean) as floats
    """
    return (
        float(tensor.min().item()),
        float(tensor.max().item()),
        float(tensor.mean().item())
    )


def debug_tensor_info(name: str, tensor):
    """Print debug information about a tensor."""
    if is_debug_enabled():
        try:
            mn, mx, mean = get_tensor_stats(tensor)
            print(f"[{name}] shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                  f"min={mn:.4f} max={mx:.4f} mean={mean:.4f}")
        except Exception as e:
            print(f"[{name}] shape={tuple(tensor.shape)} dtype={tensor.dtype} (stats failed: {e})")