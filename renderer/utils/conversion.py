"""Type conversion utilities."""

from __future__ import annotations
from typing import Union
import numpy as np


def to_torch_tensor(
    x: Union[np.ndarray, "torch.Tensor", list],
    device: str = "cuda",
    dtype: "torch.dtype" = None,
    requires_grad: bool = False
):
    """
    Convert input to PyTorch tensor.
    
    Args:
        x: Input (numpy array, torch tensor, or list)
        device: Target device
        dtype: Target dtype (default: torch.float32)
        requires_grad: Whether to enable gradient computation
    
    Returns:
        PyTorch tensor on specified device
    """
    import torch
    
    if dtype is None:
        dtype = torch.float32
    
    if isinstance(x, torch.Tensor):
        tensor = x
    else:
        tensor = torch.as_tensor(x, dtype=dtype)
    
    if device and tensor.device != torch.device(device):
        tensor = tensor.to(device)
    
    if requires_grad and not tensor.requires_grad:
        tensor.requires_grad_(True)
    
    return tensor


def to_numpy_array(
    x: Union[np.ndarray, "torch.Tensor"],
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Convert input to NumPy array.
    
    Args:
        x: Input (numpy array or torch tensor)
        dtype: Target dtype
    
    Returns:
        NumPy array
    """
    if hasattr(x, 'detach'):  # torch.Tensor
        return x.detach().cpu().numpy().astype(dtype)
    else:
        return np.asarray(x, dtype=dtype)