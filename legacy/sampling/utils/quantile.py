"""
Differentiable Quantile Utilities with PyTorch Compatibility

Provides safe quantile computation with fallback for older PyTorch versions.
"""

import torch
from typing import Union, Tuple


def safe_quantile(
    x: torch.Tensor,
    q: Union[float, torch.Tensor],
    dim: int = None,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Compute quantile with automatic fallback for compatibility.
    
    Args:
        x: Input tensor
        q: Quantile value(s) in [0, 1]
        dim: Dimension along which to compute quantile
        keepdim: Whether to keep the dimension
        
    Returns:
        Quantile tensor
        
    Notes:
        - Uses torch.quantile if available (PyTorch >= 1.7)
        - Falls back to sorting-based implementation for older versions
        - Fully differentiable in both cases
    """
    # Try native torch.quantile first
    if hasattr(torch, 'quantile'):
        try:
            return torch.quantile(x, q, dim=dim, keepdim=keepdim)
        except Exception:
            pass  # Fall through to manual implementation
    
    # Fallback implementation
    return _quantile_via_sort(x, q, dim=dim, keepdim=keepdim)


def _quantile_via_sort(
    x: torch.Tensor,
    q: Union[float, torch.Tensor],
    dim: int = None,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Compute quantile using sorting (differentiable).
    
    Args:
        x: Input tensor  
        q: Quantile value(s) in [0, 1]
        dim: Dimension along which to compute
        keepdim: Whether to keep dimension
        
    Returns:
        Quantile tensor
    """
    # Convert q to tensor
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=x.dtype, device=x.device)
    
    # Handle scalar case
    if dim is None:
        x_flat = x.reshape(-1)
        x_sorted, _ = torch.sort(x_flat)
        return _interp_quantile(x_sorted, q)
    
    # Handle multi-dimensional case
    x_sorted, _ = torch.sort(x, dim=dim)
    
    # Get indices for quantile
    n = x.shape[dim]
    indices = (q * (n - 1)).clamp(0, n - 1)
    
    # Linear interpolation between adjacent values
    idx_low = indices.long()
    idx_high = (idx_low + 1).clamp(max=n - 1)
    frac = (indices - idx_low.float()).unsqueeze(-1)
    
    # Select values
    if dim == 0 or dim == -x.ndim:
        val_low = x_sorted.index_select(dim, idx_low)
        val_high = x_sorted.index_select(dim, idx_high)
    else:
        # More complex indexing for middle dimensions
        val_low = torch.index_select(x_sorted, dim, idx_low)
        val_high = torch.index_select(x_sorted, dim, idx_high)
    
    # Interpolate
    result = val_low + frac * (val_high - val_low)
    
    if not keepdim and result.ndim > x.ndim - 1:
        result = result.squeeze(dim)
        
    return result


def _interp_quantile(x_sorted: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Interpolate quantile from sorted values.
    
    Args:
        x_sorted: Sorted tensor (1D)
        q: Quantile value(s)
        
    Returns:
        Interpolated quantile value(s)
    """
    n = x_sorted.shape[0]
    
    # Handle edge cases
    if n == 0:
        return torch.tensor(0.0, dtype=x_sorted.dtype, device=x_sorted.device)
    if n == 1:
        return x_sorted[0]
    
    # Compute fractional indices
    indices = q * (n - 1)
    idx_low = indices.long().clamp(0, n - 2)
    idx_high = (idx_low + 1).clamp(max=n - 1)
    frac = indices - idx_low.float()
    
    # Linear interpolation
    if q.ndim == 0:  # Scalar quantile
        val_low = x_sorted[idx_low]
        val_high = x_sorted[idx_high]
        return val_low + frac * (val_high - val_low)
    else:  # Multiple quantiles
        val_low = x_sorted[idx_low]
        val_high = x_sorted[idx_high]
        return val_low + frac * (val_high - val_low)


def robust_quantile_range(
    x: torch.Tensor,
    lower: float = 0.05,
    upper: float = 0.95,
    dim: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute robust quantile range (e.g., 5th-95th percentile).
    
    Useful for outlier-resistant normalization.
    
    Args:
        x: Input tensor
        lower: Lower quantile (default: 0.05 = 5th percentile)
        upper: Upper quantile (default: 0.95 = 95th percentile)
        dim: Dimension along which to compute
        
    Returns:
        (lower_val, upper_val) tuple
        
    Example:
        >>> x = torch.randn(1000)
        >>> x_min, x_max = robust_quantile_range(x, 0.05, 0.95)
        >>> x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    """
    q = torch.tensor([lower, upper], dtype=x.dtype, device=x.device)
    vals = safe_quantile(x, q, dim=dim)
    
    if vals.ndim == 0:
        # Scalar input
        return vals, vals
    elif vals.shape[0] == 2:
        return vals[0], vals[1]
    else:
        # Multi-dimensional
        return vals[0], vals[1]


# ============================================================================
# ⚡ Fast Approximate Quantile (Compatibility Alias)
# ============================================================================
def approx_quantile(x: torch.Tensor, q: float, n_bins: int = 16384) -> torch.Tensor:
    """
    ⚡ 빠른 근사 quantile (safe_quantile 재사용).
    
    대용량 텐서에서 정확한 정렬 대신 기존 구현 활용.
    n_bins는 API 호환성용 (현재 미사용).
    """
    return safe_quantile(x, q)