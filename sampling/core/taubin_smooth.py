"""
Differentiable Taubin smoothing.

Taubin smoothing = Two-pass Laplacian:
1. Forward pass: smoothing (positive λ)
2. Backward pass: inflation (negative λ)

This prevents shrinkage while maintaining differentiability.
"""

import torch
from typing import Dict
from ..utils.config import EPS_SAFE


def compute_laplacian(
    P: torch.Tensor,
    knn,
    k: int
) -> torch.Tensor:
    """
    Compute Laplacian displacement.
    
    Args:
        P: (N, 3) positions
        knn: KNN function
        k: Number of neighbors
    
    Returns:
        lap: (N, 3) Laplacian displacement
    """
    idx, w = knn(P, P, k)
    Q = P[idx]  # (N, k, 3)
    
    # Weighted centroid
    centroid = (w.unsqueeze(-1) * Q).sum(dim=1)  # (N, 3)
    
    # Laplacian
    lap = centroid - P
    
    return lap


def split_tangent_normal(
    lap: torch.Tensor,
    normals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split Laplacian into tangent and normal components.
    
    Args:
        lap: (N, 3) Laplacian displacement
        normals: (N, 3) surface normals
    
    Returns:
        lap_t: (N, 3) tangent component
        lap_n: (N, 3) normal component
    """
    # Normal component
    lap_n = (lap * normals).sum(-1, keepdim=True) * normals
    
    # Tangent component
    lap_t = lap - lap_n
    
    return lap_t, lap_n


def taubin_smooth(
    P: torch.Tensor,
    normals: torch.Tensor,
    knn,
    cfg: Dict
) -> torch.Tensor:
    """
    Apply differentiable Taubin smoothing.
    
    Args:
        P: (N, 3) point positions
        normals: (N, 3) surface normals
        knn: KNN function
        cfg: Configuration dict
    
    Returns:
        P_smooth: (N, 3) smoothed positions
    """
    iters = int(cfg.get("iters", 3))
    k = int(cfg.get("k", 24))
    lambda_smooth = float(cfg.get("lambda_smooth", 0.6))
    lambda_inflate = float(cfg.get("lambda_inflate", -0.53))
    tangent_only = bool(cfg.get("tangent_only", True))
    
    for t in range(iters):
        # ====================================================================
        # PASS 1: Smoothing (positive λ)
        # ====================================================================
        lap_smooth = compute_laplacian(P, knn, k)
        
        if tangent_only:
            lap_t, lap_n = split_tangent_normal(lap_smooth, normals)
            P_temp = P + lambda_smooth * lap_t
        else:
            P_temp = P + lambda_smooth * lap_smooth
        
        # ====================================================================
        # PASS 2: Inflation (negative λ)
        # ====================================================================
        # Recompute KNN on smoothed positions
        lap_inflate = compute_laplacian(P_temp, knn, k)
        
        if tangent_only:
            lap_t2, lap_n2 = split_tangent_normal(lap_inflate, normals)
            P = P_temp + lambda_inflate * lap_t2
        else:
            P = P_temp + lambda_inflate * lap_inflate
    
    return P


__all__ = [
    "taubin_smooth",
    "compute_laplacian",
    "split_tangent_normal",
]