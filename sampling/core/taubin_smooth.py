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


# def taubin_smooth(
#     P: torch.Tensor,
#     normals: torch.Tensor,
#     knn,
#     cfg: Dict
# ) -> torch.Tensor:
#     """
#     Apply differentiable Taubin smoothing.
    
#     Args:
#         P: (N, 3) point positions
#         normals: (N, 3) surface normals
#         knn: KNN function
#         cfg: Configuration dict
    
#     Returns:
#         P_smooth: (N, 3) smoothed positions
#     """
#     iters = int(cfg.get("iters", 3))
#     k = int(cfg.get("k", 24))
#     lambda_smooth = float(cfg.get("lambda_smooth", 0.6))
#     lambda_inflate = float(cfg.get("lambda_inflate", -0.53))
#     tangent_only = bool(cfg.get("tangent_only", True))
    
#     for t in range(iters):
#         # ====================================================================
#         # PASS 1: Smoothing (positive λ)
#         # ====================================================================
#         lap_smooth = compute_laplacian(P, knn, k)
        
#         if tangent_only:
#             lap_t, lap_n = split_tangent_normal(lap_smooth, normals)
#             P_temp = P + lambda_smooth * lap_t
#         else:
#             P_temp = P + lambda_smooth * lap_smooth
        
#         # ====================================================================
#         # PASS 2: Inflation (negative λ)
#         # ====================================================================
#         # Recompute KNN on smoothed positions
#         lap_inflate = compute_laplacian(P_temp, knn, k)
        
#         if tangent_only:
#             lap_t2, lap_n2 = split_tangent_normal(lap_inflate, normals)
#             P = P_temp + lambda_inflate * lap_t2
#         else:
#             P = P_temp + lambda_inflate * lap_inflate
    
#     return P

def taubin_smooth(
    P: torch.Tensor,
    normals: torch.Tensor,
    knn,
    cfg: Dict
) -> torch.Tensor:
    """
    Apply differentiable Taubin smoothing (two-pass Laplacian per iteration).

    Taubin smoothing prevents shrinkage by composing:
      1) a forward Laplacian smoothing pass with positive λ (lambda_smooth)
      2) a backward "inflation" pass with negative λ (lambda_inflate)

    This implementation is point-cloud friendly and differentiable. To reduce
    peak memory without adding extra config knobs, we:
      - compute the KNN graph once per iteration and reuse it in both passes
        (typical Taubin assumption: fixed connectivity within an iteration)
      - avoid in-place ops to keep autograd graphs healthy

    Args:
        P: (N, 3) point positions (will be updated)
        normals: (N, 3) surface normals (used for tangent-only projection)
        knn: KNN function returning (idx, w) for k neighbors
        cfg: Configuration dict with keys:
             - 'iters' (int, default 3)
             - 'k' (int, default 24)
             - 'lambda_smooth' (float, default 0.6)
             - 'lambda_inflate' (float, default -0.53)
             - 'tangent_only' (bool, default True)

    Returns:
        P_smooth: (N, 3) smoothed positions
    """
    iters = int(cfg.get("iters", 3))
    k = int(cfg.get("k", 24))
    lambda_smooth = float(cfg.get("lambda_smooth", 0.6))
    lambda_inflate = float(cfg.get("lambda_inflate", -0.53))
    tangent_only = bool(cfg.get("tangent_only", True))

    for t in range(iters):
        # --------------------------------------------------------------------
        # Build KNN graph ONCE per iteration and reuse it in both passes.
        # This reduces KNN calls from 2× to 1× per iteration and cuts peaks.
        # --------------------------------------------------------------------
        idx, w = knn(P, P, k)          # idx: (N, k), w: (N, k)

        # ====================================================================
        # PASS 1: Smoothing (positive λ)
        #   centroid = Σ_j w_ij * P_j,   lap = centroid - P
        #   P_temp   = P + λ_smooth * lap_tangent(or lap)
        # ====================================================================
        Q = P[idx]                      # (N, k, 3) neighbor positions
        centroid = (w.unsqueeze(-1) * Q).sum(dim=1)  # (N, 3)
        lap_smooth = centroid - P                   # (N, 3)

        if tangent_only:
            # Project Laplacian onto the tangent plane: lap_t = lap - (lap·n)n
            lap_n = (lap_smooth * normals).sum(-1, keepdim=True) * normals
            lap_t = lap_smooth - lap_n
            P_temp = P + lambda_smooth * lap_t
        else:
            P_temp = P + lambda_smooth * lap_smooth

        # ====================================================================
        # PASS 2: Inflation (negative λ)
        #   Reuse the SAME graph (idx, w) for a Taubin-style fixed connectivity
        #   within an iteration. This is the major memory/latency reduction.
        # ====================================================================
        Q2 = P_temp[idx]                                # (N, k, 3)
        centroid2 = (w.unsqueeze(-1) * Q2).sum(dim=1)  # (N, 3)
        lap_inflate = centroid2 - P_temp               # (N, 3)

        if tangent_only:
            lap2_n = (lap_inflate * normals).sum(-1, keepdim=True) * normals
            lap2_t = lap_inflate - lap2_n
            P = P_temp + lambda_inflate * lap2_t
        else:
            P = P_temp + lambda_inflate * lap_inflate

        # Clean up big temporaries promptly to help GPU memory reuse
        del Q, centroid, lap_smooth
        del Q2, centroid2, lap_inflate
        if tangent_only:
            del lap_n, lap_t, lap2_n, lap2_t

    return P



__all__ = [
    "taubin_smooth",
    "compute_laplacian",
    "split_tangent_normal",
]