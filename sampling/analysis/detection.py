"""Surface detection and probability computation."""

import torch
from typing import Tuple, Optional, Dict
from ..utils.config import EPS_PCA, MIN_PROB
from ..analysis.pca import batched_pca_surface_optimized


def soft_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    """Piecewise-differentiable quantile approximation."""
    N = x.shape[0]
    xs, _ = torch.sort(x)
    
    q = max(0.0, min(1.0, q))
    
    idx = torch.tensor(q * (N - 1), dtype=torch.float32, device=x.device)
    idx_low = torch.clamp(torch.floor(idx).long(), 0, N-1)
    idx_high = torch.clamp(idx_low + 1, 0, N-1)
    w = torch.clamp(idx - idx_low.float(), 0.0, 1.0)
    
    return (1 - w) * xs[idx_low] + w * xs[idx_high]


def compute_surface_threshold(
    surfvar: torch.Tensor, 
    thr_percentile: float,
    ema_prev: Optional[float],
    ema_beta: float
) -> Tuple[float, float, float]:
    """Compute adaptive surface threshold with EMA."""
    thr_now = soft_quantile(surfvar, thr_percentile / 100.0)
    
    if ema_prev is None:
        ema_thr = float(thr_now.detach())
    else:
        ema_thr = float(ema_beta * ema_prev + (1 - ema_beta) * thr_now.detach())
    
    return ema_thr, float(thr_now.detach()), ema_thr


def compute_surface_probability(
    surfvar: torch.Tensor,
    thr_high: float,
    soft_tau: float,
    surface_power: float
) -> torch.Tensor:
    """Compute differentiable surface probability."""
    score = torch.sigmoid(-(surfvar - thr_high) / max(soft_tau, 1e-6))
    
    surf_prob = score ** surface_power
    surf_prob = surf_prob / (surf_prob.sum() + EPS_PCA)
    
    min_p = surf_prob.max() * MIN_PROB
    surf_prob = torch.clamp(surf_prob, min=min_p)
    surf_prob = surf_prob / surf_prob.sum()
    
    return surf_prob


def compute_surface_mask_diff(
    x: torch.Tensor,
    knn,
    k_surface: int,
    thr_percentile: float,
    ema_prev: Optional[float],
    ema_beta: float,
    hysteresis: float,
    soft_tau: float,
    surface_power: float,
    state_out: Dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Classify surface-like seeds and produce differentiable sampling distribution."""
    idx, w = knn(x, x, k_surface)
    normals, surfvar, spacing = batched_pca_surface_optimized(x, idx, w)
    
    ema_thr, thr_now, ema_thr = compute_surface_threshold(
        surfvar, thr_percentile, ema_prev, ema_beta
    )
    
    band = hysteresis * max(ema_thr, 1e-6)
    thr_low, thr_high = ema_thr - band, ema_thr + band
    
    surf_prob = compute_surface_probability(surfvar, thr_high, soft_tau, surface_power)
    
    state_out.update({
        "ema_thr": float(ema_thr),
        "thr_low": float(thr_low),
        "thr_high": float(thr_high)
    })
    
    return surf_prob, normals, spacing, float(thr_low), float(thr_high)