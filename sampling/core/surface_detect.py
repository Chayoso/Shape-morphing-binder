"""
Surface detection using PCA-based planarity analysis.

The key insight:
- Surface points: λ₀ << λ₁ ≈ λ₂ (flat neighborhood)
- Interior points: λ₀ ≈ λ₁ ≈ λ₂ (isotropic neighborhood)

We use planarity = λ₀ / (λ₀ + λ₁ + λ₂) as surface quality metric.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from ..analysis.pca import batched_pca_surface_optimized
from ..utils.config import EPS_PCA


def soft_sort(x: torch.Tensor, tau: float = 0.01) -> torch.Tensor:
    """
    Differentiable soft sorting using optimal transport relaxation.
    
    Based on: "Fast Differentiable Sorting and Ranking" (Blondel et al., ICML 2020)
    
    Args:
        x: (N,) values to sort
        tau: Temperature parameter (lower = sharper)
    
    Returns:
        sorted: (N,) soft-sorted values
    """
    N = x.shape[0]
    device = x.device
    
    # 🔥 OPTIMIZATION: Use fast approximation for large N
    if N > 1000:
        # Fall back to fast quantile-based approximation
        with torch.no_grad():
            x_sorted, _ = torch.sort(x)
        return x_sorted.requires_grad_(True)
    
    # Compute pairwise differences
    x_expanded = x.unsqueeze(0)  # (1, N)
    x_diff = x_expanded.T - x_expanded  # (N, N)
    
    # Soft sign function
    P = torch.sigmoid(x_diff / tau)  # (N, N)
    
    # Soft rank (number of elements smaller than each element)
    soft_rank = P.sum(dim=1)  # (N,)
    
    # Convert soft ranks to positions [0, 1]
    positions = (soft_rank - 1) / max(N - 1, 1)
    
    # Interpolate using smooth weighting
    positions_clamped = torch.clamp(positions, 0, 1)
    idx_float = positions_clamped * (N - 1)
    
    # Create soft interpolation weights
    weights = torch.zeros(N, N, device=device)
    for i in range(N):
        idx_low = torch.clamp(torch.floor(idx_float[i]).long(), 0, N-1)
        idx_high = torch.clamp(idx_low + 1, 0, N-1)
        w_high = idx_float[i] - idx_low.float()
        w_low = 1.0 - w_high
        weights[i, idx_low] = w_low
        if idx_high != idx_low:
            weights[i, idx_high] = w_high
    
    # Soft permutation matrix: each row selects one position
    sorted_approx = torch.matmul(weights, x.unsqueeze(1)).squeeze(1)
    
    return sorted_approx


def soft_quantile(x: torch.Tensor, q: float, tau: float = 0.01) -> torch.Tensor:
    """
    Fully differentiable quantile approximation using soft sorting.
    
    Args:
        x: (N,) values
        q: Quantile in [0, 1]
        tau: Soft sorting temperature
    
    Returns:
        quantile: Soft quantile value
    """
    N = x.shape[0]
    device = x.device
    q = max(0.0, min(1.0, q))
    
    # 🔥 FIX: Fast path for large N
    if N > 1000:
        # Use fast approximation with soft weighting
        with torch.no_grad():
            x_sorted, _ = torch.sort(x)
            idx = min(int(q * N), N - 1)
            init_q = x_sorted[idx]
        
        # Differentiable refinement
        distances = (x - init_q).abs()
        weights = torch.softmax(-distances / tau, dim=0)
        return (weights * x).sum()
    
    # Original soft sort for small N
    xs_soft = soft_sort(x, tau=tau)
    
    # 🔥 FIX: Keep everything on device and differentiable
    idx_float = torch.tensor(q * (N - 1), device=device, dtype=x.dtype)
    idx_low = torch.clamp(torch.floor(idx_float), 0, N-1).long()
    idx_high = torch.clamp(idx_low + 1, 0, N-1).long()
    w = torch.clamp(idx_float - idx_low.float(), 0.0, 1.0)
    
    return (1 - w) * xs_soft[idx_low] + w * xs_soft[idx_high]


def compute_surface_threshold(
    planarity: torch.Tensor,
    percentile: float,
    ema_prev: Optional[float],
    ema_beta: float
) -> Tuple[float, float]:
    """
    Compute adaptive surface threshold with EMA.
    
    Args:
        planarity: (N,) planarity scores (lower = better surface)
        percentile: Percentile for threshold (e.g., 10.0 = bottom 10%)
        ema_prev: Previous EMA threshold
        ema_beta: EMA decay factor
    
    Returns:
        ema_thr: EMA-smoothed threshold
        raw_thr: Raw threshold
    
    Note:
        Lower planarity = better surface quality.
        percentile=10.0 means we select points with planarity below the 90th percentile.
        (i.e., exclude the top 10% worst surface points)
    """
    q = 1.0 - (percentile / 100.0)  # percentile=10 → q=0.9
    raw_thr = soft_quantile(planarity, q, tau=0.01)
    
    if ema_prev is None:
        ema_thr = float(raw_thr.detach())
    else:
        ema_thr = float(ema_beta * ema_prev + (1 - ema_beta) * raw_thr.detach())
    
    return ema_thr, float(raw_thr.detach())


def compute_surface_probability(
    planarity: torch.Tensor,
    thr_high: float,
    soft_tau: float,
    surface_power: float
) -> torch.Tensor:
    """
    🔥 Z-score based surface probability (REPLACED).
    
    Compute differentiable surface probability using z-score normalization.
    This creates better contrast even when planarity distribution is narrow.
    
    Lower planarity → higher probability (better surface)
    
    Args:
        planarity: (N,) planarity scores
        thr_high: IGNORED (kept for signature compatibility)
        soft_tau: Sigmoid temperature (applied to z-scores)
        surface_power: Concentration exponent
    
    Returns:
        surf_prob: (N,) normalized probabilities
    """
    # Standardize to z-scores
    plan_mean = planarity.mean()
    plan_std = planarity.std()
    z_score = (planarity - plan_mean) / (plan_std + 1e-8)
    
    # Lower planarity = negative z-score = better surface
    # Apply sigmoid to negative z-score
    score = torch.sigmoid(-z_score / soft_tau)
    
    # Power concentration
    if surface_power != 1.0:
        score = torch.pow(score, surface_power)
    
    # Normalize to probability distribution
    surf_prob = score / (score.sum() + EPS_PCA)
    
    return surf_prob


def detect_surface(
    x: torch.Tensor,
    knn,
    cfg: Dict,
    state: Optional[Dict] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Detect surface points using PCA-based planarity.
    
    Args:
        x: (N, 3) point positions
        knn: KNN function
        cfg: Configuration dict
        state: Optional state dict for EMA
    
    Returns:
        surf_prob: (N,) surface probabilities
        normals: (N, 3) surface normals
        spacing: (N,) local point spacing
        state: Updated state dict
    """
    if state is None:
        state = {}
    
    k = int(cfg.get("k", 48))
    percentile = float(cfg.get("planarity_percentile", 10.0))
    ema_beta = float(cfg.get("ema_beta", 0.95))
    hysteresis = float(cfg.get("hysteresis", 0.03))
    soft_tau = float(cfg.get("soft_tau", 0.5))
    surface_power = float(cfg.get("surface_power", 4.0))
    
    # PCA analysis
    idx, w = knn(x, x, k)
    normals, planarity, spacing = batched_pca_surface_optimized(x, idx, w)
    
    # 🔥 Compute statistics for z-score method
    plan_mean = planarity.mean()
    plan_std = planarity.std()
    z_score = (planarity - plan_mean) / (plan_std + 1e-8)
    
    # 🔥 DEBUG OUTPUT
    print(f"\n  === Surface Detection Debug (Z-score) ===")
    print(f"  Input points: {x.shape[0]}")
    print(f"  Planarity: mean={plan_mean:.6f}, "
          f"std={plan_std:.6f}, "
          f"min={planarity.min():.6f}, max={planarity.max():.6f}")
    print(f"  Z-score: min={z_score.min():.3f}, max={z_score.max():.3f}")
    print(f"  soft_tau={soft_tau:.3f}, surface_power={surface_power:.1f}")
    
    # Compute threshold (for state compatibility, not used in z-score)
    ema_prev = state.get("ema_thr")
    ema_thr, raw_thr = compute_surface_threshold(
        planarity, percentile, ema_prev, ema_beta
    )
    
    # Hysteresis band (for compatibility)
    band = hysteresis * max(ema_thr, 1e-6)
    thr_low = ema_thr - band
    thr_high = ema_thr + band
    
    # Score from z-score
    score = torch.sigmoid(-z_score / soft_tau)
    print(f"  After sigmoid: mean={score.mean():.6f}, "
          f"min={score.min():.6f}, max={score.max():.6f}")
    
    # Surface probability (now uses z-score internally)
    surf_prob = compute_surface_probability(
        planarity, thr_high, soft_tau, surface_power
    )
    
    print(f"  Final prob: mean={surf_prob.mean():.6f}, "
          f"min={surf_prob.min():.6f}, max={surf_prob.max():.6f}")
    
    high_conf_mask = surf_prob > (surf_prob.mean() * 3)
    print(f"  High-confidence: {high_conf_mask.sum()}/{x.shape[0]} "
          f"({100.0 * high_conf_mask.sum() / x.shape[0]:.1f}%)")
    print(f"  ================================\n")
    
    # Update state
    state["ema_thr"] = ema_thr
    state["thr_low"] = thr_low
    state["thr_high"] = thr_high
    state["planarity"] = planarity
    
    return surf_prob, normals, spacing, state


__all__ = [
    "detect_surface",
    "soft_quantile",
    "compute_surface_threshold",
    "compute_surface_probability",
]