"""
Surface Detection - Combined Metric (Planarity + Spacing) with Adaptive Threshold

Core idea:
1. PCA → planarity (lower = more planar) + spacing (higher = more open)
2. Combined metric = (1-w)*planarity + w*(1-spacing)  [lower = more surface]
3. Adaptive threshold = soft_min(percentile_thr, absolute_thr)
   - percentile_thr: Top N% guide
   - absolute_thr: μ - k*σ (quality bound)
   - If surface weak → select fewer than N% ✓
4. Score = sigmoid(-z_score / tau) ^ power
5. Uniformize probability within selected points
6. Done!

Key improvements:
- Interior points (low spacing) excluded even if planar
- Adaptive selection: can select fewer than target percentile if justified
- All operations fully differentiable
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from ..analysis.pca import batched_pca_surface_optimized
from ..utils.config import EPS_PCA
from ..utils.quantile import safe_quantile, robust_quantile_range


def soft_quantile(x: torch.Tensor, q: float, tau: float = 0.01) -> torch.Tensor:
    """
    Differentiable quantile approximation using softmax weighting.
    
    Args:
        x: (N,) values
        q: Quantile (0.0-1.0), e.g., 0.1 = 10th percentile
        tau: Temperature for softmax (smaller = sharper)
        
    Returns:
        Scalar tensor approximating q-th quantile
    """
    N = x.shape[0]
    device = x.device
    q = max(0.0, min(1.0, q))
    
    # Sort values (differentiable)
    x_sorted, _ = torch.sort(x)
    
    # Target index
    target_idx = q * (N - 1)
    positions = torch.arange(N, device=device, dtype=x.dtype)
    
    # Soft weights around target position
    distances = torch.abs(positions - target_idx)
    weights = F.softmax(-distances / tau, dim=0)
    
    # Weighted sum (fully differentiable)
    return (weights * x_sorted).sum()


def _soft_clamp_min(x: torch.Tensor, a: torch.Tensor, beta: float = 40.0) -> torch.Tensor:
    """Smooth approximation of max(x, a) using softplus."""
    return a + F.softplus(beta * (x - a)) / beta


def _uniformize_within_mask(
    p: torch.Tensor,
    mask: torch.Tensor,
    target_ratio: float = 1.10,
    alpha_max: float = 50.0,
    beta_clamp: float = 40.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Uniformize probability distribution within masked region.
    
    Goal: Make max/min ratio ≤ target_ratio within mask support.
    Method: Soft power transform + soft clamping (fully differentiable).
    
    Args:
        p: (N,) probability distribution
        mask: (N,) soft mask [0,1]
        target_ratio: Target max/min ratio (e.g., 1.10 = 10% variation)
        alpha_max: Softmax power for max estimation
        beta_clamp: Soft clamp beta
        
    Returns:
        p_flat: (N,) uniformized probability
    """
    m = mask.clamp(0, 1)
    if m.sum().item() < eps:
        return p
    
    # Masked distribution
    mass_in = (p * m).sum() + eps
    q = (p * m) / mass_in
    
    # Soft max estimation
    w = torch.softmax(alpha_max * q, dim=0) * m
    w = w / (w.sum() + eps)
    q_max_soft = (w * q).sum()
    
    # Target minimum
    q_min_target = q_max_soft / max(float(target_ratio), 1.0 + 1e-6)
    
    # Soft clamp
    q_flat = _soft_clamp_min(q, q_min_target, beta=beta_clamp)
    
    # Reproject
    p_in = q_flat * mass_in
    p_out = p * (1.0 - m)  # Preserve outside (for soft masks)
    
    p_new = p_in + p_out
    p_new = p_new / (p_new.sum() + eps)
    
    return p_new


def detect_surface(
    x: torch.Tensor,
    knn,
    cfg: Dict,
    state: Optional[Dict] = None,
    return_curvature_dirs: bool = False,
    return_score: bool = False
) -> Tuple:
    """
    Detect surface points and compute sampling probability.
    
    Pipeline:
        1. PCA analysis → planarity + spacing
        2. Combined metric: (1-w)*planarity + w*(1-spacing)
        3. Adaptive threshold: soft_min(percentile, absolute)
           - Can select fewer than target % if surface quality is low
        4. Score computation (sigmoid + power)
        5. Threshold-based selection (STE for gradients)
        6. Uniformization (flatten probability within surface)
    
    All operations are differentiable. Combined metric prevents interior selection.
    
    Args:
        x: (N, 3) point positions
        knn: KNN function
        cfg: Configuration dict with keys:
            - k: int, PCA neighbors (default: 48)
            - planarity_percentile: float, top N% guide (default: 10.0)
            - spacing_weight: float, 0-1, balance planarity/spacing (default: 0.5)
            - surface_n_sigma: float, absolute threshold = μ-k*σ (default: 1.0)
            - soft_tau: float, score sharpness (default: 0.5)
            - surface_power: float, concentration (default: 4.0)
            - ema_beta: float, threshold smoothing (default: 0.95)
            - threshold_tau: float, mask sharpness (default: 0.01)
            - uniformize_target_ratio: float, max/min ratio (default: 1.10)
        state: Optional state dict
        return_curvature_dirs: If True, return curvature directions
        return_score: If True, return unnormalized score
    
    Returns:
        Tuple of (surf_prob, normals, spacing, state, [curvature_dirs], [score])
    """
    if state is None:
        state = {}
    
    # Parameters
    k = int(cfg.get("k", 48))
    percentile = float(cfg.get("planarity_percentile", 10.0))
    soft_tau = float(cfg.get("soft_tau", 0.5))
    surface_power = float(cfg.get("surface_power", 4.0))
    ema_beta = float(cfg.get("ema_beta", 0.95))
    threshold_tau = float(cfg.get("threshold_tau", 0.01))
    
    # ========================================================================
    # Step 1: PCA Analysis
    # ========================================================================
    idx, w = knn(x, x, k)
    pca_result = batched_pca_surface_optimized(
        x, idx, w, return_principal_dirs=return_curvature_dirs
    )
    
    if return_curvature_dirs:
        normals, planarity, spacing, curvature, principal_dir1, principal_dir2, principal_curv = pca_result
    else:
        normals, planarity, spacing, curvature = pca_result
    
    # ========================================================================
    # Step 2: Combined Surface Metric (planarity + spacing)
    # ========================================================================
    # Normalize using percentile-based robust scaling (avoid collapse)
    plan_p05, plan_p95 = robust_quantile_range(planarity, lower=0.05, upper=0.95)
    plan_norm = ((planarity - plan_p05) / (plan_p95 - plan_p05 + 1e-8)).clamp(0.0, 1.0)
    
    spacing_p05, spacing_p95 = robust_quantile_range(spacing, lower=0.05, upper=0.95)
    spacing_norm = ((spacing - spacing_p05) / (spacing_p95 - spacing_p05 + 1e-8)).clamp(0.0, 1.0)
    
    # Combined surface metric (lower = more surface-like)
    # - Low planarity (flat) → surface
    # - High spacing (open side) → surface
    use_spacing = bool(cfg.get("use_spacing_score", True))
    spacing_weight = float(cfg.get("spacing_weight", 0.5))  # 0.0-1.0
    
    if use_spacing:
        # Combined: weighted sum (lower = better surface)
        surface_metric = (1.0 - spacing_weight) * plan_norm + spacing_weight * (1.0 - spacing_norm)
    else:
        # Planarity only
        surface_metric = plan_norm
    
    # ========================================================================
    # Step 3: Adaptive Threshold (percentile + absolute bound)
    # ========================================================================
    # Percentile threshold: select top N%
    q = percentile / 100.0  # e.g., 20% → q=0.2 (20th percentile)
    percentile_thr = soft_quantile(surface_metric, q, tau=0.01)
    
    # Absolute threshold: only select if surface-like enough
    # Surface metric range: [0, 1], where 0 = perfect surface, 1 = perfect interior
    # Use adaptive bound: mean - k*std (points significantly below average)
    metric_mean = surface_metric.mean()
    metric_std = surface_metric.std()
    n_sigma = float(cfg.get("surface_n_sigma", 1.0))  # How many std below mean
    absolute_thr = metric_mean - n_sigma * metric_std
    
    # Soft minimum: use stricter threshold (lower value)
    # If absolute_thr < percentile_thr: use absolute (fewer selections, surface weak)
    # If absolute_thr > percentile_thr: use percentile (more selections, surface strong)
    beta_soft_min = float(cfg.get("threshold_soft_min_beta", 20.0))
    # soft_min(a, b) = a - softplus(beta * (a - b)) / beta
    raw_thr = percentile_thr - F.softplus(beta_soft_min * (percentile_thr - absolute_thr)) / beta_soft_min
    
    # EMA smoothing over time
    ema_prev = state.get("ema_thr")
    if ema_prev is None:
        ema_thr = raw_thr
    else:
        ema_thr = ema_beta * ema_prev + (1.0 - ema_beta) * raw_thr
    
    # ========================================================================
    # Step 4: Score Computation (Z-score on combined metric)
    # ========================================================================
    # Z-score standardization for score calculation (reuse mean/std from Step 3)
    z_metric = (surface_metric - metric_mean) / (metric_std + 1e-8)
    
    # Score: lower metric → higher score
    base_score = torch.sigmoid(-z_metric / soft_tau)
    
    # Concentration (amplify differences)
    if surface_power != 1.0:
        score = torch.pow(base_score, surface_power)
    else:
        score = base_score
    
    # ========================================================================
    # Step 5: Threshold-based Selection (hard mask with soft gradient)
    # ========================================================================
    # Hard mask (forward pass) - uses combined metric
    mask_hard = (surface_metric < ema_thr).float()
    
    # Soft mask (backward pass) - differentiable
    mask_soft = torch.sigmoid(-(surface_metric - ema_thr) / threshold_tau)
    
    # Straight-Through Estimator: forward hard, backward soft
    mask = mask_hard + mask_soft - mask_soft.detach()
    # Forward: exactly 0 or 1 (hard selection)
    # Backward: smooth gradients (differentiable)
    
    # ========================================================================
    # Step 5: Probability Distribution
    # ========================================================================
    # Score-weighted probability (only within mask)
    p_raw = score * mask  # Hard mask: only selected get non-zero
    p_raw = p_raw / (p_raw.sum() + 1e-8)
    
    # 🔥 Uniformization (flatten distribution within selected points)
    use_uniformize = bool(cfg.get("uniformize_enabled", True))
    if use_uniformize:
        # Use hard mask for uniformization (clear inside/outside boundary)
        # But p_raw already has mask applied (from score * mask), so just flatten it
        p_flat = _uniformize_within_mask(
            p_raw,
            mask=mask_hard,  # Hard mask for clear inside/outside
            target_ratio=float(cfg.get("uniformize_target_ratio", 1.05)),
            alpha_max=float(cfg.get("uniformize_alpha_max", 50.0)),
            beta_clamp=float(cfg.get("uniformize_beta", 40.0)),
            eps=1e-8
        )
        # Apply mask gradient (STE) to enable backward pass
        surf_prob = mask * p_flat / (mask * p_flat).sum().clamp(min=1e-8)
    else:
        surf_prob = p_raw
    
    # 🔥 DEBUG: Check how many have non-zero probability
    with torch.no_grad():
        n_nonzero = (surf_prob > 0).sum().item()
        if cfg.get("debug", {}).get("verbose", True):
            print(f"  [DEBUG] Non-zero probabilities: {n_nonzero} / {len(surf_prob)}")
    
    # ========================================================================
    # Statistics & Logging
    # ========================================================================
    with torch.no_grad():
        # Hard threshold for counting (statistics only)
        surface_mask = (surface_metric < ema_thr)
        n_surface = int(surface_mask.sum().item())
        surface_frac = n_surface / x.shape[0]
        
        thr_value = float(ema_thr.item()) if ema_thr.numel() == 1 else float(ema_thr.mean().item())
        
        if n_surface > 0:
            probs_selected = surf_prob[surface_mask]
            prob_min = float(probs_selected.min().item())
            prob_max = float(probs_selected.max().item())
            prob_ratio = prob_max / (prob_min + 1e-9)
            
            # Additional stats
            metric_selected = surface_metric[surface_mask]
            metric_selected_mean = float(metric_selected.mean().item())
            metric_selected_std = float(metric_selected.std().item())
        else:
            prob_min = prob_max = prob_ratio = 0.0
            metric_selected_mean = metric_selected_std = 0.0
    
    if cfg.get("debug", {}).get("verbose", True):
        print(f"\n  === Surface Detection (Combined Metric: Planarity + Spacing) ===")
        print(f"  Input: {x.shape[0]} points")
        print(f"  Target: ~Top {percentile:.1f}% (adaptive, may select fewer)")
        print(f"  ")
        print(f"  Surface Metric (lower = more surface-like):")
        print(f"    Range: [{float(surface_metric.min().detach()):.4f}, {float(surface_metric.max().detach()):.4f}]")
        print(f"    Mean ± Std: {float(metric_mean.detach()):.4f} ± {float(metric_std.detach()):.4f}")
        print(f"  ")
        print(f"  Threshold Selection:")
        print(f"    Percentile ({percentile:.0f}%): {float(percentile_thr.detach()):.6f}")
        print(f"    Absolute (μ-{n_sigma:.1f}σ): {float(absolute_thr.detach()):.6f}")
        print(f"    Final (EMA): {thr_value:.6f}")
        print(f"    → Selected: {n_surface} points ({100*surface_frac:.1f}%)")
        if n_surface > 0:
            print(f"    → Metric: {metric_selected_mean:.4f} ± {metric_selected_std:.4f}")
        print(f"  ")
        print(f"  Surface probability:")
        print(f"    Mean: {surf_prob.mean():.6f}")
        print(f"    Std: {surf_prob.std():.6f}")
        print(f"    Range: [{prob_min:.4e}, {prob_max:.4e}]")
        print(f"    Ratio (max/min): {prob_ratio:.2f}x")
        print(f"  ")
        print(f"  Parameters:")
        print(f"    soft_tau: {soft_tau:.3f} (score sharpness)")
        print(f"    surface_power: {surface_power:.1f} (concentration)")
        print(f"    threshold_tau: {threshold_tau:.3f} (mask sharpness)")
        if use_spacing:
            print(f"    spacing_weight: {spacing_weight:.2f} (planarity vs spacing)")
            print(f"    → Combined metric = {1-spacing_weight:.2f}*planarity + {spacing_weight:.2f}*(1-spacing)")
        else:
            print(f"    spacing: disabled (planarity only)")
        if use_uniformize:
            print(f"    uniformize: enabled (target ratio: {cfg.get('uniformize_target_ratio', 1.05):.2f})")
        else:
            print(f"    uniformize: disabled")
        print(f"  ")
        print(f"  ✓ Fully differentiable (combined metric prevents interior selection)")
        print(f"  ================================================================\n")
    
    # Update state (detach to prevent gradient accumulation across passes)
    state["ema_thr"] = ema_thr.detach()
    state["n_surface_anchors"] = int(n_surface)
    state["surface_fraction"] = float(surface_frac)
    
    # Build returns
    base_returns = (surf_prob, normals, spacing, state)
    
    if return_curvature_dirs:
        base_returns = base_returns + (principal_dir1, principal_dir2, principal_curv)
    
    if return_score:
        base_returns = base_returns + (score,)
    
    return base_returns


__all__ = [
    "detect_surface",
    "soft_quantile",
]
