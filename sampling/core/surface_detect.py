"""
🔥 Surface Detection - Normalized Ratio-based Algorithm (ADAPTIVE ENHANCED!)

Algorithm:
1. Adaptive kNN-PCA: k = round(k0 * (1 - β·ŝ) + k_min), r = mean_nn_dist × (1.6~2.2)
   → Eigenvalue decomposition: λ₀ ≤ λ₁ ≤ λ₂, normal n = e₀ (smallest eigenvalue)
2. Ratio metrics (density-independent, 🔥 FIX 3: unified notation):
   - planarity: s = (λ₁+λ₂)/(λ₀+λ₁+λ₂+ε)  [HIGH = flat surface]
   - anisotropy: ρ = (λ₂-λ₁)/(λ₂+ε)       [HIGH = sharp edge/ridge]
3. Surface probability (calibrated):
   p_surf_raw = sigmoid((z_s - τ)/γ) + w_curv·ρ̂
   where z_s = (s - μ_s)/σ_s, τ ← quantile(z_s, q=1-p_target), p_target=0.80, γ=1.0
4. Curvature boost (percentile-gated):
   ρ̂ ← relu(ρ - P90(ρ)) / (P99 - P90), w_curv=0.4~0.6
5. Density weighting:
   w_density = spacing / mean(spacing)
6. Sampling distribution (🔥 FIX 1: separated from surface probability):
   π = normalize(p_surf_raw^α · w_density^β)  [sum=1, for Gumbel sampling]
7. Surface/interior ratio enforcement:
   Surface:Interior = 0.85:0.15 (via p_surf_raw thresholding on π)

Returns:
   - p_surf_raw: surface probability [0,1] → Stage 2 (surface-weighted density)
   - π (pi): sampling distribution (sum=1) → Stage 3 (Gumbel sampling)

All operations are fully differentiable.
Includes anti-banding: random k/r jitter (±10%), soft-TV smoothness.

Author: CHAYO (2025-10)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np
from ..analysis.pca import batched_pca_surface_optimized
from ..analysis.anti_grid import (
    compute_gridness,
    debias_planarity_anisotropy,
    compute_confidence_from_gridness
)


def detect_surface(
    x: torch.Tensor,
    knn,
    cfg: Dict,
    state: Optional[Dict] = None,
    return_curvature_dirs: bool = False,
    return_score: bool = False,
    verbose: bool = False
) -> Tuple:
    """
    🔥 REVISED: Normalized ratio-based surface detection (DIFFERENTIABLE).
    
    Algorithm (🔥 FIX 3: unified eigenvalue notation λ₀ ≤ λ₁ ≤ λ₂):
        1. kNN-PCA (k=24~48) → eigenvalues λ₀≤λ₁≤λ₂, normal n=e₀
        2. Ratio metrics:
           - planarity: s = (λ₁+λ₂)/(λ₀+λ₁+λ₂+ε)      [density-independent, HIGH=flat]
           - anisotropy: ρ = (λ₂-λ₁)/(λ₂+ε)            [sharpness, HIGH=edge/ridge]
        3. Surface probability:
           p_surf_raw = sigmoid((z_s - τ)/γ) + w_curv·ρ̂
           where z_s = (s - μ_s)/σ_s
        4. Curvature boost:
           p_surf_raw ← clamp(p_surf_raw + w_curv·ρ̂, 0, 1)
        5. Density weighting:
           w_density = spacing / mean(spacing)
        6. Sampling distribution (🔥 FIX 1: separated):
           π = normalize(p_surf_raw^α · w_density^β)  [sum=1]
    
    Args:
        x: (N, 3) point positions
        knn: KNN function
        cfg: Configuration dict with keys:
            - k: int, PCA neighbors (default: 48)
            - tau: float, sigmoid threshold τ (default: auto)
            - gamma: float, sigmoid steepness γ (default: 1.0)
            - w_curv: float, curvature boost weight (default: 0.5)
            - alpha: float, surface prob power (default: 2.5)
            - beta: float, density weight power (default: 0.7)
        state: Optional state dict for EMA tracking
        return_curvature_dirs: If True, return curvature directions
        return_score: Ignored (kept for API compatibility)
        verbose: Print diagnostics
    
    Returns:
        Tuple of (p_surf_raw, pi, normals, spacing, state, planarity, anisotropy, [curvature_dirs])
        - p_surf_raw: (N,) surface probability [0,1] → for Stage 2 (surface-weighted density)
        - pi: (N,) sampling distribution (sum=1) → for Stage 3 (Gumbel sampling)
    """
    if state is None:
        state = {}
    
    # ════════════════════════════════════════════════════════════════════════
    # Parameters (ADAPTIVE ENHANCED version)
    # ════════════════════════════════════════════════════════════════════════
    k0 = int(cfg.get("k", 48))                     # Base PCA neighbors
    k_min = int(cfg.get("k_min", 20))              # Minimum k (adaptive)
    beta_adapt = float(cfg.get("beta_adapt", 0.7))  # Planarity adaptation strength
    tau_raw = cfg.get("tau", None)
    tau = float(tau_raw) if tau_raw is not None else None  # Auto-calibrated if None
    gamma = float(cfg.get("gamma", 0.8))            # Sigmoid steepness (default 0.8)
    p_target = float(cfg.get("p_target", 0.85))    # Target surface ratio (0.85)
    w_curv = float(cfg.get("w_curv", 0.5))          # Curvature boost (0.4~0.6)
    alpha = float(cfg.get("alpha", 2.5))            # Surface prob power
    beta = float(cfg.get("beta", 0.7))              # Density weight power
    
    # Anti-banding jitter (k, r randomization)
    jitter_enabled = cfg.get("anti_banding", {}).get("enabled", True)
    jitter_k = float(cfg.get("anti_banding", {}).get("jitter_k", 0.1))  # ±10%
    jitter_r = float(cfg.get("anti_banding", {}).get("jitter_r", 0.1))  # ±10%
    
    # Surface/interior ratio enforcement
    surface_ratio = float(cfg.get("surface_ratio", 0.85))  # 0.85:0.15
    enforce_ratio = cfg.get("enforce_surface_ratio", True)
    
    N = x.shape[0]
    device = x.device
    eps = 1e-8
    
    # ════════════════════════════════════════════════════════════════════════
    # Step 0: Quick initial PCA for adaptive k calculation
    # ════════════════════════════════════════════════════════════════════════
    # Quick initial kNN for adaptive k (use small k for speed)
    k_init = min(k_min + 8, k0)
    if jitter_enabled:
        k_jitter = torch.rand(1, device=device).item() * (2 * jitter_k) - jitter_k  # [-jitter_k, +jitter_k]
        k_init = int(max(k_min, k_init * (1 + k_jitter)))
    
    idx_init, w_init = knn(x, x, k_init)
    pca_result_init = batched_pca_surface_optimized(
        x, idx_init, w_init, return_principal_dirs=False
    )
    _, _, spacing_init, _, _, planarity_init = pca_result_init[:6]
    
    # Adaptive k: k = round(k0 * (1 - β·ŝ) + k_min)
    # High planarity (flat) → lower k, Low planarity (curved) → higher k
    s_hat = planarity_init.mean()  # Global planarity estimate
    k_adaptive = int(round(k0 * (1.0 - beta_adapt * s_hat.item()) + k_min))
    k_adaptive = max(k_min, min(k_adaptive, int(N * 0.5)))  # Safety bounds
    
    if jitter_enabled:
        k_jitter_final = torch.rand(1, device=device).item() * (2 * jitter_k) - jitter_k
        k_adaptive = int(max(k_min, k_adaptive * (1 + k_jitter_final)))
    
    # Adaptive radius: r = mean_nn_dist × (1.6 ~ 2.2)
    mean_nn_dist = spacing_init.mean().item()
    r_base = mean_nn_dist * 1.9  # Default 1.9 (middle of 1.6~2.2)
    if jitter_enabled:
        r_jitter = torch.rand(1, device=device).item() * (2 * jitter_r) - jitter_r
        r_adaptive = r_base * (1 + r_jitter)
    else:
        r_adaptive = r_base
    
    # ════════════════════════════════════════════════════════════════════════
    # Step 1: kNN-PCA with Adaptive k → Normalized Ratio Metrics
    # ════════════════════════════════════════════════════════════════════════
    idx, w = knn(x, x, k_adaptive)
    pca_result = batched_pca_surface_optimized(
        x, idx, w, return_principal_dirs=return_curvature_dirs
    )
    
    if return_curvature_dirs:
        # 9 values
        normals, surfvar, spacing, curvature, anisotropy, planarity, principal_dir1, principal_dir2, principal_curv = pca_result
    else:
        # 6 values
        normals, surfvar, spacing, curvature, anisotropy, planarity = pca_result
    
    # Store mean_nn_dist for later use
    mean_nn_dist = spacing.mean()
    
    # ════════════════════════════════════════════════════════════════════════
    # Step 1.5: Anti-Grid De-biasing (OPTIONAL)
    # ════════════════════════════════════════════════════════════════════════
    anti_grid_enabled = cfg.get("anti_grid", {}).get("enabled", False)
    
    if anti_grid_enabled:
        # Compute gridness from neighbor directions
        neighbor_vec = x[idx] - x.unsqueeze(1)  # (N, k, 3)
        gridness = compute_gridness(neighbor_vec, num_bins=48)
        
        # De-bias parameters
        alpha_grid = float(cfg.get("anti_grid", {}).get("alpha_grid", 0.5))
        beta_grid = float(cfg.get("anti_grid", {}).get("beta_grid", 0.5))
        gamma_conf = float(cfg.get("anti_grid", {}).get("gamma_conf", 1.2))
        
        # De-bias planarity and anisotropy
        planarity_debias, anisotropy_debias = debias_planarity_anisotropy(
            planarity, anisotropy, gridness, alpha_grid, beta_grid
        )
        
        # Compute confidence gating
        conf = compute_confidence_from_gridness(gridness, gamma_conf, min_conf=0.2)
        
        # Use de-biased values
        planarity = planarity_debias
        anisotropy = anisotropy_debias
    else:
        gridness = None
        conf = torch.ones(N, device=device)  # No gating
    
    # ════════════════════════════════════════════════════════════════════════
    # Step 2: Surface Probability from Planarity (Z-score + Auto-Calibrated Sigmoid)
    # ════════════════════════════════════════════════════════════════════════
    # Z-score normalization: z_s = (s - μ_s)/σ_s
    s = planarity  # Alias for clarity (now de-biased if anti_grid enabled)
    s_mean = s.mean()
    s_std = torch.clamp(s.std(), min=1e-6)  # Stability floor (1e-6 as in snippet)
    z_s = (s - s_mean) / s_std
    
    # 🔥 FIX 1: Auto-calibrate τ with EMA hysteresis (prevent frame jitter)
    if tau is None:
        p_target_val = p_target  # e.g., 0.80
        tau_now = torch.quantile(z_s.detach(), 1.0 - p_target_val).item()
        
        # EMA hysteresis: tau_ema = 0.9 * tau_ema + 0.1 * tau_now
        tau_ema_key = "tau_ema_surface"
        if tau_ema_key in state:
            tau_ema = 0.9 * state[tau_ema_key] + 0.1 * tau_now
        else:
            tau_ema = tau_now  # First frame: use current value
        
        state[tau_ema_key] = tau_ema
        tau = tau_ema
    else:
        # Fixed tau (no EMA)
        tau = float(tau)
    
    # Surface probability: p_surf = sigmoid((z_s - τ)/γ)
    # HIGH planarity → HIGH z_s → HIGH p_surf
    p_surf_pre_boost = torch.sigmoid((z_s - tau) / gamma)
    
    # 🔥 Store z_s for visualization
    z_s_stored = z_s.clone()
    
    # 🔥 DEBUG: Check calibration BEFORE boost
    if verbose:
        p_above_50_pre = (p_surf_pre_boost > 0.5).float().mean().item()
        print(f"  [DEBUG] p_surf (pre-boost): mean={p_surf_pre_boost.mean():.3f}, >0.5={p_above_50_pre:.1%}")
        print(f"          tau={tau:.3f}, gamma={gamma:.2f}, z_s range=[{z_s.min():.2f}, {z_s.max():.2f}]")
        
        # Check if sigmoid is saturated
        sigmoid_input = (z_s - tau) / gamma
        saturated_low = (sigmoid_input < -3).float().mean().item()
        saturated_high = (sigmoid_input > 3).float().mean().item()
        print(f"          Sigmoid saturation: low={saturated_low:.1%}, high={saturated_high:.1%}")
    
    # ════════════════════════════════════════════════════════════════════════
    # Step 3: Curvature Boost (Percentile-Gated, Edge/Feature Enhancement)
    # ════════════════════════════════════════════════════════════════════════
    # 🔥 UPDATED: Use quantile (more accurate than sorted indexing)
    # ρ̂ ← relu(ρ - P90(ρ)) / (P99 - P90)
    rho = anisotropy  # Alias
    p90 = torch.quantile(rho.detach(), 0.90)
    p99 = torch.quantile(rho.detach(), 0.99)
    
    # Gated normalization: only boost top percentile
    rho_gate = torch.clamp((rho - p90) / (p99 - p90 + eps), 0.0, 1.0)
    
    # Store rho_gate for visualization (top 10% only >0)
    rho_gate_stored = rho_gate.clone()
    
    # Boost: p_surf ← clamp(p_surf_pre_boost + w_curv·ρ̂, 0, 1)
    # w_curv=0.4~0.6 recommended (default 0.5)
    p_surf = torch.clamp(p_surf_pre_boost + w_curv * rho_gate, 0.0, 1.0)
    
    # 🔥 Apply confidence gating (suppress grid regions)
    if anti_grid_enabled:
        p_surf = p_surf * conf  # Suppress low-confidence (high gridness) regions
    
    # ════════════════════════════════════════════════════════════════════════
    # Output: p_surf_raw only (π will be computed in pipeline after STAGE-2)
    # ════════════════════════════════════════════════════════════════════════
    # p_surf_raw ∈ [0,1] → Stage-2 (density weighting), Σ/opacity scale
    p_surf_raw = torch.clamp(p_surf, 0.0, 1.0)
    
    # ════════════════════════════════════════════════════════════════════════
    # Logging & State Updates
    # ════════════════════════════════════════════════════════════════════════
    # Get verbose flag (from parameter or config)
    if verbose is False:
        verbose = cfg.get("debug", {}).get("verbose", False)
    
    if verbose:
        p_surf_mean = p_surf_raw.mean().item()
        p_surf_above_50 = (p_surf_raw > 0.5).float().mean().item()
        
        print(f"\n  ═══ Surface Detection (p_surf_raw only) ═══")
        print(f"  Input: N = {N:,} points")
        print(f"  Adaptive k-NN:")
        print(f"    k = {k_adaptive} (base={k0}, min={k_min})")
        print(f"    ŝ (planarity mean) = {s_hat:.4f}")
        print(f"  ")
        print(f"  PCA Metrics:")
        print(f"    Planarity s:  [{planarity.min():.4f}, {planarity.max():.4f}], mean={s_mean:.4f}±{s_std:.4f}")
        p90_log = torch.quantile(anisotropy.detach(), 0.90).item()
        p99_log = torch.quantile(anisotropy.detach(), 0.99).item()
        print(f"    Anisotropy ρ: [{anisotropy.min():.4f}, {anisotropy.max():.4f}], P90={p90_log:.4f}, P99={p99_log:.4f}")
        print(f"  ")
        print(f"  Surface Probability (p_surf_raw):")
        print(f"    Formula: clamp(sigmoid((z_s - τ)/{gamma:.2f}) + {w_curv:.2f}·ρ̂, 0, 1)")
        print(f"    mean: {p_surf_mean:.4f}, >0.5: {p_surf_above_50:.2%}")
        print(f"    → Will be used in: Stage-2 (π computation), Σ/opacity scaling")
        print(f"  ")
        print(f"  ✓ Fully differentiable (no hard thresholds)")
        print(f"  ════════════════════════════════════════════════\n")
    
    # Update state
    state["last_p_surf_raw"] = p_surf_raw.detach()
    
    if anti_grid_enabled:
        state["gridness"] = gridness.detach()
        state["confidence"] = conf.detach()
    
    # ════════════════════════════════════════════════════════════════════════
    # Return Results (p_surf_raw + intermediate values for visualization)
    # ════════════════════════════════════════════════════════════════════════
    # Returns:
    #   - p_surf_raw: (N,) surface probability [0,1] → Stage-2, opacity
    #   - normals, spacing, state, planarity, anisotropy
    #   - z_s: (N,) z-score of planarity
    #   - rho_gate: (N,) curvature boost mask [0,1] (top 10% only >0)
    base_returns = (p_surf_raw, normals, spacing, state, planarity, anisotropy, z_s_stored, rho_gate_stored)
    
    if return_curvature_dirs:
        base_returns = base_returns + (principal_dir1, principal_dir2, principal_curv)
    
    return base_returns


__all__ = [
    "detect_surface",
]
