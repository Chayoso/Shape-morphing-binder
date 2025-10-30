"""
Anchor Density Estimation for Surface Sampling

This module provides differentiable density estimation for anchor points in 
physics-guided Gaussian splatting. It computes local density using kernel-based 
methods with optional surface probability weighting to guide importance sampling.

Key Features:
- Fully differentiable density computation
- Multi-scale density estimation
- Adaptive beta adjustment based on distribution uniformity
- Reduced surface bias for uniform coverage

Author: CHAYO
Version: 2.2.1 (Refactored)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

try:
    from ..utils.config import EPS_SAFE
except Exception:
    EPS_SAFE = 1e-8


# Constants for density computation
DEFAULT_CLAMP_MIN = 0.25  # Allow more sparse regions
DEFAULT_CLAMP_MAX = 4.0   # Allow denser regions
                          # Result: 16x range vs 6x (better contrast)
DEFAULT_BASE_BETA = 0.3
DEFAULT_SURF_WEIGHT_STRENGTH = 0.3
BANDWIDTH_EPS = 1e-9
DISTANCE_SAFE_THRESHOLD = 1e-12


def _compute_knn_distances(
    x: torch.Tensor,
    knn,
    k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute k-nearest neighbor indices, weights, and distances.
    
    This is a helper function to reduce code duplication across density 
    computation methods.
    
    Args:
        x: (N, 3) Point positions
        knn: KNN callable function
        k: Number of neighbors to query
        
    Returns:
        idx: (N, k) Neighbor indices
        w: (N, k) Neighbor weights
        d: (N, k) Euclidean distances to neighbors (NaN-safe)
    """
    k = int(min(k, x.shape[0]))
    idx, w = knn(x, x, k)
    d = torch.linalg.norm(x[idx] - x[:, None, :], dim=-1)
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return idx, w, d


def _rowwise_mean_bandwidth(d: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row bandwidth from k-nearest neighbor distances.
    
    Uses mean distance to k neighbors as adaptive bandwidth for kernel density
    estimation. Handles near-zero distances by replacing with row maximum.
    
    Args:
        d: (N, k) Distance matrix where each row contains distances to k neighbors
        
    Returns:
        h: (N, 1) Bandwidth for each point, guaranteed >= BANDWIDTH_EPS
        
    Notes:
        - Differentiable operation
        - Replaces distances < DISTANCE_SAFE_THRESHOLD with row max to avoid
          division by zero in kernel computation
    """
    d_max = d.max(dim=1, keepdim=True).values
    d_safe = torch.where(d < DISTANCE_SAFE_THRESHOLD, d_max, d)
    h = d_safe.mean(dim=1, keepdim=True).clamp_min(BANDWIDTH_EPS)
    return h


def _soft_kernel_density(
    d: torch.Tensor, 
    w_knn: torch.Tensor,
    surf_prob: Optional[torch.Tensor] = None,
    idx: Optional[torch.Tensor] = None,
    surf_weight_strength: float = DEFAULT_SURF_WEIGHT_STRENGTH
) -> torch.Tensor:
    """
    Compute soft Gaussian kernel density with optional surface weighting.
    
    Estimates local density using a Gaussian kernel with adaptive bandwidth.
    Optionally blends uniform and surface-weighted contributions to maintain
    coverage while respecting surface information.
    
    Args:
        d: (N, k) Distances to k neighbors
        w_knn: (N, k) KNN weights from neighbor search
        surf_prob: Optional (N,) surface probability for each point. 
                   If provided, used to weight neighbor contributions.
        idx: Optional (N, k) neighbor indices. Required if surf_prob is provided.
        surf_weight_strength: Strength of surface weighting in [0, 1].
                              0 = uniform, 1 = fully surface-weighted.
                              
    Returns:
        rho: (N,) Density estimate for each point
        
    Notes:
        - Fully differentiable
        - Surface weighting blends: (1-α)*uniform + α*surface_weighted
        - Prevents over-concentration at surface regions
        
    Raises:
        ValueError: If surf_prob is provided but idx is None
    """
    if surf_prob is not None and idx is None:
        raise ValueError("idx must be provided when surf_prob is not None")
    
    # Adaptive bandwidth Gaussian kernel
    h = _rowwise_mean_bandwidth(d)
    kern = torch.exp(-(d / h) ** 2)
    
    # Normalize KNN weights
    alpha = w_knn / (w_knn.sum(dim=1, keepdim=True) + EPS_SAFE)
    
    # Apply surface probability weighting if provided
    if surf_prob is not None and idx is not None:
        surf_weights = surf_prob[idx]  # (N, k)
        
        # Blend uniform and surface-weighted for coverage
        uniform_weights = torch.ones_like(surf_weights)
        mixed_weights = (
            (1 - surf_weight_strength) * uniform_weights + 
            surf_weight_strength * surf_weights
        )
        
        alpha = alpha * mixed_weights
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + EPS_SAFE)
    
    # Compute weighted density
    rho = (kern * alpha).sum(dim=1)
    return rho


def _normalize_density(
    rho: torch.Tensor, 
    clamp_range: Tuple[float, float] = (DEFAULT_CLAMP_MIN, DEFAULT_CLAMP_MAX)
) -> torch.Tensor:
    """
    Normalize and clamp density values for uniform coverage.
    
    Normalizes density by mean and clamps to a specified range to prevent
    over-concentration. Tighter clamping promotes more uniform sampling.
    
    Args:
        rho: (N,) Raw density values
        clamp_range: (min, max) Clamping range after normalization.
                     Default [0.25, 4.0] matches global constants.
                     
    Returns:
        rho_normalized: (N,) Normalized and clamped density values
        
    Notes:
        - Fully differentiable
        - Smaller range ratio (max/min) → more uniform sampling
        - Default 6x range vs previous 40x range significantly reduces bias
        
    Example:
        >>> rho = torch.tensor([1.0, 2.0, 5.0, 10.0])
        >>> rho_norm = _normalize_density(rho, clamp_range=(0.25, 4.0))
        >>> # Values normalized by mean and clamped to [0.5, 3.0]
    """
    rho_normalized = rho / (rho.mean() + EPS_SAFE)
    rho_normalized = rho_normalized.clamp(*clamp_range)
    return rho_normalized


def build_anchor_density(
    x: torch.Tensor,
    knn,
    k: int = 16,
    surf_prob: Optional[torch.Tensor] = None,
    use_soft_radius: bool = False,
) -> torch.Tensor:
    """
    Build single-scale anchor density with reduced surface bias.
    
    Computes local density for each anchor point using a Gaussian kernel
    with adaptive bandwidth. Optionally incorporates surface probability
    with reduced weighting to maintain uniform coverage.
    
    Args:
        x: (N, 3) Anchor point positions in 3D space
        knn: KNN search callable with signature knn(query, ref, k) -> (idx, weights)
        k: Number of nearest neighbors for density estimation. Default 16.
        surf_prob: Optional (N,) surface probability [0, 1] for each point.
                   If provided, used to weight density computation.
        use_soft_radius: Unused, kept for API compatibility
        
    Returns:
        rho: (N,) Normalized density values in range [0.25, 4.0]
        
    Notes:
        - Fully differentiable for gradient-based optimization
        - Surface weighting strength = 0.3 (30% surface, 70% uniform)
        - Density range clamped to 6x for uniform coverage
        - Handles edge cases: k clamped to actual point count
        
    Example:
        >>> anchors = torch.randn(1000, 3)
        >>> from some_knn_module import knn_search
        >>> density = build_anchor_density(anchors, knn_search, k=16)
        >>> print(density.shape)  # torch.Size([1000])
        >>> print(density.min(), density.max())  # ~0.5, ~3.0
    """
    idx, w, d = _compute_knn_distances(x, knn, k)
    
    rho = _soft_kernel_density(
        d, w, 
        surf_prob=surf_prob, 
        idx=idx, 
        surf_weight_strength=DEFAULT_SURF_WEIGHT_STRENGTH
    )
    rho = _normalize_density(rho, clamp_range=(DEFAULT_CLAMP_MIN, DEFAULT_CLAMP_MAX))
    
    return rho


def build_multiscale_density(
    x: torch.Tensor,
    knn,
    k_small: int = 8,
    k_large: int = 32,
    surf_prob: Optional[torch.Tensor] = None,
    state: Optional[Dict] = None
) -> torch.Tensor:
    """
    Multi-scale density estimation using harmonic mean.
    
    Computes local density at two scales (k_small, k_large) and combines
    them using harmonic mean for robust density estimation.
    
    Args:
        x: (N, 3) Anchor point positions
        knn: KNN search callable
        k_small: Neighbors for local scale (default: 8)
        k_large: Neighbors for global scale (default: 32)
        surf_prob: Not used (kept for compatibility)
        state: Optional state dict for EMA
        
    Returns:
        w_density: (N,) Normalized density [0.25, 4.0]
    """
    eps = 1e-8
    
    # Compute kernel density at two scales
    idx_s, w_s, d_s = _compute_knn_distances(x, knn, k_small)
    idx_l, w_l, d_l = _compute_knn_distances(x, knn, k_large)
    
    h_s = d_s[:, 1:].mean(dim=1, keepdim=True) + eps
    h_l = d_l[:, 1:].mean(dim=1, keepdim=True) + eps
    
    dens_s = torch.exp(-(d_s[:, 1:] / h_s) ** 2).mean(dim=1)
    dens_l = torch.exp(-(d_l[:, 1:] / h_l) ** 2).mean(dim=1)
    
    # Harmonic mean
    density = 2.0 / ((1.0 / (dens_s + eps)) + (1.0 / (dens_l + eps)))
    
    # Normalize to [0.25, 4.0] range
    w_density = density / (density.mean() + eps)
    w_density = torch.clamp(w_density, min=0.25, max=4.0)
    
    return w_density


def compute_adaptive_beta(
    rho_anchor: torch.Tensor,
    base_beta: float,
    cv_low: float = 0.3,
    cv_high: float = 0.6,
    beta_min_factor: float = 0.7,
    beta_max_factor: float = 1.0
) -> float:
    """
    Compute adaptive beta based on density distribution uniformity.
    
    Adjusts the density bias strength (beta) based on the coefficient of 
    variation (CV) of the density distribution. More uniform distributions
    can tolerate higher bias, while concentrated distributions need lower bias.
    
    Args:
        rho_anchor: (N,) Anchor density values
        base_beta: Base beta value to scale
        cv_low: CV threshold for maximum beta. Below this → uniform. Default 0.3.
        cv_high: CV threshold for minimum beta. Above this → concentrated. Default 0.6.
        beta_min_factor: Multiplier for minimum beta (high CV case). Default 0.7.
        beta_max_factor: Multiplier for maximum beta (low CV case). Default 1.0.
        
    Returns:
        adaptive_beta: Adjusted beta value for sampling
        
    Adaptation Strategy:
        - CV < cv_low: Uniform distribution → beta = base_beta * beta_max_factor
        - CV > cv_high: Concentrated distribution → beta = base_beta * beta_min_factor
        - cv_low <= CV <= cv_high: Linear interpolation
        
    Notes:
        - CV (Coefficient of Variation) = std / mean
        - Lower CV → more uniform → can use higher beta
        - Higher CV → more concentrated → must reduce beta to prevent holes
        
    Example:
        >>> rho = torch.randn(1000).abs() + 1.0
        >>> beta_adapted = compute_adaptive_beta(rho, base_beta=0.3)
        >>> # beta_adapted in [0.21, 0.30] depending on distribution
    """
    rho_std = rho_anchor.std()
    rho_mean = rho_anchor.mean()
    cv = rho_std / (rho_mean + EPS_SAFE)
    
    # Determine factor based on CV
    if cv < cv_low:
        factor = beta_max_factor
    elif cv > cv_high:
        factor = beta_min_factor
    else:
        # Linear interpolation
        t = (cv - cv_low) / (cv_high - cv_low)
        factor = beta_max_factor + t * (beta_min_factor - beta_max_factor)
    
    adaptive_beta = base_beta * factor
    return adaptive_beta


def prepare_sampling_cfg(
    rho_anchor: torch.Tensor,
    spacing: Optional[torch.Tensor],
    cfg_in: Optional[Dict] = None,
    *,
    beta: float = 0.7,
    gamma: float = 0.6
) -> Dict:
    """
    Prepare sampling configuration dictionary with density and spacing information.
    
    Creates or extends a configuration dictionary for importance sampling,
    incorporating anchor density and spacing bias parameters.
    
    Args:
        rho_anchor: (N,) Anchor density values (or None)
        spacing: Optional (N,) local spacing values for fallback
        cfg_in: Optional input configuration dict to extend
        beta: Density bias strength for importance sampling. Default 0.7.
        gamma: Spacing bias exponent for importance sampling. Default 0.6.
        
    Returns:
        cfg_out: Configuration dictionary with the following keys:
            - use_anchor_density: True
            - anchor_density_beta: Density bias strength
            - spacing_bias_gamma: Spacing bias exponent
            - anchor_density_values: Detached density tensor (if provided)
            - spacing_for_fallback: Detached spacing tensor (if provided)
            
    Notes:
        - Detaches tensors from computation graph for config storage
        - Preserves all keys from cfg_in if provided
        - Removes anchor_density_values if rho_anchor is not a tensor
        
    Example:
        >>> rho = torch.randn(1000)
        >>> spacing = torch.randn(1000)
        >>> cfg = prepare_sampling_cfg(rho, spacing, beta=0.3, gamma=0.6)
        >>> print(cfg['use_anchor_density'])  # True
        >>> print(cfg['anchor_density_beta'])  # 0.3
    """
    cfg_out = dict(cfg_in) if cfg_in is not None else {}
    
    cfg_out["use_anchor_density"] = True
    cfg_out["anchor_density_beta"] = float(beta)
    cfg_out["spacing_bias_gamma"] = float(gamma)

    if isinstance(rho_anchor, torch.Tensor):
        cfg_out["anchor_density_values"] = rho_anchor.detach()
    else:
        cfg_out.pop("anchor_density_values", None)

    if isinstance(spacing, torch.Tensor):
        cfg_out["spacing_for_fallback"] = spacing.detach()

    return cfg_out


def run_stage2_anchor_density(
    x: torch.Tensor,
    spacing: torch.Tensor,
    knn,
    cfg: Optional[Dict] = None,
    state: Optional[Dict] = None,
    surf_prob: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict, Dict]:
    """
    Stage 2: Compute anchor density with reduced bias for uniform coverage.
    
    Main entry point for anchor density computation in the upsampling pipeline.
    Computes density using single-scale or multi-scale method with reduced
    surface bias, adaptive beta adjustment, and comprehensive validation.
    
    Args:
        x: (N, 3) Anchor positions
        spacing: (N,) Local spacing estimate for each anchor
        knn: KNN search callable
        cfg: Optional configuration dict with keys:
            - use_multiscale_density: Use multi-scale vs single-scale (default: True)
            - stage2_k_small: k for local scale (default: 8)
            - stage2_k_large: k for global scale (default: 32)
            - stage2_k: k for single-scale (default: 16)
            - anchor_density_beta: Base beta value (default: 0.3)
            - spacing_bias_gamma: Spacing exponent (default: 0.6)
            - use_adaptive_beta: Enable adaptive beta (default: True)
            - debug: Dict with 'verbose' key for detailed logging
        state: Optional state dict to populate with statistics
        surf_prob: Optional (N,) surface probability [0, 1]
        
    Returns:
        rho_anchor: (N,) Normalized anchor density values
        cfg_out: Configuration dict for downstream sampling
        state: State dict with statistics and validation results:
            - rho_anchor_mean, std, min, max: Density statistics
            - density_cv: Coefficient of variation
            - adaptive_beta, base_beta: Beta values
            - surf_weighted: Whether surface weighting was used
            - spacing_density_corr: Correlation between spacing and 1/density
            - density_validation: Validation status string
            
    Key Improvements (v2.2.0 → v2.2.1):
        - Reduced base beta: 0.7 → 0.3 (less sampling bias)
        - Tighter clamp range: [0.25, 10.0] → [0.5, 3.0] (40x → 6x ratio)
        - Reduced surface weighting: 30-70% → 10-30%
        - Adaptive beta based on distribution uniformity
        - Comprehensive correlation-based validation
        
    Validation Logic:
        - With surface weighting: Expects negative spacing-density correlation
          (high spacing → low density, since surfaces are denser)
        - Without surface weighting: Expects positive correlation
          (high spacing → high density, inverse relationship)
        
    Notes:
        - Fully differentiable for end-to-end training
        - All computations in float32 for gradient flow
        - Statistics computed in no_grad context for efficiency
        
    Example:
        >>> anchors = torch.randn(1000, 3, requires_grad=True)
        >>> spacing = torch.rand(1000)
        >>> surf_prob = torch.sigmoid(torch.randn(1000))
        >>> 
        >>> cfg = {
        ...     'use_multiscale_density': True,
        ...     'anchor_density_beta': 0.3,
        ...     'use_adaptive_beta': True,
        ...     'debug': {'verbose': True}
        ... }
        >>> 
        >>> rho, cfg_out, state = run_stage2_anchor_density(
        ...     anchors, spacing, knn_search, 
        ...     cfg=cfg, surf_prob=surf_prob
        ... )
        >>> 
        >>> print(f"Density CV: {state['density_cv']:.3f}")
        >>> print(f"Validation: {state['density_validation']}")
        >>> 
        >>> # Backward pass works
        >>> loss = rho.mean()
        >>> loss.backward()
    """
    cfg = cfg or {}
    state = state or {}

    # Extract configuration parameters
    use_multiscale = bool(cfg.get("use_multiscale_density", True))
    k_small = int(cfg.get("stage2_k_small", 8))
    k_large = int(cfg.get("stage2_k_large", 32))
    k_legacy = int(cfg.get("stage2_k", 16))
    
    base_beta = float(cfg.get("anchor_density_beta", DEFAULT_BASE_BETA))
    gamma = float(cfg.get("spacing_bias_gamma", 0.6))
    use_adaptive_beta = bool(cfg.get("use_adaptive_beta", True))
    
    # Build density
    if use_multiscale:
        rho_anchor = build_multiscale_density(
            x, knn, 
            k_small=k_small, 
            k_large=k_large, 
            surf_prob=surf_prob,
            state=state,
        )
    else:
        rho_anchor = build_anchor_density(
            x, knn, 
            k=k_legacy, 
            surf_prob=surf_prob
        )
    
    # Final normalization
    rho_anchor = _normalize_density(
        rho_anchor, 
        clamp_range=(DEFAULT_CLAMP_MIN, DEFAULT_CLAMP_MAX)
    )
    
    # Adaptive beta adjustment
    if use_adaptive_beta:
        adaptive_beta = compute_adaptive_beta(
            rho_anchor, 
            base_beta,
            cv_low=0.3,
            cv_high=0.6,
            beta_min_factor=0.7,
            beta_max_factor=1.0
        )
    else:
        adaptive_beta = base_beta
    
    # Prepare sampling configuration
    cfg_out = prepare_sampling_cfg(
        rho_anchor, spacing, 
        cfg_in=None, 
        beta=adaptive_beta, 
        gamma=gamma
    )

    # Compute statistics and validation
    with torch.no_grad():
        _compute_density_statistics(
            rho_anchor, spacing, surf_prob, 
            base_beta, adaptive_beta, 
            use_multiscale, cfg, state
        )

    return rho_anchor, cfg_out, state


def _compute_density_statistics(
    rho_anchor: torch.Tensor,
    spacing: torch.Tensor,
    surf_prob: Optional[torch.Tensor],
    base_beta: float,
    adaptive_beta: float,
    use_multiscale: bool,
    cfg: Dict,
    state: Dict
) -> None:
    """
    Compute density statistics and validation metrics (internal helper).
    
    Populates the state dictionary with comprehensive statistics about the
    density distribution, correlation with spacing, and validation status.
    
    Args:
        rho_anchor: (N,) Computed density values
        spacing: (N,) Spacing values
        surf_prob: Optional (N,) surface probability
        base_beta: Base beta before adaptation
        adaptive_beta: Adapted beta value
        use_multiscale: Whether multi-scale was used
        cfg: Configuration dict
        state: State dict to populate (modified in-place)
        
    Side Effects:
        Modifies state dict in-place with statistics and validation results
    """
    # Basic statistics
    rho_mean = float(rho_anchor.mean().item())
    rho_std = float(rho_anchor.std().item())
    rho_min = float(rho_anchor.min().item())
    rho_max = float(rho_anchor.max().item())
    cv = rho_std / (rho_mean + EPS_SAFE)
    
    state["rho_anchor_mean"] = rho_mean
    state["rho_anchor_std"] = rho_std
    state["rho_anchor_min"] = rho_min
    state["rho_anchor_max"] = rho_max
    state["density_cv"] = cv
    state["adaptive_beta"] = adaptive_beta
    state["base_beta"] = base_beta
    state["surf_weighted"] = surf_prob is not None
    
    # Surface probability statistics
    if surf_prob is not None:
        state["surf_prob_mean"] = float(surf_prob.mean().item())
        state["surf_prob_min"] = float(surf_prob.min().item())
        state["surf_prob_max"] = float(surf_prob.max().item())
    
    # Spacing-density correlation validation
    try:
        rho_inv = 1.0 / (rho_anchor + 1e-9)
        # Manual correlation implementation for torch.corrcoef compatibility
        def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
            a_n = (a - a.mean()) / (a.std() + 1e-8)
            b_n = (b - b.mean()) / (b.std() + 1e-8)
            return float((a_n * b_n).mean().item())
        correlation = _corr(spacing, rho_inv)
        state["spacing_density_corr"] = correlation
        
        # Validate correlation sign based on surface weighting
        if surf_prob is not None:
            # With surface weighting: expect negative correlation
            if correlation < -0.2:
                state["density_validation"] = "PASS (surface-weighted)"
            elif correlation < 0.0:
                state["density_validation"] = "WEAK (surface-weighted)"
            else:
                state["density_validation"] = "FAIL (expected negative)"
        else:
            # Without surface weighting: expect positive correlation
            if correlation > 0.2:
                state["density_validation"] = "PASS"
            elif correlation > 0.0:
                state["density_validation"] = "WEAK"
            else:
                state["density_validation"] = "FAIL (expected positive)"
    except Exception as e:
        state["spacing_density_corr"] = None
        state["density_validation"] = f"ERROR: {e}"
    
    # Debug logging
    if cfg.get("debug", {}).get("verbose", False):
        _print_density_report(
            use_multiscale, surf_prob, 
            rho_mean, rho_std, rho_min, rho_max, cv,
            base_beta, adaptive_beta, 
            state.get("spacing_density_corr"),
            state.get("density_validation")
        )


def _print_density_report(
    use_multiscale: bool,
    surf_prob: Optional[torch.Tensor],
    rho_mean: float,
    rho_std: float,
    rho_min: float,
    rho_max: float,
    cv: float,
    base_beta: float,
    adaptive_beta: float,
    correlation: Optional[float],
    validation: str
) -> None:
    """Print detailed density computation report (internal helper)."""
    print(f"\n  === Stage 2: Anchor Density (Refactored v2.2.1) ===")
    print(f"  Mode: {'Multi-scale' if use_multiscale else 'Single-scale'}")
    print(f"  Surface-weighted: {surf_prob is not None}")
    print(f"  ")
    print(f"  Density Statistics:")
    print(f"    Mean: {rho_mean:.3f}, Std: {rho_std:.3f}")
    print(f"    Range: [{rho_min:.3f}, {rho_max:.3f}] = {rho_max/rho_min:.1f}x")
    print(f"    CV: {cv:.3f}")
    print(f"  ")
    print(f"  Bias Configuration (Reduced):")
    print(f"    Base beta: {base_beta:.3f}")
    print(f"    Adaptive beta: {adaptive_beta:.3f}")
    print(f"    Clamp range: [{DEFAULT_CLAMP_MIN}, {DEFAULT_CLAMP_MAX}] = "
          f"{DEFAULT_CLAMP_MAX/DEFAULT_CLAMP_MIN:.1f}x")
    print(f"  ")
    print(f"  Validation:")
    if correlation is not None:
        print(f"    Spacing-density correlation: {correlation:.3f}")
    print(f"    Status: {validation}")
    
    # Uniformity assessment
    if cv < 0.4:
        print(f"    ✓ Good uniformity (CV={cv:.3f})")
    elif cv < 0.6:
        print(f"    ⚠ Acceptable uniformity (CV={cv:.3f})")
    else:
        print(f"    ✗ High concentration (CV={cv:.3f}) - may cause holes")
    
    print(f"  =====================================================\n")


__all__ = [
    "build_anchor_density",
    "build_multiscale_density",
    "compute_adaptive_beta",
    "prepare_sampling_cfg",
    "run_stage2_anchor_density",
]