# """
# Weighted PCA-based local surface analysis (robust & differentiable).

# What’s new vs a naive PCA:
# - Safe eigendecomposition via EPS regularization (+ tiny diagonal jitter)
# - Smooth normal orientation using tanh-based sign (sign-invariant stability)
# - Surface quality 'surfvar' stabilized via z-score with std floor
# - Fully differentiable path: neighbors -> centroid -> covariance -> eig -> normal

# Returns (compat):
#     normals: (N, 3)  unit-length, consistently oriented
#     surfvar: (N,)    stabilized surface-variance in [~0, ~1/3]
#     spacing: (N,)    density-aware local point spacing

# Author: CHAYO pipeline patch (2025-10)
# """

# from typing import Tuple
# import torch
# import torch.nn.functional as F

# # Fallback EPS in case caller's config is not imported
# try:
#     from ..utils.config import EPS_SAFE, EPS_PCA, TANH_SCALE
# except Exception:
#     EPS_SAFE = 1e-8
#     EPS_PCA = 1e-10
#     TANH_SCALE = 5.0  # sharper -> closer to hard sign

# def normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     """Unit-normalize last dim safely."""
#     return v / (v.norm(dim=-1, keepdim=True) + eps)

# # ---------------------------------------------------------------------
# # Core weighted statistics
# # ---------------------------------------------------------------------

# def compute_weighted_centroid(neighbors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
#     """
#     Weighted centroid c = Σ w_i x_i  (weights are assumed normalized along dim=1).
#     neighbors: (N, k, 3), weights: (N, k)
#     returns:   (N, 3)
#     """
#     return torch.einsum("nk,nkd->nd", weights, neighbors)

# def compute_weighted_covariance(centered: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
#     """
#     Weighted covariance Σ = (1/Σw) Σ w (x-c)(x-c)^T.
#     Using sqrt(w) improves numerical stability.
#     centered: (N, k, 3), weights: (N, k)
#     returns:  (N, 3, 3)
#     """
#     sqrt_w = torch.sqrt(weights.clamp_min(0.0)).unsqueeze(-1)  # (N,k,1)
#     Xw = centered * sqrt_w                                     # (N,k,3)
#     cov = torch.einsum("nki,nkj->nij", Xw, Xw)                 # (N,3,3)
#     cov = cov / (weights.sum(dim=1, keepdim=True).unsqueeze(-1) + EPS_SAFE)
#     return cov

# # ---------------------------------------------------------------------
# # Robust eigendecomposition + normal extraction/orientation
# # ---------------------------------------------------------------------

# def _safe_eigh(cov: torch.Tensor, eps_pca: float = EPS_PCA) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Robust symmetric eigendecomposition for many small 3x3 problems.
#     Adds tiny diagonal jitter to avoid degeneracy/negative rounding.
#     cov: (N,3,3) symmetric
#     returns: evals (N,3) asc, evecs (N,3,3) columns = eigenvectors
#     """
#     # Symmetrize just in case of roundoff
#     cov = 0.5 * (cov + cov.transpose(-1, -2))
#     # Tiny diagonal jitter for stability
#     jitter = eps_pca * torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
#     evals, evecs = torch.linalg.eigh(cov + jitter)
#     # Clamp evals to avoid zero/negative issues in ratios
#     evals = torch.clamp(evals, min=eps_pca)
#     return evals, evecs

# def extract_normal_from_pca(evecs: torch.Tensor,
#                             x: torch.Tensor,
#                             centroid: torch.Tensor,
#                             tanh_scale: float = TANH_SCALE) -> torch.Tensor:
#     """
#     Smallest-eigenvector as normal, with smooth orientation.
#     evecs:    (N,3,3), columns are eigenvectors
#     x:        (N,3), original query points
#     centroid: (N,3)
#     returns:  normals (N,3), unit
#     """
#     n_raw = evecs[:, :, 0]  # smallest eigenvector (asc evals)
#     # Direction reference: from local centroid to point (fallback to global mean)
#     to_out = x - centroid
#     mask = (to_out.norm(dim=1) < 1e-9)
#     if mask.any():
#         global_c = x.mean(dim=0, keepdim=True)  # (1,3)
#         to_out = torch.where(mask.unsqueeze(-1), x - global_c, to_out)
#     # Smooth sign via tanh
#     dot = torch.einsum("nd,nd->n", n_raw, to_out)  # (N,)
#     sign = torch.tanh(tanh_scale * dot).unsqueeze(-1)  # [-1,1], smooth
#     normals = normalize(n_raw * sign)
#     return normals

# # ---------------------------------------------------------------------
# # Convenience measurements
# # ---------------------------------------------------------------------

# def compute_local_spacing(neighbors: torch.Tensor, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
#     """
#     Weighted mean distance to neighbors (scale proxy).
#     neighbors: (N,k,3), x: (N,3), weights: (N,k)
#     returns:   spacing (N,)
#     """
#     d = (neighbors - x.unsqueeze(1)).norm(dim=-1)  # (N,k)
#     spacing = torch.einsum("nk,nk->n", d, weights) / (weights.sum(dim=1) + EPS_SAFE)
#     return spacing

# # ---------------------------------------------------------------------
# # Optional soft masks (can be used upstream/downstream)
# # ---------------------------------------------------------------------

# def normal_agreement_mask(n_q: torch.Tensor, n_nb: torch.Tensor,
#                           theta_deg: float = 40.0, alpha: float = 0.1) -> torch.Tensor:
#     """
#     Soft normal-consistency mask in [0,1].
#     n_q:  (N,3), n_nb: (N,k,3)
#     """
#     cos_t = float(torch.cos(torch.tensor(theta_deg * 3.14159265 / 180.0, device=n_q.device)))
#     agree = ((n_q[:, None, :] * n_nb).sum(-1) - cos_t) / alpha  # (N,k)
#     return torch.sigmoid(agree)

# def soft_slab_weights(d_signed: torch.Tensor, sigma_local: torch.Tensor, Delta: float = 1.2) -> torch.Tensor:
#     """
#     Soft slab weight exp(-( |d| / (Delta * sigma) )^2 ), supports broadcasting.
#     d_signed:     (N,k) or (N,1)
#     sigma_local:  (N,1)
#     returns:      (N,k) weights in (0,1]
#     """
#     return torch.exp(- (d_signed.abs() / (Delta * (sigma_local + EPS_SAFE))).pow(2.0))

# # ---------------------------------------------------------------------
# # Batched PCA orchestrator (drop-in)
# # ---------------------------------------------------------------------

# def batched_pca_surface_optimized(
#     x: torch.Tensor,
#     indices: torch.Tensor,
#     weights: torch.Tensor,
#     return_principal_dirs: bool = False  # 🔥 NEW: for anisotropic jitter
# ) -> Tuple:
#     """
#     Main entry: per-point weighted PCA surface analysis.

#     Args
#     ----
#     x:        (N, 3)             query points (full cloud)
#     indices:  (N, k)             neighbor indices
#     weights:  (N, k)             normalized attention weights (sum~1 per row)
#     return_principal_dirs: bool  if True, also return principal curvature directions

#     Returns
#     -------
#     If return_principal_dirs=False (default, backward compatible):
#         normals:     (N, 3)          oriented, unit
#         surfvar:     (N,)            stabilized surface-variance (lower is more planar)
#         spacing:     (N,)            density-aware spacing
#         curvature:   (N,)            local curvature estimate (higher = more curved)
    
#     If return_principal_dirs=True (for anisotropic jitter):
#         normals:     (N, 3)          oriented, unit
#         surfvar:     (N,)            stabilized surface-variance
#         spacing:     (N,)            density-aware spacing
#         curvature:   (N,)            local curvature estimate
#         principal_dir1: (N, 3)       first principal direction (max curvature)
#         principal_dir2: (N, 3)       second principal direction (min curvature)
#         principal_curv: (N, 2)       [k1, k2] principal curvatures (normalized)
#     """
#     # 1) collect neighbors
#     neighbors = x[indices]  # (N,k,3)

#     # 2) weighted centroid
#     centroid = compute_weighted_centroid(neighbors, weights)  # (N,3)

#     # 3) center neighbors
#     centered = neighbors - centroid.unsqueeze(1)              # (N,k,3)

#     # 4) covariance
#     cov = compute_weighted_covariance(centered, weights)      # (N,3,3)

#     # 5) safe eigendecomposition
#     evals, evecs = _safe_eigh(cov, eps_pca=EPS_PCA)           # (N,3), (N,3,3)
#     # ASC order (torch.linalg.eigh guarantees ascending)
#     # evals: λ0 ≤ λ1 ≤ λ2

#     # 6) quality metric (surface variance in normal dir) with z-score stabilization
#     raw = evals[:, 0] / (evals.sum(dim=1) + EPS_SAFE)         # (N,)
#     # Row-wise statistics over batch to enhance contrast in near-degenerate regimes
#     mu  = raw.mean()
#     std = raw.std()
#     # std floor relative to robust magnitude to avoid blow-ups
#     std = torch.clamp(std, min=1e-4 * (mu.abs().detach() + 1.0))
#     surfvar_z = (raw - mu) / std

#     # We keep the original scale but lightly squash extreme outliers to [0,1]ish
#     # so downstream thresholds are stable yet compatible.
#     # Map z back to [0,1] with a smooth sigmoid; retain monotonicity.
#     surfvar = torch.sigmoid(surfvar_z)  # stabilized scalar in (0,1)

#     # 7) oriented normals
#     normals = extract_normal_from_pca(evecs, x, centroid)     # (N,3)

#     # 8) spacing
#     spacing = compute_local_spacing(neighbors, x, weights)    # (N,)
    
#     # 9) curvature estimate (principal curvature ratio)
#     # λ0 ≤ λ1 ≤ λ2
#     # For surfaces: λ0 ≈ 0 (normal direction), λ1, λ2 > 0 (tangent)
#     # curvature ∝ λ1 / λ2 (normalized by trace for stability)
#     trace = evals.sum(dim=1) + EPS_SAFE  # (N,)
#     curvature = evals[:, 1] / trace  # (N,) - middle eigenvalue ratio
#     # Higher curvature = more bent surface (edges, corners)
#     # Lower curvature = flatter surface

#     if not return_principal_dirs:
#         return normals, surfvar, spacing, curvature
    
#     # 🔥 NEW: Principal directions and anisotropic curvatures
#     # For anisotropic jitter: use tangent plane eigenvectors
#     # λ1 (smaller tangent eigenvalue) → max curvature direction (tighter)
#     # λ2 (larger tangent eigenvalue) → min curvature direction (wider)
    
#     principal_dir1 = normalize(evecs[:, :, 1])  # (N, 3) - max curvature direction
#     principal_dir2 = normalize(evecs[:, :, 2])  # (N, 3) - min curvature direction
    
#     # Principal curvatures (normalized by trace for scale invariance)
#     k1 = evals[:, 1] / trace  # (N,) - max curvature (0~0.5)
#     k2 = evals[:, 2] / trace  # (N,) - min curvature (0~0.5)
#     principal_curv = torch.stack([k1, k2], dim=1)  # (N, 2)
    
#     return normals, surfvar, spacing, curvature, principal_dir1, principal_dir2, principal_curv

# # Public API
# __all__ = [
#     "compute_weighted_centroid",
#     "compute_weighted_covariance",
#     "extract_normal_from_pca",
#     "compute_local_spacing",
#     "normal_agreement_mask",
#     "soft_slab_weights",
#     "batched_pca_surface_optimized",
# ]


"""
Weighted PCA-based local surface analysis (robust & differentiable).

What's new vs a naive PCA:
- Safe eigendecomposition via EPS regularization (+ tiny diagonal jitter)
- Smooth normal orientation using tanh-based sign (sign-invariant stability)
- Surface quality 'surfvar' stabilized via z-score with std floor
- Fully differentiable path: neighbors -> centroid -> covariance -> eig -> normal

Returns (compat):
    normals: (N, 3)  unit-length, consistently oriented
    surfvar: (N,)    stabilized surface-variance in [~0, ~1/3]
    spacing: (N,)    density-aware local point spacing

Author: CHAYO pipeline patch (2025-10)
"""

from typing import Tuple
import torch
import torch.nn.functional as F

# Fallback EPS in case caller's config is not imported
try:
    from ..utils.config import EPS_SAFE, EPS_PCA, TANH_SCALE
except Exception:
    EPS_SAFE = 1e-8
    EPS_PCA = 1e-10
    TANH_SCALE = 5.0  # sharper -> closer to hard sign

def normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Unit-normalize last dim safely."""
    return v / (v.norm(dim=-1, keepdim=True) + eps)

# ---------------------------------------------------------------------
# Core weighted statistics
# ---------------------------------------------------------------------

def compute_weighted_centroid(neighbors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted centroid c = Σ w_i x_i  (weights are assumed normalized along dim=1).
    neighbors: (N, k, 3), weights: (N, k)
    returns:   (N, 3)
    """
    return torch.einsum("nk,nkd->nd", weights, neighbors)

def compute_weighted_covariance(centered: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted covariance Σ = (1/Σw) Σ w (x-c)(x-c)^T.
    Using sqrt(w) improves numerical stability.
    centered: (N, k, 3), weights: (N, k)
    returns:  (N, 3, 3)
    """
    sqrt_w = torch.sqrt(weights.clamp_min(0.0)).unsqueeze(-1)  # (N,k,1)
    Xw = centered * sqrt_w                                     # (N,k,3)
    cov = torch.einsum("nki,nkj->nij", Xw, Xw)                 # (N,3,3)
    cov = cov / (weights.sum(dim=1, keepdim=True).unsqueeze(-1) + EPS_SAFE)
    return cov

# ---------------------------------------------------------------------
# Robust eigendecomposition + normal extraction/orientation
# ---------------------------------------------------------------------

def _safe_eigh(cov: torch.Tensor, eps_pca: float = EPS_PCA) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Robust symmetric eigendecomposition for many small 3x3 problems.
    Adds tiny diagonal jitter to avoid degeneracy/negative rounding.
    cov: (N,3,3) symmetric
    returns: evals (N,3) asc, evecs (N,3,3) columns = eigenvectors
    """
    # Symmetrize just in case of roundoff
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    # Tiny diagonal jitter for stability
    jitter = eps_pca * torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
    evals, evecs = torch.linalg.eigh(cov + jitter)
    # Clamp evals to avoid zero/negative issues in ratios
    evals = torch.clamp(evals, min=eps_pca)
    return evals, evecs

def extract_normal_from_pca(evecs: torch.Tensor,
                            x: torch.Tensor,
                            centroid: torch.Tensor,
                            tanh_scale: float = TANH_SCALE) -> torch.Tensor:
    """
    Smallest-eigenvector as normal, with smooth orientation.
    evecs:    (N,3,3), columns are eigenvectors
    x:        (N,3), original query points
    centroid: (N,3)
    returns:  normals (N,3), unit
    """
    n_raw = evecs[:, :, 0]  # smallest eigenvector (asc evals)
    # Direction reference: from local centroid to point (fallback to global mean)
    to_out = x - centroid
    mask = (to_out.norm(dim=1) < 1e-9)
    if mask.any():
        global_c = x.mean(dim=0, keepdim=True)  # (1,3)
        to_out = torch.where(mask.unsqueeze(-1), x - global_c, to_out)
    # Smooth sign via tanh
    dot = torch.einsum("nd,nd->n", n_raw, to_out)  # (N,)
    sign = torch.tanh(tanh_scale * dot).unsqueeze(-1)  # [-1,1], smooth
    normals = normalize(n_raw * sign)
    return normals

# ---------------------------------------------------------------------
# Convenience measurements
# ---------------------------------------------------------------------

def compute_local_spacing(neighbors: torch.Tensor, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted mean distance to neighbors (scale proxy).
    neighbors: (N,k,3), x: (N,3), weights: (N,k)
    returns:   spacing (N,)
    """
    d = (neighbors - x.unsqueeze(1)).norm(dim=-1)  # (N,k)
    spacing = torch.einsum("nk,nk->n", d, weights) / (weights.sum(dim=1) + EPS_SAFE)
    return spacing

# ---------------------------------------------------------------------
# Optional soft masks (can be used upstream/downstream)
# ---------------------------------------------------------------------

def normal_agreement_mask(n_q: torch.Tensor, n_nb: torch.Tensor,
                          theta_deg: float = 40.0, alpha: float = 0.1) -> torch.Tensor:
    """
    Soft normal-consistency mask in [0,1].
    n_q:  (N,3), n_nb: (N,k,3)
    """
    cos_t = float(torch.cos(torch.tensor(theta_deg * 3.14159265 / 180.0, device=n_q.device)))
    agree = ((n_q[:, None, :] * n_nb).sum(-1) - cos_t) / alpha  # (N,k)
    return torch.sigmoid(agree)

def soft_slab_weights(d_signed: torch.Tensor, sigma_local: torch.Tensor, Delta: float = 1.2) -> torch.Tensor:
    """
    Soft slab weight exp(-( |d| / (Delta * sigma) )^2 ), supports broadcasting.
    d_signed:     (N,k) or (N,1)
    sigma_local:  (N,1)
    returns:      (N,k) weights in (0,1]
    """
    return torch.exp(- (d_signed.abs() / (Delta * (sigma_local + EPS_SAFE))).pow(2.0))

# ---------------------------------------------------------------------
# Batched PCA orchestrator (drop-in)
# ---------------------------------------------------------------------

def batched_pca_surface_optimized(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    return_principal_dirs: bool = False  # 🔥 NEW: for anisotropic jitter
) -> Tuple:
    """
    Main entry: per-point weighted PCA surface analysis.

    Args
    ----
    x:        (N, 3)             query points (full cloud)
    indices:  (N, k)             neighbor indices
    weights:  (N, k)             normalized attention weights (sum~1 per row)
    return_principal_dirs: bool  if True, also return principal curvature directions

    Returns
    -------
    If return_principal_dirs=False (default):
        normals:     (N, 3)          oriented, unit
        surfvar:     (N,)            stabilized surface-variance (lower is more planar)
        spacing:     (N,)            density-aware spacing
        curvature:   (N,)            local curvature estimate (higher = more curved)
        anisotropy:  (N,)            🔥 NEW: ρ = (λ₂-λ₁)/(λ₂+ε), sharpness [0,1]
        planarity:   (N,)            🔥 NEW: s = (λ₁+λ₂)/(λ₀+λ₁+λ₂+ε), surface quality [0,1]
    
    If return_principal_dirs=True (for anisotropic jitter):
        normals:     (N, 3)          oriented, unit
        surfvar:     (N,)            stabilized surface-variance
        spacing:     (N,)            density-aware spacing
        curvature:   (N,)            local curvature estimate
        anisotropy:  (N,)            🔥 NEW: ρ = (λ₂-λ₁)/(λ₂+ε), sharpness [0,1]
        planarity:   (N,)            🔥 NEW: s = (λ₁+λ₂)/(λ₀+λ₁+λ₂+ε), surface quality [0,1]
        principal_dir1: (N, 3)       first principal direction (max curvature)
        principal_dir2: (N, 3)       second principal direction (min curvature)
        principal_curv: (N, 2)       [k1, k2] principal curvatures (normalized)
    """
    # 1) collect neighbors
    neighbors = x[indices]  # (N,k,3)

    # 2) weighted centroid
    centroid = compute_weighted_centroid(neighbors, weights)  # (N,3)

    # 3) center neighbors
    centered = neighbors - centroid.unsqueeze(1)              # (N,k,3)

    # 4) covariance
    cov = compute_weighted_covariance(centered, weights)      # (N,3,3)

    # 5) safe eigendecomposition
    evals, evecs = _safe_eigh(cov, eps_pca=EPS_PCA)           # (N,3), (N,3,3)
    # ASC order (torch.linalg.eigh guarantees ascending)
    # evals: λ0 ≤ λ1 ≤ λ2

    # 6) quality metric (surface variance in normal dir) - FULLY DETACHED
    # 🔥 ULTIMATE FIX: Detach the entire surfvar computation from gradient graph
    # This is a feature engineering step, not a learnable transformation
    with torch.no_grad():
        raw = evals[:, 0] / (evals.sum(dim=1) + EPS_SAFE)  # (N,)
        
        # Z-score normalization
        mu = raw.mean()
        std = torch.clamp(raw.std(), min=1e-3)
        surfvar_z = (raw - mu) / std
        surfvar_z = torch.clamp(surfvar_z, min=-10.0, max=10.0)
        
        # Sigmoid mapping
        surfvar = torch.sigmoid(surfvar_z)  # (N,) in [0,1]
    
    # Reattach to graph (but values are fixed, no backward through this)
    surfvar = surfvar.detach().requires_grad_(False)

    # 7) oriented normals
    normals = extract_normal_from_pca(evecs, x, centroid)     # (N,3)

    # 8) spacing
    spacing = compute_local_spacing(neighbors, x, weights)    # (N,)
    
    # 9) curvature estimate (principal curvature ratio)
    # λ0 ≤ λ1 ≤ λ2
    # For surfaces: λ0 ≈ 0 (normal direction), λ1, λ2 > 0 (tangent)
    # curvature ∝ λ1 / λ2 (normalized by trace for stability)
    trace = evals.sum(dim=1) + EPS_SAFE  # (N,)
    curvature = evals[:, 1] / trace  # (N,) - middle eigenvalue ratio
    # Higher curvature = more bent surface (edges, corners)
    # Lower curvature = flatter surface
    
    # 🔥 NEW: PCA-based geometric features (DIFFERENTIABLE!)
    # These are density-independent normalized ratios
    lam0, lam1, lam2 = evals[:, 0], evals[:, 1], evals[:, 2]  # (N,) each
    
    # Anisotropy: ρ = (λ₁ - λ₂)/(λ₁ + ε)
    # Measures sharpness along the normal direction vs tangent plane
    # High ρ → sharp edge/feature, Low ρ → uniform/flat
    anisotropy = (lam2 - lam1) / (lam2 + EPS_SAFE)  # (N,) in [0,1]
    
    # Planarity: s = (λ₂+λ₃)/(λ₁+λ₂+λ₃+ε)
    # Density-independent surface quality metric
    # High s → more planar (surface-like), Low s → more scattered (volumetric)
    planarity = (lam1 + lam2) / (trace + EPS_SAFE)  # (N,) in [0,1]

    if not return_principal_dirs:
        return normals, surfvar, spacing, curvature, anisotropy, planarity
    
    # 🔥 NEW: Principal directions and anisotropic curvatures
    # Eigenvalues: λ₀ ≤ λ₁ ≤ λ₂ (ascending order from eigh)
    # - λ₀ ≈ 0 (normal direction, 가장 작음)
    # - λ₁, λ₂ > 0 (tangent plane, λ₁ < λ₂)
    #
    # Principal curvature interpretation:
    # - λ₁ (smaller tangent eigenvalue) → 더 구부러진 방향 (tighter)
    # - λ₂ (larger tangent eigenvalue) → 덜 구부러진 방향 (wider)
    
    principal_dir1 = normalize(evecs[:, :, 1])  # (N, 3) - e₁ direction (λ₁, smaller → more curved)
    principal_dir2 = normalize(evecs[:, :, 2])  # (N, 3) - e₂ direction (λ₂, larger → less curved)
    
    # Principal curvatures (normalized by trace for scale invariance)
    # k1 < k2 이지만, k1 방향이 더 구부러진 방향 (eigenvalue가 작음)
    k1 = evals[:, 1] / trace  # (N,) - λ₁/trace (0~0.5, smaller value)
    k2 = evals[:, 2] / trace  # (N,) - λ₂/trace (0~0.5, larger value)
    principal_curv = torch.stack([k1, k2], dim=1)  # (N, 2)
    
    # 🔥 Anisotropy and planarity (already computed above, reuse)
    # anisotropy, planarity are already defined in the main path
    
    return normals, surfvar, spacing, curvature, anisotropy, planarity, principal_dir1, principal_dir2, principal_curv

# Public API
__all__ = [
    "compute_weighted_centroid",
    "compute_weighted_covariance",
    "extract_normal_from_pca",
    "compute_local_spacing",
    "normal_agreement_mask",
    "soft_slab_weights",
    "batched_pca_surface_optimized",
]
