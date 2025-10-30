"""
Deformation-based Covariance Construction from F-field Interpolation.

This module handles covariance construction for PREDICTED/SIMULATED meshes
using physics-guided F-field (deformation gradient) interpolation.

For TARGET mesh covariance, use utils/curvature_covariance.py instead.

Key Features:
- F-field interpolation with graph Laplacian smoothing
- Polar decomposition (rotation removal)
- Density-based adaptive scaling
- Differentiable end-to-end

Author: PhysMorph-GS Team
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from ..utils.config import EPS_SAFE


# ============================================================================
# Differentiable Utilities
# ============================================================================

def soft_top_k(
    distances: torch.Tensor,
    k: int, 
    tau: float = 0.1,
    largest: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable soft top-k selection.
    
    Uses temperature-scaled softmax to create smooth, differentiable weights.
    For k << M, this approximates hard top-k selection while maintaining gradients.
    
    Args:
        distances: (N, M) distance matrix
        k: Number of neighbors
        tau: Temperature for softmax (lower = sharper selection)
        largest: If True, select largest values; else smallest
    
    Returns:
        indices: (N, k) hard-selected indices (for compatibility)
        weights: (N, k) soft attention weights
    """
    N, M = distances.shape
    
    # Forward: Hard top-k selection
    if largest:
        topk_values, topk_indices = torch.topk(distances, k=k, dim=1)
    else:
        topk_values, topk_indices = torch.topk(-distances, k=k, dim=1)
        topk_values = -topk_values
    
    # Re-gather distances WITH gradients
    topk_distances = torch.gather(distances, 1, topk_indices)
    
    # Soft weights (fully differentiable)
    if largest:
        logits = topk_distances / tau
    else:
        logits = -topk_distances / tau
    
    soft_weights = F.softmax(logits, dim=1)  # (N, k)
    
    return topk_indices, soft_weights


def gumbel_select_k(
    probs: torch.Tensor,
    K: int,
    tau: float = 0.1,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable K-selection using Gumbel-Softmax.
    
    Args:
        probs: (N,) selection probabilities (or uniform if all equal)
        K: Number of elements to select
        tau: Temperature
        seed: Random seed
    
    Returns:
        selected_indices: (K,) hard indices (with soft gradients)
        selection_weights: (K,) soft selection weights for selected items
    """
    N = probs.shape[0]
    device = probs.device
    
    # Generate Gumbel noise (detached from graph)
    generator = torch.Generator(device=device).manual_seed(seed)
    u = torch.rand(N, generator=generator, device=device)
    u = torch.clamp(u, 1e-10, 1 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    
    # Gumbel-Softmax trick (this part is differentiable w.r.t. probs)
    safe_probs = torch.clamp(probs, min=1e-10)
    logits = (safe_probs.log() + gumbel) / max(tau, 1e-6)
    soft_scores = F.softmax(logits, dim=0)  # (N,) - differentiable
    
    # Hard selection for indices (no grad)
    with torch.no_grad():
        _, idx_hard = torch.topk(soft_scores, k=K, largest=True)
    
    # Gather soft weights for selected indices (differentiable)
    soft_weights_selected = soft_scores[idx_hard]  # (K,) - maintains gradients
    
    # Renormalize
    weights_normalized = soft_weights_selected / (soft_weights_selected.sum() + EPS_SAFE)
    
    return idx_hard, weights_normalized


# ============================================================================
# F-field Smoothing (Graph Laplacian)
# ============================================================================

def select_graph_nodes(x: torch.Tensor, K: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select K nodes for graph smoothing using differentiable sampling.
    
    Args:
        x: (N, 3) point positions
        K: Number of nodes to select
        seed: Random seed for reproducibility
    
    Returns:
        Xn: (K, 3) selected node positions
        sel: (K,) selected indices
    """
    N = x.shape[0]
    device = x.device
    
    # Uniform probability for now (can be made adaptive)
    probs = torch.ones(N, device=device) / N
    
    # Differentiable selection
    sel, selection_weights = gumbel_select_k(probs, K, tau=0.1, seed=seed)
    
    # Gather selected points
    Xn = x[sel]
    
    return Xn, sel


def build_graph_laplacian(Xn, node_knn, K, device, dtype):
    D_nodes = torch.cdist(Xn, Xn, p=2)
    k_node = min(node_knn, K - 1)
    
    j_nodes, attn_k = soft_top_k(D_nodes, k=k_node + 1, tau=0.05, largest=False)
    
    # Build sparse W
    W = torch.zeros(K, K, device=device, dtype=dtype)
    row_idx = torch.arange(K, device=device).unsqueeze(1).expand(-1, k_node + 1)
    W[row_idx, j_nodes] = attn_k
    
    # Remove diagonal
    W.fill_diagonal_(0)
    
    # Normalize
    W = W / (W.sum(dim=1, keepdim=True) + EPS_SAFE)
    
    # Laplacian
    return torch.diag(W.sum(dim=1)) - W


def build_interpolation_weights(x, Xn, point_knn, N, K, device, dtype):
    D_p2n = torch.cdist(x, Xn, p=2)
    k_point = min(point_knn, K)
    
    j_hard, attn_k = soft_top_k(D_p2n, k=k_point, tau=0.05, largest=False)
    
    # Gaussian on selected
    d_k = torch.gather(D_p2n, 1, j_hard)
    h = d_k.mean(dim=1, keepdim=True) + EPS_SAFE
    gauss_k = torch.exp(-(d_k / h) ** 2)
    
    combined_k = attn_k * gauss_k
    combined_k = combined_k / (combined_k.sum(dim=1, keepdim=True) + EPS_SAFE)
    
    # Expand to sparse W
    W = torch.zeros(N, K, device=device, dtype=dtype)
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, k_point)
    W[row_idx, j_hard] = combined_k
    
    return W


def smooth_F_field(x_low: torch.Tensor, F_low: torch.Tensor, cfg: Dict) -> torch.Tensor:
    """
    Smooth F-field using differentiable graph Laplacian.
    
    Args:
        x_low: (N, 3) anchor positions
        F_low: (N, 3, 3) deformation gradients
        cfg: Configuration dict
    
    Returns:
        F_smooth: (N, 3, 3) smoothed F-field
    """
    K = min(int(cfg.get("num_nodes", 180)), x_low.shape[0])
    node_knn = int(cfg.get("node_knn", 8))
    point_knn = int(cfg.get("point_knn", 8))
    lam = float(cfg.get("lambda_lap", 1e-2))
    seed = int(cfg.get("seed", 42))
    
    device, dtype = F_low.device, F_low.dtype
    N = x_low.shape[0]
    
    # Select graph nodes (differentiable)
    Xn, sel = select_graph_nodes(x_low, K, seed=seed)
    
    # Build Laplacian (differentiable)
    L = build_graph_laplacian(Xn, node_knn, K, device, dtype)
    
    # Build interpolation weights (differentiable)
    W = build_interpolation_weights(x_low, Xn, point_knn, N, K, device, dtype)
    
    # Solve: (W^T W + λ L) Y = W^T F
    WtW = torch.einsum('nk,nm->km', W, W)
    A = WtW + lam * L
    
    F_flat = F_low.reshape(N, 9)
    rhs = torch.einsum('nk,nr->kr', W, F_flat)
    
    # Add small regularization for numerical stability
    A = A + 1e-8 * torch.eye(K, device=device, dtype=dtype)
    
    Y = torch.linalg.solve(A, rhs)
    
    F_smooth_flat = torch.einsum('nk,kr->nr', W, Y)
    
    return F_smooth_flat.reshape(N, 3, 3)


# ============================================================================
# Polar Decomposition
# ============================================================================

def polar_decomposition(F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Polar decomposition: F = R S
    
    Strategy: Always keep S positive definite (all eigenvalues > 0)
    R can be rotation + reflection (det(R) = ±1)
    
    For covariance Σ = S·Σ₀·S (rotation R removed!):
    - We only use the stretch component S, not rotation R
    - This captures pure deformation (compression/extension) without orientation artifacts
    - Σ is positive definite as long as S is positive definite ✓
    
    Args:
        F: (N, 3, 3) deformation gradient
    
    Returns:
        R: (N, 3, 3) orthogonal matrices (rotation or rotation+reflection)
        S: (N, 3, 3) symmetric positive definite stretch matrices
    """
    # SVD: F = U Σ V^T
    U, sigma, Vt = torch.linalg.svd(F)  # sigma: (N, 3) - always positive
    
    # Rotation (or rotation+reflection): R = U V^T
    R = torch.bmm(U, Vt)
    # det(R) can be ±1, but that's OK for covariance computation
    
    # Stretch: S = V Σ V^T (always positive definite since sigma > 0)
    V = Vt.transpose(-2, -1)
    sigma_diag = torch.diag_embed(sigma)  # Diagonal with positive values
    S = torch.bmm(torch.bmm(V, sigma_diag), Vt)
    
    # S is guaranteed positive definite:
    # - Eigenvalues of S are exactly sigma (all positive from SVD)
    # - S is symmetric by construction
    
    return R, S


# ============================================================================
# Deformation-based Covariance Construction
# ============================================================================

def build_deformation_covariance(
    points: torch.Tensor,
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    knn,
    cfg: Dict,
    learnable_cov_module=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build covariance matrices via F-field interpolation + optional learnable refinement.
    
    This function is for PREDICTED/SIMULATED meshes only (Episode > 0).
    For TARGET mesh, use utils.curvature_covariance instead.

    Args:
        points: (M, 3) upsampled point positions
        x_low: (N, 3) anchor positions (sparse particles from simulation)
        F_low: (N, 3, 3) deformation gradients at anchors
        knn: KNN function
        cfg: Configuration dict with keys:
            - sigma0: float (base scale, default 0.08)
            - k_F: int (neighbors for interpolation, default 32)
            - use_F_smoothing: bool
            - use_adaptive_scale: bool
            - use_polar_decomposition: bool
            - voxel_size: float (for eigenvalue bounds)
            - kappa_max: float (condition number limit, default 40.0)
            - learnable: dict (learnable covariance config)
            - density: dict (density-based scaling config)
        learnable_cov_module: Optional LearnableCovariance module
    
    Returns:
        cov: (M, 3, 3) covariance matrices
        F_interp: (M, 3, 3) interpolated F-field
        idx: (M, k_F) KNN indices
    """
    sigma0 = float(cfg.get("sigma0", 0.08))
    k_F = int(cfg.get("k_F", 32))
    use_F_smoothing = bool(cfg.get("use_F_smoothing", True))
    use_adaptive_scale = bool(cfg.get("use_adaptive_scale", False))
    use_polar = bool(cfg.get("use_polar_decomposition", True))

    # 1) Optional F-field smoothing
    F_smooth = smooth_F_field(x_low, F_low, cfg.get("F_smooth", {})) if use_F_smoothing else F_low

    # 2) Interpolate F to upsampled points
    idx, w = knn(points, x_low, k_F)
    F_neighbors = F_smooth[idx]  # (M, k_F, 3, 3)
    F_interp = torch.einsum('mk,mkrc->mrc', w, F_neighbors)  # (M, 3, 3)
    
    # DEBUG: Check F values
    with torch.no_grad():
        if torch.isnan(F_interp).any() or torch.isinf(F_interp).any():
            print(f"[ERROR] Invalid F_interp!")
            print(f"  NaN: {torch.isnan(F_interp).sum().item()}")
            print(f"  Inf: {torch.isinf(F_interp).sum().item()}")
        
        F_det = torch.det(F_interp)
        print(f"[F-field Debug]")
        print(f"  F_interp range: [{F_interp.min():.6f}, {F_interp.max():.6f}]")
        print(f"  det(F) range: [{F_det.min():.6f}, {F_det.max():.6f}]")
        if (F_det < 0).any():
            print(f"  ⚠ WARNING: {(F_det < 0).sum().item()} negative determinants (reflection!)")

    # 3) Local spacing for adaptive scale
    local_spacing = None
    if use_adaptive_scale:
        neighbor_anchors = x_low[idx]
        dists = torch.norm(neighbor_anchors - points.unsqueeze(1), dim=-1)  # (M, k)
        if dists.shape[1] >= 2:
            d2 = torch.topk(dists, k=2, largest=False).values[:, 1]
            local_spacing = d2.clamp(min=1e-6)
        else:
            local_spacing = dists[:, 0].clamp(min=1e-6)

    # 4) Anchor density interpolation
    rho_cfg = cfg.get("density", {})
    rho_anchor = rho_cfg.get("rho_anchor", None)
    rho_pts = None
    if isinstance(rho_anchor, torch.Tensor) and rho_anchor.shape[0] == x_low.shape[0]:
        rho_anchor = rho_anchor.to(device=points.device, dtype=points.dtype)
        rho_pts = (w * rho_anchor[idx]).sum(dim=1)
        rho_pts = (rho_pts / (rho_pts.mean() + EPS_SAFE)).clamp(0.25, 4.0)

    # 5) Build covariance
    if use_polar:
        # Base sigma at each point
        if use_adaptive_scale and local_spacing is not None:
            sigma_adaptive = sigma0 * torch.clamp(local_spacing / (local_spacing.mean() + EPS_SAFE), 0.3, 2.0)
        else:
            sigma_adaptive = torch.full((points.shape[0],), sigma0, device=points.device, dtype=points.dtype)

        # Density-based scale adjustment
        if rho_pts is not None and bool(rho_cfg.get("use_scale_prior", False)):
            kappa = float(rho_cfg.get("scale_kappa", 0.15))
            max_up = float(rho_cfg.get("scale_max_up", 0.12))
            alpha = float(rho_cfg.get("scale_smooth_alpha", 8.0))
            allow_shrink = bool(rho_cfg.get("allow_shrink", False))
            inv = (rho_pts + 1e-3).pow(-kappa)
            s_factor = _smooth_scaleFactor(inv, allow_shrink=allow_shrink, max_up=max_up, alpha=alpha)
            
            with torch.no_grad():
                print(f"    [Density-Scale Debug]")
                print(f"      rho_pts: min={rho_pts.min():.3f}, mean={rho_pts.mean():.3f}, max={rho_pts.max():.3f}")
                print(f"      s_factor: min={s_factor.min():.3f}, mean={s_factor.mean():.3f}, max={s_factor.max():.3f}")
            
            sigma_adaptive = sigma_adaptive * s_factor

        # Polar decomposition
        R, S = polar_decomposition(F_interp)
        
        with torch.no_grad():
            print(f"[Polar Decomposition Debug]")
            print(f"  S range: [{S.min():.6f}, {S.max():.6f}]")

        # Get eigenvalues of S
        S_eigvals = torch.linalg.eigvalsh(S)  # (M, 3), sorted ascending
        
        # Clamp eigenvalues
        voxel_size = cfg.get("voxel_size", 0.5)
        s_min = max(0.4 * voxel_size, 0.01)
        s_max = min(6.0 * voxel_size, 1.0)
        S_eigvals_clamped = S_eigvals.clamp(s_min, s_max)
        
        # Condition number limiting
        kappa_max = cfg.get("kappa_max", 40.0)
        kappa = S_eigvals_clamped[:, 2] / (S_eigvals_clamped[:, 0] + 1e-8)
        needs_fix = kappa > kappa_max
        if needs_fix.any():
            target_min = S_eigvals_clamped[:, 2] / kappa_max
            S_eigvals_clamped[:, 0] = torch.where(
                needs_fix,
                torch.maximum(S_eigvals_clamped[:, 0], target_min),
                S_eigvals_clamped[:, 0]
            )
        
        # Reconstruct S with clamped eigenvalues
        U_s, _, Vt_s = torch.linalg.svd(S)
        S_fixed = U_s @ torch.diag_embed(S_eigvals_clamped) @ Vt_s
        
        # Build covariance: Σ = S·Σ₀·S (rotation removed!)
        Sigma0 = (sigma_adaptive.view(-1, 1, 1) ** 2) * torch.eye(3, device=points.device).unsqueeze(0)
        S_Sigma0 = torch.bmm(S_fixed, Sigma0)
        cov = torch.bmm(S_Sigma0, S_fixed)
        
        # Stabilization
        eps_physics = 1e-6
        eye_reg = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov = cov + eps_physics * eye_reg
        
    else:
        # Direct FF^T path
        if use_adaptive_scale and local_spacing is not None:
            sigma_adapt = sigma0 * torch.clamp(local_spacing / (local_spacing.mean() + EPS_SAFE), 0.3, 2.0)
            cov = (sigma_adapt.view(-1, 1, 1) ** 2) * torch.matmul(F_interp, F_interp.transpose(-2, -1))
        else:
            cov = (sigma0 ** 2) * torch.matmul(F_interp, F_interp.transpose(-2, -1))
        
        # Regularization
        batch_size = cov.shape[0]
        if batch_size > 200000:
            eps_physics = 2e-4
        elif batch_size > 100000:
            eps_physics = 1e-4
        else:
            eps_physics = 2e-5
        
        eye_reg = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov = cov + eps_physics * eye_reg

        # Density-based scaling
        if rho_pts is not None and bool(rho_cfg.get("use_scale_prior", False)):
            kappa = float(rho_cfg.get("scale_kappa", 0.15))
            max_up = float(rho_cfg.get("scale_max_up", 0.12))
            alpha = float(rho_cfg.get("scale_smooth_alpha", 8.0))
            allow_shrink = bool(rho_cfg.get("allow_shrink", False))
            inv = (rho_pts + 1e-3).pow(-kappa)
            s_factor = _smooth_scaleFactor(inv, allow_shrink=allow_shrink, max_up=max_up, alpha=alpha)
            cov = (s_factor.view(-1, 1, 1) ** 2) * cov

    # 6) Symmetry enforcement
    cov = 0.5 * (cov + cov.transpose(-2, -1))
    
    # 7) Diagonal dominance enforcement
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)  # (M, 3)
    d0, d1, d2 = diag[:, 0], diag[:, 1], diag[:, 2]
    
    factor = 0.5
    bound_01 = factor * torch.sqrt(torch.clamp(d0 * d1, min=1e-10))
    bound_02 = factor * torch.sqrt(torch.clamp(d0 * d2, min=1e-10))
    bound_12 = factor * torch.sqrt(torch.clamp(d1 * d2, min=1e-10))
    
    c01 = torch.clamp(cov[:, 0, 1], -bound_01, bound_01)
    c02 = torch.clamp(cov[:, 0, 2], -bound_02, bound_02)
    c12 = torch.clamp(cov[:, 1, 2], -bound_12, bound_12)
    
    cov_new = torch.zeros_like(cov)
    cov_new[:, 0, 0] = d0
    cov_new[:, 1, 1] = d1
    cov_new[:, 2, 2] = d2
    cov_new[:, 0, 1] = c01
    cov_new[:, 1, 0] = c01
    cov_new[:, 0, 2] = c02
    cov_new[:, 2, 0] = c02
    cov_new[:, 1, 2] = c12
    cov_new[:, 2, 1] = c12
    
    cov = cov_new
    
    # 8) Adaptive regularization
    batch_size = cov.shape[0]
    diag_mean = torch.diagonal(cov, dim1=-2, dim2=-1).mean()
    
    if batch_size > 200000:
        eps_ratio = 0.15
    elif batch_size > 100000:
        eps_ratio = 0.10
    else:
        eps_ratio = 0.05
    
    eps_extra = eps_ratio * diag_mean
    eps_extra = torch.clamp(eps_extra, min=1e-4, max=2e-3)
    
    eye_extra = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
    cov = cov + eps_extra * eye_extra
    
    # DEBUG
    with torch.no_grad():
        print(f"[Deformation Covariance Debug]")
        print(f"  cov range: [{cov.min():.6f}, {cov.max():.6f}]")
        cov_det = torch.det(cov)
        print(f"  det(cov) range: [{cov_det.min():.6e}, {cov_det.max():.6e}]")

    # 9) Learnable refinement (optional)
    learnable_cfg = cfg.get("learnable", {})
    use_learnable = bool(learnable_cfg.get("enabled", False))
    
    if use_learnable and learnable_cov_module is not None:
        alpha = float(learnable_cfg.get("alpha", 0.3))
        cov = learnable_cov_module(cov_physics=cov, alpha=alpha)
        cov = 0.5 * (cov + cov.transpose(-2, -1))

    return cov, F_interp, idx


def _smooth_scaleFactor(inv: torch.Tensor,
                        allow_shrink: bool,
                        max_up: float = 0.15,
                        alpha: float = 8.0) -> torch.Tensor:
    """
    Smooth, everywhere differentiable scale factor in a bounded range.

    inv = (rho + eps)^(-kappa)   # sparse(ρ↓) → inv↑

    - allow_shrink=False (only-enlarge):
        s ∈ (1, 1+max_up),  s = 1 + max_up * sigmoid( alpha * (inv - 1) )
    - allow_shrink=True (two-sided):
        s ∈ (1-max_up, 1+max_up),  s = 1 + max_up * tanh( alpha * (inv - 1) )

    alpha controls the "sharpness" (α↑ → clamp-like).
    """
    if allow_shrink:
        s = 1.0 + max_up * torch.tanh(alpha * (inv - 1.0))
    else:
        s = 1.0 + max_up * torch.sigmoid(alpha * (inv - 1.0))
    return s


__all__ = [
    "build_deformation_covariance",
    "smooth_F_field",
    "polar_decomposition",
    "select_graph_nodes",
    "build_graph_laplacian",
    "build_interpolation_weights",
    "soft_top_k",
    "gumbel_select_k",
]

