"""
Covariance construction from F-field interpolation.
Unified version with polar decomposition + spectral alignment.
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
    
    Args:
        F: (N, 3, 3) deformation gradient
    
    Returns:
        R: (N, 3, 3) rotation matrices
        S: (N, 3, 3) symmetric stretch matrices
    """
    # SVD: F = U Σ V^T
    U, sigma, Vt = torch.linalg.svd(F)  # sigma: (N, 3)
    
    # Rotation: R = U V^T
    R = torch.bmm(U, Vt)
    
    # Handle reflection (det(R) < 0)
    det_R = torch.det(R)
    reflection_mask = det_R < 0
    
    if reflection_mask.any():
        # Flip last column of U for reflected cases
        U_fixed = U.clone()
        U_fixed[reflection_mask, :, -1] *= -1
        R[reflection_mask] = torch.bmm(U_fixed[reflection_mask], Vt[reflection_mask])
        sigma[reflection_mask, -1] *= -1  # Flip corresponding singular value
    
    # Stretch: S = V Σ V^T
    V = Vt.transpose(-2, -1)
    sigma_diag = torch.diag_embed(sigma)
    S = torch.bmm(torch.bmm(V, sigma_diag), Vt)
    
    return R, S


def build_covariance_polar(
    F_interp: torch.Tensor,
    sigma0: float,
    use_adaptive_scale: bool = False,
    local_spacing: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Build covariance using polar decomposition: Σ = R S Σ₀ S R^T
    
    Args:
        F_interp: (M, 3, 3) interpolated deformation gradients
        sigma0: Base Gaussian scale
        use_adaptive_scale: Use adaptive sigma based on local spacing
        local_spacing: (M,) local spacing (if adaptive)
    
    Returns:
        cov: (M, 3, 3) covariance matrices
    """
    M = F_interp.shape[0]
    device = F_interp.device
    
    # Polar decomposition
    R, S = polar_decomposition(F_interp)
    
    # Base covariance (isotropic)
    if use_adaptive_scale and local_spacing is not None:
        sigma_adaptive = sigma0 * torch.clamp(
            local_spacing / local_spacing.mean(), 0.3, 2.0
        )
        Sigma0 = (sigma_adaptive.unsqueeze(-1).unsqueeze(-1) ** 2) * \
                 torch.eye(3, device=device).unsqueeze(0).expand(M, 3, 3)
    else:
        Sigma0 = (sigma0 ** 2) * torch.eye(3, device=device).unsqueeze(0).expand(M, 3, 3)
    
    # Covariance: Σ = R S Σ₀ S R^T
    S_Sigma0 = torch.bmm(S, Sigma0)
    S_Sigma0_S = torch.bmm(S_Sigma0, S)
    cov = torch.bmm(torch.bmm(R, S_Sigma0_S), R.transpose(-2, -1))
    
    return cov


# ============================================================================
# Covariance Construction
# ============================================================================

def build_covariance(
    points: torch.Tensor,
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    knn,
    cfg: Dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build covariance matrices via F-field interpolation.
    
    Args:
        points: (M, 3) upsampled points
        x_low: (N, 3) anchor positions
        F_low: (N, 3, 3) deformation gradients
        knn: KNN function
        cfg: Configuration dict
    
    Returns:
        cov: (M, 3, 3) covariance matrices
        F_interp: (M, 3, 3) interpolated F-field
        idx: (M, k) neighbor indices
    """
    sigma0 = float(cfg.get("sigma0", 0.08))
    k_F = int(cfg.get("k_F", 32))
    use_F_smoothing = bool(cfg.get("use_F_smoothing", True))
    use_adaptive_scale = bool(cfg.get("use_adaptive_scale", False))
    use_polar = bool(cfg.get("use_polar_decomposition", True))  # ✅ Default ON
    
    # Smooth F-field
    if use_F_smoothing:
        F_smooth_cfg = cfg.get("F_smooth", {})
        F_smooth = smooth_F_field(x_low, F_low, F_smooth_cfg)
    else:
        F_smooth = F_low
    
    # Interpolate F to upsampled points
    idx, w = knn(points, x_low, k_F)
    F_neighbors = F_smooth[idx]  # (M, k, 3, 3)
    F_interp = torch.einsum('mk,mkrc->mrc', w, F_neighbors)  # (M, 3, 3)
    
    # Compute local spacing for adaptive scale
    local_spacing = None
    if use_adaptive_scale:
        neighbor_points = points[idx]  # (M, k, 3)
        dists = torch.norm(neighbor_points - points.unsqueeze(1), dim=-1)  # (M, k)
        local_spacing = dists[:, 1].clamp(min=1e-6)  # Nearest neighbor distance
    
    # Build covariance
    if use_polar:
        # ✅ Polar decomposition method (RECOMMENDED)
        cov = build_covariance_polar(
            F_interp, sigma0, use_adaptive_scale, local_spacing
        )
    else:
        # Original method: Σ = σ² F F^T
        if use_adaptive_scale and local_spacing is not None:
            sigma_adaptive = sigma0 * torch.clamp(
                local_spacing / local_spacing.mean(), 0.3, 2.0
            )
            sigma_adaptive = sigma_adaptive.unsqueeze(-1).unsqueeze(-1)  # (M, 1, 1)
            cov = (sigma_adaptive ** 2) * torch.matmul(F_interp, F_interp.transpose(-2, -1))
        else:
            cov = (sigma0 ** 2) * torch.matmul(F_interp, F_interp.transpose(-2, -1))
    
    return cov, F_interp, idx


# ============================================================================
# 🔥 Spectral Covariance Alignment (CVPR 2026)
# ============================================================================

def curvature_to_eigenvalues(
    kappa1: torch.Tensor,
    kappa2: torch.Tensor,
    sigma0: float,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Convert principal curvatures to target eigenvalues.
    
    Geometric motivation (from differential geometry):
    Surface curvature affects Gaussian kernel spread when projected onto tangent plane.
    
    High curvature → Sharp feature → Small Gaussian spread
    Low curvature → Flat region → Large Gaussian spread
    
    Theoretical basis:
    For a surface S with curvatures κ₁, κ₂, a Gaussian kernel G(x; σ)
    when restricted to S has effective spread:
    
        λᵢ = σ₀ / √(1 + κᵢ²)
    
    This comes from projecting 3D Gaussian onto curved 2-manifold.
    
    Args:
        kappa1, kappa2: (N,) principal curvatures (in 1/units, e.g., 1/cm)
        sigma0: Base scale (in same units as geometry)
        eps: Numerical stability
    
    Returns:
        eigenvalues: (N, 3) target eigenvalues [λ₁, λ₂, λ₃]
    """
    # Geometric scaling law
    lambda1 = sigma0 / torch.sqrt(1 + kappa1**2 + eps)
    lambda2 = sigma0 / torch.sqrt(1 + kappa2**2 + eps)
    lambda3 = torch.full_like(lambda1, sigma0)  # Along normal (unchanged)
    
    eigenvalues = torch.stack([lambda1, lambda2, lambda3], dim=-1)  # (N, 3)
    
    return eigenvalues


def build_eigenvector_frame(
    principal_dirs: torch.Tensor,
    normals: torch.Tensor
) -> torch.Tensor:
    """
    Build target eigenvector frame from principal curvature directions.
    
    Geometric meaning:
    Principal curvature directions {t₁, t₂} and normal {n} form an
    orthonormal frame that captures local surface geometry.
    
    This frame becomes the eigenvector basis for anisotropic Gaussians,
    naturally aligning ellipsoidal splats with surface geometry.
    
    Args:
        principal_dirs: (N, 3, 2) principal tangent directions [t₁, t₂]
                        (from eigen-decomposition of shape operator)
        normals: (N, 3) surface normals
    
    Returns:
        Q: (N, 3, 3) orthonormal frame [v₁ | v₂ | v₃]
           where vᵢ are eigenvectors of target covariance
    """
    t1 = principal_dirs[:, :, 0]  # (N, 3) - max curvature direction
    t2 = principal_dirs[:, :, 1]  # (N, 3) - min curvature direction
    n = normals  # (N, 3) - surface normal
    
    # Build rotation matrix Q = [t1 | t2 | n]
    Q = torch.stack([t1, t2, n], dim=-1)  # (N, 3, 3)
    
    return Q


def matrix_log_SO3(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Matrix logarithm on SO(3) using Rodrigues formula.
    
    For rotation matrix R ∈ SO(3), computes log(R) ∈ so(3).
    
    Formula:
        log(R) = (θ / (2 sin θ)) × (R - R^T)
        where θ = arccos((tr(R) - 1) / 2)
    
    Args:
        R: (N, 3, 3) rotation matrices
        eps: Numerical stability
    
    Returns:
        log_R: (N, 3, 3) skew-symmetric matrices (Lie algebra so(3))
    """
    N = R.shape[0]
    device = R.device
    
    # Trace of R
    trace_R = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # (N,)
    
    # Rotation angle: θ = arccos((tr(R) - 1) / 2)
    cos_theta = (trace_R - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + eps, 1 - eps)
    theta = torch.arccos(cos_theta)  # (N,)
    
    # For small angles, use first-order approximation
    small_angle = theta < eps
    
    # Skew-symmetric part: (R - R^T) / 2
    skew = (R - R.transpose(-2, -1)) / 2  # (N, 3, 3)
    
    # Scale factor: θ / (2 sin θ)
    scale = torch.zeros_like(theta)
    scale[~small_angle] = theta[~small_angle] / (2 * torch.sin(theta[~small_angle] + eps))
    scale[small_angle] = 0.5  # lim_{θ→0} θ/(2 sin θ) = 0.5
    
    # log(R) = scale * (R - R^T)
    log_R = scale.view(N, 1, 1) * skew
    
    return log_R


def geodesic_distance_SO3(Q1: torch.Tensor, Q2: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance on SO(3) Riemannian manifold.
    
    The geodesic distance between two rotations is the angle of the
    relative rotation:
    
        d(Q1, Q2) = ||log(Q1^T Q2)||_F = θ
    
    where θ is the rotation angle of Q1^T Q2.
    
    This is the *natural* distance on SO(3) respecting its geometry.
    
    Args:
        Q1, Q2: (N, 3, 3) rotation matrices
    
    Returns:
        dist: (N,) geodesic distances (in radians)
    """
    # Relative rotation
    R_rel = torch.bmm(Q1.transpose(-2, -1), Q2)  # (N, 3, 3)
    
    # Matrix logarithm
    log_R = matrix_log_SO3(R_rel)  # (N, 3, 3)
    
    # Frobenius norm
    dist = torch.norm(log_R.reshape(log_R.shape[0], -1), dim=-1)  # (N,)
    
    return dist


def build_target_covariance(
    curvature_data: Dict[str, torch.Tensor],
    sigma0: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build target covariance from curvature data using spectral decomposition.
    
    Mathematical formulation:
        Σ★ = Q Λ Q^T
    
    where:
    - Λ = diag(λ₁², λ₂², λ₃²): eigenvalues from curvature
    - Q = [t₁ | t₂ | n]: eigenvectors from principal directions
    
    Args:
        curvature_data: Dict with keys:
            - 'kappa1': (N,) first principal curvature (larger |κ|)
            - 'kappa2': (N,) second principal curvature (smaller |κ|)
            - 'principal_dirs': (N, 3, 2) principal directions [t₁, t₂]
            - 'normals': (N, 3) surface normals
        sigma0: Base Gaussian scale
    
    Returns:
        Sigma_target: (N, 3, 3) target covariance matrices
        Lambda_target: (N, 3) target eigenvalues (standard deviations, not variances!)
        Q_target: (N, 3, 3) target eigenvectors
    """
    # Target eigenvalues from curvature
    Lambda_target = curvature_to_eigenvalues(
        curvature_data['kappa1'],
        curvature_data['kappa2'],
        sigma0
    )  # (N, 3)
    
    # Target eigenvectors from geometry
    Q_target = build_eigenvector_frame(
        curvature_data['principal_dirs'],
        curvature_data['normals']
    )  # (N, 3, 3)
    
    # Reconstruct target covariance: Σ = Q Λ² Q^T
    # (Note: λ are standard deviations, so Σ needs λ²)
    Lambda_diag = torch.diag_embed(Lambda_target ** 2)  # (N, 3, 3)
    Sigma_target = torch.bmm(torch.bmm(Q_target, Lambda_diag), Q_target.transpose(-2, -1))
    
    return Sigma_target, Lambda_target, Q_target


def spectral_alignment_loss(
    Sigma_current: torch.Tensor,
    Lambda_target: torch.Tensor,
    Q_target: torch.Tensor,
    lambda_rot: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute spectral alignment loss for covariance matrices.
    
    Loss formulation:
        L = L_eigenvalue + λ_rot × L_eigenvector
    
    where:
    - L_eigenvalue: ||Λ_current - Λ_target²||²  (scale alignment)
    - L_eigenvector: d_geodesic(Q_current, Q_target)  (orientation alignment)
    
    Args:
        Sigma_current: (N, 3, 3) current covariance matrices
        Lambda_target: (N, 3) target eigenvalues (std devs)
        Q_target: (N, 3, 3) target eigenvectors
        lambda_rot: Weight for eigenvector alignment
    
    Returns:
        loss_total: Total spectral alignment loss
        loss_eigenvalue: Eigenvalue alignment loss
        loss_eigenvector: Eigenvector alignment loss
    """
    # Eigen-decomposition of current covariance (differentiable!)
    Lambda_current_sq, Q_current = torch.linalg.eigh(Sigma_current)  # (N, 3), (N, 3, 3)
    
    # Convert to standard deviations (eigenvalues of Σ are variances)
    Lambda_current = torch.sqrt(torch.clamp(Lambda_current_sq, min=1e-8))  # (N, 3)
    
    # Eigenvalue loss (L2 on standard deviations)
    loss_eigenvalue = ((Lambda_current - Lambda_target) ** 2).mean()
    
    # Eigenvector loss (geodesic distance on SO(3))
    geodesic_dists = geodesic_distance_SO3(Q_current, Q_target)  # (N,)
    loss_eigenvector = geodesic_dists.mean()
    
    # Total loss
    loss_total = loss_eigenvalue + lambda_rot * loss_eigenvector
    
    return loss_total, loss_eigenvalue, loss_eigenvector


__all__ = [
    "build_covariance",
    "smooth_F_field",
    "select_graph_nodes",
    "build_graph_laplacian",
    "build_interpolation_weights",
    "soft_top_k",
    "gumbel_select_k",
    # Polar decomposition
    "polar_decomposition",
    "build_covariance_polar",
    # Spectral alignment
    "curvature_to_eigenvalues",
    "build_eigenvector_frame",
    "build_target_covariance",
    "spectral_alignment_loss",
    "geodesic_distance_SO3",
    "matrix_log_SO3",
]