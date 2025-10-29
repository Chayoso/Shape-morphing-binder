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
    
    Returns:x
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

def build_covariance_polar(
    F_interp: torch.Tensor,
    sigma0: float,
    use_adaptive_scale: bool = False,
    local_spacing: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Build covariance using polar decomposition: Σ = S·Σ₀·S (rotation removed!)
    
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
    
    # 🔥 Covariance: Σ = S·Σ₀·S (ROTATION REMOVED!)
    # Remove rotation R to avoid unwanted orientation artifacts
    # Only use stretch S to capture physical deformation (compression/extension)
    S_Sigma0 = torch.bmm(S, Sigma0)
    cov = torch.bmm(S_Sigma0, S)  # No R! Pure stretch
    
    return cov


# ============================================================================
# Covariance Construction
# ============================================================================

def build_covariance(
    points: torch.Tensor,
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    knn,
    cfg: Dict,
    learnable_cov_module=None  # 🔥 NEW: Optional learnable covariance module
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build covariance matrices via F-field interpolation + optional learnable refinement.

    cfg knobs (all optional):
      cfg["sigma0"]: float (default 0.08)
      cfg["k_F"]: int (default 32)
      cfg["use_F_smoothing"]: bool
      cfg["use_adaptive_scale"]: bool
      cfg["use_polar_decomposition"]: bool

      # 🔥 NEW: Learnable covariance
      cfg["learnable"] = {
         "enabled": bool,               # Enable learnable covariance
         "alpha": float (0.0-1.0),      # Mixing weight (0=pure learnable, 1=pure physics)
      }

      # Density prior (anchor → per-point interpolation):
      cfg["density"] = {
         "rho_anchor": (N,) torch on x_low device, normalized ~[0.25,4.0]
         "use_scale_prior": True/False,
         "scale_kappa": 0.15,            # sensitivity (0.10~0.20)
         "scale_max_up": 0.12,           # max +% enlarge of sigma (0.10~0.15)
         "scale_smooth_alpha": 8.0,      # smoothness (6~10)
         "allow_shrink": False,          # if True, also allows ≤1 scaling (two-sided)
      }
    """
    sigma0 = float(cfg.get("sigma0", 0.08))
    k_F = int(cfg.get("k_F", 32))
    use_F_smoothing = bool(cfg.get("use_F_smoothing", True))
    use_adaptive_scale = bool(cfg.get("use_adaptive_scale", False))
    use_polar = bool(cfg.get("use_polar_decomposition", True))  # default ON

    # 1) (optional) smoothing
    F_smooth = smooth_F_field(x_low, F_low, cfg.get("F_smooth", {})) if use_F_smoothing else F_low

    # 2) interpolate F to upsampled points
    idx, w = knn(points, x_low, k_F)               # idx: (M,k) over x_low
    F_neighbors = F_smooth[idx]                    # (M,k,3,3)
    F_interp = torch.einsum('mk,mkrc->mrc', w, F_neighbors)  # (M,3,3)
    
    # 🔥 DEBUG: Check F values
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

    # 3) local spacing for adaptive scale (avoid self-zero)
    local_spacing = None
    if use_adaptive_scale:
        neighbor_anchors = x_low[idx]
        dists = torch.norm(neighbor_anchors - points.unsqueeze(1), dim=-1)  # (M,k)
        if dists.shape[1] >= 2:
            d2 = torch.topk(dists, k=2, largest=False).values[:, 1]
            local_spacing = d2.clamp(min=1e-6)
        else:
            local_spacing = dists[:, 0].clamp(min=1e-6)

    # ---- (D) NEW: anchor-density → per-point density (interpolate with same weights w)
    rho_cfg = cfg.get("density", {})
    rho_anchor = rho_cfg.get("rho_anchor", None)   # (N,) on x_low
    rho_pts = None
    if isinstance(rho_anchor, torch.Tensor) and rho_anchor.shape[0] == x_low.shape[0]:
        rho_anchor = rho_anchor.to(device=points.device, dtype=points.dtype)
        rho_pts = (w * rho_anchor[idx]).sum(dim=1)                        # (M,)
        rho_pts = (rho_pts / (rho_pts.mean() + EPS_SAFE)).clamp(0.25, 4.0)

    # 4) covariance (polar or direct) + smooth density prior
    if use_polar:
        # 🔥 CRITICAL: Check if we should use curvature-based initialization
        # For target mesh (F≈Identity), use curvature instead of F-field!
        curvature_cfg = cfg.get("curvature_cov", {})
        use_curvature = bool(curvature_cfg.get("enabled", False))
        curvature_data = cfg.get("curvature_data", None)  # From pipeline
        
        # 🔥 Check if this is Episode 0 (initial frame) - use curvature-based covariance
        # More reliable than F≈I check which can have numerical issues
        episode = cfg.get("episode", -1)  # -1 means target/initial
        is_initial_frame = (episode == 0) or (episode == -1)  # Episode 0 or target mesh
        
        # Base sigma at each point
        if use_adaptive_scale and local_spacing is not None:
            sigma_adaptive = sigma0 * torch.clamp(local_spacing / (local_spacing.mean() + EPS_SAFE), 0.3, 2.0)
        else:
            sigma_adaptive = torch.full((points.shape[0],), sigma0, device=points.device, dtype=points.dtype)

        # (E1) Smooth density-scale prior (no dead-zone)
        if rho_pts is not None and bool(rho_cfg.get("use_scale_prior", False)):
            kappa = float(rho_cfg.get("scale_kappa", 0.15))
            max_up = float(rho_cfg.get("scale_max_up", 0.12))
            alpha = float(rho_cfg.get("scale_smooth_alpha", 8.0))
            allow_shrink = bool(rho_cfg.get("allow_shrink", False))
            inv = (rho_pts + 1e-3).pow(-kappa)  # sparse(ρ↓) → inv↑
            s_factor = _smooth_scaleFactor(inv, allow_shrink=allow_shrink, max_up=max_up, alpha=alpha)
            
            # Debug: Check correlation
            with torch.no_grad():
                print(f"    [Density-Scale Debug]")
                print(f"      rho_pts: min={rho_pts.min():.3f}, mean={rho_pts.mean():.3f}, max={rho_pts.max():.3f}")
                print(f"      inv: min={inv.min():.3f}, mean={inv.mean():.3f}, max={inv.max():.3f}")
                print(f"      s_factor: min={s_factor.min():.3f}, mean={s_factor.mean():.3f}, max={s_factor.max():.3f}")
                
                # Check correlation: rho↑ should → s_factor↓
                if len(rho_pts) > 10:
                    corr_matrix = torch.corrcoef(torch.stack([rho_pts, s_factor]))
                    corr = corr_matrix[0, 1].item()
                    print(f"      corr(rho_pts, s_factor): {corr:.3f}")
                    if corr < -0.3:
                        print(f"      ✓ CORRECT: High density → small scale")
                    elif corr > 0.3:
                        print(f"      ✗ WRONG: High density → LARGE scale (BUG!)")
                    else:
                        print(f"      ⚠ WEAK correlation")
            
            sigma_adaptive = sigma_adaptive * s_factor

        # 🔥 CRITICAL: Use curvature-based covariance for initial frame (Episode 0)
        if use_curvature and is_initial_frame and curvature_data is not None:
            print(f"[Curvature-Based Covariance] ✓ ACTIVE (Episode {episode}: Initial Frame)")
            
            # Build anisotropic covariance from curvature
            Sigma_curv, Lambda_curv, Q_curv = build_target_covariance(
                curvature_data, sigma0  # Use base sigma0
            )
            
            # Apply density-based scaling to curvature eigenvalues
            if rho_pts is not None and bool(rho_cfg.get("use_scale_prior", False)):
                # Scale all eigenvalues uniformly by s_factor
                s_factor_3d = s_factor.unsqueeze(-1)  # (M, 1)
                Lambda_curv_scaled = Lambda_curv * s_factor_3d  # (M, 3)
                
                # Rebuild covariance with scaled eigenvalues
                Lambda_diag = torch.diag_embed(Lambda_curv_scaled ** 2)
                cov = torch.bmm(torch.bmm(Q_curv, Lambda_diag), Q_curv.transpose(-2, -1))
                
                print(f"  Lambda (scaled): [{Lambda_curv_scaled.min():.4f}, {Lambda_curv_scaled.max():.4f}]")
            else:
                cov = Sigma_curv
                print(f"  Lambda (base): [{Lambda_curv.min():.4f}, {Lambda_curv.max():.4f}]")
            
            print(f"  Anisotropy (max/min): {(Lambda_curv.max() / (Lambda_curv.min() + 1e-8)):.2f}")
            
            # Small stabilization
            eps_physics = 1e-6
            eye_reg = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
            cov = cov + eps_physics * eye_reg
            
        else:
            # Normal F-field based covariance
            # Build Σ = S·Σ₀·S (ROTATION REMOVED!)
            Sigma0 = (sigma_adaptive.view(-1,1,1) ** 2) * torch.eye(3, device=points.device).unsqueeze(0)
            R, S = polar_decomposition(F_interp)  # Still get R for debug, but don't use it
            
            # 🔥 DEBUG: Check polar decomposition results
            with torch.no_grad():
                print(f"[Polar Decomposition Debug (F-based)]")
                print(f"  ⚠ NOTE: R is NOT used in covariance (rotation removed!)")
                print(f"  R range (all elements): [{R.min():.6f}, {R.max():.6f}]")
                print(f"  S range (all elements): [{S.min():.6f}, {S.max():.6f}]")
                
                # Check R orthogonality (for verification only, R is not used)
                R_RT = torch.bmm(R, R.transpose(-2, -1))
                eye_err = (R_RT - torch.eye(3, device=R.device).unsqueeze(0)).abs().max()
                print(f"  R orthogonality error: {eye_err:.6e} (R not used)")
                
                # Check S properties (should be symmetric positive definite)
                S_diag = torch.diagonal(S, dim1=-2, dim2=-1)
                print(f"  S diagonal range: [{S_diag.min():.6f}, {S_diag.max():.6f}]")
                
                # Check S eigenvalues (most important!)
                try:
                    S_eigvals = torch.linalg.eigvalsh(S)  # (M, 3), sorted ascending
                    print(f"  S eigenvalues: [{S_eigvals.min():.6f}, {S_eigvals.max():.6f}]")
                    if (S_eigvals < 0).any():
                        print(f"  ❌ ERROR: {(S_eigvals < 0).sum().item()} negative S eigenvalues!")
                    elif (S_eigvals < 1e-6).any():
                        print(f"  ⚠ WARNING: {(S_eigvals < 1e-6).sum().item()} near-zero S eigenvalues!")
                    else:
                        print(f"  ✓ S is positive definite (all eigenvalues > 0)")
                except Exception as e:
                    print(f"  ⚠ Could not compute S eigenvalues: {e}")
            
            # 🔥 CRITICAL: Condition number limiting (Kerbl et al., Zwicker et al.)
            # Prevent needle-like splats that cause blocky artifacts
            
            # Get eigenvalues of S (stretch components)
            S_eigvals = torch.linalg.eigvalsh(S)  # (M, 3), sorted ascending
            
            # Estimate voxel_size from point spacing if not provided
            voxel_size = cfg.get("voxel_size", 0.5)  # Default from grid_dx
            
            # Clamp eigenvalues to reasonable range (Kerbl et al., 2023)
            s_min = max(0.4 * voxel_size, 0.01)     # Lower bound: 0.4 * voxel_size
            s_max = min(6.0 * voxel_size, 1.0)      # Upper bound: 6 * voxel_size
            S_eigvals_clamped = S_eigvals.clamp(s_min, s_max)
            
            # Condition number limiting: κ = λ_max / λ_min ≤ κ_max
            kappa_max = cfg.get("kappa_max", 40.0)  # Typical: 30-50
            kappa = S_eigvals_clamped[:, 2] / (S_eigvals_clamped[:, 0] + 1e-8)
            
            # If κ > κ_max, increase smallest eigenvalue
            needs_fix = kappa > kappa_max
            if needs_fix.any():
                target_min = S_eigvals_clamped[:, 2] / kappa_max
                S_eigvals_clamped[:, 0] = torch.where(
                    needs_fix,
                    torch.maximum(S_eigvals_clamped[:, 0], target_min),
                    S_eigvals_clamped[:, 0]
                )
            
            # Reconstruct S with clamped eigenvalues
            # S = V diag(λ) V^T where V are eigenvectors of original S
            U_s, _, Vt_s = torch.linalg.svd(S)
            S_fixed = U_s @ torch.diag_embed(S_eigvals_clamped) @ Vt_s
            
            # Build covariance with fixed S (ROTATION REMOVED!)
            S_Sigma0 = torch.bmm(S_fixed, Sigma0)
            cov = torch.bmm(S_Sigma0, S_fixed)  # No R! Pure stretch
            
            # Small numerical stabilization
            eps_physics = 1e-6
            eye_reg = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
            cov = cov + eps_physics * eye_reg
        
        # 🔥 DEBUG: Check final covariance before learnable
        with torch.no_grad():
            print(f"[Physics Covariance Debug]")
            print(f"  cov range (all elements): [{cov.min():.6f}, {cov.max():.6f}]")
            cov_diag = torch.diagonal(cov, dim1=-2, dim2=-1)
            print(f"  cov diagonal range: [{cov_diag.min():.6f}, {cov_diag.max():.6f}]")
            
            cov_det = torch.det(cov)
            print(f"  det(cov) range: [{cov_det.min():.6e}, {cov_det.max():.6e}]")
            
            # Check eigenvalues (definitive test for positive definiteness!)
            try:
                cov_eigvals = torch.linalg.eigvalsh(cov)  # (M, 3), sorted ascending
                print(f"  cov eigenvalues: [{cov_eigvals.min():.6e}, {cov_eigvals.max():.6e}]")
                if (cov_eigvals < 0).any():
                    print(f"  ❌ ERROR: {(cov_eigvals < 0).sum().item()} negative eigenvalues!")
                    print(f"    → Covariance is NOT positive definite!")
                elif (cov_eigvals < 1e-8).any():
                    print(f"  ⚠ WARNING: {(cov_eigvals < 1e-8).sum().item()} near-zero eigenvalues!")
                else:
                    print(f"  ✓ Covariance is positive definite (all eigenvalues > 0)")
            except Exception as e:
                print(f"  ⚠ Could not compute eigenvalues: {e}")

    else:
        # Direct FF^T path
        if use_adaptive_scale and local_spacing is not None:
            sigma_adapt = sigma0 * torch.clamp(local_spacing / (local_spacing.mean() + EPS_SAFE), 0.3, 2.0)
            cov = (sigma_adapt.view(-1,1,1) ** 2) * torch.matmul(F_interp, F_interp.transpose(-2, -1))
        else:
            cov = (sigma0 ** 2) * torch.matmul(F_interp, F_interp.transpose(-2, -1))
        
        # 🔥 CRITICAL: Add strong regularization for numerical stability (especially 300k+ batches)
        batch_size = cov.shape[0]
        if batch_size > 200000:
            eps_physics = 2e-4  # Very strong for 200k+
        elif batch_size > 100000:
            eps_physics = 1e-4  # Strong for 100k-200k
        else:
            eps_physics = 2e-5  # Normal for <100k
        
        eye_reg = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov = cov + eps_physics * eye_reg

        # (E1-direct) Smooth density-scale prior on Σ
        if rho_pts is not None and bool(rho_cfg.get("use_scale_prior", False)):
            kappa = float(rho_cfg.get("scale_kappa", 0.15))
            max_up = float(rho_cfg.get("scale_max_up", 0.12))
            alpha = float(rho_cfg.get("scale_smooth_alpha", 8.0))
            allow_shrink = bool(rho_cfg.get("allow_shrink", False))
            inv = (rho_pts + 1e-3).pow(-kappa)
            s_factor = _smooth_scaleFactor(inv, allow_shrink=allow_shrink, max_up=max_up, alpha=alpha)
            
            # Debug: Check correlation (same as polar path)
            with torch.no_grad():
                print(f"    [Density-Scale Debug - Direct Path]")
                print(f"      rho_pts: min={rho_pts.min():.3f}, mean={rho_pts.mean():.3f}, max={rho_pts.max():.3f}")
                print(f"      inv: min={inv.min():.3f}, mean={inv.mean():.3f}, max={inv.max():.3f}")
                print(f"      s_factor: min={s_factor.min():.3f}, mean={s_factor.mean():.3f}, max={s_factor.max():.3f}")
                
                # Check correlation: rho↑ should → s_factor↓
                if len(rho_pts) > 10:
                    corr_matrix = torch.corrcoef(torch.stack([rho_pts, s_factor]))
                    corr = corr_matrix[0, 1].item()
                    print(f"      corr(rho_pts, s_factor): {corr:.3f}")
                    if corr < -0.3:
                        print(f"      ✓ CORRECT: High density → small scale")
                    elif corr > 0.3:
                        print(f"      ✗ WRONG: High density → LARGE scale (BUG!)")
                    else:
                        print(f"      ⚠ WEAK correlation")
            
            cov = (s_factor.view(-1,1,1) ** 2) * cov

    # 5) numeric symmetry guard (helps tiny asymmetries from float ops)
    cov = 0.5 * (cov + cov.transpose(-2, -1))
    
    # 🔥 5.5) CRITICAL: Enforce diagonal dominance to guarantee positive definite
    # Problem: FF^T/Polar can have off-diagonal elements that are too large
    # Solution: Rebuild matrix with clamped off-diagonals (non-inplace for autograd)
    
    # Extract diagonal
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)  # (M, 3)
    d0, d1, d2 = diag[:, 0], diag[:, 1], diag[:, 2]  # (M,)
    
    # Compute safe bounds for off-diagonals
    # For 3×3 matrix to be strictly positive definite with det >> 0:
    # We use 0.5 factor (very conservative) to ensure det(Σ) is well above zero
    # This prevents near-singular matrices in large batches (300k+)
    factor = 0.5  # Conservative for numerical stability
    bound_01 = factor * torch.sqrt(torch.clamp(d0 * d1, min=1e-10))  # (M,)
    bound_02 = factor * torch.sqrt(torch.clamp(d0 * d2, min=1e-10))
    bound_12 = factor * torch.sqrt(torch.clamp(d1 * d2, min=1e-10))
    
    # Extract and clamp off-diagonals
    c01 = torch.clamp(cov[:, 0, 1], -bound_01, bound_01)  # (M,)
    c02 = torch.clamp(cov[:, 0, 2], -bound_02, bound_02)
    c12 = torch.clamp(cov[:, 1, 2], -bound_12, bound_12)
    
    # Rebuild covariance matrix (non-inplace, autograd-safe)
    cov_new = torch.zeros_like(cov)
    cov_new[:, 0, 0] = d0
    cov_new[:, 1, 1] = d1
    cov_new[:, 2, 2] = d2
    cov_new[:, 0, 1] = c01
    cov_new[:, 1, 0] = c01  # Symmetry
    cov_new[:, 0, 2] = c02
    cov_new[:, 2, 0] = c02
    cov_new[:, 1, 2] = c12
    cov_new[:, 2, 1] = c12
    
    cov = cov_new  # Replace with clean matrix
    
    # 🔥 5.6) Add adaptive regularization for extra safety
    # CRITICAL: Large batches (300k+) need strong regularization to prevent
    # determinant → 0 due to numerical precision limits (float32)
    # Strategy: eps proportional to diagonal magnitude (adaptive)
    
    batch_size = cov.shape[0]
    diag_mean = torch.diagonal(cov, dim1=-2, dim2=-1).mean()  # Average diagonal
    
    if batch_size > 200000:
        eps_ratio = 0.15  # 15% of diagonal for 200k+ (very strong)
    elif batch_size > 100000:
        eps_ratio = 0.10  # 10% for 100k-200k (strong)
    else:
        eps_ratio = 0.05  # 5% for <100k (normal)
    
    eps_extra = eps_ratio * diag_mean  # Adaptive to scale
    eps_extra = torch.clamp(eps_extra, min=1e-4, max=2e-3)  # Safety bounds
    
    eye_extra = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
    cov = cov + eps_extra * eye_extra
    
    # Diagnostic check (no-grad for printing only)
    with torch.no_grad():
        diag_cov = torch.diagonal(cov, dim1=-2, dim2=-1)
        diag_min = diag_cov.min()
        
        # Check for REAL issues: negative diagonal (not just negative elements!)
        # Note: Off-diagonal elements CAN be negative in positive definite matrices
        if diag_min < 0:
            cov_min = cov.min()
            cov_max = cov.max()
            print(f"[ERROR] Physics cov has NEGATIVE DIAGONAL!")
            print(f"  Diagonal range: [{diag_min:.6e}, {diag_cov.max():.6e}]")
            print(f"  Matrix range: [{cov_min:.6e}, {cov_max:.6e}]")
            print(f"  This indicates numerical instability in F-field!")
        # Off-diagonal can be negative - this is NORMAL for covariance matrices!

    # 🔥 6) Learnable covariance refinement (hybrid mode)
    learnable_cfg = cfg.get("learnable", {})
    use_learnable = bool(learnable_cfg.get("enabled", False))
    
    if use_learnable and learnable_cov_module is not None:
        alpha = float(learnable_cfg.get("alpha", 0.3))
        
        # Apply hybrid mixing: α·Σ_physics + (1-α)·Σ_learnable
        cov = learnable_cov_module(cov_physics=cov, alpha=alpha)
        
        # Symmetry guard after learnable refinement
        cov = 0.5 * (cov + cov.transpose(-2, -1))

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
    # 🔥 DEBUG: Print curvature BEFORE scaling
    with torch.no_grad():
        print(f"  [Curvature INPUT to eigenvalues]")
        print(f"    kappa1: mean={kappa1.mean():.6f}, range=[{kappa1.min():.6f}, {kappa1.max():.6f}]")
        print(f"    kappa2: mean={kappa2.mean():.6f}, range=[{kappa2.min():.6f}, {kappa2.max():.6f}]")
        print(f"    sigma0: {sigma0:.6f}")
    
    # Geometric scaling law
    lambda1 = sigma0 / torch.sqrt(1 + kappa1**2 + eps)
    lambda2 = sigma0 / torch.sqrt(1 + kappa2**2 + eps)
    lambda3 = torch.full_like(lambda1, sigma0 * 0.12)  # 🔥 Along normal (thin: 12% of sigma0)
    
    # 🔥 DEBUG: Print lambda values (NO CLAMPING - trust curvature!)
    with torch.no_grad():
        print(f"  [Lambda (curvature-based, no clamp)]")
        print(f"    lambda1: mean={lambda1.mean():.6f}, range=[{lambda1.min():.6f}, {lambda1.max():.6f}]")
        print(f"    lambda2: mean={lambda2.mean():.6f}, range=[{lambda2.min():.6f}, {lambda2.max():.6f}]")
    
    # 🔥 NO CLAMPING: Let curvature naturally determine Gaussian sizes
    # Curvature-based scaling already provides physically meaningful bounds
    
    eigenvalues = torch.stack([lambda1, lambda2, lambda3], dim=-1)  # (N, 3)
    
    # 🔥 DEBUG: Print actual lambda values AFTER curvature application
    with torch.no_grad():
        print(f"  [Lambda After Curvature]")
        print(f"    lambda1 (curvature-based): mean={lambda1.mean():.6f}, range=[{lambda1.min():.6f}, {lambda1.max():.6f}]")
        print(f"    lambda2 (curvature-based): mean={lambda2.mean():.6f}, range=[{lambda2.min():.6f}, {lambda2.max():.6f}]")
        print(f"    lambda3 (normal/thin): mean={lambda3.mean():.6f}, range=[{lambda3.min():.6f}, {lambda3.max():.6f}]")
        print(f"    Anisotropy (max/min): {(lambda1.max() / (lambda3.min() + 1e-8)):.2f}")
    
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
        # symmetric: smoothly approaches both bounds
        s = 1.0 + max_up * torch.tanh(alpha * (inv - 1.0))
    else:
        # lower-bound 1.0 is enforced smoothly via sigmoid
        s = 1.0 + max_up * torch.sigmoid(alpha * (inv - 1.0))
    return s


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