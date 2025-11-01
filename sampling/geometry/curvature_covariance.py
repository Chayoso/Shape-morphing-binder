"""
Curvature-based Covariance Initialization (TARGET ONLY!)
Revised: curvature proxy direction fix, PCA tangent frame (optional),
robust anisotropy normalization, footprint clamp, iso ramp, EMA.

Author: CHAYO (2025-10) + patch
"""

import torch
import numpy as np
from typing import Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Import project's FAISS-based Hybrid KNN
# ──────────────────────────────────────────────────────────────────────────────
try:
    from ..analysis.knn import HybridFAISSKNN   # relative to sampling/geometry/
    _HAS_FAISS = True
except Exception as _e:
    HybridFAISSKNN = None
    _HAS_FAISS = False

def _robust_aniso_norm(rho: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """MAD-based continuous normalization + smooth compression (1-exp(-·))."""
    # NaN check
    if torch.isnan(rho).any():
        print("[WARN] _robust_aniso_norm: rho contains NaN, returning zeros")
        return torch.zeros_like(rho)
    
    med = rho.median()
    mad = torch.clamp(torch.median(torch.abs(rho - med)), min=eps)  # safe clamping
    z = torch.clamp((rho - med) / (1.4826 * mad + eps), min=0.0)  # positive tail only
    return 1.0 - torch.exp(-z)  # smooth compression to [0,1)

def _curv_proxy_from_planarity(s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    planarity s∈[0,1] → curvature proxy c = 1 - s, positive part of z-score.
    
    Note: planarity represents curvature "magnitude" only, not sign (convex/concave).
    This is intentional - Gaussian scale depends only on curvature magnitude,
    and direction info is already captured by the normal vector.
    """
    # NaN check
    if torch.isnan(s).any():
        print("[WARN] _curv_proxy_from_planarity: s contains NaN, returning zeros")
        return torch.zeros_like(s)
    
    c = 1.0 - s  # s↑(flat) → c↓(low curvature), sign-agnostic
    m = c.mean()
    sd = torch.clamp(c.std(), min=eps)
    return torch.clamp((c - m) / (sd + eps), min=0.0)  # positive part only (curvature magnitude)

def _tangent_frame_gram(normals_t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Legacy Gram-Schmidt-based frame (fast, may cause jitter)."""
    n = normals_t / (normals_t.norm(dim=1, keepdim=True) + eps)
    ref = torch.zeros_like(n); ref[:, 2] = 1.0
    parallel_mask = (torch.abs(n[:, 2]) > 0.9)
    ref[parallel_mask, 2] = 0.0; ref[parallel_mask, 0] = 1.0
    t1 = ref - (ref * n).sum(dim=1, keepdim=True) * n
    t1 = t1 / (t1.norm(dim=1, keepdim=True) + eps)
    t2 = torch.cross(n, t1, dim=1)
    return torch.stack([t1, t2, n], dim=2)  # (N,3,3)

# ──────────────────────────────────────────────────────────────────────────────
# Memory-safe FAISS KNN-based PCA frame (signature preserved)
# ──────────────────────────────────────────────────────────────────────────────
def _faiss_knn_indices_batched(
    points_t: torch.Tensor,     # (N,3), device
    k_fetch: int,               # Requested neighbor count (recommend k + margin for self-removal)
    knn: "HybridFAISSKNN",
    batch_q: int
) -> torch.Tensor:
    """
    Build FAISS index once, then query in batches to get (N, k_fetch) neighbor indices.
    Returns tensor with same device/dtype (long) as points_t.
    """
    device = points_t.device
    N, D = points_t.shape
    idx_all = torch.empty((N, k_fetch), dtype=torch.long, device=device)

    # Use internal index cache (lower-level method but stable in our codebase)
    data_np = points_t.detach().cpu().float().numpy()
    M = data_np.shape[0]
    nlist = min(knn.nlist, max(4, M // 100))
    nprobe = min(knn.nprobe, nlist)
    cache_key = (M, D, nlist, int(points_t.untyped_storage().data_ptr()), knn._epoch)
    index = knn._build_index(points_t, D, nlist, nprobe, cache_key)

    for s in range(0, N, batch_q):
        e = min(s + batch_q, N)
        q_np = data_np[s:e]  # (b, D)
        d_np, i_np = index.search(q_np, k_fetch)
        idx_all[s:e] = torch.from_numpy(i_np).to(device=device, dtype=torch.long)

    return idx_all

def _tangent_frame_pca(
    points_t: torch.Tensor,
    normals_t: torch.Tensor,
    k: int = 16,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Legacy name/signature preserved.
    Internal implementation replaced with FAISS KNN + batched PCA to prevent OOM.
    - Over-fetch with k_fetch = k + 8 to remove self-neighbors, then trim.
    - Batch size automatically determined based on VRAM budget (~512MB).
    """
    if not _HAS_FAISS or HybridFAISSKNN is None:
        raise RuntimeError(
            "Cannot run _tangent_frame_pca without FAISS/HybridFAISSKNN. "
            "hybrid_knn.py must be in the same package and faiss must be installed."
        )

    device = points_t.device
    dtype  = points_t.dtype
    N = points_t.shape[0]
    k = int(k)

    # 1) KNN index (may include self) — with margin for self-removal
    k_fetch = min(k + 8, max(k + 1, points_t.shape[0]))  # margin for self-removal
    knn = HybridFAISSKNN(use_faiss=True, use_ivf=True, nlist=100, nprobe=10)

    # Query batch size (adjust based on memory). Default ~50k, lower if VRAM is tight.
    batch_q = 50000 if N >= 50000 else max(8192, (N // 4) + 1)
    idx_knn_fetch = _faiss_knn_indices_batched(points_t, k_fetch, knn, batch_q)  # (N,k_fetch)

    # 2) Remove self-neighbors and select top k (vectorized)
    #    Compare with row indices [0..N-1] and remove matching indices (True)
    row_ids = torch.arange(N, device=device).unsqueeze(1).expand(-1, k_fetch)
    mask = (idx_knn_fetch != row_ids)

    # Use position indices as weights to gather True values to front
    pos = torch.arange(k_fetch, device=device).unsqueeze(0).expand(N, -1)
    pos_masked = torch.where(mask, pos, k_fetch * torch.ones_like(pos))
    # Select k smallest positions (= valid neighbors)
    _, relpos = torch.topk(-pos_masked, k=min(k, k_fetch), dim=1)  # negative to select min
    neigh_idx = idx_knn_fetch.gather(1, relpos)  # (N,k)

    # 3) Batched PCA (auto-batching based on VRAM budget)
    #    Approx (B*k*3*4 bytes) + margin. Estimate with 512MB budget.
    bytes_budget = 512 * 1024 * 1024
    bytes_per_row = k * 3 * 4 * 4  # estimate with margin
    batch_pca = max(1024, min(N, bytes_budget // max(bytes_per_row, 1)))

    R_out = torch.empty((N, 3, 3), dtype=dtype, device=device)
    n = normals_t / (normals_t.norm(dim=1, keepdim=True) + eps)

    for s in range(0, N, batch_pca):
        e = min(s + batch_pca, N)
        b = e - s

        neigh_idx_b = neigh_idx[s:e]                 # (b,k)
        P = points_t[neigh_idx_b]                    # (b,k,3)
        mu = P.mean(dim=1, keepdim=True)             # (b,1,3)
        Q = P - mu                                   # (b,k,3)
        
        # Validate Q (prevent case where all points are identical)
        Q_norm = torch.norm(Q, dim=-1)  # (b,k)
        invalid_points = (Q_norm.max(dim=1)[0] < eps)  # (b,) - invalid if all neighbors equal center
        
        # Covariance and eigendecomposition
        C = torch.matmul(Q.transpose(1, 2), Q) / (Q.shape[1] - 1 + eps)   # (b,3,3)
        C = 0.5 * (C + C.transpose(1, 2))  # symmetrize
        
        # PSD guarantee: add tiny regularization (numerical stability)
        eye_reg = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * (eps * 10)  # (1,3,3)
        C = C + eye_reg  # shift all eigenvalues by eps*10
        
        # Invalid check: NaN/Inf/Zero-variance/Singular
        nan_mask = torch.isnan(C).any(dim=1).any(dim=1) | torch.isinf(C).any(dim=1).any(dim=1)  # (b,)
        zero_var_mask = invalid_points | (torch.det(C).abs() < eps**3)  # (b,) - singular or zero variance
        
        invalid_mask = nan_mask | zero_var_mask  # (b,)
        if invalid_mask.any():
            num_bad = invalid_mask.sum().item()
            if num_bad > 0:
                print(f"[WARN] PCA batch [{s}:{e}] contains {num_bad}/{b} invalid covariance matrices, replacing with identity")
                if nan_mask.any():
                    print(f"  - NaN/Inf: {nan_mask.sum().item()} matrices")
                if zero_var_mask.any():
                    print(f"  - Zero-variance/Singular: {zero_var_mask.sum().item()} matrices")
            
            # Replace only invalid matrices with identity
            if invalid_mask.all():
                # If all invalid, replace entire batch with identity
                C = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(b, 3, 3).clone()
            else:
                C_good = C[~invalid_mask]
                num_bad = invalid_mask.sum().item()
                C_fixed = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_bad, 3, 3).clone()
                C_new = torch.empty_like(C)
                C_new[invalid_mask] = C_fixed
                C_new[~invalid_mask] = C_good
                C = C_new
        
        # Additional: check singular matrices with very small eigenvalues (numerical stability)
        # However, hard to check before eigh, so handle after eigh
        
        # Eigendecomposition (safe for symmetric matrices)
        # CUSOLVER may raise RuntimeError or _LinAlgError
        try:
            evals, evecs = torch.linalg.eigh(C)          # (b,3), (b,3,3)
        except (RuntimeError, Exception) as ex:  # use 'ex' to avoid variable name conflict
            # Handle CUSOLVER error or other exceptions
            error_str = str(ex)
            if "CUSOLVER" in error_str or "LinAlg" in error_str or "INVALID_VALUE" in error_str:
                print(f"[ERROR] eigh failed (CUSOLVER/LinAlg) in batch [{s}:{e}]: {error_str[:200]}")
            else:
                print(f"[ERROR] eigh failed in batch [{s}:{e}]: {error_str[:200]}")
            
            # Output C statistics (for debugging)
            C_dets = torch.det(C)
            print(f"  C stats: min={C.min().item():.6e}, max={C.max().item():.6e}, "
                  f"has_nan={torch.isnan(C).any().item()}, has_inf={torch.isinf(C).any().item()}, "
                  f"det_range=[{C_dets.min().item():.6e}, {C_dets.max().item():.6e}]")
            
            # Check count of negative determinants (abnormal)
            neg_det_count = (C_dets < 0).sum().item()
            if neg_det_count > 0:
                print(f"  WARNING: {neg_det_count}/{b} matrices have negative determinant (numerical error)")
            
            # Replace all matrices with identity (fallback)
            print(f"  → Fallback: using identity matrices for all {b} points in this batch")
            evals = torch.ones((b, 3), device=device, dtype=dtype)
            evecs = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(b, 3, 3).clone()

        t1 = evecs[..., 2]                            # maximum variance axis
        t2 = evecs[..., 1]
        n_b = n[s:e]

        # Orthogonal realignment (numerical stability)
        t1 = t1 - (t1 * n_b).sum(-1, keepdim=True) * n_b
        t1 = t1 / (t1.norm(dim=1, keepdim=True) + eps)
        t2 = torch.cross(n_b, t1)

        R = torch.stack([t1, t2, n_b], dim=2)        # (b,3,3)
        R_out[s:e] = R

        # Memory cleanup
        del neigh_idx_b, P, Q, C, evals, evecs, t1, t2, n_b, R
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return R_out

# ──────────────────────────────────────────────────────────────────────────────
# Below: create_curvature_based_covariance_star is unchanged (frame selection calls above function)
# ──────────────────────────────────────────────────────────────────────────────

def create_curvature_based_covariance_star(
    points: np.ndarray,
    normals: np.ndarray,
    planarity: np.ndarray,
    anisotropy: np.ndarray,
    sigma_params: dict = None,
    principal_curv: np.ndarray = None,  # NEW: (N,2) [k1, k2] from STAGE 1 PCA
    principal_dirs: tuple = None  # NEW: (dir1, dir2) each (N,3) from STAGE 1
) -> np.ndarray:
    """
    Create target covariance Σ★ based on planarity and anisotropy (TARGET ONLY!)
    - Curvature: Use STAGE 1 principal_curv directly (if available) → else planarity-based proxy
    - Directions: Use STAGE 1 principal_dirs directly (if available) → else PCA/Gram-Schmidt
    - ρ normalization: MAD-based continuous + smooth compression
    - Frame: 'frame'='pca'|'gram'|'external' (default pca)  ※ internal pca is FAISS-based
    - Lower bound: max(min_px_radius/fx·z, sigma_floor) for screen footprint
    - Initial stabilization: iso_ramp blending
    - Temporal stabilization: EMA (optional) — state stored/reused in sigma_params dict
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points_t = torch.from_numpy(points).float().to(device)
    normals_t = torch.from_numpy(normals).float().to(device)
    s  = torch.from_numpy(planarity).float().to(device)
    rho = torch.from_numpy(anisotropy).float().to(device)
    
    # FIX: Input data validation (improved error message)
    nan_mask_points = torch.isnan(points_t).any(dim=1)  # (N,)
    nan_mask_normals = torch.isnan(normals_t).any(dim=1)  # (N,)
    
    if nan_mask_points.any() or nan_mask_normals.any():
        num_nan = (nan_mask_points | nan_mask_normals).sum().item()
        print(f"[ERROR] Found {num_nan}/{points.shape[0]} points with NaN in positions/normals")
        print(f"  - Check STAGE 1-5 output for numerical errors")
        print(f"  - Possible causes: degenerate geometry, extreme smoothing, invalid F-field")
        raise ValueError(f"points or normals contains NaN values ({num_nan} points affected)")
    
    # planarity and anisotropy NaNs replaced with midpoint (less critical)
    if torch.isnan(s).any():
        num_nan_s = torch.isnan(s).sum().item()
        print(f"[WARN] planarity contains {num_nan_s} NaN values, replacing with 0.5")
        s = torch.nan_to_num(s, nan=0.5)
    if torch.isnan(rho).any():
        num_nan_rho = torch.isnan(rho).sum().item()
        print(f"[WARN] anisotropy contains {num_nan_rho} NaN values, replacing with 0.5")
        rho = torch.nan_to_num(rho, nan=0.5)

    N = points.shape[0]; eps = 1e-6
    if sigma_params is None: sigma_params = {}

    # Hyperparameters
    sigma_n0   = float(sigma_params.get('sigma_n0', 0.015))
    sigma_t0   = float(sigma_params.get('sigma_t0', 0.022))
    a          = float(sigma_params.get('a', 1.2))
    b          = float(sigma_params.get('b', 0.30))
    u          = float(sigma_params.get('u', 0.20))
    sigma_floor_base = float(sigma_params.get('sigma_floor', 0.004))
    sigma_ceil = float(sigma_params.get('sigma_ceil', 0.080))
    frame_mode = sigma_params.get('frame', 'pca')
    pca_k      = int(sigma_params.get('pca_k', 16))
    iso_ramp   = float(sigma_params.get('iso_ramp', 0.4))
    ema_decay  = sigma_params.get('ema_decay', 0.85)  # float or None

    # ========================================================================
    # NEW: Use principal curvature from STAGE 1 (if available)
    # ========================================================================
    use_stage1_curv = (principal_curv is not None and len(principal_curv) == N)
    
    if use_stage1_curv:
        # k1, k2 are already normalized to [0, 0.5] range (divided by trace)
        k1_k2 = torch.from_numpy(principal_curv).float().to(device)  # (N, 2)
        k1, k2 = k1_k2[:, 0], k1_k2[:, 1]  # (N,), (N,)
        
        # Curvature magnitude: k_mean = (k1 + k2) / 2, anisotropy: k_aniso = |k1 - k2|
        kappa_hat = (k1 + k2) / 2.0  # mean curvature (0~0.5)
        rho_hat = torch.abs(k1 - k2)  # curvature anisotropy (0~0.5)
        
        # Normalize with z-score (same range as legacy proxy)
        kappa_hat = torch.clamp((kappa_hat - kappa_hat.mean()) / (kappa_hat.std() + eps), min=0.0)
        rho_hat = torch.clamp((rho_hat - rho_hat.mean()) / (rho_hat.std() + eps), min=0.0)
        
        print(f"[INFO] Using STAGE 1 principal curvatures (k1, k2) directly")
        print(f"  k_mean range: [{(k1+k2).min().item()/2:.4f}, {(k1+k2).max().item()/2:.4f}]")
        print(f"  k_aniso range: [{torch.abs(k1-k2).min().item():.4f}, {torch.abs(k1-k2).max().item():.4f}]")
    else:
        # Legacy: derive from planarity/anisotropy
        kappa_hat = _curv_proxy_from_planarity(s)
        rho_hat   = _robust_aniso_norm(rho)
        print(f"[INFO] Using planarity-based curvature proxy (STAGE 1 curv not available)")

    # ========================================================================
    # Frame computation
    # ========================================================================
    use_stage1_dirs = (principal_dirs is not None and len(principal_dirs) == 2)
    
    if use_stage1_dirs:
        # Use principal directions computed in STAGE 1
        dir1_np, dir2_np = principal_dirs
        if len(dir1_np) == N and len(dir2_np) == N:
            t1 = torch.from_numpy(dir1_np).float().to(device)  # (N, 3)
            t2 = torch.from_numpy(dir2_np).float().to(device)  # (N, 3)
            R = torch.stack([t1, t2, normals_t], dim=2)  # (N, 3, 3)
            print(f"[INFO] Using STAGE 1 principal directions for tangent frame")
        else:
            print(f"[WARN] STAGE 1 principal_dirs size mismatch, falling back to PCA")
            use_stage1_dirs = False
    
    if not use_stage1_dirs:
        # Legacy: PCA or Gram-Schmidt
        if frame_mode == 'external':
            R = sigma_params['tangent_R'].to(device)  # (N,3,3)
        elif frame_mode == 'gram':
            R = _tangent_frame_gram(normals_t)
        else:
            # Default: PCA (replaced with FAISS-based, no OOM)
            R = _tangent_frame_pca(points_t, normals_t, k=pca_k)

    # Curvature/anisotropy scaling
    sigma_n  = sigma_n0 / (1.0 + a * kappa_hat)           # curvature↑ → normal thinner
    sigma_tB = sigma_t0 * (1.0 + b * kappa_hat)           # curvature↑ → tangent thicker
    sigma_t1 = sigma_tB
    sigma_t2 = sigma_tB * (1.0 + u * rho_hat)

    # Lower bound based on screen footprint
    cam_z = sigma_params.get('cam_z', None)
    fx    = float(sigma_params.get('fx', 1200.0))
    min_px_radius = float(sigma_params.get('min_px_radius', 0.9))
    if cam_z is not None:
        cam_z_t = torch.as_tensor(cam_z, dtype=torch.float32, device=device)
        rmin_world = (min_px_radius / fx) * torch.clamp(cam_z_t, min=eps)
        sigma_floor = torch.maximum(rmin_world, torch.tensor(sigma_floor_base, device=device))
    else:
        sigma_floor = torch.full((N,), sigma_floor_base, device=device)

    # Convert sigma_ceil to Tensor (type matching for clamp)
    sigma_ceil_t = torch.full((N,), sigma_ceil, device=device)

    # Clamp function
    def _clamp(v):
        return torch.clamp(v, min=sigma_floor, max=sigma_ceil_t)

    sigma_t1 = _clamp(sigma_t1)
    sigma_t2 = _clamp(sigma_t2)
    sigma_n  = _clamp(sigma_n)

    # Stack scales
    scales = torch.stack([sigma_t1, sigma_t2, sigma_n], dim=1)  # (N,3)

    # EMA (optional): store/reuse state in sigma_params dict
    prev = sigma_params.get('ema_prev_scales', None)
    if prev is not None and ema_decay is not None:
        d = float(ema_decay)
        prev = prev.to(device)
        scales = d*prev + (1.0 - d)*scales
    sigma_params['ema_prev_scales'] = scales.detach()

    # Initial isotropic blending (iso_ramp)
    if iso_ramp is not None and 0.0 < iso_ramp < 1.0:
        sigma_iso = torch.full_like(scales, sigma_t0)
        scales = (1.0 - iso_ramp)*sigma_iso + iso_ramp*scales

    # Σ = R diag(scales^2) R^T
    S = torch.diag_embed(scales**2)
    cov = R @ S @ R.transpose(1, 2)
    cov = 0.5*(cov + cov.transpose(1, 2)) + 1e-6*torch.eye(3, device=cov.device)

    return cov.detach().cpu().numpy()
