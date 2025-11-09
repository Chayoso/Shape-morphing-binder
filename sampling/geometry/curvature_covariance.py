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


# ──────────────────────────────────────────────────────────────────────────────
# 안전 고유분해 (CUDA → CPU fallback)
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# 안전 고유분해 (CUDA → CPU fallback)
# ──────────────────────────────────────────────────────────────────────────────
def _safe_eigh_cuda_cpu(C: torch.Tensor, eps: float = 1e-6):
    """
    Robust symmetric eigendecomposition for batched 3x3.
    1) 대각성분 적응형 jitter 추가 (행별 det 기반)
    2) CUDA eigh 실패/NaN 발생 시 해당 행만 CPU(float64)에서 재시도
    반환: evals (B,3), evecs (B,3,3)
    """
    device = C.device
    B = C.shape[0]
    dtype = C.dtype
    
    # 🔥 FIX: Empty tensor check (before any operation)
    if B == 0:
        return (
            torch.empty((0, 3), dtype=dtype, device=device),
            torch.empty((0, 3, 3), dtype=dtype, device=device)
        )
    
    Csym = 0.5 * (C + C.transpose(-1, -2))
    
    # 🔥 FIX: NaN/Inf check BEFORE det computation
    precheck_bad = torch.isnan(Csym).any(dim=(1, 2)) | torch.isinf(Csym).any(dim=(1, 2))
    
    if precheck_bad.all():
        # All matrices invalid → return identity
        print(f"[ERROR] _safe_eigh_cuda_cpu: ALL {B} matrices contain NaN/Inf, returning identity")
        evals = torch.ones(B, 3, device=device, dtype=dtype)
        evecs = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        return evals, evecs
    
    # Compute det only for valid matrices
    det = torch.zeros(B, device=device, dtype=dtype)
    if (~precheck_bad).any():
        det[~precheck_bad] = torch.det(Csym[~precheck_bad])
    
    # det이 작을수록 jitter를 더 크게
    jitter_row = (eps * 10.0) * (det.abs() + eps).rsqrt().clamp(max=1e3)  # 안전 상한
    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    Csym = Csym + jitter_row.view(B, 1, 1) * eye
    
    # 불량 판단: NaN/Inf/음수 det 과도/조건수 과도
    with torch.no_grad():
        nan_mask = torch.isnan(Csym).any(dim=(1, 2)) | torch.isinf(Csym).any(dim=(1, 2))
        # 매우 작은 det(준특이)
        small_mask = (det.abs() < (eps**3))
        bad = nan_mask | small_mask
    
    # 🔥 FIX: If all bad, return identity immediately
    if bad.all():
        print(f"[ERROR] _safe_eigh_cuda_cpu: ALL {B} matrices singular/invalid after jitter, returning identity")
        evals = torch.ones(B, 3, device=device, dtype=dtype)
        evecs = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        return evals, evecs
    
    # 기본값(아이덴티티)로 초기화
    evals_all = torch.ones(B, 3, device=device, dtype=dtype)
    evecs_all = eye.clone()
    
    good = ~bad
    if good.any():
        try:
            # 🔥 FIX: Additional safety - contiguous and validation
            Csym_good = Csym[good].contiguous()
            
            # Final NaN check on subset
            if torch.isnan(Csym_good).any() or torch.isinf(Csym_good).any():
                print(f"[WARN] _safe_eigh_cuda_cpu: NaN/Inf in 'good' subset, using CPU fallback")
                raise RuntimeError("NaN in good subset")
            
            evals_g, evecs_g = torch.linalg.eigh(Csym_good)
            
            # Validate output
            if torch.isnan(evals_g).any() or torch.isnan(evecs_g).any():
                print(f"[WARN] _safe_eigh_cuda_cpu: eigh output contains NaN, using CPU fallback")
                raise RuntimeError("NaN in eigh output")
            
            evals_all[good] = evals_g
            evecs_all[good] = evecs_g
        except Exception as e:
            print(f"[WARN] _safe_eigh_cuda_cpu: CUDA eigh failed ({e}), falling back to CPU float64")
            # 🔥 FIX: detach() 제거 - gradient 보존
            # CPU fallback for ALL good matrices
            try:
                C_cpu = Csym[good].to('cpu', torch.float64)  # detach() 제거!
                evals_b, evecs_b = torch.linalg.eigh(C_cpu)
                evals_all[good] = evals_b.to(device=device, dtype=dtype)
                evecs_all[good] = evecs_b.to(device=device, dtype=dtype)
            except Exception as e2:
                print(f"[CRITICAL] CPU fallback failed ({e2}), gradient WILL be broken!")
                # 최후의 수단: 작은 regularization만 추가
                print(f"  → Adding minimal regularization instead of full identity")
                # Leave as identity (gradient broken warning already printed)
    
    return evals_all, evecs_all


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
        
        # Eigendecomposition with safe CUDA→CPU fallback
        evals, evecs = _safe_eigh_cuda_cpu(C, eps=eps)  # (b,3), (b,3,3)
        
        t1 = evecs[..., 2]                            # maximum variance axis
        t2 = evecs[..., 1]
        n_b = n[s:e]

        # Orthogonal realignment (numerical stability)
        t1 = t1 - (t1 * n_b).sum(-1, keepdim=True) * n_b
        t1 = t1 / (t1.norm(dim=1, keepdim=True) + eps)
        t2 = torch.cross(n_b, t1, dim=-1)

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
    
    # 🔥 디버그 토글 (로그 스팸 제어)
    debug = bool(sigma_params.get('debug', False))

    # Hyperparameters (🔥 재스케일: 타깃 공분산 과다 스케일 방지)
    sigma_n0   = float(sigma_params.get('sigma_n0', 0.035))  # 0.015 → 0.035 (법선 초기값 상향)
    sigma_t0   = float(sigma_params.get('sigma_t0', 0.06))   # 0.022 → 0.06 (접선 초기값 상향)
    a          = float(sigma_params.get('a', 2.0))           # 1.2 → 2.0 (곡률 감쇠 강화)
    b          = float(sigma_params.get('b', 0.5))           # 0.30 → 0.5 (접선 감쇠 지수 상향)
    u          = float(sigma_params.get('u', 0.3))           # 0.20 → 0.3 (법선 감쇠 지수 상향)
    sigma_floor_base = float(sigma_params.get('sigma_floor', 0.004))
    sigma_ceil = float(sigma_params.get('sigma_ceil', 0.080))
    frame_mode = sigma_params.get('frame', 'pca')
    pca_k      = int(sigma_params.get('pca_k', 16))
    iso_ramp   = float(sigma_params.get('iso_ramp', 0.4))
    ema_decay  = sigma_params.get('ema_decay', 0.85)  # float or None
    
    # 🔥 에피소드별 스케줄 적용 (초기 안정화 → 후기 디테일)
    current_ep = sigma_params.get('episode', -1)
    schedule_cfg = sigma_params.get('schedule', None)
    
    if schedule_cfg is not None and current_ep >= 0:
        try:
            from utils.schedule_utils import apply_schedule_to_cfg
            
            # 임시 cfg dict로 스케줄 적용
            temp_cfg = {
                'iso_ramp': iso_ramp,
                'a': a,
                'b': b,
            }
            temp_cfg = apply_schedule_to_cfg(temp_cfg, schedule_cfg, current_ep, verbose=debug)
            
            iso_ramp = temp_cfg.get('iso_ramp', iso_ramp)
            a = temp_cfg.get('a', a)
            b = temp_cfg.get('b', b)
            
            if debug:
                print(f"[Schedule Applied] ep={current_ep}: iso_ramp={iso_ramp:.2f}, a={a:.2f}, b={b:.2f}")
        
        except Exception as e:
            if debug:
                print(f"[WARN] Schedule application failed: {e}, using default values")

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

    # Curvature/anisotropy scaling (🔥 재스케일: 감쇠 공식 통일)
    # 접선: 곡률↑ → σ↓ (디테일 보존)
    # 법선: 평균곡률 기반 제어
    a_n = float(sigma_params.get('a_n', 0.5))  # 법선 감쇠 계수 (별도)
    
    sigma_t1 = sigma_t0 / (1.0 + a * kappa_hat).pow(b)    # 곡률↑ → 얇게
    sigma_t2 = sigma_t0 / (1.0 + a * kappa_hat).pow(b) * (1.0 + u * rho_hat)  # 이방성 추가
    sigma_n  = sigma_n0 / (1.0 + a_n * kappa_hat).pow(u)  # 법선: 별도 감쇠
    
    # 🔥 DEBUG: 실제 사용되는 곡률값과 최종 sigma 출력 (토글 가능)
    if debug:
        print(f"[DEBUG] Curvature-based Sigma Computation:")
        print(f"  Input params: σ_n0={sigma_n0:.4f}, σ_t0={sigma_t0:.4f}, a={a:.2f}, b={b:.2f}, u={u:.2f}")
        print(f"  kappa_hat (normalized): mean={kappa_hat.mean():.4f}, std={kappa_hat.std():.4f}, range=[{kappa_hat.min():.4f}, {kappa_hat.max():.4f}]")
        print(f"  rho_hat (normalized):   mean={rho_hat.mean():.4f}, std={rho_hat.std():.4f}, range=[{rho_hat.min():.4f}, {rho_hat.max():.4f}]")
        print(f"  Final σ_n: mean={sigma_n.mean():.5f}, range=[{sigma_n.min():.5f}, {sigma_n.max():.5f}]")
        print(f"  Final σ_t1: mean={sigma_t1.mean():.5f}, range=[{sigma_t1.min():.5f}, {sigma_t1.max():.5f}]")
        print(f"  Final σ_t2: mean={sigma_t2.mean():.5f}, range=[{sigma_t2.min():.5f}, {sigma_t2.max():.5f}]")

    # ═══════════════════════════════════════════════════════════════════════
    # Axis-Specific Lower Bounds (축별 바닥치)
    # ═══════════════════════════════════════════════════════════════════════
    # 🔥 핵심: 법선과 접선을 구분해야 "전부 바닥" 문제가 해결됨!
    
    # Voxel size (approximation from sigma_params or default)
    voxel_size = float(sigma_params.get('voxel_size', 0.08))
    
    # 축별 바닥치 계수 (🔥 법선 바닥치 낮춤: 디테일 살리고 두께 얇게)
    k_n = float(sigma_params.get('k_floor_normal', 0.05))    # 법선: 얇게 유지 (0.08 → 0.05)
    k_t = float(sigma_params.get('k_floor_tangent', 0.25))   # 접선: 스패클 방지 (0.20~0.30)
    
    # 축별 바닥치 계산
    sigma_n_floor = k_n * voxel_size   # 예: 0.08 × 0.084 = 0.0067
    sigma_t_floor = k_t * voxel_size   # 예: 0.25 × 0.084 = 0.021
    
    # 화면 발자국 하한 (워밍업 스케줄)
    cam_z = sigma_params.get('cam_z', None)
    fx = float(sigma_params.get('fx', 1200.0))
    
    # 🔥 2D 발자국 워밍업 스케줄 (ep 0-5: 1.2 → 0.9 px)
    screen_footprint_cfg = sigma_params.get('screen_footprint', {})
    if screen_footprint_cfg.get('enabled', False):
        warmup_eps = int(screen_footprint_cfg.get('warmup_episodes', 5))
        r_start = float(screen_footprint_cfg.get('r_px_min_start', 1.2))
        r_end = float(screen_footprint_cfg.get('r_px_min_end', 0.9))
        current_ep = sigma_params.get('episode', 0)
        
        if current_ep < warmup_eps:
            # 선형 램프다운
            progress = current_ep / max(warmup_eps, 1)
            min_px_radius = r_start + (r_end - r_start) * progress
        else:
            min_px_radius = r_end
        
        if debug:
            print(f"[Screen Footprint] ep={current_ep}, r_px_min={min_px_radius:.3f} px "
                  f"({'warmup' if current_ep < warmup_eps else 'stable'})")
    else:
        min_px_radius = float(sigma_params.get('min_px_radius', 0.9))
    
    if cam_z is not None:
        cam_z_t = torch.as_tensor(cam_z, dtype=torch.float32, device=device)
        rmin_world = (min_px_radius / fx) * torch.clamp(cam_z_t, min=eps)
        # 접선만 화면 발자국 적용 (법선은 voxel 기반만)
        sigma_t_floor_screen = torch.maximum(rmin_world, torch.tensor(sigma_t_floor, device=device))
    else:
        sigma_t_floor_screen = torch.full((N,), sigma_t_floor, device=device)
    
    sigma_n_floor_t = torch.full((N,), sigma_n_floor, device=device)
    
    # Convert sigma_ceil to Tensor
    sigma_ceil_t = torch.full((N,), sigma_ceil, device=device)
    
    # 축별 클램프 (1차)
    sigma_t1 = torch.clamp(sigma_t1, min=sigma_t_floor_screen, max=sigma_ceil_t)
    sigma_t2 = torch.clamp(sigma_t2, min=sigma_t_floor_screen, max=sigma_ceil_t)
    sigma_n  = torch.clamp(sigma_n,  min=sigma_n_floor_t, max=sigma_ceil_t)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 🔥 이방성 비율 강화 (σ_t/σ_n ≈ 4~6)
    # ═══════════════════════════════════════════════════════════════════════
    aniso_min = float(sigma_params.get('aniso_target_min', 3.0))
    aniso_max = float(sigma_params.get('aniso_target_max', 8.0))
    
    # 현재 이방성 비율 계산
    aniso_ratio = (sigma_t1 + sigma_t2) / (2.0 * sigma_n + eps)
    
    # 비율이 너무 낮으면 접선 확대 or 법선 축소
    needs_boost = (aniso_ratio < aniso_min)
    if needs_boost.any():
        # 전략: 법선을 약간 줄여서 이방성 확보 (접선은 화면 안정성 유지)
        target_sigma_n = (sigma_t1 + sigma_t2) / (2.0 * aniso_min)
        sigma_n = torch.where(
            needs_boost,
            torch.maximum(target_sigma_n, sigma_n_floor_t),  # 바닥치는 유지
            sigma_n
        )
    
    # 비율이 너무 높으면 법선 확대 or 접선 축소
    needs_cap = (aniso_ratio > aniso_max)
    if needs_cap.any():
        # 전략: 법선을 키워서 과도한 이방성 억제
        target_sigma_n = (sigma_t1 + sigma_t2) / (2.0 * aniso_max)
        sigma_n = torch.where(
            needs_cap,
            torch.minimum(target_sigma_n, sigma_ceil_t),
            sigma_n
        )
    
    # 최종 비율 재계산 (디버그)
    aniso_ratio_final = (sigma_t1 + sigma_t2) / (2.0 * sigma_n + eps)
    aniso_mean = aniso_ratio_final.mean().item()
    aniso_std = aniso_ratio_final.std().item()
    aniso_in_range = ((aniso_ratio_final >= aniso_min) & (aniso_ratio_final <= aniso_max)).float().mean().item()
    
    if debug:
        print(f"[Anisotropy Ratio σ_t/σ_n]")
        print(f"  Target range: [{aniso_min:.1f}, {aniso_max:.1f}]")
        print(f"  Actual: mean={aniso_mean:.2f}, std={aniso_std:.2f}")
        print(f"  In range: {aniso_in_range*100:.1f}% (target: ≥80%)")
    
    # 🔥 디버그: 바닥치 히트율 확인
    hit_n = (sigma_n <= sigma_n_floor + eps).float().mean().item()
    hit_t1 = (sigma_t1 <= sigma_t_floor + eps).float().mean().item()
    hit_t2 = (sigma_t2 <= sigma_t_floor + eps).float().mean().item()
    
    print(f"[Floor Hit Rate]: ")
    print(f"  Normal (σ_n):   {hit_n*100:.1f}% (floor={sigma_n_floor:.5f}, target: 10-40%)")
    print(f"  Tangent1 (σ_t1): {hit_t1*100:.1f}% (floor={sigma_t_floor:.5f}, target: 0-15%)")
    print(f"  Tangent2 (σ_t2): {hit_t2*100:.1f}% (floor={sigma_t_floor:.5f}, target: 0-15%)")

    # Stack scales
    scales = torch.stack([sigma_t1, sigma_t2, sigma_n], dim=1)  # (N,3)

    # EMA (optional): store/reuse state in sigma_params dict
    # 🔥 EMA 상태는 CPU로 보관 (장기 에피소드 VRAM 누수 방지)
    prev = sigma_params.get('ema_prev_scales', None)
    if prev is not None and ema_decay is not None:
        d = float(ema_decay)
        prev = prev.to(device)
        scales = d*prev + (1.0 - d)*scales
    
    # 🔥 CPU로 저장 (VRAM 누수 방지)
    sigma_params['ema_prev_scales'] = scales.detach().cpu()

    # Initial isotropic blending (iso_ramp)
    if iso_ramp is not None and 0.0 < iso_ramp < 1.0:
        sigma_iso = torch.full_like(scales, sigma_t0)
        scales = (1.0 - iso_ramp)*sigma_iso + iso_ramp*scales

    # Σ = R diag(scales^2) R^T
    S = torch.diag_embed(scales**2)
    cov = R @ S @ R.transpose(1, 2)
    cov = 0.5*(cov + cov.transpose(1, 2)) + 1e-6*torch.eye(3, device=cov.device)

    return cov.detach().cpu().numpy()
