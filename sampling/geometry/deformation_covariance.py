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


def build_edge_aware_weights(
    Xn: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
    curvature: Optional[torch.Tensor] = None,
    j_nodes: torch.Tensor = None,
    distances: torch.Tensor = None,
    gating_cfg: Optional[Dict] = None
) -> torch.Tensor:
    """
    🔥 엣지-어웨어 가중치 계산 (곡률·노멀각·거리 기반).
    
    핵심: 엣지/곡률 경계에서 smoothing 약화 → 디테일 보존
    
    Args:
        Xn: (K, 3) 노드 위치
        normals: (K, 3) 노멀 벡터 (선택)
        curvature: (K,) 곡률 값 (선택)
        j_nodes: (K, k) 이웃 인덱스
        distances: (K, k) 이웃 거리
        gating_cfg: 게이팅 설정
    
    Returns:
        weights: (K, k) 엣지-어웨어 가중치
    """
    if gating_cfg is None or not gating_cfg.get('use_curvature', False):
        # 기본: 거리 기반만
        sigma_x = distances.mean() + EPS_SAFE
        w_dist = torch.exp(-(distances ** 2) / (sigma_x ** 2))
        return w_dist
    
    K, k = j_nodes.shape
    device, dtype = Xn.device, Xn.dtype
    
    # 1) 거리 가중
    sigma_x = distances.mean() + EPS_SAFE
    w_dist = torch.exp(-(distances ** 2) / (sigma_x ** 2))
    
    # 2) 노멀각 가중 (선택)
    w_angle = torch.ones_like(w_dist)
    if normals is not None:
        # normals: (K, 3), j_nodes: (K, k)
        n_i = normals.unsqueeze(1).expand(-1, k, -1)  # (K, k, 3)
        n_j = normals[j_nodes]  # (K, k, 3)
        
        cos_angle = (n_i * n_j).sum(dim=-1).clamp(-1.0, 1.0)  # (K, k)
        theta_rad = torch.acos(cos_angle)  # [0, π]
        theta_deg = theta_rad * 180.0 / 3.14159265
        
        # 각도 임계치 (기본: 22.5도)
        normal_angle_thresh = float(gating_cfg.get('normal_angle_deg', 22.5))
        sigma_theta = torch.deg2rad(torch.tensor(normal_angle_thresh, device=device))
        w_angle = torch.exp(-(theta_rad ** 2) / (sigma_theta ** 2))
    
    # 3) 곡률 차이 가중 (선택)
    w_curv = torch.ones_like(w_dist)
    if curvature is not None:
        kappa_i = curvature.unsqueeze(1).expand(-1, k)  # (K, k)
        kappa_j = curvature[j_nodes]  # (K, k)
        d_kappa = (kappa_i - kappa_j).abs()
        
        # 곡률 임계치 (무단위, 기본: 0.10)
        kappa_thresh = float(gating_cfg.get('kappa_thresh', 0.10))
        w_curv = torch.exp(-(d_kappa ** 2) / (kappa_thresh ** 2))
    
    # 4) 결합 가중치
    min_weight = float(gating_cfg.get('min_weight', 0.10))
    softness = float(gating_cfg.get('weight_softness', 0.5))
    
    w_combined = w_dist * w_angle * w_curv
    w_combined = w_combined.pow(softness)  # 부드럽게
    w_combined = min_weight + (1.0 - min_weight) * w_combined  # 최소 가중 보존
    
    return w_combined


def build_graph_laplacian(
    Xn, node_knn, K, device, dtype,
    normals: Optional[torch.Tensor] = None,
    curvature: Optional[torch.Tensor] = None,
    gating_cfg: Optional[Dict] = None
):
    """
    🔥 엣지-어웨어 그래프 라플라시안 (개선됨).
    
    Args:
        Xn: (K, 3) 노드 위치
        node_knn: 이웃 수
        K: 노드 총 개수
        device, dtype: 텐서 속성
        normals: (K, 3) 노멀 벡터 (선택)
        curvature: (K,) 곡률 값 (선택)
        gating_cfg: 엣지-어웨어 게이팅 설정
    
    Returns:
        L: (K, K) 라플라시안 행렬
    """
    D_nodes = torch.cdist(Xn, Xn, p=2)
    k_node = min(node_knn, K - 1)
    
    j_nodes, attn_k = soft_top_k(D_nodes, k=k_node + 1, tau=0.05, largest=False)
    
    # 🔥 엣지-어웨어 가중치 계산
    d_selected = torch.gather(D_nodes, 1, j_nodes)  # (K, k+1)
    weights = build_edge_aware_weights(
        Xn, normals, curvature, j_nodes, d_selected, gating_cfg
    )
    
    # Build sparse W
    W = torch.zeros(K, K, device=device, dtype=dtype)
    row_idx = torch.arange(K, device=device).unsqueeze(1).expand(-1, k_node + 1)
    W[row_idx, j_nodes] = weights
    
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


def smooth_F_field(
    x_low: torch.Tensor, 
    F_low: torch.Tensor, 
    cfg: Dict,
    normals: Optional[torch.Tensor] = None,
    curvature: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    🔥 엣지-어웨어 F-field smoothing (개선됨).
    
    개선사항:
    - 곡률/노멀각 기반 게이팅 → 엣지 보존
    - 에피소드별 lambda_lap 스케줄 지원
    
    Args:
        x_low: (N, 3) anchor positions
        F_low: (N, 3, 3) deformation gradients
        cfg: Configuration dict
        normals: (N, 3) 노멀 벡터 (선택, 엣지 검출용)
        curvature: (N,) 곡률 값 (선택, 엣지 검출용)
    
    Returns:
        F_smooth: (N, 3, 3) smoothed F-field
    """
    K = min(int(cfg.get("num_nodes", 180)), x_low.shape[0])
    node_knn = int(cfg.get("node_knn", 8))
    point_knn = int(cfg.get("point_knn", 8))
    lam = float(cfg.get("lambda_lap", 1e-2))
    seed = int(cfg.get("seed", 42))
    gating_cfg = cfg.get("gating", None)  # 🔥 NEW: 엣지 게이팅 설정
    
    device, dtype = F_low.device, F_low.dtype
    N = x_low.shape[0]
    
    # Select graph nodes (differentiable)
    Xn, sel = select_graph_nodes(x_low, K, seed=seed)
    
    # 🔥 선택된 노드의 노멀/곡률 (있으면)
    Xn_normals = normals[sel] if normals is not None else None
    Xn_curvature = curvature[sel] if curvature is not None else None
    
    # Build Laplacian (엣지-어웨어, differentiable)
    L = build_graph_laplacian(
        Xn, node_knn, K, device, dtype,
        normals=Xn_normals,
        curvature=Xn_curvature,
        gating_cfg=gating_cfg
    )
    
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
# Soft Clamping Utilities (Gradient-Friendly)
# ============================================================================

def soft_clamp_min(x: torch.Tensor, lo: float, beta: float = 4.0) -> torch.Tensor:
    """Soft minimum clamp: x' = lo + softplus(x - lo)"""
    return lo + F.softplus(x - lo, beta=beta)

def soft_clamp_minmax(x: torch.Tensor, lo: float, hi: Optional[float] = None, beta: float = 4.0) -> torch.Tensor:
    """Soft min-max clamp (gradient-friendly)."""
    y = soft_clamp_min(x, lo, beta)
    if hi is not None:
        y = hi - F.softplus(hi - y, beta=beta)
    return y


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
    
    # 🔥 NaN/Inf 안전장치: sigma 클램핑 (수치 안정성)
    sigma = torch.clamp(sigma, min=0.01, max=3.0)  # 물리적으로 합리적인 범위
    
    # Rotation (or rotation+reflection): R = U V^T
    R = torch.bmm(U, Vt)
    # det(R) can be ±1, but that's OK for covariance computation
    
    # Stretch: S = V Σ V^T (always positive definite since sigma > 0)
    V = Vt.transpose(-2, -1)
    sigma_diag = torch.diag_embed(sigma)  # Diagonal with positive values
    S = torch.bmm(torch.bmm(V, sigma_diag), Vt)
    
    # 🔥 대칭화 + 안전장치 (수치 오차 제거)
    S = 0.5 * (S + S.transpose(-2, -1))  # 완벽한 대칭 보장
    
    # 🔥 추가 안전장치: 대각 최소값 보장
    diag_S = torch.diagonal(S, dim1=-2, dim2=-1)  # (N, 3)
    min_diag = 0.01  # 최소 대각 원소
    need_fix = (diag_S < min_diag).any(dim=-1)  # (N,)
    if need_fix.any():
        eye = torch.eye(3, device=S.device, dtype=S.dtype).unsqueeze(0)
        fix_amount = torch.clamp(min_diag - diag_S.min(dim=-1).values, min=0.0)  # (N,)
        S = S + fix_amount.view(-1, 1, 1) * eye
    
    return R, S


def polar_with_sv_soft_clamp(
    F_in: torch.Tensor,
    s_min: float = 0.30,
    s_max: Optional[float] = None,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    🔥 Polar decomposition + SV soft-clamp (특이값만, 원소별 클램프 금지).
    
    Args:
        F_in: (..., 3, 3) deformation gradient
        s_min: 최소 특이값 (soft)
        s_max: 최대 특이값 (soft, None=무제한)
        eps: 수치 안정성
    
    Returns:
        R: (..., 3, 3) proper rotation (det(R) > 0, SO(3))
        H: (..., 3, 3) SPD stretch (soft-clamped)
        S_cl: (..., 3) clamped singular values
    """
    # SVD: F = U Σ V^T
    U, S, Vh = torch.linalg.svd(F_in, full_matrices=False)  # S: (..., 3)
    
    # Proper rotation 보장 (det(R) > 0)
    R = U @ Vh
    detR = torch.det(R)
    neg = detR < 0
    if neg.any():
        # 마지막 축 뒤집어 SO(3)로 보정
        Vh = Vh.clone()
        Vh[neg, -1, :] *= -1
        R = U @ Vh
    
    # 🔥 SV soft-clamp (경직 대신 완만)
    S_cl = soft_clamp_minmax(S, s_min, s_max)
    
    # SPD stretch H = V Σ V^T
    H = Vh.transpose(-2, -1) @ torch.diag_embed(S_cl) @ Vh
    
    # 수치적 SPD 보장 (대칭화 + 작은 regularization)
    H = 0.5 * (H + H.transpose(-2, -1))
    eye = torch.eye(3, device=H.device, dtype=H.dtype).unsqueeze(0)
    H = H + eps * eye
    
    return R, H, S_cl


# ============================================================================
# Log-Euclidean Smoothing (SPD-only)
# ============================================================================

def spd_log(H: torch.Tensor) -> torch.Tensor:
    """Log of SPD matrix: log(H) in Lie algebra."""
    evals, evecs = torch.linalg.eigh(H)  # SPD → real eigenvalues
    evals = evals.clamp_min(1e-9)
    L = evecs @ torch.diag_embed(torch.log(evals)) @ evecs.transpose(-2, -1)
    return L

def spd_exp(L: torch.Tensor) -> torch.Tensor:
    """Exp of symmetric matrix: exp(L) back to SPD."""
    evals, evecs = torch.linalg.eigh(L)
    H = evecs @ torch.diag_embed(torch.exp(evals)) @ evecs.transpose(-2, -1)
    return 0.5 * (H + H.transpose(-2, -1))  # 수치 대칭화

def laplacian_smooth_tensor(
    T: torch.Tensor,
    idx: torch.Tensor,
    w: torch.Tensor,
    lam: float = 0.04,
    iters: int = 1
) -> torch.Tensor:
    """Laplacian smoothing on tensor field."""
    X = T
    for _ in range(iters):
        neigh = X[idx]  # (M, k, ...)
        avg = (w.unsqueeze(-1).unsqueeze(-1) * neigh).sum(dim=1)
        X = (1.0 - lam) * X + lam * avg
    return X

def smooth_H_only(
    H: torch.Tensor,
    points: torch.Tensor,
    knn,
    lam: float = 0.04,
    iters: int = 1,
    k: int = 6
) -> torch.Tensor:
    """
    🔥 SPD-only smoothing in log-Euclidean space.
    
    Args:
        H: (M, 3, 3) SPD stretch matrices
        points: (M, 3) positions
        knn: KNN function
        lam: Laplacian strength
        iters: Smoothing iterations
        k: Number of neighbors
    
    Returns:
        Hs: (M, 3, 3) smoothed SPD
    """
    L = spd_log(H)  # (M, 3, 3) - log space
    
    k_actual = min(k, points.shape[0] - 1)
    idx, w = knn(points, points, k_actual)  # (M, k), (M, k)
    
    Ls = laplacian_smooth_tensor(L, idx, w, lam, iters)  # 로그 영역 스무딩
    Hs = spd_exp(Ls)  # 복귀 → SPD
    
    return Hs


# ============================================================================
# Multi-Scale F-field Interpolation
# ============================================================================

def interpolate_F_multiscale(
    points: torch.Tensor,
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    knn,
    k_coarse: int = 64,
    k_fine: int = 16,
    blend_mode: str = 'adaptive'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-scale F-field interpolation for large deformations.

    Strategy:
    - Coarse scale (k=64): Captures global shape transformation (sphere → bunny outline)
    - Fine scale (k=16): Captures local details (bunny ears, nose, tail)
    - Adaptive blending based on deformation magnitude

    This improves quality for large deformations without adding particles.

    Args:
        points: (M, 3) query positions
        x_low: (N, 3) anchor positions
        F_low: (N, 3, 3) deformation gradients at anchors
        knn: KNN function
        k_coarse: Neighbors for global deformation (default 64)
        k_fine: Neighbors for local details (default 16)
        blend_mode: 'adaptive' (based on det(F)) or 'fixed' (50/50)

    Returns:
        F_interp: (M, 3, 3) blended F-field
        blend_weights: (M,) blend weights (0=fine, 1=coarse) for visualization
    """
    M = points.shape[0]
    device = points.device

    # 1) Coarse-scale interpolation (global)
    k_c = min(k_coarse, x_low.shape[0])
    idx_c, w_c = knn(points, x_low, k_c)
    F_neighbors_c = F_low[idx_c]  # (M, k_c, 3, 3)
    F_coarse = torch.einsum('mk,mkrc->mrc', w_c, F_neighbors_c)  # (M, 3, 3)

    # 2) Fine-scale interpolation (local)
    k_f = min(k_fine, x_low.shape[0])
    idx_f, w_f = knn(points, x_low, k_f)
    F_neighbors_f = F_low[idx_f]  # (M, k_f, 3, 3)
    F_fine = torch.einsum('mk,mkrc->mrc', w_f, F_neighbors_f)  # (M, 3, 3)

    # 3) Adaptive blending
    if blend_mode == 'adaptive':
        # Measure deformation magnitude via det(F)
        # det(F) ≈ 1 → small deformation → use fine
        # det(F) far from 1 → large deformation → use coarse

        det_coarse = torch.det(F_coarse).abs()  # (M,)
        det_fine = torch.det(F_fine).abs()  # (M,)

        # Deviation from identity (1.0)
        dev_coarse = (det_coarse - 1.0).abs()
        dev_fine = (det_fine - 1.0).abs()

        # Blend weight: higher deviation → use coarse (more global context)
        # Sigmoid to smooth transition
        tau = 0.2  # Temperature for smooth blending
        logits = (dev_coarse - dev_fine) / (tau + EPS_SAFE)
        alpha = torch.sigmoid(logits)  # (M,) in [0, 1]

        # Clamp to reasonable range (don't fully trust one scale)
        alpha = torch.clamp(alpha, 0.2, 0.8)

    elif blend_mode == 'fixed':
        # Fixed 50/50 blend
        alpha = torch.full((M,), 0.5, device=device, dtype=points.dtype)

    else:
        raise ValueError(f"Unknown blend_mode: {blend_mode}")

    # 4) Blend F-fields
    alpha_expanded = alpha.view(-1, 1, 1)  # (M, 1, 1)
    F_interp = alpha_expanded * F_coarse + (1.0 - alpha_expanded) * F_fine

    return F_interp, alpha


# ============================================================================
# Deformation-based Covariance Construction
# ============================================================================

def build_deformation_covariance(
    points: torch.Tensor,
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    knn,
    cfg: Dict,
    learnable_cov_module=None,
    x_low_normals: Optional[torch.Tensor] = None,  # 🔥 NEW: 엣지 검출용
    x_low_curvature: Optional[torch.Tensor] = None  # 🔥 NEW: 엣지 검출용
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build covariance matrices via F-field interpolation + optional learnable refinement.
    
    🔥 개선: 엣지-어웨어 F-smoothing 지원
    
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
        x_low_normals: (N, 3) 노멀 벡터 (엣지-어웨어 smoothing용, 선택)
        x_low_curvature: (N,) 곡률 값 (엣지-어웨어 smoothing용, 선택)
    
    Returns:
        cov: (M, 3, 3) covariance matrices
        F_interp: (M, 3, 3) interpolated F-field
        idx: (M, k_F) KNN indices
    """
    sigma0 = float(cfg.get("sigma0", 0.08))
    k_F = int(cfg.get("k_F", 32))
    use_F_smoothing = bool(cfg.get("use_F_smoothing", False))  # 🔥 FIX: Changed default from True to False
    use_adaptive_scale = bool(cfg.get("use_adaptive_scale", False))
    use_polar = bool(cfg.get("use_polar_decomposition", True))

    # 🔥 DEBUG: Print what we received
    print(f"[DEBUG] use_F_smoothing = {use_F_smoothing} (from cfg: {cfg.get('use_F_smoothing', 'KEY_MISSING')})")

    # 🔥 F-field 분포 체크 (Root cause 디버깅)
    with torch.no_grad():
        F_det = torch.det(F_low)
        print(f"\n[F-field Stats - Input]")
        print(f"  F_low range: [{F_low.min():.3f}, {F_low.max():.3f}]")
        print(f"  det(F_low): [{F_det.min():.3f}, {F_det.max():.3f}]")
        print(f"  Negative det: {(F_det < 0).sum().item()} / {len(F_det)}")
        if (F_det < 0).any():
            print(f"  ⚠️  WARNING: Negative determinants (reflections/inversions)!")
    
    # 1) Optional F-field smoothing (🔥 엣지-어웨어)
    if use_F_smoothing:
        F_smooth = smooth_F_field(
            x_low, F_low, cfg.get("F_smooth", {}),
            normals=x_low_normals,  # 🔥 NEW
            curvature=x_low_curvature  # 🔥 NEW
        )
        
        # 🔥 Smoothing 전후 비교
        with torch.no_grad():
            print(f"[F-field Stats - After Smoothing]")
            print(f"  F_smooth range: [{F_smooth.min():.3f}, {F_smooth.max():.3f}]")
            F_det_smooth = torch.det(F_smooth)
            print(f"  det(F_smooth): [{F_det_smooth.min():.3f}, {F_det_smooth.max():.3f}]")
    else:
        F_smooth = F_low

    # 2) Interpolate F to upsampled points
    use_multiscale = bool(cfg.get("use_multiscale_F", False))

    if use_multiscale:
        # Multi-scale interpolation (coarse + fine)
        k_coarse = int(cfg.get("k_F_coarse", 64))
        k_fine = int(cfg.get("k_F_fine", 16))
        blend_mode = cfg.get("multiscale_blend_mode", "adaptive")

        F_interp, blend_weights = interpolate_F_multiscale(
            points, x_low, F_smooth, knn,
            k_coarse=k_coarse,
            k_fine=k_fine,
            blend_mode=blend_mode
        )

        with torch.no_grad():
            print(f"\n[Multi-scale F-interpolation]")
            print(f"  Coarse scale: k={k_coarse} (global)")
            print(f"  Fine scale: k={k_fine} (local)")
            print(f"  Blend mode: {blend_mode}")
            print(f"  Blend weights (α): mean={blend_weights.mean():.3f}, range=[{blend_weights.min():.3f}, {blend_weights.max():.3f}]")

        # Create dummy idx for compatibility
        idx, _ = knn(points, x_low, k_F)
    else:
        # Single-scale interpolation (original)
        idx, w = knn(points, x_low, k_F)
        F_neighbors = F_smooth[idx]  # (M, k_F, 3, 3)
        F_interp = torch.einsum('mk,mkrc->mrc', w, F_neighbors)  # (M, 3, 3)
    
    # 🔥 SV Soft-Clamp (원소별 경직 클램프 금지!)
    s_min = float(cfg.get("sv_min", 0.30))
    s_max = cfg.get("sv_max", None)  # None = 무제한
    
    # DEBUG: F_interp 전
    with torch.no_grad():
        F_det_before = torch.det(F_interp)
        print(f"[F-field Debug - Before SV Clamp]")
        print(f"  F_interp range: [{F_interp.min():.6f}, {F_interp.max():.6f}]")
        print(f"  det(F) range: [{F_det_before.min():.6f}, {F_det_before.max():.6f}]")
        neg_det_count = (F_det_before < 0).sum().item()
        if neg_det_count > 0:
            print(f"  ⚠️  {neg_det_count} negative determinants (will be corrected)")

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

        # 🔥 Polar decomposition + SV soft-clamp (경직 클램프 제거!)
        R, H, S_cl = polar_with_sv_soft_clamp(F_interp, s_min=s_min, s_max=s_max)
        
        with torch.no_grad():
            print(f"[Polar + SV Soft-Clamp Debug]")
            print(f"  SV clamped: [{S_cl.min():.6f}, {S_cl.max():.6f}]")
            print(f"  H (stretch) range: [{H.min():.6f}, {H.max():.6f}]")
            
            # 포화율 체크
            sat_min = (S_cl <= s_min + 1e-4).float().mean().item()
            sat_max = (S_cl >= (s_max - 1e-4) if s_max else 0.0).float().mean().item() if s_max else 0.0
            print(f"  SV saturation: min={sat_min*100:.1f}%, max={sat_max*100:.1f}%")
            if sat_min > 0.20:
                print(f"  ⚠️  WARNING: >20% SVs at minimum (over-clamped, loss of anisotropy)")
        
        # 🔥 (Optional) SPD-only smoothing (mode="spd_only")
        smooth_mode = cfg.get("F_smooth", {}).get("mode", "full")
        if smooth_mode == "spd_only":
            # R은 보존, H만 log-Euclidean smoothing
            H_smooth_cfg = cfg.get("F_smooth", {})
            lam_H = float(H_smooth_cfg.get("lambda_H", 0.02))
            H = smooth_H_only(H, points, knn, lam=lam_H, iters=1, k=6)
            
            with torch.no_grad():
                print(f"  [SPD-only Smoothing] H smoothed with λ={lam_H:.3f}")
        
        S_fixed = H  # H가 이미 SPD + smoothed
        
        # Build covariance: Σ = S·Σ₀·S (rotation removed!)
        Sigma0 = (sigma_adaptive.view(-1, 1, 1) ** 2) * torch.eye(3, device=points.device).unsqueeze(0)
        S_Sigma0 = torch.bmm(S_fixed, Sigma0)
        cov = torch.bmm(S_Sigma0, S_fixed)
        
        # 🔥 Root Cause 디버깅 + SV 동적범위 분석
        with torch.no_grad():
            print(f"[Deformation Covariance Debug]")
            print(f"  sigma_adaptive: [{sigma_adaptive.min():.6f}, {sigma_adaptive.max():.6f}]")
            print(f"  S_fixed range: [{S_fixed.min():.6f}, {S_fixed.max():.6f}]")
            
            # SV 동적범위
            sv_p05 = torch.quantile(S_cl, 0.05, dim=0)
            sv_p50 = torch.quantile(S_cl, 0.50, dim=0)
            sv_p95 = torch.quantile(S_cl, 0.95, dim=0)
            print(f"  SV quantiles [p05, p50, p95]:")
            for i in range(3):
                print(f"    σ_{i}: [{sv_p05[i]:.4f}, {sv_p50[i]:.4f}, {sv_p95[i]:.4f}]")
            
            print(f"  cov range: [{cov.min():.6f}, {cov.max():.6f}]")
            cov_det = torch.det(cov)
            print(f"  det(cov) range: [{cov_det.min():.6e}, {cov_det.max():.6e}]")
        
        # Stabilization
        eps_physics = 1e-5  # 🔥 증가: 1e-6 → 1e-5
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
    "interpolate_F_multiscale",
    "smooth_F_field",
    "polar_decomposition",
    "select_graph_nodes",
    "build_graph_laplacian",
    "build_interpolation_weights",
    "soft_top_k",
    "gumbel_select_k",
]

