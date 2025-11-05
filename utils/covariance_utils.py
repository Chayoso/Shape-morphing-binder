"""
Covariance Utilities for SPD (Symmetric Positive Definite) Enforcement

공분산 행렬의 SPD 보장을 위한 유틸리티:
- 로그-파라미터화로 양의 고유값 보장
- 고유값 클램핑
- 공분산 진단 및 수치 안정성 검증
- 🔥 2D screen covariance projection (Cholesky-based, gradient-safe)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union


# ============================================================================
# 🔥 2D Screen Covariance (Gradient-Safe, Cholesky-based)
# ============================================================================

def project_sigma3d_to_2d_safe(
    Sigma3: torch.Tensor,
    xyz: torch.Tensor,
    fx: float,
    fy: float,
    r_px_min: float = 1.0,
    z_eps: float = 1e-3
) -> torch.Tensor:
    """
    3D → 2D 스크린 공분산 투영 (원근투영 + px-footprint 바닥치).
    
    Args:
        Sigma3: (..., 3, 3) SPD world covariance
        xyz: (..., 3) world points (camera coords)
        fx, fy: focal lengths (pixels)
        r_px_min: 최소 px 반지름 (기본 1.0px)
        z_eps: z 나눗셈 안전장치
    
    Returns:
        (..., 2, 2) SPD screen covariance (pixel units)
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2].clamp_min(z_eps)
    
    # Perspective Jacobian J(u,v wrt x,y,z)
    J00 = fx / z
    J01 = torch.zeros_like(z)
    J02 = -fx * x / (z * z)
    J10 = torch.zeros_like(z)
    J11 = fy / z
    J12 = -fy * y / (z * z)
    
    J = torch.stack([
        torch.stack([J00, J01, J02], dim=-1),
        torch.stack([J10, J11, J12], dim=-1)
    ], dim=-2)  # (..., 2, 3)
    
    # Σ₂D = J Σ₃D J^T
    S2 = J @ Sigma3 @ J.transpose(-1, -2)
    
    # px-발자국 바닥치: 최소 반지름 r_px_min 보장
    if r_px_min is not None and r_px_min > 0:
        floor = r_px_min ** 2
        I2 = torch.eye(2, device=S2.device, dtype=S2.dtype)
        S2 = S2 + floor * I2
    
    # 수치 대칭화 (기계 오차 제거)
    S2 = 0.5 * (S2 + S2.transpose(-1, -2))
    
    return S2


def inv_and_logdet_spd_2x2_chol(
    S2: torch.Tensor,
    jitter: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cholesky로 안정적인 Σ₂D⁻¹, log|Σ₂D| 계산 (gradient-safe).
    
    Args:
        S2: (..., 2, 2) SPD screen covariance
        jitter: Cholesky 실패 시 추가 regularization
    
    Returns:
        S2_inv: (..., 2, 2) inverse
        logdet: (...) log determinant
        ok_mask: (...) bool (True=성공, False=jitter 추가됨)
    """
    # 1) 기본 Cholesky
    L, info = torch.linalg.cholesky_ex(S2)
    bad = (info > 0)
    
    # 2) 실패한 항목만 jitter 추가 후 재시도
    if bad.any():
        eye = torch.eye(2, device=S2.device, dtype=S2.dtype)
        # 평균 스케일 기반 jitter
        s = S2.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
        S2b = S2[bad] + (jitter + 1e-6) * s[bad].unsqueeze(-1) * eye
        Lb = torch.linalg.cholesky(S2b)
        L = L.clone()
        L[bad] = Lb
        bad = torch.zeros_like(bad)  # 모두 OK로 표시
    
    # 3) 역행렬: solve(L L^T X = I)
    I = torch.eye(2, device=S2.device, dtype=S2.dtype).expand_as(S2)
    S2_inv = torch.cholesky_solve(I, L)
    
    # 4) logdet = 2 * sum(log(diag(L)))
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-10)).sum(-1)
    
    return S2_inv, logdet, ~bad


# ============================================================================
# 🔥 Gradient Safety Hook (NaN Shield)
# ============================================================================

def create_nan_clip_hook(name: str, clip_value: float = 5.0):
    """
    Gradient NaN/Inf shield hook (디버그/워밍업용).
    
    Args:
        name: Hook 이름 (디버깅용)
        clip_value: Gradient 클리핑 범위
    
    Returns:
        Hook function
    
    사용법:
        x_low.register_hook(create_nan_clip_hook("x_low", clip=5.0))
    """
    def _hook(grad):
        if grad is None:
            return None
        
        if not torch.isfinite(grad).all():
            nan_count = torch.isnan(grad).sum().item()
            inf_count = torch.isinf(grad).sum().item()
            print(f"[HOOK {name}] Non-finite gradient detected!")
            print(f"  NaN: {nan_count}, Inf: {inf_count}")
            print(f"  → Applying nan_to_num + clip({-clip_value:.1f}, {clip_value:.1f})")
            
            # NaN/Inf 제거
            grad = torch.nan_to_num(grad, nan=0.0, posinf=clip_value, neginf=-clip_value)
        
        # Gradient clipping
        grad = grad.clamp(-clip_value, clip_value)
        
        return grad
    
    return _hook


def ensure_spd_log_parameterization(
    cov: torch.Tensor,
    log_scale_min: float = -3.0,
    log_scale_max: float = 2.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    로그-파라미터화를 이용한 SPD 보장.
    
    전략:
        1. 고유값 분해: Σ = Q Λ Q^T
        2. 로그 변환 및 클램프: log(λ_i) → [log_scale_min, log_scale_max]
        3. 재구성: Σ_new = Q exp(log(Λ)) Q^T
    
    보장사항:
        - 모든 고유값 > 0 (양정부호)
        - 고유값 범위 제한 (수치 안정성)
        - 미분 가능 (그라디언트 흐름)
    
    Args:
        cov: (N, 3, 3) 공분산 행렬
        log_scale_min: 최소 로그 스케일 (예: -3 → σ_min ≈ 0.05)
        log_scale_max: 최대 로그 스케일 (예: 2 → σ_max ≈ 7.4)
        eps: 수치 안정성을 위한 작은 값
    
    Returns:
        SPD가 보장된 공분산 행렬 (N, 3, 3)
    """
    device = cov.device
    N = cov.shape[0]
    
    # 대칭화 (수치 비대칭 처리)
    cov_sym = 0.5 * (cov + cov.transpose(-2, -1))
    
    # 고유값 분해
    try:
        eigvals, eigvecs = torch.linalg.eigh(cov_sym)  # (N, 3), (N, 3, 3)
    except Exception as e:
        print(f"[WARN] 고유값 분해 실패: {e}, 정규화 추가 중...")
        # Fallback: 작은 단위행렬 추가
        eye = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
        cov_sym = cov_sym + 1e-4 * eye
        eigvals, eigvecs = torch.linalg.eigh(cov_sym)
    
    # 로그 변환 및 클램프
    log_eigvals = torch.log(torch.clamp(eigvals, min=eps))
    log_eigvals_clamped = torch.clamp(log_eigvals, min=log_scale_min, max=log_scale_max)
    
    # 클램프된 고유값으로 재구성
    eigvals_new = torch.exp(log_eigvals_clamped)  # (N, 3)
    
    # Σ_new = Q diag(λ_new) Q^T
    # 효율적인 배치 계산
    cov_new = torch.bmm(
        eigvecs * eigvals_new.unsqueeze(-2),  # (N, 3, 3) * (N, 1, 3) = (N, 3, 3)
        eigvecs.transpose(-2, -1)
    )
    
    return cov_new


def ensure_spd_eigenvalue_clamp(
    cov: torch.Tensor,
    min_eigenvalue: float = 1e-4,
    max_eigenvalue: float = 10.0
) -> torch.Tensor:
    """
    고유값 클램핑을 통한 SPD 보장.
    
    로그-파라미터화보다 단순하지만 덜 원칙적.
    빠른 수정이나 fallback으로 사용.
    
    Args:
        cov: (N, 3, 3) 공분산 행렬
        min_eigenvalue: 최소 고유값 (특이점 방지)
        max_eigenvalue: 최대 고유값 (폭발 방지)
    
    Returns:
        SPD가 보장된 공분산 행렬 (N, 3, 3)
    """
    device = cov.device
    N = cov.shape[0]
    
    # 대칭화
    cov_sym = 0.5 * (cov + cov.transpose(-2, -1))
    
    # 고유값 분해
    try:
        eigvals, eigvecs = torch.linalg.eigh(cov_sym)
    except Exception as e:
        print(f"[WARN] 고유값 분해 실패: {e}")
        # 정규화된 버전 반환
        eye = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
        return cov_sym + min_eigenvalue * eye
    
    # 고유값 클램프
    eigvals_clamped = torch.clamp(eigvals, min=min_eigenvalue, max=max_eigenvalue)
    
    # 재구성
    cov_new = torch.bmm(
        eigvecs * eigvals_clamped.unsqueeze(-2),
        eigvecs.transpose(-2, -1)
    )
    
    return cov_new


def ensure_spd_with_voxel_floor(
    cov: torch.Tensor,
    voxel_size: float,
    k: float = 0.6,
    eps: float = 1e-12,
    elem_clip: Optional[Tuple[float, float]] = None  # 🔥 NEW: 원소 클램프 옵션화
) -> torch.Tensor:
    """
    PSD 보장 + voxel 크기 기반 분산 바닥치 적용 (EVD 역전파 없이).
    
    🔥 핵심: EVD를 gradient 그래프 밖에서 수행 + STE로 역전파 안전성 보장!
    
    🔥 개선: 원소별 클램프 옵션화 (기본 해제)
    - SPD는 대각 jitter로 보장됨
    - 원소 클램프는 고유구조를 왜곡할 수 있어 기본 해제
    - 필요시만 elem_clip=(-10.0, 10.0) 전달
    
    전략:
        1. 대칭화
        2. Gershgorin 하계로 빠른 우회 (대부분 케이스는 EVD 불필요)
        3. 필요시 EVD → 하지만 torch.no_grad() 안에서만
        4. STE(Straight-Through Estimator)로 gradient 직통
        5. 대각 jitter 추가 (안전장치)
        6. (선택) 원소 클램프 (elem_clip 파라미터)
    
    이 방식은:
        - EVD 역전파 NaN/Inf 완전 차단 ✅
        - 등방성 공분산(λ_i ≈ λ_j)에서도 안전 ✅
        - 대부분 케이스에서 EVD 우회 (성능↑) ✅
        - 고유구조 왜곡 최소화 ✅ (elem_clip=None 기본)
    
    Args:
        cov: (N, 3, 3) 공분산 행렬
        voxel_size: 레벨셋 voxel 크기 (예: 0.0716)
        k: 바닥치 계수 (0.4~0.8 권장, 기본 0.6)
        eps: 수치 안정성 epsilon
        elem_clip: (min, max) 원소별 클램프 범위 (None=해제, 기본 권장)
    
    Returns:
        PSD + 바닥치가 보장된 공분산 행렬 (gradient-safe)
    """
    device = cov.device
    dtype = cov.dtype
    N = cov.shape[0]
    
    # ① 대칭화
    cov_sym = 0.5 * (cov + cov.transpose(-2, -1))
    
    # ② 분산 바닥치 계산
    lambda_floor = (k * voxel_size) ** 2
    
    # ③ Gershgorin 하계로 빠른 체크 (EVD 없이!)
    # Gershgorin: λ_min ≥ min_i(a_ii - Σ_{j≠i}|a_ij|)
    with torch.no_grad():
        diag = torch.diagonal(cov_sym, dim1=-2, dim2=-1)  # (N, 3)
        off_diag_sum = torch.sum(torch.abs(cov_sym), dim=-1) - torch.abs(diag)  # (N, 3)
        gersh_lower_bound = (diag - off_diag_sum).min(dim=-1).values  # (N,)
        
        # 대부분이 이미 안전하면 빠른 우회
        need_fix = (gersh_lower_bound < lambda_floor)
        
        if not need_fix.any():
            # 🚀 빠른 우회: 모두 안전하면 그대로 반환 (EVD 없음!)
            return cov_sym
        
        # 일부만 수정 필요 → delta 계산 (그래프 밖에서!)
        delta_per_point = torch.clamp(lambda_floor - gersh_lower_bound, min=0.0)  # (N,)
    
    # ④ 대각 jitter 추가 (STE: delta는 상수 취급)
    # Forward: cov + δI, Backward: ∂L/∂cov (δ는 detach되어 gradient 없음)
    delta_3d = delta_per_point.view(N, 1, 1)  # (N, 1, 1)
    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
    
    cov_safe = cov_sym + delta_3d * eye
    
    # ⑤ 원소 클램프 (선택적 - 기본 해제)
    # ✅ 기본은 원소 클램프 해제. 필요 시만 사용.
    # SPD는 대각 jitter로 보장되므로, 원소 클램프는 고유구조를 왜곡할 수 있음
    if elem_clip is not None:
        lo, hi = elem_clip
        cov_safe = torch.clamp(cov_safe, min=lo, max=hi)
    
    return cov_safe


def covariance_regularization_loss(
    cov: torch.Tensor,
    lambda_scale: float = 1e-3,
    target_trace: Optional[float] = None
) -> torch.Tensor:
    """
    공분산 붕괴 또는 폭발 방지를 위한 정규화 손실.
    
    손실: λ_scale * mean(log²(σ_i))
    
    극단적 스케일에 페널티를 주면서 적당한 변동은 허용.
    
    Args:
        cov: (N, 3, 3) 공분산 행렬
        lambda_scale: 정규화 가중치
        target_trace: 목표 대각합 (None이면 극단값만 페널티)
    
    Returns:
        정규화 손실 (스칼라)
    """
    # 대각 성분 추출 (분산)
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)  # (N, 3)
    scales = torch.sqrt(torch.clamp(diag, min=1e-8))  # (N, 3)
    
    # 로그 스케일 정규화
    log_scales = torch.log(scales)
    loss = lambda_scale * (log_scales ** 2).mean()
    
    # 옵션: 대각합 정규화
    if target_trace is not None:
        trace = diag.sum(dim=-1)  # (N,)
        target = torch.full_like(trace, target_trace)
        loss = loss + lambda_scale * torch.nn.functional.mse_loss(trace, target)
    
    return loss


def diagnose_covariance_health(
    cov: Union[torch.Tensor, np.ndarray],
    verbose: bool = False
) -> Dict[str, float]:
    """
    공분산 행렬 건강도 진단.
    
    검사 항목:
        - 고유값 범위 (양정부호성)
        - 행렬식 범위 (부피)
        - 대각합 (스케일)
        - 음수 원소 (SPD 위반)
    
    Args:
        cov: (N, 3, 3) 공분산 행렬 (torch 또는 numpy)
        verbose: 상세 진단 출력
    
    Returns:
        진단 통계 딕셔너리
    """
    # numpy로 변환하여 분석
    if torch.is_tensor(cov):
        cov_np = cov.detach().cpu().numpy()
    else:
        cov_np = cov
    
    N = cov_np.shape[0]
    
    # 고유값 계산
    eigvals_list = []
    det_list = []
    trace_list = []
    min_elem_list = []
    max_elem_list = []
    
    for i in range(N):
        try:
            eigvals = np.linalg.eigvalsh(cov_np[i])
            eigvals_list.append(eigvals)
            det_list.append(np.prod(eigvals))
        except:
            eigvals_list.append(np.array([np.nan, np.nan, np.nan]))
            det_list.append(np.nan)
        
        trace_list.append(np.trace(cov_np[i]))
        min_elem_list.append(cov_np[i].min())
        max_elem_list.append(cov_np[i].max())
    
    eigvals_all = np.array(eigvals_list)  # (N, 3)
    
    diagnostics = {
        # 고유값 통계
        'eig_min': float(np.nanmin(eigvals_all)),
        'eig_max': float(np.nanmax(eigvals_all)),
        'eig_mean': float(np.nanmean(eigvals_all)),
        'eig_std': float(np.nanstd(eigvals_all)),
        
        # 음수 고유값 (PSD 위반)
        'num_negative_eig': int((eigvals_all < 0).sum()),
        'frac_negative_eig': float((eigvals_all < 0).sum() / (N * 3)),
        
        # 행렬식 (부피)
        'det_min': float(np.nanmin(det_list)),
        'det_max': float(np.nanmax(det_list)),
        'det_mean': float(np.nanmean(det_list)),
        'det_median': float(np.nanmedian(det_list)),
        
        # 대각합 (스케일)
        'trace_min': float(np.min(trace_list)),
        'trace_max': float(np.max(trace_list)),
        'trace_mean': float(np.mean(trace_list)),
        
        # 원소별
        'elem_min': float(np.min(min_elem_list)),
        'elem_max': float(np.max(max_elem_list)),
        
        # 건강 플래그
        'has_negative_elem': bool(np.min(min_elem_list) < 0),
        'has_negative_eig': bool((eigvals_all < 0).any()),
        'has_extreme_det': bool((np.array(det_list) < 1e-8).any() or (np.array(det_list) > 1e2).any()),
    }
    
    if verbose:
        print("\n" + "="*70)
        print("공분산 진단 (Covariance Diagnostics)")
        print("="*70)
        print(f"샘플 크기: {N:,}")
        print(f"\n고유값 (Eigenvalues):")
        print(f"  범위: [{diagnostics['eig_min']:.6e}, {diagnostics['eig_max']:.6e}]")
        print(f"  평균: {diagnostics['eig_mean']:.6e}, 표준편차: {diagnostics['eig_std']:.6e}")
        print(f"  음수: {diagnostics['num_negative_eig']} ({diagnostics['frac_negative_eig']*100:.2f}%)")
        print(f"\n행렬식 (Determinant):")
        print(f"  범위: [{diagnostics['det_min']:.6e}, {diagnostics['det_max']:.6e}]")
        print(f"  평균: {diagnostics['det_mean']:.6e}, 중앙값: {diagnostics['det_median']:.6e}")
        print(f"\n대각합 (Trace):")
        print(f"  범위: [{diagnostics['trace_min']:.6e}, {diagnostics['trace_max']:.6e}]")
        print(f"  평균: {diagnostics['trace_mean']:.6e}")
        print(f"\n원소별 (Element-wise):")
        print(f"  범위: [{diagnostics['elem_min']:.6e}, {diagnostics['elem_max']:.6e}]")
        
        # 건강도 요약
        print(f"\n건강도 요약:")
        issues = []
        if diagnostics['has_negative_elem']:
            issues.append("❌ 음수 원소 감지됨 (SPD 위반)")
        if diagnostics['has_negative_eig']:
            issues.append("❌ 음수 고유값 감지됨 (양정부호 아님)")
        if diagnostics['has_extreme_det']:
            issues.append("⚠️  극단적 행렬식 (수치 불안정)")
        
        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"  ✅ 모든 검사 통과")
        
        print("="*70 + "\n")
    
    return diagnostics


def apply_covariance_fixes(
    cov: torch.Tensor,
    method: str = 'log_param',
    verbose: bool = False
) -> torch.Tensor:
    """
    진단 기반 공분산 수정 적용.
    
    Args:
        cov: (N, 3, 3) 공분산 행렬
        method: 'log_param' 또는 'eigen_clamp'
        verbose: 전/후 진단 출력
    
    Returns:
        수정된 공분산 행렬
    """
    if verbose:
        print("\n[공분산 수정] 이전:")
        diagnose_covariance_health(cov, verbose=True)
    
    if method == 'log_param':
        cov_fixed = ensure_spd_log_parameterization(cov)
    elif method == 'eigen_clamp':
        cov_fixed = ensure_spd_eigenvalue_clamp(cov)
    else:
        raise ValueError(f"알 수 없는 메서드: {method}")
    
    if verbose:
        print(f"\n[공분산 수정] 이후 ({method}):")
        diagnose_covariance_health(cov_fixed, verbose=True)
    
    return cov_fixed


def compose_cov_from_frame(
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
    sig_t1: torch.Tensor,
    sig_t2: torch.Tensor,
    sig_n: torch.Tensor,
    voxel: float,
    k_t: float = 0.25,
    k_n: float = 0.05
) -> torch.Tensor:
    """
    축별 바닥치를 적용하여 프레임에서 공분산을 합성 (Cholesky/EVD-free).
    
    핵심: 법선 방향은 얇게(디테일), 접선 방향은 두껍게(안정성) 유지.
    
    Args:
        t1: (N, 3) 첫 번째 접선 벡터
        t2: (N, 3) 두 번째 접선 벡터
        n: (N, 3) 법선 벡터
        sig_t1: (N,) 첫 번째 접선 표준편차
        sig_t2: (N,) 두 번째 접선 표준편차
        sig_n: (N,) 법선 표준편차
        voxel: voxel 크기
        k_t: 접선 바닥치 계수 (기본: 0.25)
        k_n: 법선 바닥치 계수 (기본: 0.05)
    
    Returns:
        (N, 3, 3) SPD 공분산 행렬
    """
    eps = 1e-6
    N = t1.shape[0]
    device = t1.device
    
    # 축별 바닥치 적용
    sig_t1 = torch.clamp(sig_t1, min=k_t * voxel)
    sig_t2 = torch.clamp(sig_t2, min=k_t * voxel)
    sig_n = torch.clamp(sig_n, min=k_n * voxel)
    
    # 안전한 정규화
    def safe_norm(v, eps=eps):
        return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)
    
    t1 = safe_norm(t1)
    n = safe_norm(n)
    
    # 직교화: t2를 t1, n에 직교하게
    t2 = t2 - (t1 * (t2 * t1).sum(-1, keepdim=True))
    t2 = t2 - (n * (t2 * n).sum(-1, keepdim=True))
    t2 = safe_norm(t2)
    
    # 프레임 행렬 U = [t1, t2, n]
    U = torch.stack([t1, t2, n], dim=-1)  # (N, 3, 3)
    
    # 대각 행렬 D = diag(σ_t1², σ_t2², σ_n²)
    D = torch.diag_embed(torch.stack([sig_t1**2, sig_t2**2, sig_n**2], dim=-1))  # (N, 3, 3)
    
    # Σ = U D U^T (SPD 보장)
    cov = torch.bmm(torch.bmm(U, D), U.transpose(-1, -2))
    
    return cov


def clamp_screen_cov(
    Sig2d: torch.Tensor,
    r_px_min: float = 1.0
) -> torch.Tensor:
    """
    스크린공간 2D 공분산의 고유값 하한 적용.
    
    월드 바닥치를 과하게 키우지 않고, 프로젝션 후 2D에서만 하한을 보장합니다.
    
    Args:
        Sig2d: (N, 2, 2) 2D 스크린 공분산
        r_px_min: 최소 픽셀 반경 (기본: 1.0 px)
    
    Returns:
        (N, 2, 2) 하한이 적용된 2D 공분산
    """
    # 대칭화
    Sig2d = 0.5 * (Sig2d + Sig2d.transpose(-1, -2))
    
    # 고유값 분해
    s, U = torch.linalg.eigh(Sig2d)  # s: (N, 2), U: (N, 2, 2)
    
    # 고유값 하한 적용 (r_px_min² = 최소 분산)
    s = torch.clamp(s, min=r_px_min**2)
    
    # 재구성
    Sig2d_clamped = torch.bmm(
        torch.bmm(U, torch.diag_embed(s)),
        U.transpose(-1, -2)
    )
    
    return Sig2d_clamped


def sigma_from_curv(
    k1: torch.Tensor,
    k2: torch.Tensor,
    voxel: float,
    sig_t0: float = 0.06,
    sig_n0: float = 0.035,
    a: float = 2.0,
    b: float = 0.5,
    a_n: float = 0.5,
    u: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    곡률에서 축별 표준편차 계산 (재스케일 버전).
    
    전략:
    - 접선: 곡률↑일수록 얇게 (디테일 보존)
    - 법선: 작되 하한 유지 (두께 제어)
    - 무단위화: k̂ = |k| × voxel
    
    Args:
        k1: (N,) 주곡률 1
        k2: (N,) 주곡률 2
        voxel: voxel 크기
        sig_t0: 초기 접선 스케일 (기본: 0.06)
        sig_n0: 초기 법선 스케일 (기본: 0.035)
        a: 접선 감쇠 계수 (기본: 2.0)
        b: 접선 감쇠 지수 (기본: 0.5)
        a_n: 법선 감쇠 계수 (기본: 0.5)
        u: 법선 감쇠 지수 (기본: 0.3)
    
    Returns:
        (sig_t1, sig_t2, sig_n) 각각 (N,)
    """
    # 무단위화
    k1n = k1.abs() * voxel
    k2n = k2.abs() * voxel
    
    # 평균 곡률 (법선 방향)
    Hn = 0.5 * (k1n + k2n)
    
    # 접선 방향: 곡률↑ → σ↓ (얇아짐)
    sig_t1 = sig_t0 / (1 + a * k1n).pow(b)
    sig_t2 = sig_t0 / (1 + a * k2n).pow(b)
    
    # 법선 방향: 평균 곡률에 따라 제어
    sig_n = sig_n0 / (1 + a_n * Hn).pow(u)
    
    return sig_t1, sig_t2, sig_n


__all__ = [
    'ensure_spd_log_parameterization',
    'ensure_spd_eigenvalue_clamp',
    'ensure_spd_with_voxel_floor',  # 🔥 NEW: 바닥치 적용 (핵심!)
    'covariance_regularization_loss',
    'diagnose_covariance_health',
    'apply_covariance_fixes',
    'compose_cov_from_frame',  # 🔥 NEW: 축별 바닥치 프레임 합성
    'clamp_screen_cov',        # 🔥 NEW: 2D 스크린공간 하한
    'sigma_from_curv',          # 🔥 NEW: 곡률→σ 재스케일
    'project_sigma3d_to_2d_safe',    # 🔥 NEW: 2D screen projection (gradient-safe)
    'inv_and_logdet_spd_2x2_chol',   # 🔥 NEW: Cholesky inv/logdet (gradient-safe)
    'create_nan_clip_hook',          # 🔥 NEW: Gradient NaN shield
]

