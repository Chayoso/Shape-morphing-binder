"""
Gradient Management Utilities for Multi-Task Learning

Implements:
- EMA-based gradient scaling
- PCGrad (gradient projection for conflict resolution)
- Gradient diagnostics and monitoring
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_gradient_statistics(
    dLdF: np.ndarray,
    dLdx: np.ndarray
) -> Dict[str, float]:
    """
    Compute gradient statistics for monitoring.
    
    Args:
        dLdF: (N, 3, 3) gradient w.r.t. deformation gradient
        dLdx: (N, 3) gradient w.r.t. positions
    
    Returns:
        Dictionary with gradient norms and statistics
    """
    g_F = np.linalg.norm(dLdF)
    g_x = np.linalg.norm(dLdx)
    g_total = np.sqrt(g_F**2 + g_x**2)
    
    return {
        'grad_F_norm': float(g_F),
        'grad_x_norm': float(g_x),
        'grad_total_norm': float(g_total),
        'grad_F_mean': float(np.abs(dLdF).mean()),
        'grad_x_mean': float(np.abs(dLdx).mean()),
    }


def compute_gradient_cosine_similarity(
    dLdF_physics: np.ndarray,
    dLdx_physics: np.ndarray,
    dLdF_render: np.ndarray,
    dLdx_render: np.ndarray
) -> float:
    """
    Compute cosine similarity between physics and render gradients.
    
    Args:
        dLdF_physics: Physics gradients for F
        dLdx_physics: Physics gradients for x
        dLdF_render: Render gradients for F
        dLdx_render: Render gradients for x
    
    Returns:
        Cosine similarity in [-1, 1]
    """
    # Flatten all gradients
    g_phys = np.concatenate([dLdF_physics.flatten(), dLdx_physics.flatten()])
    g_render = np.concatenate([dLdF_render.flatten(), dLdx_render.flatten()])
    
    # Compute cosine similarity
    dot = np.dot(g_phys, g_render)
    norm_phys = np.linalg.norm(g_phys) + 1e-12
    norm_render = np.linalg.norm(g_render) + 1e-12
    
    cosine = dot / (norm_phys * norm_render)
    return float(np.clip(cosine, -1.0, 1.0))


def _flatten_pair(dLdF: np.ndarray, dLdx: np.ndarray) -> Tuple[np.ndarray, int]:
    """Flatten F and x gradients into single vector."""
    gF = dLdF.reshape(-1)
    gx = dLdx.reshape(-1)
    return np.concatenate([gF, gx], axis=0), gF.size


def _blockwise_whiten(
    dLdF: np.ndarray, 
    dLdx: np.ndarray, 
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    블록별 스케일 정규화: F와 x를 각각 자기 L2 노름으로 나눠 동일 단위화.
    
    이렇게 하면 F(변형 그래디언트)와 x(위치)의 물리 단위 차이가 
    PCGrad 투영에서 왜곡을 일으키지 않습니다.
    
    Returns:
        (dLdF_normalized, dLdx_normalized, {'nF': norm_F, 'nx': norm_x})
    """
    nF = max(np.linalg.norm(dLdF), eps)
    nx = max(np.linalg.norm(dLdx), eps)
    return dLdF / nF, dLdx / nx, {'nF': nF, 'nx': nx}


def _clip_by_quantile(arr: np.ndarray, q: float = 0.999) -> np.ndarray:
    """
    분위수 기반 클리핑: outlier 억제.
    
    렌더 그라디언트에서 극단값(예: 급경사 실루엣 픽셀)이 
    전체 투영을 왜곡하는 것을 방지합니다.
    
    Args:
        arr: 입력 배열
        q: 분위수 임계치 (기본: 99.9%)
    
    Returns:
        클리핑된 배열
    """
    thr = np.quantile(np.abs(arr), q)
    if thr <= 0:
        return arr
    return np.clip(arr, -thr, thr)


def pcgrad_projection(
    dLdF_render: np.ndarray,
    dLdx_render: np.ndarray,
    dLdF_physics: np.ndarray,
    dLdx_physics: np.ndarray,
    conflict_threshold: float = -0.1,
    adapt_threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    🔥 Metric-aware PCGrad with quantile clipping.
    
    개선사항:
    - 블록별 정규화 후 투영 (F, x 스케일 정합)
    - 분위수(기본 99.9%) 클리핑으로 outlier 억제
    - 적응형 임계치 지원 (에피소드 진행도에 따라 조정 가능)
    
    References:
        Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020
    
    Args:
        dLdF_render: Render gradients for F
        dLdx_render: Render gradients for x
        dLdF_physics: Physics gradients for F (reference)
        dLdx_physics: Physics gradients for x (reference)
        conflict_threshold: Threshold for conflict detection (default: -0.1)
        adapt_threshold: 적응형 임계치 (None이면 conflict_threshold 사용)
    
    Returns:
        Tuple of (dLdF_projected, dLdx_projected, info_dict)
    """
    # 1) 블록별 정규화 (화이트닝)
    rF_w, rx_w, stats_r = _blockwise_whiten(dLdF_render, dLdx_render)
    pF_w, px_w, stats_p = _blockwise_whiten(dLdF_physics, dLdx_physics)
    
    # 2) 플래튼
    g_r, nF = _flatten_pair(rF_w, rx_w)
    g_p, _  = _flatten_pair(pF_w, px_w)
    
    # 3) 분위수 클리핑 (렌더만 - outlier 억제)
    g_r = _clip_by_quantile(g_r, q=0.999)
    
    # 4) 코사인/충돌 판정
    dot = float(np.dot(g_r, g_p))
    denom = (np.linalg.norm(g_r) + 1e-12) * (np.linalg.norm(g_p) + 1e-12)
    cosine = float(np.clip(dot / denom, -1.0, 1.0))
    
    # 임계치 선택 (적응형 우선)
    thr = conflict_threshold if adapt_threshold is None else adapt_threshold
    
    info = {
        'pcgrad_cosine': cosine, 
        'pcgrad_applied': False, 
        'pcgrad_projection_scale': 0.0,
        'pcgrad_threshold_used': thr
    }
    
    # 5) 충돌 시 투영 (음의 성분만 제거 - 과투영 방지)
    if cosine < thr:
        # 표준 PCGrad: 음의 내적 성분만 제거
        proj = (min(0.0, dot) / (np.dot(g_p, g_p) + 1e-12)) * g_p
        g_r_proj = g_r - proj
        
        info['pcgrad_applied'] = True
        info['pcgrad_projection_scale'] = float(np.linalg.norm(proj) / (np.linalg.norm(g_r) + 1e-12))
    else:
        g_r_proj = g_r
    
    # 6) 원래 스케일로 복원
    gF_proj = g_r_proj[:nF].reshape(dLdF_render.shape) * stats_r['nF']
    gx_proj = g_r_proj[nF:].reshape(dLdx_render.shape) * stats_r['nx']
    
    return gF_proj, gx_proj, info


def ema_gradient_scaling(
    g_render_norm: float,
    g_physics_norm: float,
    target_ratio: float,
    ema_state: Dict,
    ema_beta: float = 0.8,
    power: float = 0.7,
    min_gain: float = 0.1,
    max_gain: float = 50000.0
) -> Tuple[float, Dict[str, float]]:
    """
    🔥 EMA-based adaptive gradient scaling with dynamic upper bound.
    
    개선사항:
    - 동적 상한: 물리 신호가 약한 구간의 gain 폭주 방지
    - after_gain_ratio 상태 저장: 다음 스텝 활용 가능
    
    Target: ||g_render|| / ||g_physics|| ≈ target_ratio (e.g., 0.2-0.5)
    
    Args:
        g_render_norm: Current render gradient norm
        g_physics_norm: Current physics gradient norm
        target_ratio: Desired ratio (e.g., 0.3 for render to be 30% of physics)
        ema_state: EMA state dictionary (will be updated in-place)
        ema_beta: EMA smoothing factor (0.8 = smooth, 0.0 = no smoothing)
        power: Power for scaling (< 1.0 for conservative adjustment)
        min_gain: Minimum gain clamp
        max_gain: Maximum gain clamp
    
    Returns:
        Tuple of (gain, info_dict)
    """
    eps = 1e-12
    g_phys = max(g_physics_norm, eps)
    g_ren = max(g_render_norm, eps)
    
    # Compute target norm
    target_norm = target_ratio * g_phys
    
    # Compute raw gain
    raw = (target_norm / g_ren) ** power
    
    # 🔒 동적 상한: 물리 신호가 약한 구간의 폭주 방지
    # 렌더 대비 물리가 10배 이상 크면 상한을 그에 맞춰 낮춤
    dyn_cap = max(10.0, 10.0 * (g_phys / g_ren))
    raw = float(np.clip(raw, min_gain, min(max_gain, dyn_cap)))
    
    # Apply EMA smoothing
    if 'render_gain_ema' not in ema_state:
        ema_state['render_gain_ema'] = raw
        gain = raw
    else:
        gain = ema_beta * ema_state['render_gain_ema'] + (1.0 - ema_beta) * raw
        ema_state['render_gain_ema'] = gain
    
    # 🔎 after_gain_ratio 상태 기록 (다음 스텝/로그에서 활용)
    after_ratio = (g_ren * gain) / g_phys
    ema_state['after_gain_ratio'] = float(after_ratio)
    
    info = {
        'gain_raw': raw,
        'gain_smoothed': gain,
        'target_ratio': target_ratio,
        'actual_ratio': g_ren / g_phys,
        'target_norm': target_norm,
        'after_gain_ratio': after_ratio,  # 🔥 NEW: 게인 적용 후 비율
        'dyn_cap_used': dyn_cap,          # 🔥 NEW: 동적 상한값
    }
    
    return gain, info


def adaptive_target_ratio_schedule(
    episode: int,
    total_episodes: int,
    warmup_episodes: int = 5,
    rampup_episodes: int = 15,
    warmup_ratio: float = 0.10,   # 🔥 0.05 → 0.10 (초기부터 렌더 신호 강화)
    rampup_ratio: float = 0.35,   # 🔥 0.30 → 0.35 (중기 목표 상향)
    full_ratio: float = 0.40       # 🔥 0.50 → 0.40 (안정적 균형점)
) -> float:
    """
    Compute adaptive target ratio based on training progress.
    
    🔥 공분산 효과 극대화 패치: 목표 비율 전반적 상향
    
    Schedule (updated):
        - Episodes 0-5:   warmup_ratio (0.10) - 초기부터 렌더 신호 작동
        - Episodes 5-15:  ramp from 0.10 to 0.35 - 점진적 강화
        - Episodes 15+:   full_ratio (0.35-0.40) - 물리와 거의 대등ㅋ
    
    Args:
        episode: Current episode number
        total_episodes: Total number of episodes
        warmup_episodes: Episodes for warmup phase
        rampup_episodes: Episodes to complete ramp-up
        warmup_ratio: Ratio during warmup (0.10 권장)
        rampup_ratio: Ratio at end of ramp-up (0.35 권장)
        full_ratio: Final ratio (0.35-0.40 권장)
    
    Returns:
        Target ratio for current episode
    """
    if episode < warmup_episodes:
        # Warmup: 초기부터 적정 렌더 신호
        return warmup_ratio
    elif episode < rampup_episodes:
        # Ramp-up: linear interpolation
        progress = (episode - warmup_episodes) / (rampup_episodes - warmup_episodes)
        return warmup_ratio + (rampup_ratio - warmup_ratio) * progress
    else:
        # Full training: 물리와 거의 대등한 균형
        return full_ratio


def diagnose_gradient_health(
    dLdF: np.ndarray,
    dLdx: np.ndarray,
    grad_type: str = "unknown"
) -> Dict[str, float]:
    """
    Diagnose gradient health (NaN, Inf, extreme values).
    
    Args:
        dLdF: Gradient w.r.t. F
        dLdx: Gradient w.r.t. x
        grad_type: Gradient type for logging (e.g., "physics", "render")
    
    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        'has_nan_F': bool(np.isnan(dLdF).any()),
        'has_inf_F': bool(np.isinf(dLdF).any()),
        'has_nan_x': bool(np.isnan(dLdx).any()),
        'has_inf_x': bool(np.isinf(dLdx).any()),
        'max_abs_F': float(np.abs(dLdF).max()),
        'max_abs_x': float(np.abs(dLdx).max()),
        'min_abs_F': float(np.abs(dLdF[dLdF != 0]).min()) if (dLdF != 0).any() else 0.0,
        'min_abs_x': float(np.abs(dLdx[dLdx != 0]).min()) if (dLdx != 0).any() else 0.0,
    }
    
    # Check for extreme values
    extreme_threshold = 1e6
    diagnostics['has_extreme_F'] = bool(np.abs(dLdF).max() > extreme_threshold)
    diagnostics['has_extreme_x'] = bool(np.abs(dLdx).max() > extreme_threshold)
    
    # Overall health
    is_healthy = (
        not diagnostics['has_nan_F'] and
        not diagnostics['has_nan_x'] and
        not diagnostics['has_inf_F'] and
        not diagnostics['has_inf_x'] and
        not diagnostics['has_extreme_F'] and
        not diagnostics['has_extreme_x']
    )
    diagnostics['is_healthy'] = is_healthy
    
    if not is_healthy:
        print(f"\n⚠️  [WARN] Gradient health issue detected ({grad_type}):")
        if diagnostics['has_nan_F'] or diagnostics['has_nan_x']:
            print(f"  - NaN detected: F={diagnostics['has_nan_F']}, x={diagnostics['has_nan_x']}")
        if diagnostics['has_inf_F'] or diagnostics['has_inf_x']:
            print(f"  - Inf detected: F={diagnostics['has_inf_F']}, x={diagnostics['has_inf_x']}")
        if diagnostics['has_extreme_F'] or diagnostics['has_extreme_x']:
            print(f"  - Extreme values: |F|_max={diagnostics['max_abs_F']:.2e}, |x|_max={diagnostics['max_abs_x']:.2e}")
    
    return diagnostics


__all__ = [
    'compute_gradient_statistics',
    'compute_gradient_cosine_similarity',
    'pcgrad_projection',
    'ema_gradient_scaling',
    'adaptive_target_ratio_schedule',
    'diagnose_gradient_health',
]

