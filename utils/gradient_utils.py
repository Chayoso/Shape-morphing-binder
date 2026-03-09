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
        print(f"\n[WARN] Gradient health issue ({grad_type}):")
        if diagnostics['has_nan_F'] or diagnostics['has_nan_x']:
            print(f"  - NaN detected: F={diagnostics['has_nan_F']}, x={diagnostics['has_nan_x']}")
        if diagnostics['has_inf_F'] or diagnostics['has_inf_x']:
            print(f"  - Inf detected: F={diagnostics['has_inf_F']}, x={diagnostics['has_inf_x']}")
        if diagnostics['has_extreme_F'] or diagnostics['has_extreme_x']:
            print(f"  - Extreme values: |F|_max={diagnostics['max_abs_F']:.2e}, |x|_max={diagnostics['max_abs_x']:.2e}")
    
    return diagnostics


def normalize_and_combine_gradients(
    dLdF_physics: np.ndarray,
    dLdx_physics: np.ndarray,
    dLdF_render: np.ndarray,
    dLdx_render: np.ndarray,
    w_physics: float = 0.7,
    w_render: float = 0.3,
    magnitude_strategy: str = 'physics',
    render_F_ratio: float = 0.0,
    w_render_F: float = -1.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Normalize and combine physics and render gradients.

    When render_F_ratio > 0 and physics F gradients are zero,
    F channel is handled separately: render drives F directly,
    scaled relative to physics x magnitude.

    Args:
        dLdF_physics: (N, 3, 3) physics gradients for deformation gradient
        dLdx_physics: (N, 3) physics gradients for positions
        dLdF_render: (N, 3, 3) render gradients for deformation gradient
        dLdx_render: (N, 3) render gradients for positions
        w_physics: Weight for physics gradients
        w_render: Weight for render gradients (used for x channel)
        magnitude_strategy: How to set final magnitude for x channel
        render_F_ratio: F gradient magnitude = ratio * ||dL_phys/dx||.
            When > 0 and physics F grads are zero, bypasses normalization for F.
        w_render_F: Separate render weight for F channel. If < 0, uses w_render.

    Returns:
        Tuple of (dLdF_combined, dLdx_combined, info_dict)
    """
    eps = 1e-12

    # Compute gradient norms
    g_F_phys = np.linalg.norm(dLdF_physics)
    g_x_phys = np.linalg.norm(dLdx_physics)
    g_phys_total = np.sqrt(g_F_phys**2 + g_x_phys**2)

    g_F_render = np.linalg.norm(dLdF_render)
    g_x_render = np.linalg.norm(dLdx_render)
    g_render_total = np.sqrt(g_F_render**2 + g_x_render**2)

    # Check for zero gradients
    if g_phys_total < eps:
        return dLdF_render * w_render, dLdx_render * w_render, {
            'g_physics_norm': 0.0,
            'g_render_norm': float(g_render_total),
            'g_combined_norm': float(g_render_total * w_render),
            'ratio_before': np.inf,
            'ratio_after': np.inf,
            'magnitude_scale': w_render,
        }

    if g_render_total < eps:
        return dLdF_physics, dLdx_physics, {
            'g_physics_norm': float(g_phys_total),
            'g_render_norm': 0.0,
            'g_combined_norm': float(g_phys_total),
            'ratio_before': 0.0,
            'ratio_after': 0.0,
            'magnitude_scale': 1.0,
        }

    # ═══════════════════════════════════════════════════════════════
    # F CHANNEL: Direct render-driven when physics F is zero
    # ═══════════════════════════════════════════════════════════════
    use_direct_F = (render_F_ratio > 0) and (g_F_phys < eps) and (g_F_render > eps)

    if use_direct_F:
        # Physics produces zero F gradient → render drives F directly
        # Scale: ||dLdF_combined|| = render_F_ratio * ||dLdx_phys||
        target_F_mag = render_F_ratio * g_x_phys
        dLdF_render_unit = dLdF_render / g_F_render
        dLdF_combined = target_F_mag * dLdF_render_unit

        # F-DIRECT: render drives F (physics dL/dF=0)
    else:
        # Original logic for F when physics contributes
        dLdF_combined = None  # Will be set below

    # ═══════════════════════════════════════════════════════════════
    # X CHANNEL: Standard normalization (physics + render combined)
    # ═══════════════════════════════════════════════════════════════
    # Normalize x to unit vectors
    phys_den_x = g_x_phys if g_x_phys > eps else max(g_x_render, eps)
    render_den_x = max(g_x_render, eps)
    dLdx_phys_unit = dLdx_physics / phys_den_x
    dLdx_render_unit = dLdx_render / render_den_x

    dLdx_combined_unit = w_physics * dLdx_phys_unit + w_render * dLdx_render_unit

    # x magnitude strategy
    if magnitude_strategy == 'physics':
        target_x = g_x_phys if g_x_phys > eps else max(g_x_render, eps)
    elif magnitude_strategy == 'weighted':
        target_x = w_physics * g_x_phys + w_render * g_x_render
    elif magnitude_strategy == 'max':
        target_x = max(g_x_phys, g_x_render)
    elif magnitude_strategy in ('normalize', 'rms'):
        target_x = np.sqrt((g_x_phys**2 + g_x_render**2) / 2.0) if (g_x_phys > eps or g_x_render > eps) else eps
    else:
        raise ValueError(f"Unknown magnitude_strategy: {magnitude_strategy}")

    dLdx_combined = target_x * dLdx_combined_unit

    # If F wasn't handled by direct mode, use original combined logic
    if dLdF_combined is None:
        phys_den_F = g_F_phys if g_F_phys > eps else max(g_F_render, eps)
        render_den_F = max(g_F_render, eps)
        w_F = w_render_F if w_render_F >= 0 else w_render
        dLdF_phys_unit = dLdF_physics / phys_den_F
        dLdF_render_unit_norm = dLdF_render / render_den_F
        dLdF_combined_unit = w_physics * dLdF_phys_unit + w_F * dLdF_render_unit_norm

        if magnitude_strategy == 'physics':
            target_F = g_F_phys if g_F_phys > eps else g_F_render
        elif magnitude_strategy == 'weighted':
            target_F = w_physics * g_F_phys + w_F * g_F_render
        elif magnitude_strategy == 'max':
            target_F = max(g_F_phys, g_F_render)
        elif magnitude_strategy in ('normalize', 'rms'):
            target_F = np.sqrt((g_F_phys**2 + g_F_render**2) / 2.0) if (g_F_phys > eps or g_F_render > eps) else eps
        else:
            target_F = g_F_phys if g_F_phys > eps else g_F_render

        dLdF_combined = target_F * dLdF_combined_unit

    # Diagnostics
    g_F_combined = np.linalg.norm(dLdF_combined)
    g_x_combined = np.linalg.norm(dLdx_combined)
    g_combined_total = np.sqrt(g_F_combined**2 + g_x_combined**2)

    ratio_before = g_render_total / (g_phys_total + eps)
    ratio_after = g_combined_total / (g_phys_total + eps)

    info = {
        'g_physics_norm': float(g_phys_total),
        'g_render_norm': float(g_render_total),
        'g_combined_norm': float(g_combined_total),
        'ratio_before': float(ratio_before),
        'ratio_after': float(ratio_after),
        'w_physics': w_physics,
        'w_render': w_render,
        'magnitude_strategy': magnitude_strategy,
        'magnitude_scale': float(g_F_combined / max(g_F_render, eps)),
        'direct_F_mode': use_direct_F,
        'render_F_ratio': render_F_ratio,
    }

    return dLdF_combined, dLdx_combined, info


__all__ = [
    'compute_gradient_statistics',
    'compute_gradient_cosine_similarity',
    'pcgrad_projection',
    'diagnose_gradient_health',
    'normalize_and_combine_gradients',
]
