"""
Surface-aware control-space guidance utilities.

This module converts renderer-derived particle gradients into control-space
penalties that are injected into the differentiable MPM backward pass.

The intended use is:
  surface observation gradient -> smoothed vector field / tensor field
  -> injected (dL/dx, dL/dF) penalty -> Adam updates dFc inside physics
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_control_guidance_penalty(x, grad_x, cfg, focus_mask=None, core_mask=None, support_mask=None):
    """
    Convert particle-space observation gradients into injected control penalties.

    Returns:
        penalty: {'dLdF': (N,3,3), 'dLdx': (N,3)} or None
        meta: dict of norms / activity statistics
    """
    x = np.asarray(x, dtype=np.float32)
    g = np.asarray(grad_x, dtype=np.float32).copy()
    N = x.shape[0]

    zero_meta = {
        'control_guidance_applied': 0,
        'control_guidance_active_count': 0,
        'control_guidance_active_frac': 0.0,
        'control_guidance_core_count': 0,
        'control_guidance_core_frac': 0.0,
        'control_guidance_support_count': 0,
        'control_guidance_support_frac': 0.0,
        'control_guidance_dLdx_norm': 0.0,
        'control_guidance_dLdF_norm': 0.0,
        'control_guidance_dLdx_max': 0.0,
        'control_guidance_dLdF_max': 0.0,
        'control_guidance_focus_count': 0,
        'control_guidance_focus_frac': 0.0,
        'control_guidance_shell_lift_weight': float(cfg.get('shell_lift_weight', 0.0)),
        'control_guidance_shell_lift_k': int(cfg.get('shell_lift_k', 24)),
        'control_guidance_shell_lift_diffusion_iters': int(cfg.get('shell_lift_diffusion_iters', 1)),
        'control_guidance_partner_k': int(cfg.get('smooth_k', 64)),
        'control_guidance_diffusion_iters': int(cfg.get('diffusion_iters', 2)),
    }
    if N == 0 or g.shape != (N, 3):
        return None, zero_meta

    eps = 1e-8
    support = np.ones(N, dtype=bool)
    if support_mask is not None:
        sm = np.asarray(support_mask, dtype=bool).reshape(-1)
        if sm.shape[0] == N and sm.any():
            support = sm

    core = support.copy()
    if core_mask is not None:
        cm = np.asarray(core_mask, dtype=bool).reshape(-1)
        if cm.shape[0] == N and cm.any():
            core = cm & support

    support_idx = np.flatnonzero(support)
    core_idx = np.flatnonzero(core)
    n_support = int(support_idx.size)
    n_core = int(core_idx.size)
    zero_meta['control_guidance_core_count'] = n_core
    zero_meta['control_guidance_core_frac'] = float(n_core / max(N, 1))
    zero_meta['control_guidance_support_count'] = n_support
    zero_meta['control_guidance_support_frac'] = float(n_support / max(N, 1))
    if n_support < 8 or n_core < 8:
        return None, zero_meta

    core_norms = np.linalg.norm(g[core_idx], axis=1)
    active_core = core_norms > 1e-10
    n_active_core = int(active_core.sum())
    if n_active_core < 8:
        return None, zero_meta

    if bool(cfg.get('normalize', True)):
        scale = float(np.median(core_norms[active_core]))
        scale = max(scale, eps)
        g /= scale

    x_sup = x[support_idx]
    g_seed = np.zeros((n_support, 3), dtype=np.float32)
    support_lookup = np.full(N, -1, dtype=np.int64)
    support_lookup[support_idx] = np.arange(n_support, dtype=np.int64)
    core_local = support_lookup[core_idx]
    g_seed[core_local] = g[core_idx]

    shell_lift_weight = float(cfg.get('shell_lift_weight', 0.0))
    shell_lift_k = int(cfg.get('shell_lift_k', 24))
    shell_lift_diff_iters = int(cfg.get('shell_lift_diffusion_iters', 1))
    if shell_lift_weight > 0.0 and shell_lift_diff_iters >= 0 and n_support >= 8:
        k_eff_shell = min(max(shell_lift_k + 1, 2), n_support)
        _, idx_shell = cKDTree(x_sup).query(x_sup, k=k_eff_shell)
        nbr_shell = idx_shell[:, 1:] if idx_shell.ndim == 2 and idx_shell.shape[1] > 1 else idx_shell[:, :1]
        g_shell = g_seed.copy()
        for _ in range(max(shell_lift_diff_iters, 0)):
            g_shell = g_shell[nbr_shell].mean(axis=1).astype(np.float32)
        g_sup = g_seed + shell_lift_weight * g_shell
    else:
        g_sup = g_seed

    sup_norms = np.linalg.norm(g_sup, axis=1)
    active_sup = sup_norms > 1e-10
    n_active = int(active_sup.sum())
    if n_active < 8:
        return None, zero_meta

    smooth_k = int(cfg.get('smooth_k', 64))
    diff_iters = int(cfg.get('diffusion_iters', 2))
    k_eff = min(max(smooth_k + 1, 2), n_support)
    _, idx = cKDTree(x_sup).query(x_sup, k=k_eff)
    nbr = idx[:, 1:] if idx.ndim == 2 and idx.shape[1] > 1 else idx[:, :1]

    g_diff = g_sup.copy()
    for _ in range(max(diff_iters, 0)):
        g_diff = g_diff[nbr].mean(axis=1).astype(np.float32)

    dx = x_sup[nbr] - x_sup[:, None, :]
    dg = g_diff[nbr] - g_diff[:, None, :]
    dxs = np.maximum(np.sum(dx ** 2, axis=2, keepdims=True), eps)
    J = (dg[:, :, :, None] * dx[:, :, None, :] / dxs[:, :, :, None]).mean(axis=1)

    if bool(cfg.get('symmetrize_tensor', True)):
        J = 0.5 * (J + J.transpose(0, 2, 1))

    focus_count = 0
    focus_frac = 0.0
    if focus_mask is not None:
        focus_mask = np.asarray(focus_mask, dtype=bool).reshape(-1)
        if focus_mask.shape[0] == N:
            focus_local = focus_mask[support_idx]
            focus_count = int(focus_local.sum())
            focus_frac = float(focus_count / max(n_support, 1))
            focus_boost = float(cfg.get('ear_focus_boost', 0.0))
            if focus_count > 0 and focus_boost > 0:
                scale = 1.0 + focus_boost
                g_diff[focus_local] *= scale
                J[focus_local] *= scale

    dLdx_weight = float(cfg.get('dLdx_weight', 1.0))
    dLdF_weight = float(cfg.get('dLdF_weight', 0.15))
    dLdx_pen_sup = (dLdx_weight * g_diff).astype(np.float32)
    dLdF_pen_sup = (dLdF_weight * J).astype(np.float32)

    clip_dx = float(cfg.get('clip_dLdx_norm', 0.25))
    if clip_dx > 0:
        dx_norm = np.linalg.norm(dLdx_pen_sup, axis=1, keepdims=True)
        dLdx_pen_sup *= np.minimum(1.0, clip_dx / np.maximum(dx_norm, eps))

    clip_dF = float(cfg.get('clip_dLdF_norm', 0.10))
    if clip_dF > 0:
        dF_norm = np.linalg.norm(dLdF_pen_sup.reshape(n_support, -1), axis=1, keepdims=True)
        dLdF_pen_sup *= np.minimum(1.0, clip_dF / np.maximum(dF_norm, eps))[:, None]

    dLdx_pen = np.zeros((N, 3), dtype=np.float32)
    dLdF_pen = np.zeros((N, 3, 3), dtype=np.float32)
    dLdx_pen[support_idx] = dLdx_pen_sup
    dLdF_pen[support_idx] = dLdF_pen_sup

    meta = {
        'control_guidance_applied': 1,
        'control_guidance_active_count': n_active,
        'control_guidance_active_frac': float(n_active / max(n_support, 1)),
        'control_guidance_core_count': n_core,
        'control_guidance_core_frac': float(n_core / max(N, 1)),
        'control_guidance_support_count': n_support,
        'control_guidance_support_frac': float(n_support / max(N, 1)),
        'control_guidance_dLdx_norm': float(np.linalg.norm(dLdx_pen)),
        'control_guidance_dLdF_norm': float(np.linalg.norm(dLdF_pen)),
        'control_guidance_dLdx_max': float(np.linalg.norm(dLdx_pen, axis=1).max()),
        'control_guidance_dLdF_max': float(np.linalg.norm(dLdF_pen.reshape(N, -1), axis=1).max()),
        'control_guidance_focus_count': focus_count,
        'control_guidance_focus_frac': focus_frac,
        'control_guidance_shell_lift_weight': shell_lift_weight,
        'control_guidance_shell_lift_k': int(min(max(shell_lift_k, 1), max(n_support - 1, 1))),
        'control_guidance_shell_lift_diffusion_iters': int(shell_lift_diff_iters),
        'control_guidance_partner_k': int(k_eff - 1),
        'control_guidance_diffusion_iters': int(diff_iters),
    }
    return {'dLdF': dLdF_pen, 'dLdx': dLdx_pen}, meta


__all__ = ['build_control_guidance_penalty']
