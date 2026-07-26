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


def _apply_patchwise_balance(render_grad, phys_grad, patch_ids, cfg):
    """Locally rebalance render guidance against shell physics gradients."""
    eps = 1e-8
    scale_min = float(cfg.get('patch_scale_min', 0.25))
    scale_max = float(cfg.get('patch_scale_max', 4.0))
    use_pcgrad = bool(cfg.get('pcgrad', True))
    min_active = int(cfg.get('patch_min_active', 4))

    g_out = np.asarray(render_grad, dtype=np.float32).copy()
    p_out = np.asarray(phys_grad, dtype=np.float32)
    patch_ids = np.asarray(patch_ids, dtype=np.int32).reshape(-1)
    scales = np.ones((g_out.shape[0],), dtype=np.float32)
    patch_scale_values = []
    conflict_patches = 0

    for patch_id in np.unique(patch_ids):
        mask = patch_ids == patch_id
        if int(mask.sum()) < min_active:
            continue

        g_patch = g_out[mask]
        p_patch = p_out[mask]
        g_norm = np.linalg.norm(g_patch, axis=1)
        p_norm = np.linalg.norm(p_patch, axis=1)
        g_active = g_norm > 1e-10
        p_active = p_norm > 1e-10
        if int(g_active.sum()) < min_active or int(p_active.sum()) < min_active:
            continue

        s = float(np.median(p_norm[p_active]) / max(np.median(g_norm[g_active]), eps))
        s = float(np.clip(s, scale_min, scale_max))
        g_patch = (s * g_patch).astype(np.float32)
        scales[mask] = s
        patch_scale_values.append(s)

        if use_pcgrad:
            dot = float(np.sum(g_patch * p_patch))
            if dot < 0.0:
                denom = float(np.sum(p_patch * p_patch)) + eps
                g_patch = (g_patch - (dot / denom) * p_patch).astype(np.float32)
                conflict_patches += 1

        g_out[mask] = g_patch

    if patch_scale_values:
        scale_arr = np.asarray(patch_scale_values, dtype=np.float32)
        meta = {
            'control_guidance_patch_balanced': 1,
            'control_guidance_patch_count': int(scale_arr.size),
            'control_guidance_patch_scale_mean': float(scale_arr.mean()),
            'control_guidance_patch_scale_min': float(scale_arr.min()),
            'control_guidance_patch_scale_max': float(scale_arr.max()),
            'control_guidance_patch_conflict_count': int(conflict_patches),
        }
    else:
        meta = {
            'control_guidance_patch_balanced': 0,
            'control_guidance_patch_count': 0,
            'control_guidance_patch_scale_mean': 1.0,
            'control_guidance_patch_scale_min': 1.0,
            'control_guidance_patch_scale_max': 1.0,
            'control_guidance_patch_conflict_count': 0,
        }
    return g_out, meta


def _apply_global_pcgrad(render_grad, phys_grad):
    eps = 1e-8
    g_out = np.asarray(render_grad, dtype=np.float32).copy()
    p_out = np.asarray(phys_grad, dtype=np.float32)
    dot = float(np.sum(g_out * p_out))
    if dot < 0.0:
        denom = float(np.sum(p_out * p_out)) + eps
        g_out = (g_out - (dot / denom) * p_out).astype(np.float32)
        return g_out, 1
    return g_out, 0


def premerge_surface_obs_gradient(grad_x, phys_grad_x, cfg, support_mask=None, surface_graph=None):
    """
    Rebalance the current episode's diffused render gradient against the current
    physics gradient before storing it for the next episode.

    This avoids relying purely on previous-episode physics gradients when
    render guidance is converted into the next control penalty.
    """
    g = np.asarray(grad_x, dtype=np.float32).copy()
    zero_meta = {
        'same_step_merge_applied': 0,
        'same_step_merge_support_count': 0,
        'same_step_merge_patch_count': 0,
        'same_step_merge_scale_mean': 1.0,
        'same_step_merge_scale_min': 1.0,
        'same_step_merge_scale_max': 1.0,
        'same_step_merge_conflict_count': 0,
    }
    if phys_grad_x is None or g.ndim != 2 or g.shape[1] != 3:
        return g, zero_meta

    p = np.asarray(phys_grad_x, dtype=np.float32)
    if p.shape != g.shape:
        return g, zero_meta

    support = np.ones((g.shape[0],), dtype=bool)
    if support_mask is not None:
        sm = np.asarray(support_mask, dtype=bool).reshape(-1)
        if sm.shape[0] == g.shape[0] and sm.any():
            support = sm
    support_idx = np.flatnonzero(support)
    zero_meta['same_step_merge_support_count'] = int(support_idx.size)
    if support_idx.size < 8:
        return g, zero_meta

    g_sup = g[support_idx]
    p_sup = p[support_idx]
    g_norm = np.linalg.norm(g_sup, axis=1)
    p_norm = np.linalg.norm(p_sup, axis=1)
    if int((g_norm > 1e-10).sum()) < 8 or int((p_norm > 1e-10).sum()) < 8:
        return g, zero_meta

    patch_ids = None
    if surface_graph is not None:
        graph_idx = np.asarray(surface_graph.get('indices', []), dtype=np.int64).reshape(-1)
        graph_patch_ids = np.asarray(surface_graph.get('patch_ids', []), dtype=np.int32).reshape(-1)
        if graph_idx.shape[0] == support_idx.size and np.array_equal(graph_idx, support_idx) and graph_patch_ids.shape[0] == support_idx.size:
            patch_ids = graph_patch_ids

    if patch_ids is not None and bool(cfg.get('patch_balance', True)):
        g_sup, patch_meta = _apply_patchwise_balance(g_sup, p_sup, patch_ids, cfg)
        meta = {
            'same_step_merge_applied': 1,
            'same_step_merge_support_count': int(support_idx.size),
            'same_step_merge_patch_count': int(patch_meta['control_guidance_patch_count']),
            'same_step_merge_scale_mean': float(patch_meta['control_guidance_patch_scale_mean']),
            'same_step_merge_scale_min': float(patch_meta['control_guidance_patch_scale_min']),
            'same_step_merge_scale_max': float(patch_meta['control_guidance_patch_scale_max']),
            'same_step_merge_conflict_count': int(patch_meta['control_guidance_patch_conflict_count']),
        }
    elif bool(cfg.get('pcgrad', True)):
        g_sup, conflict = _apply_global_pcgrad(g_sup, p_sup)
        meta = {
            'same_step_merge_applied': 1,
            'same_step_merge_support_count': int(support_idx.size),
            'same_step_merge_patch_count': 1,
            'same_step_merge_scale_mean': 1.0,
            'same_step_merge_scale_min': 1.0,
            'same_step_merge_scale_max': 1.0,
            'same_step_merge_conflict_count': int(conflict),
        }
    else:
        return g, zero_meta

    g[support_idx] = g_sup
    return g, meta


def build_control_guidance_penalty(x, grad_x, cfg, focus_mask=None, core_mask=None, support_mask=None, surface_graph=None, phys_grad_x=None):
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
        'control_guidance_partner_k': int(cfg.get('smooth_k', 64)),
        'control_guidance_diffusion_iters': int(cfg.get('diffusion_iters', 2)),
        'control_guidance_patch_balanced': 0,
        'control_guidance_patch_count': 0,
        'control_guidance_patch_scale_mean': 1.0,
        'control_guidance_patch_scale_min': 1.0,
        'control_guidance_patch_scale_max': 1.0,
        'control_guidance_patch_conflict_count': 0,
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

    graph_nbr = None
    graph_w = None
    graph_patch_ids = None
    if surface_graph is not None:
        graph_idx = np.asarray(surface_graph.get('indices', []), dtype=np.int64).reshape(-1)
        if graph_idx.shape[0] == n_support and np.array_equal(graph_idx, support_idx):
            graph_nbr = np.asarray(surface_graph.get('neighbors'), dtype=np.int64)
            graph_w = np.asarray(surface_graph.get('weights'), dtype=np.float32)
            graph_patch_ids = np.asarray(surface_graph.get('patch_ids', []), dtype=np.int32).reshape(-1)

    g_sup = g_seed

    sup_norms = np.linalg.norm(g_sup, axis=1)
    active_sup = sup_norms > 1e-10
    n_active = int(active_sup.sum())
    if n_active < 8:
        return None, zero_meta

    smooth_k = int(cfg.get('smooth_k', 64))
    diff_iters = int(cfg.get('diffusion_iters', 2))
    if graph_nbr is not None and graph_w is not None:
        k_eff = min(max(smooth_k, 1), graph_nbr.shape[1])
        nbr = graph_nbr[:, :k_eff]
        w_diff = graph_w[:, :k_eff]
        w_diff = w_diff / np.maximum(w_diff.sum(axis=1, keepdims=True), eps)
    else:
        k_eff = min(max(smooth_k + 1, 2), n_support)
        _, idx = cKDTree(x_sup).query(x_sup, k=k_eff)
        nbr = idx[:, 1:] if idx.ndim == 2 and idx.shape[1] > 1 else idx[:, :1]
        w_diff = np.full((n_support, nbr.shape[1]), 1.0 / max(nbr.shape[1], 1), dtype=np.float32)

    g_diff = g_sup.copy()
    for _ in range(max(diff_iters, 0)):
        g_diff = (w_diff[:, :, None] * g_diff[nbr]).sum(axis=1).astype(np.float32)

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

    patch_meta = {
        'control_guidance_patch_balanced': 0,
        'control_guidance_patch_count': 0,
        'control_guidance_patch_scale_mean': 1.0,
        'control_guidance_patch_scale_min': 1.0,
        'control_guidance_patch_scale_max': 1.0,
        'control_guidance_patch_conflict_count': 0,
    }
    if phys_grad_x is not None and graph_patch_ids is not None and graph_patch_ids.shape[0] == n_support:
        phys_grad_x = np.asarray(phys_grad_x, dtype=np.float32)
        if phys_grad_x.shape == (N, 3) and bool(cfg.get('patch_balance', True)):
            g_diff, patch_meta = _apply_patchwise_balance(g_diff, phys_grad_x[support_idx], graph_patch_ids, cfg)

    dx = x_sup[nbr] - x_sup[:, None, :]
    dg = g_diff[nbr] - g_diff[:, None, :]
    dxs = np.maximum(np.sum(dx ** 2, axis=2, keepdims=True), eps)
    J = (dg[:, :, :, None] * dx[:, :, None, :] / dxs[:, :, :, None]).mean(axis=1)

    if bool(cfg.get('symmetrize_tensor', True)):
        J = 0.5 * (J + J.transpose(0, 2, 1))

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
        'control_guidance_partner_k': int(k_eff if graph_nbr is not None else (k_eff - 1)),
        'control_guidance_diffusion_iters': int(diff_iters),
    }
    meta.update(patch_meta)
    return {'dLdF': dLdF_pen, 'dLdx': dLdx_pen}, meta


__all__ = ['build_control_guidance_penalty', 'premerge_surface_obs_gradient']
