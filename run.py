"""
run.py — Decoupled Render-Guided Physics Morphing

3-Stage Pipeline:
  A. Physics Rollout — MPM/EndLayerMassLoss (C++ Adam, physics-only)
  B. Observation-Guided Correction — coarse deformation field optimized by render loss
  C. Immediate Mass-Loss Gate — reject correction if the target-grid mass loss jumps too much

Usage:
  Single:  python run.py -c configs/experiment.yaml [--png]
  Batch:   python run.py -c configs/batch.yaml [--png] [--skip-existing]
"""

import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse, json, numpy as np, yaml
from pathlib import Path
from scipy.spatial import cKDTree

from sampling import default_cfg
from sampling.utils.config_adapter import adapt_config
from utils.physics_utils import build_opt_input, initialize_point_clouds, initialize_grids, initialize_comp_graph
from utils.rendering_utils import setup_renderer, generate_target_observation
from utils.training_loop import run_episode, isochoric_project
from utils.visualize import create_episode_visualization, create_per_view_visualization
from utils.deformation_field import CoarseDeformationField
from utils.alpha_losses import compute_dt_map
from utils.surface_utils import build_fixed_surface_mask
from utils.control_guidance import build_control_guidance_penalty


def compute_sigma0(pos, scale=0.5):
    dd, _ = cKDTree(pos).query(pos, k=2)
    return float(dd[:,1].mean()) * scale


def _deep_update(dst, src):
    """Recursively update nested dictionaries."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def setup_cameras(base_cam, multi_cfg=None):
    """Build camera layouts from a small preset vocabulary."""
    if isinstance(multi_cfg, int):
        multi_cfg = {'num_cameras': multi_cfg}
    multi_cfg = multi_cfg or {}

    lookat = base_cam.get('lookat', {})
    eye = np.array(lookat.get('eye', [20, -25, 12.5]))
    target = np.array(lookat.get('target', [0, 0, 0]))
    offset = eye - target
    dist = float(np.linalg.norm(offset))
    base_elev = np.degrees(np.arctan2(offset[2], np.sqrt(offset[0]**2 + offset[1]**2)))
    base_azim = np.degrees(np.arctan2(offset[1], offset[0]))
    preset = str(multi_cfg.get('preset', 'ring')).lower()
    num_cameras = int(multi_cfg.get('num_cameras', 8))
    low_elev_offset = float(multi_cfg.get('low_elev_offset', 0.0))
    high_elev_offset = float(multi_cfg.get('high_elev_offset', 30.0))
    top_elev = float(multi_cfg.get('top_elev', 85.0))
    high_azim_offset = float(multi_cfg.get('high_azim_offset', 0.0))
    top_azim = float(multi_cfg.get('top_azim', base_azim))
    num_low = int(multi_cfg.get('num_low', max(1, num_cameras // 2)))
    num_high = int(multi_cfg.get('num_high', max(1, num_cameras - num_low)))
    ring_view_weight = float(multi_cfg.get('ring_view_weight', 1.0))
    low_view_weight = float(multi_cfg.get('low_view_weight', 1.0))
    high_view_weight = float(multi_cfg.get('high_view_weight', 1.0))
    top_view_weight = float(multi_cfg.get('top_view_weight', 1.0))

    def cam(elev, azim):
        e, a = np.radians(elev), np.radians(azim)
        pos = target + dist * np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
        c = base_cam.copy()
        c['lookat'] = {'eye': pos.tolist(), 'target': target.tolist(), 'up': [0, 0, 1]}
        return c, pos

    def add_ring(elev, count, prefix, weight, azim_offset=0.0):
        step = 360.0 / max(count, 1)
        for i in range(count):
            c, e = cam(elev, base_azim + azim_offset + step * i)
            cams.append(c)
            eyes.append(e)
            labels.append(f'{prefix}-{i}')
            weights.append(weight)

    cams, eyes, labels, weights = [], [], [], []
    if preset == 'ring':
        add_ring(base_elev + low_elev_offset, num_cameras, 'Ring', ring_view_weight)
        summary = f"{num_cameras} ring"
    elif preset == 'ring_top':
        add_ring(base_elev + low_elev_offset, num_cameras, 'Ring', ring_view_weight)
        c, e = cam(top_elev, top_azim)
        cams.append(c); eyes.append(e); labels.append('Top')
        weights.append(top_view_weight)
        summary = f"{num_cameras} ring + top"
    elif preset == 'dual_ring':
        add_ring(base_elev + low_elev_offset, num_low, 'Low', low_view_weight)
        add_ring(base_elev + high_elev_offset, num_high, 'High', high_view_weight, azim_offset=high_azim_offset)
        summary = f"{num_low} low + {num_high} high"
    elif preset == 'dual_ring_top':
        add_ring(base_elev + low_elev_offset, num_low, 'Low', low_view_weight)
        add_ring(base_elev + high_elev_offset, num_high, 'High', high_view_weight, azim_offset=high_azim_offset)
        c, e = cam(top_elev, top_azim)
        cams.append(c); eyes.append(e); labels.append('Top')
        weights.append(top_view_weight)
        summary = f"{num_low} low + {num_high} high + top"
    else:
        raise ValueError(f"Unknown multi_view preset: {preset}")

    print(f"  Cameras: {summary} (preset={preset}, base_elev={base_elev:.0f}°)")
    return cams, np.array(eyes), labels, np.array(weights, dtype=np.float32)


# ─── Stage B: Observation-Guided Correction ──────────────────────────────

def _make_field_opt_cfg(src, fallback, prefix=''):
    def pick(name, default):
        return src.get(f'{prefix}{name}', src.get(name, fallback.get(name, default)))

    return {
        'lr': float(pick('lr', 0.01)),
        'num_iters': int(pick('num_iters', 3)),
        'w_smooth': float(pick('w_smooth', 0.1)),
        'w_l2': float(pick('w_l2', 0.01)),
        'max_disp': float(pick('max_disp', 1.0)),
        'symmetry_strength': float(pick('symmetry_strength', 0.0)),
        'symmetry_start_iter': int(pick('symmetry_start_iter', 0)),
        'grad_symmetry_strength': float(pick('grad_symmetry_strength', 0.0)),
        'grad_symmetry_start_iter': int(pick('grad_symmetry_start_iter', 0)),
        'ear_grad_boost': float(pick('ear_grad_boost', 0.0)),
        'ear_grad_start_iter': int(pick('ear_grad_start_iter', 0)),
        'ear_focus_height_frac': float(pick('ear_focus_height_frac', 0.58)),
        'ear_focus_side_frac': float(pick('ear_focus_side_frac', 0.18)),
        'ear_focus_side_max_frac': float(pick('ear_focus_side_max_frac', 0.50)),
        'ear_focus_height_axis': int(pick('ear_focus_height_axis', 2)),
        'ear_focus_lateral_axis': int(pick('ear_focus_lateral_axis', 0)),
    }


def _build_mirror_partner_indices(x, axis, center=None, active_mask=None):
    """Nearest-neighbor mirror correspondences for particle-space gradient symmetrization."""
    N = x.shape[0]
    partner_idx = np.arange(N, dtype=np.int64)
    if N == 0:
        return partner_idx, {'mirror_partner_mean_dist': 0.0, 'mirror_partner_max_dist': 0.0}

    axis = int(axis)
    center = float(center) if center is not None else float(0.5 * (x[:, axis].min() + x[:, axis].max()))
    if active_mask is None:
        active_mask = np.ones((N,), dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    active_ids = np.flatnonzero(active_mask)
    if active_ids.size == 0:
        return partner_idx, {'mirror_partner_mean_dist': 0.0, 'mirror_partner_max_dist': 0.0}

    active_pts = x[active_ids]
    mirrored_pts = active_pts.copy()
    mirrored_pts[:, axis] = 2.0 * center - mirrored_pts[:, axis]

    dist, nn = cKDTree(active_pts).query(mirrored_pts, k=1)
    partner_idx[active_ids] = active_ids[np.asarray(nn, dtype=np.int64)]

    return partner_idx, {
        'mirror_partner_mean_dist': float(np.asarray(dist, dtype=np.float32).mean()),
        'mirror_partner_max_dist': float(np.asarray(dist, dtype=np.float32).max()),
    }


def _apply_mirror_gradient_symmetry(grad_x, partner_idx, axis, strength):
    """
    Hard-project or softly blend particle-space gradients into a mirror-symmetric subspace.

    For a mirror across `axis`:
      g_axis(-x) = -g_axis(x)
      g_other(-x) =  g_other(x)
    """
    if strength <= 0:
        return grad_x

    axis = int(axis)
    mirrored = grad_x[partner_idx].copy()
    mirrored[:, axis] *= -1.0
    projected = 0.5 * (grad_x + mirrored)
    return ((1.0 - strength) * grad_x + strength * projected).astype(np.float32)


def _build_ear_focus_mask(
    x,
    active_mask=None,
    lateral_axis=0,
    height_axis=2,
    height_frac=0.58,
    side_frac=0.18,
    side_max_frac=0.50,
    center=None,
):
    """
    Build a bilateral upper-side mask that biases correction toward two-ear splitting.

    The intent is to amplify gradients only on particles that already lie in the
    plausible ear support region: upper part of the bunny and away from the centerline.
    """
    N = x.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=bool), {'ear_focus_count': 0, 'ear_focus_frac': 0.0}

    if active_mask is None:
        active_mask = np.ones((N,), dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    lateral_axis = int(lateral_axis)
    height_axis = int(height_axis)
    bbox_min = x.min(axis=0)
    bbox_max = x.max(axis=0)
    span = np.maximum(bbox_max - bbox_min, 1e-6)
    z_norm = (x[:, height_axis] - bbox_min[height_axis]) / span[height_axis]

    center = float(center) if center is not None else float(0.5 * (bbox_min[lateral_axis] + bbox_max[lateral_axis]))
    side_dist = np.abs(x[:, lateral_axis] - center) / span[lateral_axis]

    mask = (
        active_mask
        & (z_norm >= float(height_frac))
        & (side_dist >= float(side_frac))
        & (side_dist <= float(side_max_frac))
    )

    return mask, {
        'ear_focus_count': int(mask.sum()),
        'ear_focus_frac': float(mask.mean()),
    }


def _prepare_render_force_penalty(x, grad_x, force_cfg):
    """
    Convert a renderer-derived particle gradient into a physics-side penalty signal.

    This keeps the physics rollout as the main optimizer, but lets the renderer act
    like a weak, spatially-shaped external force on the next episode.
    """
    if grad_x is None or x is None or not bool(force_cfg.get('enabled', False)):
        return None, {
            'render_force_applied': 0,
            'render_force_norm': 0.0,
            'render_force_focus_frac': 0.0,
            'render_force_focus_count': 0,
        }

    g = np.asarray(grad_x, dtype=np.float32).copy()
    N = g.shape[0]

    # Normalize first so the injected signal behaves like a force direction field
    # rather than an unbounded episode-dependent gradient magnitude.
    if bool(force_cfg.get('normalize', True)):
        norms = np.linalg.norm(g, axis=1)
        valid = norms > 1e-8
        if valid.any():
            scale = float(np.median(norms[valid]))
            scale = max(scale, 1e-8)
            g /= scale

    # Optional bunny-specific bilateral upper-ear emphasis.
    ear_boost = float(force_cfg.get('ear_boost', 0.0))
    focus_meta = {'ear_focus_count': 0, 'ear_focus_frac': 0.0}
    if ear_boost > 0.0:
        focus_mask, focus_meta = _build_ear_focus_mask(
            x,
            active_mask=None,
            lateral_axis=int(force_cfg.get('ear_focus_lateral_axis', 0)),
            height_axis=int(force_cfg.get('ear_focus_height_axis', 2)),
            height_frac=float(force_cfg.get('ear_focus_height_frac', 0.58)),
            side_frac=float(force_cfg.get('ear_focus_side_frac', 0.18)),
            side_max_frac=float(force_cfg.get('ear_focus_side_max_frac', 0.50)),
            center=force_cfg.get('ear_focus_center', None),
        )
        if focus_meta['ear_focus_count'] > 0:
            g[focus_mask] *= (1.0 + ear_boost)

    # Per-particle clipping keeps the injected signal weak and force-like.
    clip_norm = float(force_cfg.get('clip_norm', 0.2))
    if clip_norm > 0:
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        g *= np.minimum(1.0, clip_norm / np.maximum(norms, 1e-8))

    return {
        'dLdF': np.zeros((N, 3, 3), dtype=np.float32),
        'dLdx': g.astype(np.float32),
    }, {
        'render_force_applied': 1,
        'render_force_norm': float(np.linalg.norm(g)),
        'render_force_focus_frac': float(focus_meta['ear_focus_frac']),
        'render_force_focus_count': int(focus_meta['ear_focus_count']),
    }


def _optimize_field(
    stage_name,
    base_x,
    field,
    F_e,
    sigma_corr,
    opacity,
    renderers,
    campos_list,
    targets,
    rcfg,
    color,
    lcfg,
    Fp,
    opt_cfg,
    target_dt_maps=None,
    target_depths=None,
    target_curvatures=None,
    target_concavities=None,
    view_weights=None,
    particle_mask=None,
):
    from utils.training_loop import compute_multiview_gradients, compute_multiview_metrics

    lr = float(opt_cfg.get('lr', 0.01))
    num_iters = int(opt_cfg.get('num_iters', 3))
    w_smooth = float(opt_cfg.get('w_smooth', 0.1))
    w_l2 = float(opt_cfg.get('w_l2', 0.01))
    max_disp = float(opt_cfg.get('max_disp', 1.0))
    symmetry_strength = float(opt_cfg.get('symmetry_strength', 0.0))
    symmetry_start_iter = int(opt_cfg.get('symmetry_start_iter', 0))
    symmetry_axis = opt_cfg.get('symmetry_axis', None)
    grad_symmetry_strength = float(opt_cfg.get('grad_symmetry_strength', 0.0))
    grad_symmetry_start_iter = int(opt_cfg.get('grad_symmetry_start_iter', 0))
    grad_symmetry_axis = opt_cfg.get('grad_symmetry_axis', symmetry_axis)
    grad_symmetry_center = opt_cfg.get('grad_symmetry_center', None)
    ear_grad_boost = float(opt_cfg.get('ear_grad_boost', 0.0))
    ear_grad_start_iter = int(opt_cfg.get('ear_grad_start_iter', 0))

    best_obj = float('inf')
    best_u = field.u.copy()
    last_grad = None
    grad_sym_meta = {'mirror_partner_mean_dist': 0.0, 'mirror_partner_max_dist': 0.0}
    grad_partner_idx = None
    if grad_symmetry_axis is not None and grad_symmetry_strength > 0:
        grad_partner_idx, grad_sym_meta = _build_mirror_partner_indices(
            base_x,
            axis=int(grad_symmetry_axis),
            center=grad_symmetry_center,
            active_mask=getattr(field, 'active_mask', None),
        )

    ear_focus_meta = {'ear_focus_count': 0, 'ear_focus_frac': 0.0}
    ear_focus_mask = None
    if ear_grad_boost > 0:
        ear_focus_mask, ear_focus_meta = _build_ear_focus_mask(
            base_x,
            active_mask=getattr(field, 'active_mask', None),
            lateral_axis=int(opt_cfg.get('ear_focus_lateral_axis', 0)),
            height_axis=int(opt_cfg.get('ear_focus_height_axis', 2)),
            height_frac=float(opt_cfg.get('ear_focus_height_frac', 0.58)),
            side_frac=float(opt_cfg.get('ear_focus_side_frac', 0.18)),
            side_max_frac=float(opt_cfg.get('ear_focus_side_max_frac', 0.50)),
            center=grad_symmetry_center,
        )

    for it in range(num_iters):
        x_def = field.forward(base_x)
        dLdx, rm, _, _ = compute_multiview_gradients(
            x_def, F_e, sigma_corr, opacity, targets, renderers, campos_list,
            rcfg, color, lcfg, Fp, target_dt_maps=target_dt_maps,
            target_depths=target_depths, target_curvatures=target_curvatures, target_concavities=target_concavities, return_alpha_list=False,
            view_weights=view_weights, particle_mask=particle_mask,
        )
        if dLdx is None:
            break
        if grad_partner_idx is not None and it >= grad_symmetry_start_iter:
            dLdx = _apply_mirror_gradient_symmetry(
                dLdx, grad_partner_idx, axis=int(grad_symmetry_axis),
                strength=grad_symmetry_strength,
            )
        if ear_focus_mask is not None and ear_focus_meta['ear_focus_count'] > 0 and it >= ear_grad_start_iter:
            dLdx = dLdx.copy()
            dLdx[ear_focus_mask] *= (1.0 + ear_grad_boost)

        obs_val = float(rm.get('loss_total_obj_mv', rm.get('loss_total_mv', float('inf'))))
        smooth_val = field.field_smoothness_loss()
        l2_val = float((field.u ** 2).sum())
        corr_obj = obs_val + w_smooth * smooth_val + w_l2 * l2_val

        if corr_obj < best_obj:
            best_obj = corr_obj
            best_u = field.u.copy()

        grad_u = field.backward(dLdx)
        last_grad = dLdx
        grad_u += w_smooth * field.field_smoothness_grad()
        grad_u += w_l2 * 2.0 * field.u

        field.update(grad_u, lr=lr)
        if symmetry_axis is not None and symmetry_strength > 0 and it >= symmetry_start_iter:
            field.project_mirror_symmetry(axis=int(symmetry_axis), strength=symmetry_strength)
        field.clip_displacements(max_disp)

        if it == 0 or it == num_iters - 1:
            print(
                f"    [{stage_name}] it={it}: total={obs_val:.6f}, "
                f"smooth={smooth_val:.6f}, l2={l2_val:.6f}, "
                f"active={field.active_fraction():.2f}, max_disp={field.max_displacement():.4f}"
            )

    x_final = field.forward(base_x)
    rm_final, _, _ = compute_multiview_metrics(
        x_final, F_e, sigma_corr, opacity, targets, renderers, campos_list,
        rcfg, color, lcfg, Fp, target_dt_maps=target_dt_maps,
        target_depths=target_depths, target_curvatures=target_curvatures, target_concavities=target_concavities, return_alpha_list=False,
        view_weights=view_weights, particle_mask=particle_mask,
    )
    final_obs = float(rm_final.get('loss_total_obj_mv', rm_final.get('loss_total_mv', float('inf'))))
    final_smooth = field.field_smoothness_loss()
    final_l2 = float((field.u ** 2).sum())
    final_obj = final_obs + w_smooth * final_smooth + w_l2 * final_l2
    if final_obj < best_obj:
        best_obj = final_obj
        best_u = field.u.copy()

    field.u = best_u
    x_best = field.forward(base_x)
    rm_best, _, _ = compute_multiview_metrics(
        x_best, F_e, sigma_corr, opacity, targets, renderers, campos_list,
        rcfg, color, lcfg, Fp, target_dt_maps=target_dt_maps,
        target_depths=target_depths, target_curvatures=target_curvatures, target_concavities=target_concavities, return_alpha_list=False,
        view_weights=view_weights, particle_mask=particle_mask,
    )

    return x_best, rm_best, {
        'objective': float(best_obj),
        'max_disp': float(field.max_displacement()),
        'iters': int(num_iters),
        'active_frac': float(field.active_fraction()),
        'grad_symmetry_strength': float(grad_symmetry_strength),
        'grad_symmetry_axis': int(grad_symmetry_axis) if grad_symmetry_axis is not None else -1,
        'mirror_partner_mean_dist': float(grad_sym_meta['mirror_partner_mean_dist']),
        'mirror_partner_max_dist': float(grad_sym_meta['mirror_partner_max_dist']),
        'ear_grad_boost': float(ear_grad_boost),
        'ear_focus_count': int(ear_focus_meta['ear_focus_count']),
        'ear_focus_frac': float(ear_focus_meta['ear_focus_frac']),
    }, last_grad


def _resolve_local_symmetry_axis(local_cfg, bbox_center, seed_pts):
    axis_cfg = str(local_cfg.get('symmetry_axis', 'auto')).lower()
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis_cfg in axis_map:
        return axis_map[axis_cfg]

    candidate_axes = []
    axes_cfg = str(local_cfg.get('symmetry_candidate_axes', 'xy')).lower()
    for ch in axes_cfg:
        if ch in axis_map and axis_map[ch] not in candidate_axes:
            candidate_axes.append(axis_map[ch])
    if not candidate_axes:
        candidate_axes = [0, 1]

    offsets = np.abs(seed_pts.mean(axis=0) - bbox_center)
    return int(max(candidate_axes, key=lambda ax: float(offsets[ax])))


def _compute_local_refine_start_ep(local_cfg, num_eps):
    if 'start_ep' in local_cfg:
        return max(0, int(local_cfg.get('start_ep', 0)))
    start_frac = float(local_cfg.get('start_frac', 0.0))
    if num_eps is None:
        return 0
    return max(0, int(round(start_frac * float(num_eps))))


def _select_local_refine_region(x, grad_x, local_cfg):
    if grad_x is None or x.shape[0] == 0:
        return None

    scores = np.linalg.norm(grad_x, axis=1)
    if not np.isfinite(scores).any() or float(scores.max()) <= 1e-10:
        return None

    bbox_min = x.min(axis=0)
    bbox_max = x.max(axis=0)
    span = np.maximum(bbox_max - bbox_min, 1e-6)
    z_norm = (x[:, 2] - bbox_min[2]) / span[2]

    z_min_frac = float(local_cfg.get('z_min_frac', 0.55))
    score_pct = float(local_cfg.get('score_percentile', 88.0))
    bbox_padding = float(local_cfg.get('bbox_padding', 1.25))
    active_margin = float(local_cfg.get('active_margin', 0.75))
    max_extent_frac = float(local_cfg.get('max_extent_frac', 0.4))
    min_extent_frac = float(local_cfg.get('min_extent_frac', 0.0))
    min_seed = int(local_cfg.get('min_seed_particles', max(64, x.shape[0] // 300)))
    min_active = int(local_cfg.get('min_active_particles', max(128, x.shape[0] // 80)))

    eligible = z_norm >= z_min_frac
    eligible_scores = scores[eligible] if eligible.any() else scores
    thresh = np.percentile(eligible_scores, score_pct)
    seed_mask = eligible & (scores >= thresh)

    if int(seed_mask.sum()) < min_seed:
        relaxed = z_norm >= max(z_min_frac - 0.15, 0.0)
        relaxed_scores = scores[relaxed] if relaxed.any() else scores
        thresh = np.percentile(relaxed_scores, min(score_pct, 80.0))
        seed_mask = relaxed & (scores >= thresh)

    if int(seed_mask.sum()) < min_seed:
        topk = min(max(min_seed, 1), x.shape[0])
        seed_mask = np.zeros((x.shape[0],), dtype=bool)
        seed_mask[np.argsort(scores)[-topk:]] = True

    if not seed_mask.any():
        return None

    seed_pts = x[seed_mask]
    roi_min = seed_pts.min(axis=0) - bbox_padding
    roi_max = seed_pts.max(axis=0) + bbox_padding

    center = 0.5 * (roi_min + roi_max)
    max_half = 0.5 * span * max_extent_frac
    min_half = 0.5 * span * min_extent_frac
    half = np.maximum(0.5 * (roi_max - roi_min), min_half)
    half = np.minimum(half, np.maximum(max_half, 1e-3))
    roi_min = np.maximum(center - half, bbox_min - bbox_padding)
    roi_max = np.minimum(center + half, bbox_max + bbox_padding)

    support_boxes = [(roi_min.copy(), roi_max.copy())]
    bilateral = bool(local_cfg.get('bilateral', False))
    symmetry_axis = -1
    symmetry_center = 0.0
    if bilateral:
        symmetry_axis = _resolve_local_symmetry_axis(local_cfg, 0.5 * (bbox_min + bbox_max), seed_pts)
        symmetry_center = float(local_cfg.get('symmetry_center', 0.5 * (bbox_min[symmetry_axis] + bbox_max[symmetry_axis])))
        mirror_min = roi_min.copy()
        mirror_max = roi_max.copy()
        mirror_min[symmetry_axis] = 2.0 * symmetry_center - roi_max[symmetry_axis]
        mirror_max[symmetry_axis] = 2.0 * symmetry_center - roi_min[symmetry_axis]
        mirror_min = np.maximum(mirror_min, bbox_min - bbox_padding)
        mirror_max = np.minimum(mirror_max, bbox_max + bbox_padding)
        support_boxes.append((mirror_min, mirror_max))
        roi_min = np.minimum(roi_min, mirror_min)
        roi_max = np.maximum(roi_max, mirror_max)

    active_mask = np.zeros((x.shape[0],), dtype=bool)
    for box_min, box_max in support_boxes:
        active_mask |= np.all((x >= (box_min - active_margin)) & (x <= (box_max + active_margin)), axis=1)
    if int(active_mask.sum()) < min_active:
        expand = active_margin + 0.5 * bbox_padding
        active_mask = np.zeros((x.shape[0],), dtype=bool)
        for box_min, box_max in support_boxes:
            active_mask |= np.all((x >= (box_min - expand)) & (x <= (box_max + expand)), axis=1)

    if int(active_mask.sum()) == 0:
        return None

    seed_balance = 0.0
    active_balance = 0.0
    seed_left = seed_right = active_left = active_right = 0
    if bilateral and symmetry_axis >= 0:
        seed_coord = seed_pts[:, symmetry_axis] - symmetry_center
        active_coord = x[active_mask, symmetry_axis] - symmetry_center
        seed_left = int((seed_coord < 0).sum())
        seed_right = int((seed_coord > 0).sum())
        active_left = int((active_coord < 0).sum())
        active_right = int((active_coord > 0).sum())
        seed_balance = abs(seed_left - seed_right) / max(seed_left + seed_right, 1)
        active_balance = abs(active_left - active_right) / max(active_left + active_right, 1)

    return {
        'seed_mask': seed_mask,
        'active_mask': active_mask,
        'bbox_min': roi_min.astype(np.float32),
        'bbox_max': roi_max.astype(np.float32),
        'seed_count': int(seed_mask.sum()),
        'active_count': int(active_mask.sum()),
        'active_frac': float(active_mask.mean()),
        'bbox_diag': float(np.linalg.norm(roi_max - roi_min)),
        'score_mean': float(scores[seed_mask].mean()),
        'z_min_frac': z_min_frac,
        'bilateral': int(bilateral),
        'symmetry_axis': symmetry_axis,
        'symmetry_center': float(symmetry_center),
        'seed_left_count': seed_left,
        'seed_right_count': seed_right,
        'active_left_count': active_left,
        'active_right_count': active_right,
        'seed_balance': float(seed_balance),
        'active_balance': float(active_balance),
    }

def render_correction(
    x_phys, F_e, sigma0, opacity,
    renderers, campos_list, targets, rcfg, color, lcfg, Fp,
    correction_cfg,
    target_dt_maps=None,
    target_depths=None,
    target_curvatures=None,
    target_concavities=None,
    view_weights=None,
    particle_mask=None,
    ep=None,
    num_eps=None,
):
    """
    Stage B: Optimize a coarse deformation field via manual gradient projection.

    1. Deform particles: x' = x + W @ u  (trilinear interp)
    2. Render x' → get dL/dx' from multi-view alpha loss
    3. Project to field: dL/du = W^T @ dL/dx'
    4. Add regularization gradient
    5. Update field: u -= lr * dL/du
    6. Clip displacements (trust-region)

    Returns:
        x_corrected: (N, 3) numpy
        correction_metrics: dict
    """
    N = x_phys.shape[0]
    grid_res = int(correction_cfg.get('grid_res', 8))
    sigma0_factor = float(correction_cfg.get('sigma0_factor', 1.0))
    sigma_corr = sigma0 * sigma0_factor

    if len(renderers) == 0 or len(targets) == 0:
        return x_phys, {}

    # Global coarse field
    global_opt_cfg = _make_field_opt_cfg(correction_cfg, correction_cfg)
    field = CoarseDeformationField(grid_res=grid_res)
    field.set_bbox(x_phys, padding=2.0)
    field.precompute_weights(x_phys)

    x_global, rm_global, global_meta, global_grad = _optimize_field(
        'CorrGlobal', x_phys, field, F_e, sigma_corr, opacity,
        renderers, campos_list, targets, rcfg, color, lcfg, Fp,
        global_opt_cfg, target_dt_maps=target_dt_maps, target_depths=target_depths, target_curvatures=target_curvatures,
        target_concavities=target_concavities,
        view_weights=view_weights, particle_mask=particle_mask,
    )

    x_corrected = x_global
    rm_best = rm_global
    best_obj = global_meta['objective']
    local_meta = {
        'corr_local_enabled': 0,
        'corr_local_applied': 0,
        'corr_local_ready': 0,
        'corr_local_delayed': 0,
        'corr_local_start_ep': 0,
        'corr_local_seed_count': 0,
        'corr_local_active_count': 0,
        'corr_local_active_frac': 0.0,
        'corr_local_bbox_diag': 0.0,
        'corr_local_gain': 0.0,
        'corr_local_bilateral': 0,
        'corr_local_grad_symmetry_strength': 0.0,
        'corr_local_grad_symmetry_axis': -1,
        'corr_local_mirror_partner_mean_dist': 0.0,
        'corr_local_mirror_partner_max_dist': 0.0,
        'corr_local_ear_grad_boost': 0.0,
        'corr_local_ear_focus_count': 0,
        'corr_local_ear_focus_frac': 0.0,
    }

    local_cfg = correction_cfg.get('local_refine', {}) or {}
    if bool(local_cfg.get('enabled', False)):
        local_meta['corr_local_enabled'] = 1
        start_ep = _compute_local_refine_start_ep(local_cfg, num_eps)
        local_meta['corr_local_start_ep'] = int(start_ep)
        local_ready = ep is None or ep >= start_ep
        local_meta['corr_local_ready'] = int(local_ready)
        local_meta['corr_local_delayed'] = int(not local_ready)
        region = _select_local_refine_region(x_global, global_grad, local_cfg) if local_ready else None
        if region is not None and region['active_count'] > 0:
            local_field = CoarseDeformationField(grid_res=int(local_cfg.get('grid_res', grid_res + 4)))
            local_field.bbox_min = region['bbox_min'].copy()
            local_field.bbox_max = region['bbox_max'].copy()
            local_field.precompute_weights(x_global, active_mask=region['active_mask'])

            local_opt_cfg = _make_field_opt_cfg(local_cfg, correction_cfg)
            if region['symmetry_axis'] >= 0 and float(local_cfg.get('symmetry_strength', 0.0)) > 0:
                local_opt_cfg['symmetry_axis'] = int(region['symmetry_axis'])
            if region['symmetry_axis'] >= 0 and float(local_cfg.get('grad_symmetry_strength', 0.0)) > 0:
                local_opt_cfg['grad_symmetry_axis'] = int(region['symmetry_axis'])
                local_opt_cfg['grad_symmetry_center'] = float(region['symmetry_center'])
            x_local, rm_local, local_stage_meta, _ = _optimize_field(
                'CorrLocal', x_global, local_field, F_e, sigma_corr, opacity,
                renderers, campos_list, targets, rcfg, color, lcfg, Fp,
                local_opt_cfg, target_dt_maps=target_dt_maps, target_depths=target_depths, target_curvatures=target_curvatures,
                target_concavities=target_concavities,
                view_weights=view_weights, particle_mask=particle_mask,
            )

            x_corrected = x_local
            rm_best = rm_local
            best_obj = float(local_stage_meta['objective'])
            local_meta.update({
                'corr_local_applied': 1,
                'corr_local_seed_count': region['seed_count'],
                'corr_local_active_count': region['active_count'],
                'corr_local_active_frac': region['active_frac'],
                'corr_local_bbox_diag': region['bbox_diag'],
                'corr_local_score_mean': region['score_mean'],
                'corr_local_z_min_frac': region['z_min_frac'],
                'corr_local_bilateral': region['bilateral'],
                'corr_local_symmetry_axis': int(region['symmetry_axis']),
                'corr_local_symmetry_center': float(region['symmetry_center']),
                'corr_local_symmetry_strength': float(local_opt_cfg.get('symmetry_strength', 0.0)),
                'corr_local_seed_left_count': int(region['seed_left_count']),
                'corr_local_seed_right_count': int(region['seed_right_count']),
                'corr_local_active_left_count': int(region['active_left_count']),
                'corr_local_active_right_count': int(region['active_right_count']),
                'corr_local_seed_balance': float(region['seed_balance']),
                'corr_local_active_balance': float(region['active_balance']),
                'corr_local_grid_res': int(local_field.grid_res),
                'corr_local_max_disp': float(local_stage_meta['max_disp']),
                'corr_local_objective': float(local_stage_meta['objective']),
                'corr_local_gain': float(rm_global.get('loss_total_mv', 0.0) - rm_local.get('loss_total_mv', 0.0)),
                'corr_local_grad_symmetry_strength': float(local_stage_meta.get('grad_symmetry_strength', 0.0)),
                'corr_local_grad_symmetry_axis': int(local_stage_meta.get('grad_symmetry_axis', -1)),
                'corr_local_mirror_partner_mean_dist': float(local_stage_meta.get('mirror_partner_mean_dist', 0.0)),
                'corr_local_mirror_partner_max_dist': float(local_stage_meta.get('mirror_partner_max_dist', 0.0)),
                'corr_local_ear_grad_boost': float(local_stage_meta.get('ear_grad_boost', 0.0)),
                'corr_local_ear_focus_count': int(local_stage_meta.get('ear_focus_count', 0)),
                'corr_local_ear_focus_frac': float(local_stage_meta.get('ear_focus_frac', 0.0)),
            })
            print(
                f"    [CorrLocal] seeds={region['seed_count']}, active={region['active_count']}/{N} "
                f"({100.0 * region['active_frac']:.1f}%), bbox_diag={region['bbox_diag']:.3f}, "
                f"gain={local_meta['corr_local_gain']:+.4f}, bilateral={region['bilateral']}, "
                f"grad_sym={local_meta['corr_local_grad_symmetry_strength']:.2f}, "
                f"ear_boost={local_meta['corr_local_ear_grad_boost']:.2f}"
            )
        elif not local_ready:
            print(f"    [CorrLocal] delayed until ep>={start_ep}")
        else:
            print("    [CorrLocal] skipped (no stable high-error upper-region ROI found)")

    metrics = {
        'corr_loss_total_mv': float(rm_best.get('loss_total_mv', float('inf'))),
        'corr_loss_obj_mv': float(rm_best.get('loss_total_obj_mv', rm_best.get('loss_total_mv', float('inf')))),
        'corr_loss_weighted_mv': float(rm_best.get('loss_weighted_mv', rm_best.get('loss_total_mv', float('inf')))),
        'corr_loss_hardmax_mv': float(rm_best.get('loss_hardmax_mv', 0.0)),
        'corr_loss_topk_mv': float(rm_best.get('loss_topk_mv', 0.0)),
        'corr_loss_bce_mv': float(rm_best.get('loss_bce_mv', float('inf'))),
        'corr_loss_iou_mv': float(rm_best.get('loss_iou_mv', 0.0)),
        'corr_loss_dt_mv': float(rm_best.get('loss_dt_mv', 0.0)),
        'corr_loss_depth_mv': float(rm_best.get('loss_depth_mv', 0.0)),
        'corr_loss_edge_mv': float(rm_best.get('loss_edge_mv', 0.0)),
        'corr_objective': best_obj,
        'corr_max_disp': float(global_meta['max_disp']),
        'corr_iters': int(global_meta['iters']),
        'corr_global_active_frac': float(global_meta['active_frac']),
        'corr_global_grid_res': int(grid_res),
        'corr_sigma0': float(sigma_corr),
    }
    metrics.update(local_meta)

    return x_corrected, metrics


# ─── Stage C: Immediate Mass-Loss Gate ───────────────────────────────────

def immediate_mass_loss_gate(cg, x_phys, x_corrected, pc, tau=0.1):
    """
    Accept correction only if the immediate end-layer mass loss does not
    degrade too much after overwriting the current positions.

    This is an immediate geometric consistency check, not a rollout-aware
    physics validation.

    Args:
        cg: computation graph
        x_phys: (N, 3) positions after physics
        x_corrected: (N, 3) corrected positions
        pc: point cloud object
        tau: max allowed mass-loss increase (fraction)

    Returns:
        x_accepted: (N, 3) — either corrected or original
        accepted: bool
        mass_delta: float
        mass_before: float
        mass_after: float
    """
    mass_before = float(cg.end_layer_mass_loss())

    pos_view = pc.get_positions_view()
    original = pos_view.copy()
    pos_view[:] = x_corrected
    mass_after = float(cg.end_layer_mass_loss())
    pos_view[:] = original

    delta = (mass_after - mass_before) / (abs(mass_before) + 1e-8)

    if delta <= tau:
        return x_corrected, True, float(delta), mass_before, mass_after
    return x_phys, False, float(delta), mass_before, mass_after


def apply_accepted_correction(pc, x_accepted, correction_cfg, F_accept=None):
    """
    Commit an accepted correction to the promoted state.

    By default we zero kinematics and absorb the current render-time total
    deformation into the point cloud to avoid carrying a stale external Fp.
    """
    pc.set_positions(np.ascontiguousarray(x_accepted.astype(np.float32)))

    if correction_cfg.get('reset_kinematics_on_accept', True):
        pc.reset_kinematics()
    else:
        vel_damp = float(correction_cfg.get('velocity_damping', 0.5))
        if vel_damp > 0:
            v = pc.get_velocities_view()
            v[:] *= (1.0 - vel_damp)

    if F_accept is not None and correction_cfg.get('absorb_fp_on_accept', True):
        pc.set_def_grads(np.ascontiguousarray(F_accept.astype(np.float32)))


# ─── Main Loop ───────────────────────────────────────────────────────────

def run_single(cfg, png=False):
    """Run a single experiment with 3-stage decoupled pipeline."""
    out = Path(cfg.get('output_dir', 'output/run')); out.mkdir(parents=True, exist_ok=True)

    # ── Physics init ──────────────────────────────────────────────────────
    import diffmpm_bindings
    opt = build_opt_input(cfg)
    in_pc, tgt_pc = initialize_point_clouds(opt, cfg=cfg)
    in_grid, tgt_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(in_pc, in_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(tgt_pc, tgt_grid)
    cg = initialize_comp_graph(in_pc, in_grid, tgt_grid)

    resume_cfg = cfg.get('resume', {}) or {}
    resume_Fp = None
    if resume_cfg.get('enabled', False):
        ckpt_path = resume_cfg.get('checkpoint_path')
        if not ckpt_path:
            raise ValueError("resume.enabled=true but resume.checkpoint_path is missing")
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = np.load(ckpt_path)
        pc0 = cg.get_point_cloud(0)
        resume_pos = np.asarray(ckpt['positions'], dtype=np.float32)
        resume_vel = np.asarray(ckpt['velocities'], dtype=np.float32) if 'velocities' in ckpt.files else None
        resume_dFc = np.asarray(ckpt['dFc'], dtype=np.float32) if 'dFc' in ckpt.files else None
        resume_Fp = np.asarray(ckpt['Fp'], dtype=np.float32) if 'Fp' in ckpt.files else None

        if resume_pos.shape != np.asarray(pc0.get_positions(), dtype=np.float32).shape:
            raise ValueError(
                f"Resume checkpoint position shape mismatch: {resume_pos.shape} "
                f"vs {np.asarray(pc0.get_positions(), dtype=np.float32).shape}"
            )

        pc0.set_positions(np.ascontiguousarray(resume_pos))
        if resume_vel is not None:
            v = pc0.get_velocities_view()
            if v.shape != resume_vel.shape:
                raise ValueError(f"Resume velocity shape mismatch: {resume_vel.shape} vs {v.shape}")
            v[:] = resume_vel
        if resume_dFc is not None:
            pc0.set_dFc(np.ascontiguousarray(resume_dFc))

        print(
            f"[Resume] Loaded {ckpt_path} "
            f"(N={resume_pos.shape[0]:,}, has_vel={resume_vel is not None}, "
            f"has_dFc={resume_dFc is not None}, has_Fp={resume_Fp is not None})"
        )

    x0 = np.array(in_pc.get_positions(), dtype=np.float32)
    N = x0.shape[0]
    if resume_cfg.get('enabled', False):
        x0 = np.array(cg.get_point_cloud(0).get_positions(), dtype=np.float32)
        N = x0.shape[0]

    # ── Config ────────────────────────────────────────────────────────────
    rcfg = cfg.get('render', {})
    sigma0 = compute_sigma0(x0, float(rcfg.get('sigma0_scale', 0.5))) if rcfg.get('sigma0', 'auto') == 'auto' else float(rcfg['sigma0'])
    opacity = float(rcfg.get('opacity', 0.95))
    color = rcfg.get('particle_color', [0.27, 0.51, 0.71])

    pcfg = cfg.get('plasticity', {})
    lcfg = cfg.get('loss_weights', {})
    cam_cfg = cfg.get('camera', {})
    correction_cfg = cfg.get('correction', {})
    render_force_cfg = cfg.get('render_force', {}) or {}
    surface_cfg = cfg.get('surface_aware', {}) or {}
    control_guidance_cfg = cfg.get('control_guidance', {}) or {}

    eta_v = float(pcfg.get('eta_v', 0.5))
    max_v = float(pcfg.get('max_v', 5.0))
    max_aniso = float(pcfg.get('max_anisotropy', 1.5))
    damping = float(pcfg.get('damping', 0.0))
    use_correction = bool(correction_cfg.get('enabled', False))
    mass_gate_tau = float(correction_cfg.get('mass_gate_tau', correction_cfg.get('acceptance_tau', 0.1)))
    debug_cfg = cfg.get('debug', {}) or {}
    debug_gradient_eps = {int(e) for e in debug_cfg.get('gradient_heatmap_eps', [])}
    render_force_enabled = bool(render_force_cfg.get('enabled', False))
    render_force_start_ep = int(render_force_cfg.get('start_ep', 1))
    control_guidance_enabled = bool(control_guidance_cfg.get('enabled', False))
    control_guidance_start_ep = int(control_guidance_cfg.get('start_ep', 1))

    print(f"[Init] N={N:,}, loss={cg.end_layer_mass_loss():.1f}")
    print(f"[Config] eta_v={eta_v}, correction={'ON' if use_correction else 'OFF'}")

    surface_mask = None
    render_surface_mask = None
    surface_meta = {
        'surface_particle_count': int(N),
        'surface_particle_frac': 1.0,
        'render_surface_particle_count': int(N),
        'render_surface_particle_frac': 1.0,
        'surface_recon_resolution': int(surface_cfg.get('recon_resolution', 64)),
    }
    if bool(surface_cfg.get('enabled', False)):
        recon = build_fixed_surface_mask(
            x0,
            resolution=int(surface_cfg.get('recon_resolution', 64)),
            padding=float(surface_cfg.get('recon_padding', 2.0)),
            sigma=float(surface_cfg.get('recon_sigma', 1.5)),
            level_ratio=float(surface_cfg.get('recon_level_ratio', 0.02)),
            threshold_mult=float(surface_cfg.get('surface_threshold_mult', 1.75)),
            min_surface_frac=float(surface_cfg.get('min_surface_frac', 0.08)),
            max_surface_frac=float(surface_cfg.get('max_surface_frac', 0.40)),
        )
        surface_mask = np.asarray(recon['surface_mask'], dtype=bool)
        render_recon = build_fixed_surface_mask(
            x0,
            resolution=int(surface_cfg.get('recon_resolution', 64)),
            padding=float(surface_cfg.get('recon_padding', 2.0)),
            sigma=float(surface_cfg.get('recon_sigma', 1.5)),
            level_ratio=float(surface_cfg.get('recon_level_ratio', 0.02)),
            threshold_mult=float(surface_cfg.get('render_surface_threshold_mult', surface_cfg.get('surface_threshold_mult', 1.75))),
            min_surface_frac=float(surface_cfg.get('render_min_surface_frac', surface_cfg.get('min_surface_frac', 0.08))),
            max_surface_frac=float(surface_cfg.get('render_max_surface_frac', surface_cfg.get('max_surface_frac', 0.40))),
        )
        render_surface_mask = np.asarray(render_recon['surface_mask'], dtype=bool)
        surface_meta.update({
            'surface_particle_count': int(surface_mask.sum()),
            'surface_particle_frac': float(surface_mask.mean()),
            'render_surface_particle_count': int(render_surface_mask.sum()),
            'render_surface_particle_frac': float(render_surface_mask.mean()),
            'surface_distance_threshold': float(recon.get('surface_distance_threshold', 0.0)),
            'surface_distance_mean': float(recon.get('surface_distance_mean', 0.0)),
            'render_surface_distance_threshold': float(render_recon.get('surface_distance_threshold', 0.0)),
        })
        np.savez_compressed(
            out / 'surface_recon.npz',
            surface_mask=surface_mask.astype(np.uint8),
            render_surface_mask=render_surface_mask.astype(np.uint8),
            verts=np.zeros((0, 3), dtype=np.float32) if recon.get('verts') is None else recon['verts'],
            faces=np.zeros((0, 3), dtype=np.int32) if recon.get('faces') is None else recon['faces'],
            origin=recon.get('origin', np.zeros((3,), dtype=np.float32)),
            spacing=np.asarray([recon.get('spacing', 0.0)], dtype=np.float32),
            level=np.asarray([recon.get('level', 0.0)], dtype=np.float32),
            density_max=np.asarray([recon.get('density_max', 0.0)], dtype=np.float32),
            surface_distance_threshold=np.asarray([recon.get('surface_distance_threshold', 0.0)], dtype=np.float32),
            surface_fraction=np.asarray([recon.get('surface_fraction', 1.0)], dtype=np.float32),
            render_surface_distance_threshold=np.asarray([render_recon.get('surface_distance_threshold', 0.0)], dtype=np.float32),
            render_surface_fraction=np.asarray([render_recon.get('surface_fraction', 1.0)], dtype=np.float32),
        )
        print(
            f"[Surface] enabled, recon={surface_meta['surface_recon_resolution']}^3, "
            f"mask={surface_meta['surface_particle_count']:,}/{N:,} "
            f"({100.0 * surface_meta['surface_particle_frac']:.1f}%), "
            f"render_mask={surface_meta['render_surface_particle_count']:,}/{N:,} "
            f"({100.0 * surface_meta['render_surface_particle_frac']:.1f}%)"
        )
    else:
        render_surface_mask = surface_mask

    # ── Cameras + targets ─────────────────────────────────────────────────
    rs = default_cfg()
    rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))
    sim = cfg.get('simulation', {})
    rs['physics_grid'] = {
        'grid_min': sim.get('grid_min_point', [-16]*3),
        'grid_max': sim.get('grid_max_point', [16]*3),
        'grid_dx': sim.get('grid_dx', 1.0),
    }

    multi_cfg = cfg.get('multi_view', {})
    multi = multi_cfg.get('enabled', False)
    renderers, campos_list, targets, target_dt_maps, target_depths, target_curvatures, target_concavities, cam_eyes, cam_labels = [], [], [], [], [], [], [], np.array([]), []
    view_weights = []

    if multi:
        all_cams, cam_eyes, cam_labels, all_view_weights = setup_cameras(cam_cfg, multi_cfg)
        tpos = np.array(tgt_pc.get_positions(), dtype=np.float32)
        active_eyes, active_labels = [], []
        for cam, view_w, eye, label in zip(all_cams, all_view_weights, cam_eyes, cam_labels):
            r, p = setup_renderer(cam, rcfg, training_mode=True)
            if r:
                obs = generate_target_observation(
                    tpos, rs, r, p['campos'], rcfg, color,
                    target_mesh_path=cfg.get('target_mesh_path'), cam_cfg=cam
                )
                if obs is not None and obs.get('alpha') is not None:
                    t = obs['alpha']
                    renderers.append(r); campos_list.append(p['campos']); targets.append(t)
                    target_dt_maps.append(compute_dt_map(t))
                    target_depths.append(obs.get('depth'))
                    target_curvatures.append(obs.get('curvature'))
                    target_concavities.append(obs.get('concavity'))
                    view_weights.append(float(view_w))
                    active_eyes.append(eye)
                    active_labels.append(label)
        cam_eyes = np.asarray(active_eyes, dtype=np.float32) if active_eyes else np.array([])
        cam_labels = active_labels
        print(f"[Cameras] {len(renderers)} views")
    else:
        r, p = setup_renderer(cam_cfg, rcfg, training_mode=True)
        if r:
            tpos = np.array(tgt_pc.get_positions(), dtype=np.float32)
            obs = generate_target_observation(
                tpos, rs, r, p['campos'], rcfg, color,
                target_mesh_path=cfg.get('target_mesh_path'), cam_cfg=cam_cfg
            )
            if obs is not None and obs.get('alpha') is not None:
                t = obs['alpha']
                renderers.append(r); campos_list.append(p['campos']); targets.append(t)
                target_dt_maps.append(compute_dt_map(t))
                target_depths.append(obs.get('depth'))
                target_curvatures.append(obs.get('curvature'))
                target_concavities.append(obs.get('concavity'))
                cam_eyes = np.array([p['campos']]); cam_labels = ['Primary']
                view_weights = [1.0]

    if targets:
        from utils.io_utils import save_image_png, save_depth_png
        save_image_png(out / 'target_alpha.png', targets[0].numpy())
        if target_depths and target_depths[0] is not None:
            td = target_depths[0]
            td_np = td.cpu().numpy() if hasattr(td, 'cpu') else np.asarray(td, dtype=np.float32)
            if (td_np > 0).any():
                save_depth_png(out / 'target_depth.png', td_np, bits=16)
        if target_curvatures and target_curvatures[0] is not None:
            tc = target_curvatures[0]
            tc_np = tc.cpu().numpy() if hasattr(tc, 'cpu') else np.asarray(tc, dtype=np.float32)
            if np.isfinite(tc_np).any():
                from utils.io_utils import save_image_png
                denom = max(float(tc_np.max()), 1e-8)
                save_image_png(out / 'target_curvature.png', np.clip(tc_np / denom, 0.0, 1.0))
        if target_concavities and target_concavities[0] is not None:
            ts = target_concavities[0]
            ts_np = ts.cpu().numpy() if hasattr(ts, 'cpu') else np.asarray(ts, dtype=np.float32)
            if np.isfinite(ts_np).any():
                from utils.io_utils import save_image_png
                save_image_png(out / 'target_concavity.png', np.clip(0.5 + 0.5 * ts_np, 0.0, 1.0))

    # ── Train ─────────────────────────────────────────────────────────────
    Fp = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)) if resume_Fp is None else resume_Fp.copy()
    num_eps = int(opt.num_animations)
    episode_offset = int(resume_cfg.get('episode_offset', 0))
    total_sched_eps = int(resume_cfg.get('total_episodes', episode_offset + num_eps))
    history = []

    alpha_start = float(cfg.get('optimization', {}).get('initial_alpha', 0.01))
    alpha_end = float(cfg.get('optimization', {}).get('final_alpha', alpha_start * 0.1))

    if episode_offset > 0:
        print(
            f"\nTraining: {num_eps} episodes (global ep {episode_offset}..{episode_offset + num_eps - 1}), "
            f"correction={'ON' if use_correction else 'OFF'}\n"
        )
    else:
        print(f"\nTraining: {num_eps} episodes, correction={'ON' if use_correction else 'OFF'}\n")

    prev_surface_obs_x = None
    prev_surface_obs_grad = None

    for ep_local in range(num_eps):
        ep = ep_local + episode_offset
        # LR decay
        t = ep / max(total_sched_eps - 1, 1)
        opt.initial_alpha = alpha_end + 0.5 * (alpha_start - alpha_end) * (1 + np.cos(np.pi * t))
        if ep % 10 == 0:
            print(f"  [LR] alpha={opt.initial_alpha:.6f}")

        capture_debug_viz = ep in debug_gradient_eps
        capture_images = png or capture_debug_viz
        render_penalty = None
        control_guidance_meta = {
            'control_guidance_applied': 0,
            'control_guidance_render_gain': 0.0,
            'control_guidance_physics_weight': 1.0,
            'control_guidance_dLdx_norm': 0.0,
            'control_guidance_dLdF_norm': 0.0,
            'control_guidance_dLdx_max': 0.0,
            'control_guidance_dLdF_max': 0.0,
            'control_guidance_active_count': 0,
            'control_guidance_active_frac': 0.0,
            'control_guidance_focus_count': 0,
            'control_guidance_focus_frac': 0.0,
            'control_guidance_partner_k': int(control_guidance_cfg.get('smooth_k', 64)),
            'control_guidance_diffusion_iters': int(control_guidance_cfg.get('diffusion_iters', 2)),
        }
        render_force_meta = {
            'render_force_applied': 0,
            'render_force_gain': 0.0,
            'render_force_physics_weight': 1.0,
            'render_force_norm': 0.0,
            'render_force_focus_frac': 0.0,
            'render_force_focus_count': 0,
        }
        try:
            cg.set_render_gain(1.0)
            cg.set_physics_weight(1.0)
        except Exception:
            pass

        if control_guidance_enabled and ep >= control_guidance_start_ep and prev_surface_obs_grad is not None:
            focus_mask = None
            if float(control_guidance_cfg.get('ear_focus_boost', 0.0)) > 0.0:
                focus_mask, focus_meta = _build_ear_focus_mask(
                    prev_surface_obs_x,
                    active_mask=render_surface_mask,
                    lateral_axis=int(control_guidance_cfg.get('ear_focus_lateral_axis', 0)),
                    height_axis=int(control_guidance_cfg.get('ear_focus_height_axis', 2)),
                    height_frac=float(control_guidance_cfg.get('ear_focus_height_frac', 0.58)),
                    side_frac=float(control_guidance_cfg.get('ear_focus_side_frac', 0.18)),
                    side_max_frac=float(control_guidance_cfg.get('ear_focus_side_max_frac', 0.50)),
                    center=control_guidance_cfg.get('ear_focus_center', None),
                )
                control_guidance_meta['control_guidance_focus_count'] = int(focus_meta['ear_focus_count'])
                control_guidance_meta['control_guidance_focus_frac'] = float(focus_meta['ear_focus_frac'])

            render_penalty, cg_meta = build_control_guidance_penalty(
                prev_surface_obs_x,
                prev_surface_obs_grad,
                control_guidance_cfg,
                focus_mask=focus_mask,
                core_mask=surface_mask,
                support_mask=render_surface_mask,
            )
            control_guidance_meta.update(cg_meta)
            if render_penalty is not None:
                render_gain = float(control_guidance_cfg.get('render_gain', 1.0))
                render_gain_start = float(control_guidance_cfg.get('render_gain_start', render_gain))
                render_gain_warmup_eps = int(control_guidance_cfg.get('render_gain_warmup_eps', 0))
                if render_gain_warmup_eps > 0:
                    rel_ep = max(ep - control_guidance_start_ep, 0)
                    a = min(rel_ep / max(render_gain_warmup_eps, 1), 1.0)
                    render_gain = render_gain_start + a * (render_gain - render_gain_start)
                physics_weight = float(control_guidance_cfg.get('physics_weight', 1.0))
                control_guidance_meta['control_guidance_render_gain'] = render_gain
                control_guidance_meta['control_guidance_physics_weight'] = physics_weight
                try:
                    cg.set_render_gain(render_gain)
                    cg.set_physics_weight(physics_weight)
                except Exception:
                    pass
        elif render_force_enabled and ep >= render_force_start_ep and prev_surface_obs_grad is not None:
            render_penalty, rf_meta = _prepare_render_force_penalty(
                prev_surface_obs_x, prev_surface_obs_grad, render_force_cfg
            )
            render_force_meta.update(rf_meta)
            if render_penalty is not None:
                render_gain = float(render_force_cfg.get('render_gain', 0.1))
                physics_weight = float(render_force_cfg.get('physics_weight', 1.0))
                render_force_meta['render_force_gain'] = render_gain
                render_force_meta['render_force_physics_weight'] = physics_weight
                try:
                    cg.set_render_gain(render_gain)
                    cg.set_physics_weight(physics_weight)
                except Exception:
                    pass

        # ── Stage A: Physics Rollout ──────────────────────────────────────
        losses, dFp, direction, cohesion, dLdx_norms, bce_list, alpha_list, diffused, phys_grad, x_episode = run_episode(
            ep, cg, opt, sigma0, opacity,
            renderers, campos_list, targets, rcfg, color,
            out, capture_images, Fp, pcfg, lcfg,
            render_penalty=render_penalty,
            target_dt_maps=target_dt_maps, target_depths=target_depths, target_curvatures=target_curvatures,
            target_concavities=target_concavities,
            view_weights=view_weights, particle_mask=render_surface_mask, view_labels=cam_labels,
        )
        losses.update(control_guidance_meta)
        losses.update(render_force_meta)
        losses.update(surface_meta)
        prev_surface_obs_x = None if x_episode is None else np.asarray(x_episode, dtype=np.float32).copy()
        prev_surface_obs_grad = None if diffused is None else np.asarray(diffused, dtype=np.float32).copy()

        # Save dFc before promote
        dFc_raw = np.array(cg.get_point_cloud(0).get_dFc(), dtype=np.float32)
        dFc_norms_viz = np.linalg.norm(dFc_raw.reshape(N, -1), axis=1)

        # Promote
        cg.promote_last_as_initial(carry_grid=False)
        pc = cg.get_point_cloud(0)

        # Impulse (from render direction, if available)
        if eta_v > 0 or cohesion is not None:
            v = pc.get_velocities_view()
            imp = np.zeros((N, 3), dtype=np.float32)
            if direction is not None and eta_v > 0:
                imp -= (eta_v * direction).astype(np.float32)
            if cohesion is not None:
                imp += cohesion
            v[:] = np.clip(v + imp, -max_v, max_v)
            losses['impulse_mean'] = float(np.linalg.norm(imp, axis=1).mean())

        # ── Stage B: Observation-Guided Correction ────────────────────────
        every_n = int(correction_cfg.get('every_n_eps', 1))
        accepted_correction = False
        skip_dfp_update = False
        if use_correction and len(renderers) > 0 and ep % every_n == 0:
            x_phys = np.array(pc.get_positions(), dtype=np.float32)
            try:
                F_e = pc.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy().astype(np.float32)
            except Exception:
                F_e = np.array(pc.get_def_grads(), dtype=np.float32)
            F_accept = np.matmul(F_e, Fp).astype(np.float32)

            x_corrected, corr_metrics = render_correction(
                x_phys, F_e, sigma0, opacity,
                renderers, campos_list, targets, rcfg, color, lcfg, Fp,
                correction_cfg, target_dt_maps=target_dt_maps, target_depths=target_depths, target_curvatures=target_curvatures,
                target_concavities=target_concavities,
                view_weights=view_weights, particle_mask=surface_mask,
                ep=ep, num_eps=num_eps,
            )
            losses.update(corr_metrics)

            # ── Stage C: Immediate Mass-Loss Gate ────────────────────────
            x_accepted, accepted, mass_delta, mass_before, mass_after = immediate_mass_loss_gate(
                cg, x_phys, x_corrected, pc, tau=mass_gate_tau
            )

            if accepted:
                apply_accepted_correction(pc, x_accepted, correction_cfg, F_accept=F_accept)
                accepted_correction = True
                skip_dfp_update = bool(correction_cfg.get('skip_dfp_update_on_accept', True))
                if correction_cfg.get('absorb_fp_on_accept', True):
                    Fp = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
                    losses['correction_fp_absorbed'] = 1
                else:
                    losses['correction_fp_absorbed'] = 0
                losses['correction_accepted'] = 1
            else:
                losses['correction_fp_absorbed'] = 0
                losses['correction_accepted'] = 0
            losses['correction_phys_delta'] = mass_delta
            losses['correction_mass_gate_delta'] = mass_delta
            losses['correction_mass_before'] = mass_before
            losses['correction_mass_after'] = mass_after
            print(f"  [MassGate] {'ACCEPT' if accepted else 'REJECT'} (Δmass={mass_delta:+.4f})")

        # ── Fp update ─────────────────────────────────────────────────────
        if dFp is not None and not skip_dfp_update:
            Fp = np.matmul(np.eye(3)[None] + dFp, Fp).astype(np.float32)
            if damping > 0:
                Fp = ((1 - damping) * Fp + damping * np.eye(3)[None]).astype(np.float32)
            Fp = isochoric_project(Fp, max_aniso)
            losses['dFp_applied'] = 1

            norms = np.linalg.norm((Fp - np.eye(3)[None]).reshape(N, -1), axis=1)
            losses['Fp_dev'] = float(norms.mean())
            if (ep + 1) % 10 == 0 or ep == num_eps - 1:
                np.save(out / f'Fp_ep{ep:03d}.npy', Fp)
        else:
            losses['dFp_applied'] = 0
            if accepted_correction:
                losses['Fp_dev'] = float(
                    np.linalg.norm((Fp - np.eye(3)[None]).reshape(N, -1), axis=1).mean()
                )

        # ── Save ──────────────────────────────────────────────────────────
        if bce_list: losses['per_view_bce'] = bce_list
        history.append(losses)
        with open(out / 'losses.json', 'w') as f:
            json.dump(history, f, indent=2)

        # ── Checkpoint ────────────────────────────────────────────────────
        if ep % 5 == 0 or ep == num_eps - 1:
            ckpt_dir = out / 'checkpoints'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ckpt_dir / f'ckpt_ep{ep:03d}.npz',
                positions=np.array(pc.get_positions(), dtype=np.float32),
                velocities=np.array(pc.get_velocities_view(), dtype=np.float32),
                dFc=np.array(pc.get_dFc(), dtype=np.float32),
                Fp=Fp.copy(),
            )
            print(f"  [Checkpoint] ep{ep:03d}")

        # ── Viz ───────────────────────────────────────────────────────────
        if capture_images and ((png and (ep % 5 == 0 or ep == num_eps - 1)) or capture_debug_viz):
            x_viz = np.array(pc.get_positions(), dtype=np.float32)
            dLdx_dirs = None
            if dLdx_norms is not None and diffused is not None:
                d_n = np.linalg.norm(diffused, axis=1, keepdims=True)
                dLdx_dirs = np.zeros_like(diffused, dtype=np.float32)
                np.divide(diffused, d_n, out=dLdx_dirs, where=d_n > 1e-8)
            try:
                fp_n = np.linalg.norm((Fp - np.eye(3)[None]).reshape(N, -1), axis=1)
                phys_n = np.linalg.norm(phys_grad, axis=1) if phys_grad is not None else None
                create_episode_visualization(ep, x_viz, dLdx_norms, dFc_norms_viz, out,
                    cam_eye=cam_cfg.get('lookat', {}).get('eye'),
                    cam_target=cam_cfg.get('lookat', {}).get('target'),
                    cam_positions=cam_eyes,
                    dLdx_directions=dLdx_dirs,
                    Fp_norms=fp_n,
                    phys_norms=phys_n)
                if alpha_list:
                    tgt_nps = [t.cpu().numpy() if hasattr(t, 'cpu') else np.array(t) for t in targets]
                    create_per_view_visualization(ep, alpha_list, tgt_nps, bce_list, cam_labels, out)
                print(f"  [Viz] ep{ep:03d}")
            except Exception as e:
                import traceback; print(f"  [Viz] {e}"); traceback.print_exc()

    np.save(out / 'Fp_final.npy', Fp)
    print(f"\nDone. {out}")


# ─── Batch / Main ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--png', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if 'experiments' in cfg:
        import copy
        defaults = cfg.get('defaults', {})
        experiments = cfg['experiments']
        base_out = cfg.get('output_base', 'output')

        print(f"=== Batch mode: {len(experiments)} experiments ===\n")
        for i, exp in enumerate(experiments):
            ecfg = copy.deepcopy(defaults)
            ecfg['target_mesh_path'] = exp.get('target_mesh_path', defaults.get('target_mesh_path'))
            ecfg['input_mesh_path'] = defaults.get('input_mesh_path', 'assets/isosphere.obj')
            ecfg['output_dir'] = f"{base_out}/{exp['name']}"

            # Per-experiment overrides
            if 'lambda_render' in exp:
                ecfg.setdefault('plasticity', {})['lambda_render'] = exp['lambda_render']
            if 'num_cameras' in exp:
                ecfg.setdefault('multi_view', {})['num_cameras'] = exp['num_cameras']
            if 'num_animations' in exp:
                ecfg.setdefault('optimization', {})['num_animations'] = exp['num_animations']
            for section in [
                'plasticity', 'multi_view', 'optimization', 'correction',
                'loss_weights', 'render', 'camera', 'simulation', 'upsample',
                'surface_aware', 'control_guidance', 'render_force', 'debug', 'resume'
            ]:
                if section in exp:
                    base_section = ecfg.get(section, {})
                    if isinstance(base_section, dict) and isinstance(exp[section], dict):
                        ecfg[section] = _deep_update(copy.deepcopy(base_section), exp[section])
                    else:
                        ecfg[section] = copy.deepcopy(exp[section])

            if args.skip_existing and (Path(ecfg['output_dir']) / 'losses.json').exists():
                print(f"[{i+1}/{len(experiments)}] SKIP {exp['name']}")
                continue

            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(experiments)}] {exp['name']}")
            print(f"  target: {ecfg.get('target_mesh_path')}")
            print(f"  correction: {'ON' if exp.get('correction', {}).get('enabled') else 'OFF'}")
            if 'multi_view' in ecfg:
                print(f"  cameras: {ecfg['multi_view'].get('preset', 'ring')} / {ecfg['multi_view'].get('num_cameras', 'n/a')}")
            print(f"{'='*60}\n")

            run_single(ecfg, png=args.png)

        print(f"\n=== All {len(experiments)} experiments complete ===")
    else:
        run_single(cfg, png=args.png)


if __name__ == '__main__':
    main()
