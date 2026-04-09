"""
Training Loop — Decoupled Physics + Render Correction

Stage A: Physics rollout (EndLayerMassLoss, C++ Adam)
Stage B/C handled in run.py (deformation field correction + acceptance)
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from utils.io_utils import save_image_png
from utils.alpha_losses import combined_alpha_loss


# ── Rendering ────────────────────────────────────────────────────────────

def _pca_normals(x, k=16):
    from scipy.spatial import cKDTree
    N = x.shape[0]
    _, idx = cKDTree(x).query(x, k=min(k, N))
    normals = np.zeros_like(x)
    for i in range(N):
        c = x[idx[i]] - x[idx[i]].mean(0)
        normals[i] = np.linalg.eigh(c.T @ c)[1][:, 0]
    normals[((x - x.mean(0)) * normals).sum(1) < 0] *= -1
    return normals.astype(np.float32)


def _prepare_shared_render_inputs(x, F_e, sigma0, opacity, Fp=None, require_grad=True):
    """Prepare tensors shared across all views for the same particle state."""
    N = x.shape[0]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_t = torch.from_numpy(x).float().to(dev)
    if require_grad:
        x_t.requires_grad_(True)

    F_r = np.matmul(F_e, Fp).astype(np.float32) if Fp is not None else np.asarray(F_e, dtype=np.float32)

    with torch.no_grad():
        F_t = torch.from_numpy(F_r).float().to(dev)
        eye = torch.eye(3, device=dev)
        S0 = (sigma0 ** 2) * eye
        cov = torch.bmm(torch.matmul(F_t, S0[None].expand(N, -1, -1)), F_t.transpose(1, 2))
        cov += 1e-6 * eye[None]
        opacity_t = torch.full((N, 1), opacity, device=dev)

    return {
        'device': dev,
        'x_t': x_t,
        'cov': cov,
        'opacity': opacity_t,
        'num_points': N,
    }


def _constant_rgb(num_points, color, device):
    color_t = torch.tensor(color, dtype=torch.float32, device=device)
    return color_t.unsqueeze(0).expand(num_points, -1).contiguous()


def _render_prepared(renderer, x_t, cov, rgb, opacity, return_torch=True):
    pred = renderer.render(
        x_t, cov, rgb=rgb, opacity=opacity,
        prefer_cov_precomp=True, return_torch=return_torch
    )
    alpha = pred.get('alpha')
    if alpha is None:
        return None, pred
    if isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha).float().to(x_t.device)
    if alpha.dim() == 3:
        alpha = alpha[0]
    return alpha, pred


def _compute_view_losses(pred_alpha, pred_dict, target, lcfg, target_rgb=None, target_dt=None):
    """Compute per-view observation loss and metrics."""
    loss_alpha, metrics = _alpha_loss(pred_alpha, target, lcfg, dt_map=target_dt)

    loss_rgb = torch.tensor(0.0, device=pred_alpha.device)
    w_rgb = float(lcfg.get('w_rgb', 0.0))
    if w_rgb > 0 and target_rgb is not None and pred_dict.get('image') is not None:
        pred_img = pred_dict['image']
        if isinstance(pred_img, np.ndarray):
            pred_img = torch.from_numpy(pred_img).float().to(pred_alpha.device)
        tgt_img = target_rgb.to(pred_alpha.device)
        if tgt_img.shape != pred_img.shape and tgt_img.dim() == 3 and pred_img.dim() == 3:
            tgt_img = F.interpolate(
                tgt_img.permute(2, 0, 1).unsqueeze(0),
                size=pred_img.shape[:2], mode='bilinear', align_corners=False
            )[0].permute(1, 2, 0)
        loss_rgb = w_rgb * ((pred_img - tgt_img) ** 2).mean()
        metrics['loss_rgb'] = float(loss_rgb.item())

    loss_total = loss_alpha + loss_rgb
    metrics['loss_total'] = float(loss_total.item())
    return loss_total, metrics


def _resolve_view_weights(num_views, view_weights, dtype, device=None):
    if view_weights is None:
        if device is None:
            return np.ones((num_views,), dtype=np.float32)
        return torch.ones((num_views,), dtype=dtype, device=device)

    if device is None:
        w = np.asarray(view_weights[:num_views], dtype=np.float32)
        if w.shape[0] != num_views:
            raise ValueError(f"view_weights length mismatch: expected {num_views}, got {w.shape[0]}")
        return np.clip(w, 1e-8, None)

    w = torch.as_tensor(view_weights[:num_views], dtype=dtype, device=device)
    if w.numel() != num_views:
        raise ValueError(f"view_weights length mismatch: expected {num_views}, got {w.numel()}")
    return torch.clamp(w, min=1e-8)


def _get_topk_count(num_views, cfg):
    k = int(cfg.get('mv_topk_k', 0))
    frac = float(cfg.get('mv_topk_frac', 0.0))
    if k <= 0 and frac > 0:
        k = int(np.ceil(num_views * frac))
    return min(max(k, 0), num_views)


def _aggregate_multiview_tensors(loss_tensors, view_weights, cfg):
    losses = torch.stack(loss_tensors)
    weights = _resolve_view_weights(losses.numel(), view_weights, losses.dtype, losses.device)

    mean_loss = losses.mean()
    weighted_mean = (weights * losses).sum() / weights.sum()
    objective = weighted_mean

    hardmax = losses.max()
    hardmax_w = float(cfg.get('mv_hardmax_w', 0.0))
    if hardmax_w > 0:
        objective = objective + hardmax_w * hardmax

    topk_k = _get_topk_count(losses.numel(), cfg)
    topk_w = float(cfg.get('mv_topk_w', 0.0))
    if topk_k > 0:
        topk_mean = torch.topk(losses, k=topk_k).values.mean()
        if topk_w > 0:
            objective = objective + topk_w * topk_mean
    else:
        topk_mean = torch.zeros((), dtype=losses.dtype, device=losses.device)

    return objective, {
        'loss_total_mv': float(mean_loss.item()),
        'loss_weighted_mv': float(weighted_mean.item()),
        'loss_hardmax_mv': float(hardmax.item()),
        'loss_topk_mv': float(topk_mean.item()),
        'loss_total_obj_mv': float(objective.item()),
        'mv_topk_k': int(topk_k),
    }


def _aggregate_multiview_values(loss_values, view_weights, cfg):
    losses = np.asarray(loss_values, dtype=np.float32)
    weights = _resolve_view_weights(len(loss_values), view_weights, np.float32, device=None)

    mean_loss = float(losses.mean())
    weighted_mean = float((weights * losses).sum() / weights.sum())
    objective = weighted_mean

    hardmax = float(losses.max())
    hardmax_w = float(cfg.get('mv_hardmax_w', 0.0))
    if hardmax_w > 0:
        objective += hardmax_w * hardmax

    topk_k = _get_topk_count(len(loss_values), cfg)
    if topk_k > 0:
        topk_mean = float(np.sort(losses)[-topk_k:].mean())
        topk_w = float(cfg.get('mv_topk_w', 0.0))
        if topk_w > 0:
            objective += topk_w * topk_mean
    else:
        topk_mean = 0.0

    return {
        'loss_total_mv': mean_loss,
        'loss_weighted_mv': weighted_mean,
        'loss_hardmax_mv': hardmax,
        'loss_topk_mv': topk_mean,
        'loss_total_obj_mv': objective,
        'mv_topk_k': int(topk_k),
    }


def render(x, F_e, sigma0, opacity, renderer, campos, rcfg, color, training=True, Fp=None):
    """Render particles. F_render = F_e · Fp if Fp provided."""
    from renderer import compute_shading
    N, dev = x.shape[0], torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_t = torch.from_numpy(x).float().to(dev)
    F_r = np.matmul(F_e, Fp).astype(np.float32) if Fp is not None else F_e
    F_t = torch.from_numpy(F_r).float().to(dev)
    if training:
        x_t.requires_grad_(True); F_t.requires_grad_(True)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        S0 = (sigma0**2) * torch.eye(3, device=dev)
        cov = torch.bmm(torch.matmul(F_t, S0[None].expand(N,-1,-1)), F_t.transpose(1,2))
        cov += 1e-6 * torch.eye(3, device=dev)[None]
        rgb = torch.from_numpy(compute_shading(
            x, _pca_normals(x), camera_pos=campos,
            light_cfg=rcfg.get('lighting',{}), albedo_color=color, model='phong'
        )).float().to(dev)
        pred = renderer.render(x_t, cov, rgb=rgb,
                               opacity=torch.full((N,1), opacity, device=dev),
                               prefer_cov_precomp=True, return_torch=training)

    alpha = pred.get('alpha')
    if alpha is not None:
        if isinstance(alpha, np.ndarray):
            alpha = torch.from_numpy(alpha).float().to(dev)
        if alpha.dim() == 3:
            alpha = alpha[0]
    return alpha, pred, x_t, F_t


# ── Gradient extraction ──────────────────────────────────────────────────

def _alpha_loss(pred, target, cfg, dt_map=None):
    tgt = target.to(pred.device)
    if tgt.shape != pred.shape:
        tgt = F.interpolate(tgt[None,None], size=pred.shape, mode='bilinear', align_corners=False)[0,0]
    dt_resized = None
    if dt_map is not None:
        dt_resized = dt_map.to(pred.device)
        if dt_resized.shape != pred.shape:
            dt_resized = F.interpolate(
                dt_resized[None, None], size=pred.shape, mode='bilinear', align_corners=False
            )[0, 0]
    return combined_alpha_loss(pred, tgt,
        w_bce=float(cfg.get('w_bce',1)), w_iou=float(cfg.get('w_iou',1)),
        w_dt=float(cfg.get('w_dt', 0.0)), w_edge=float(cfg.get('w_edge', 0.0)),
        dt_map=dt_resized,
        mask_threshold=float(cfg.get('masked_threshold',0)),
        edge_band_width=int(cfg.get('edge_band_width', 5)),
        edge_threshold=float(cfg.get('edge_threshold', 0.02)))


def compute_view_gradient(
    x, F_e, sigma0, opacity, target, renderer, campos, rcfg, color, lcfg,
    Fp=None, target_rgb=None, target_dt=None
):
    """Single view → dL/dx from alpha + RGB loss. Each call is graph-independent."""
    pred_alpha, pred_dict, x_t, _ = render(x, F_e, sigma0, opacity, renderer, campos, rcfg, color, True, Fp)
    if pred_alpha is None: return None, {}, None
    pred_np = pred_alpha.detach().cpu().numpy().copy()

    # Alpha loss
    loss_alpha, m = _alpha_loss(pred_alpha, target, lcfg, dt_map=target_dt)

    # RGB loss (if target_rgb provided)
    loss_rgb = torch.tensor(0.0, device=pred_alpha.device)
    w_rgb = float(lcfg.get('w_rgb', 0.0))
    if w_rgb > 0 and target_rgb is not None and pred_dict.get('image') is not None:
        pred_img = pred_dict['image']  # (H, W, 3) torch tensor
        if isinstance(pred_img, np.ndarray):
            pred_img = torch.from_numpy(pred_img).float().to(pred_alpha.device)
        tgt_img = target_rgb.to(pred_alpha.device)
        if tgt_img.shape != pred_img.shape:
            # Resize target to match pred
            if tgt_img.dim() == 3 and pred_img.dim() == 3:
                tgt_img = F.interpolate(
                    tgt_img.permute(2,0,1).unsqueeze(0),
                    size=pred_img.shape[:2], mode='bilinear', align_corners=False
                )[0].permute(1,2,0)
        loss_rgb = w_rgb * ((pred_img - tgt_img) ** 2).mean()
        m['loss_rgb'] = float(loss_rgb.item())

    loss = loss_alpha + loss_rgb
    loss.backward()
    dLdx = x_t.grad.detach().cpu().numpy().astype(np.float32) if x_t.grad is not None else None
    m['dLdx_norm'] = float(np.linalg.norm(dLdx)) if dLdx is not None else 0.0
    m['loss_total'] = float(loss.item())
    return dLdx, m, pred_np


def compute_multiview_gradients(
    x, F_e, sigma0, opacity, targets, renderers, campos_list, rcfg, color, lcfg,
    Fp=None, target_rgbs=None, target_dt_maps=None, return_alpha_list=False, view_weights=None
):
    """8 views → summed dL/dx from alpha + RGB loss."""
    from renderer import compute_shading

    shared = _prepare_shared_render_inputs(x, F_e, sigma0, opacity, Fp=Fp, require_grad=True)
    x_t = shared['x_t']
    cov = shared['cov']
    opacity_t = shared['opacity']

    need_rgb = float(lcfg.get('w_rgb', 0.0)) > 0 and target_rgbs is not None and any(
        t is not None for t in target_rgbs[:len(renderers)]
    )

    normals = _pca_normals(x) if need_rgb else None
    base_rgb = _constant_rgb(shared['num_points'], color, shared['device']) if not need_rgb else None

    bce_list = []
    alpha_list = [] if return_alpha_list else None
    total_list, iou_list, dt_list, edge_list, rgb_list = [], [], [], [], []
    loss_tensors = []
    first_m = {}
    valid_views = 0

    for v, (rend, tgt, cam) in enumerate(zip(renderers, targets, campos_list)):
        tgt_rgb = target_rgbs[v] if target_rgbs and v < len(target_rgbs) else None
        tgt_dt = target_dt_maps[v] if target_dt_maps and v < len(target_dt_maps) else None

        if need_rgb:
            rgb = torch.from_numpy(compute_shading(
                x, normals, camera_pos=cam,
                light_cfg=rcfg.get('lighting', {}), albedo_color=color, model='phong'
            )).float().to(shared['device'])
        else:
            rgb = base_rgb

        pred_alpha, pred_dict = _render_prepared(
            rend, x_t, cov, rgb, opacity_t, return_torch=True
        )
        if pred_alpha is None:
            continue

        loss_total, m = _compute_view_losses(
            pred_alpha, pred_dict, tgt, lcfg, target_rgb=tgt_rgb, target_dt=tgt_dt
        )

        bce_list.append(float(m.get('loss_bce', 0)))
        iou_list.append(float(m.get('loss_iou', 0)))
        dt_list.append(float(m.get('loss_dt', 0)))
        edge_list.append(float(m.get('loss_edge', 0)))
        rgb_list.append(float(m.get('loss_rgb', 0)))
        total_list.append(float(m.get('loss_total', 0)))
        loss_tensors.append(loss_total)
        if return_alpha_list:
            alpha_list.append(pred_alpha.detach().cpu().numpy().copy())
        if valid_views == 0:
            first_m = m
        valid_views += 1

    if valid_views == 0:
        return None, {}, bce_list, ([] if return_alpha_list else None)

    loss_obj, agg = _aggregate_multiview_tensors(loss_tensors, view_weights, lcfg)
    loss_obj.backward()

    if x_t.grad is None:
        return None, {}, bce_list, ([] if return_alpha_list else None)

    dLdx_sum = x_t.grad.detach().cpu().numpy().astype(np.float32)
    metrics = {
        'loss_bce': first_m.get('loss_bce', 0),
        'loss_iou': first_m.get('loss_iou', 0),
        'loss_dt': first_m.get('loss_dt', 0),
        'loss_bce_mv': sum(bce_list) / valid_views,
        'loss_iou_mv': sum(iou_list) / valid_views,
        'loss_dt_mv': sum(dt_list) / valid_views,
        'loss_edge_mv': sum(edge_list) / valid_views,
        'loss_rgb': first_m.get('loss_rgb', 0),
        'loss_rgb_mv': sum(rgb_list) / valid_views,
        'loss_total': first_m.get('loss_total', 0),
        'n_views': valid_views,
        'dLdx_norm': float(np.linalg.norm(dLdx_sum)),
    }
    metrics.update(agg)
    extra = []
    if abs(metrics['loss_total_obj_mv'] - metrics['loss_total_mv']) > 1e-6:
        extra.append(f"obj={metrics['loss_total_obj_mv']:.4f}")
    if metrics['loss_iou_mv'] > 0:
        extra.append(f"iou={metrics['loss_iou_mv']:.4f}")
    if metrics['loss_dt_mv'] != 0:
        extra.append(f"dt={metrics['loss_dt_mv']:.4f}")
    if metrics['loss_edge_mv'] > 0:
        extra.append(f"edge={metrics['loss_edge_mv']:.4f}")
    if metrics['loss_rgb_mv'] > 0:
        extra.append(f"rgb={metrics['loss_rgb_mv']:.4f}")
    extra_str = ", " + ", ".join(extra) if extra else ""
    print(f"  [MV] {valid_views} views, total={metrics['loss_total_mv']:.4f}, bce={metrics['loss_bce_mv']:.4f}{extra_str}")
    return dLdx_sum, metrics, bce_list, alpha_list


def compute_multiview_metrics(
    x, F_e, sigma0, opacity, targets, renderers, campos_list, rcfg, color, lcfg,
    Fp=None, target_rgbs=None, target_dt_maps=None, return_alpha_list=False, view_weights=None
):
    """Multi-view observation metrics without backward pass."""
    from renderer import compute_shading

    shared = _prepare_shared_render_inputs(x, F_e, sigma0, opacity, Fp=Fp, require_grad=False)
    x_t = shared['x_t']
    cov = shared['cov']
    opacity_t = shared['opacity']

    need_rgb = float(lcfg.get('w_rgb', 0.0)) > 0 and target_rgbs is not None and any(
        t is not None for t in target_rgbs[:len(renderers)]
    )

    normals = _pca_normals(x) if need_rgb else None
    base_rgb = _constant_rgb(shared['num_points'], color, shared['device']) if not need_rgb else None

    bce_list = []
    alpha_list = [] if return_alpha_list else None
    total_list, iou_list, dt_list, edge_list, rgb_list = [], [], [], [], []
    first_m = {}
    valid_views = 0

    with torch.no_grad():
        for v, (rend, tgt, cam) in enumerate(zip(renderers, targets, campos_list)):
            tgt_rgb = target_rgbs[v] if target_rgbs and v < len(target_rgbs) else None
            tgt_dt = target_dt_maps[v] if target_dt_maps and v < len(target_dt_maps) else None

            if need_rgb:
                rgb = torch.from_numpy(compute_shading(
                    x, normals, camera_pos=cam,
                    light_cfg=rcfg.get('lighting', {}), albedo_color=color, model='phong'
                )).float().to(shared['device'])
            else:
                rgb = base_rgb

            pred_alpha, pred_dict = _render_prepared(
                rend, x_t, cov, rgb, opacity_t, return_torch=True
            )
            if pred_alpha is None:
                continue

            _, m = _compute_view_losses(
                pred_alpha, pred_dict, tgt, lcfg, target_rgb=tgt_rgb, target_dt=tgt_dt
            )
            bce_list.append(float(m.get('loss_bce', 0)))
            iou_list.append(float(m.get('loss_iou', 0)))
            dt_list.append(float(m.get('loss_dt', 0)))
            edge_list.append(float(m.get('loss_edge', 0)))
            rgb_list.append(float(m.get('loss_rgb', 0)))
            total_list.append(float(m.get('loss_total', 0)))
            if return_alpha_list:
                alpha_list.append(pred_alpha.detach().cpu().numpy().copy())
            if valid_views == 0:
                first_m = m
            valid_views += 1

    if valid_views == 0:
        return {}, bce_list, ([] if return_alpha_list else None)

    metrics = {
        'loss_bce': first_m.get('loss_bce', 0),
        'loss_iou': first_m.get('loss_iou', 0),
        'loss_dt': first_m.get('loss_dt', 0),
        'loss_bce_mv': sum(bce_list) / valid_views,
        'loss_iou_mv': sum(iou_list) / valid_views,
        'loss_dt_mv': sum(dt_list) / valid_views,
        'loss_edge_mv': sum(edge_list) / valid_views,
        'loss_rgb': first_m.get('loss_rgb', 0),
        'loss_rgb_mv': sum(rgb_list) / valid_views,
        'loss_total': first_m.get('loss_total', 0),
        'n_views': valid_views,
    }
    metrics.update(_aggregate_multiview_values(total_list, view_weights, lcfg))
    return metrics, bce_list, alpha_list


# ── Diffusion + Plasticity ───────────────────────────────────────────────

def diffuse_and_compute_plasticity(x, dLdx, cfg):
    """
    Volume diffusion → direction field d + dFp (adaptive).
    Returns: (dFp, d_weighted, diffused_dLdx, metrics)
    """
    from scipy.spatial import cKDTree
    N, eps = x.shape[0], 1e-8

    eta = float(cfg.get('eta', 0.01))
    smooth_k = int(cfg.get('smooth_k', 128))
    clip_pct = float(cfg.get('clip_pct', 95))
    diff_iters = int(cfg.get('diffusion_iters', 5))
    ada_max = float(cfg.get('adaptive_eta_max', 5.0))

    g = dLdx.copy()
    norms = np.linalg.norm(g, axis=1)
    n_act = int((norms > 1e-10).sum())
    if n_act < 10:
        z3, z33 = np.zeros((N,3), np.float32), np.zeros((N,3,3), np.float32)
        return z33, z3, z3, {'active': n_act, 'diffused_ratio': 0, 'dFp_norm': 0}

    # Clip outliers
    thresh = np.percentile(norms[norms > 1e-10], clip_pct)
    if thresh > eps: g *= np.minimum(1.0, thresh / (norms + eps))[:, None]

    # KNN + volume diffusion
    _, idx = cKDTree(x).query(x, k=min(smooth_k + 1, N))
    for _ in range(diff_iters):
        g = g[idx].mean(axis=1).astype(np.float32)

    # Direction field
    d_n = np.linalg.norm(g, axis=1, keepdims=True)
    d = (g / np.maximum(d_n, eps)).astype(np.float32)
    d[d_n.squeeze() <= eps] = 0
    med = float(np.median(d_n[d_n.squeeze() > eps]))

    # Magnitude-weighted (for impulse)
    d_w = (d * np.clip(d_n / (med + eps), 0, 3)).astype(np.float32) if med > eps else d.copy()

    n_diff = int((np.linalg.norm(d, axis=1) > 0.5).sum())

    # Symmetric Jacobian → dFp (adaptive eta)
    dx = x[idx] - x[:, None, :]
    dd = d[idx] - d[:, None, :]
    dxs = np.maximum(np.sum(dx**2, axis=2, keepdims=True), eps)
    J = (dd[:,:,:,None] * dx[:,:,None,:] / dxs[:,:,:,None]).mean(1)
    J = 0.5 * (J + J.transpose(0,2,1))

    eta_p = eta * np.clip(d_n.squeeze() / (med + eps), 0, ada_max) if med > eps else np.full(N, eta)
    dFp = (eta_p[:, None, None] * J).astype(np.float32)

    # Diffused gradient (for render penalty injection)
    diffused = g.copy()  # already diffused

    metrics = {
        'active': n_act, 'diffused_ratio': n_diff / N,
        'dFp_norm': float(np.linalg.norm(dFp)),
        'dFp_max': float(np.abs(dFp).max()),
    }
    print(f"    [Diff] active={n_act:,}, diffused={n_diff:,}/{N:,} ({100*n_diff/N:.0f}%)")
    return dFp, d_w, diffused, metrics


# ── Cohesion ─────────────────────────────────────────────────────────────

def compute_cohesion(x, k=32, strength=0.1, sparse_threshold=1.5):
    """
    Cohesion heuristic for locally sparse regions.

    This is not a floating-particle detector. It identifies particles whose
    local neighborhood spacing is unusually large relative to the global median
    and whose neighbors are also sparse, then applies a weak pull toward the
    local centroid.
    """
    from scipy.spatial import cKDTree

    N = x.shape[0]
    k_eff = min(k + 1, N)
    if k_eff <= 2:
        z = np.zeros((N, 3), dtype=np.float32)
        return z, {
            'cohesion_affected_count': 0,
            'cohesion_affected_ratio': 0.0,
            'cohesion_score_mean': 1.0,
            'cohesion_pull_mean': 0.0,
        }

    dd, idx = cKDTree(x).query(x, k=k_eff)
    local_spacing = dd[:, 1:k_eff].mean(axis=1)
    global_spacing = float(np.median(local_spacing))
    global_spacing = max(global_spacing, 1e-8)

    spacing_ratio = local_spacing / global_spacing
    neighbor_spacing = local_spacing[idx[:, 1:k_eff]].mean(axis=1)
    neighbor_ratio = neighbor_spacing / global_spacing

    # Average, rather than multiply, to avoid exploding the score.
    sparse_region_score = 0.5 * (spacing_ratio + neighbor_ratio)

    disp = x[idx[:, 1:k_eff]].mean(axis=1) - x
    disp_norm = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-8
    excess = np.clip(sparse_region_score - sparse_threshold, 0.0, None)
    pull_mag = strength * excess
    imp = (disp / disp_norm * pull_mag[:, None]).astype(np.float32)

    affected = pull_mag > 0
    n_aff = int(affected.sum())
    aff_ratio = float(n_aff / max(N, 1))
    if n_aff:
        print(f"    [Cohesion] affected={n_aff:,}/{N:,} ({100.0 * aff_ratio:.1f}%), "
              f"threshold={sparse_threshold:.2f}")

    return imp, {
        'cohesion_affected_count': n_aff,
        'cohesion_affected_ratio': aff_ratio,
        'cohesion_score_mean': float(sparse_region_score.mean()),
        'cohesion_pull_mean': float(pull_mag.mean()),
    }


# ── Fp projection ────────────────────────────────────────────────────────

def isochoric_project(Fp, max_aniso=1.5):
    U, S, Vt = np.linalg.svd(Fp)
    if max_aniso > 0:
        S = np.maximum(S, S.max(axis=1, keepdims=True) / max_aniso)
    S /= S.prod(axis=1, keepdims=True) ** (1/3)
    return np.einsum('nij,nj,njk->nik', U, S, Vt).astype(np.float32)


# ── Episode ──────────────────────────────────────────────────────────────

def run_episode(
    ep, cg, opt, sigma0, opacity,
    renderers, campos_list, targets, rcfg, color,
    out_dir, png, Fp, cfg, lcfg,
    render_penalty=None, target_rgbs=None, target_dt_maps=None, view_weights=None,
    view_labels=None,
):
    """
    One episode:
      1. Physics (+ render penalty if available)
      2. Multi-view render → diffused gradient
      3. Plasticity (dFp) + direction (impulse) + diffused (next penalty)

    Returns: losses, dFp, direction, cohesion_imp, dLdx_norms, bce_list, alpha_list, diffused_grad
    """
    num_ts = int(opt.num_timesteps)
    pc0 = cg.get_point_cloud(0)
    x0 = np.array(pc0.get_positions(), dtype=np.float32).copy()
    N = x0.shape[0]

    # ── Physics ───────────────────────────────────────────────────────────
    if render_penalty is not None:
        res = cg.run_e2e_pass_batched(opt, render_penalty['dLdF'], render_penalty['dLdx'], has_render_grads=True)
        loss_phys = res['loss_physics']
    else:
        cg.run_optimization(opt)
        loss_phys = cg.end_layer_mass_loss()

    # Extract physics gradient (for visualization + diagnostics)
    try:
        _, phys_dx_raw = cg.get_last_layer_phys_gradients()
        phys_grad = np.array(phys_dx_raw, dtype=np.float32).reshape(N, 3)
    except:
        phys_grad = np.zeros((N, 3), dtype=np.float32)

    pc = cg.get_point_cloud(num_ts - 1)
    x = np.ascontiguousarray(np.array(pc.get_positions(), dtype=np.float32))
    try: F_e = pc.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy()
    except: F_e = np.array(pc.get_def_grads(), dtype=np.float32)
    F_e = np.ascontiguousarray(F_e.astype(np.float32))

    det = np.linalg.det(F_e)
    dx_m = float(np.linalg.norm(x - x0, axis=1).mean())

    # dFc tracking (from control layer 0, where Adam optimizes)
    pc0 = cg.get_point_cloud(0)
    dFc = np.array(pc0.get_dFc(), dtype=np.float32)
    dFc_norms = np.linalg.norm(dFc.reshape(N, -1), axis=1)
    dFc_mean = float(dFc_norms.mean())
    dFc_max = float(dFc_norms.max())

    pen_str = " +penalty" if render_penalty is not None else ""
    print(f"\n[Ep {ep:03d}] phys={loss_phys:.1f}{pen_str}, J=[{det.min():.3f},{det.max():.3f}], dx={dx_m:.4f}, ||dFc||={dFc_mean:.6f}")

    # ── Multi-view render ─────────────────────────────────────────────────
    dFp = direction = cohesion_imp = diffused = dLdx = None
    rm, bce_list, alpha_list = {}, [], []

    if len(renderers) > 0 and len(targets) > 0:
        dLdx, rm, bce_list, alpha_list = compute_multiview_gradients(
            x, F_e, sigma0, opacity, targets, renderers, campos_list, rcfg, color, lcfg, Fp,
            target_rgbs=target_rgbs, target_dt_maps=target_dt_maps, return_alpha_list=png,
            view_weights=view_weights)

        if dLdx is not None:
            dFp, direction, diffused, pm = diffuse_and_compute_plasticity(x, dLdx, cfg)
            rm.update(pm)

        cs = float(cfg.get('cohesion_strength', 0))
        if cs > 0:
            cohesion_imp, cm = compute_cohesion(
                x,
                int(cfg.get('cohesion_k', 32)),
                cs,
                sparse_threshold=float(cfg.get('cohesion_sparse_threshold', 1.5)),
            )
            rm.update(cm)

    # ── Alpha metric (primary view) ───────────────────────────────────────
    alpha_mse = 0.0
    if len(renderers) > 0 and len(targets) > 0:
        with torch.no_grad():
            pa, pred, _, _ = render(x, F_e, sigma0, opacity, renderers[0], campos_list[0], rcfg, color, False, Fp)
        if pa is not None:
            tgt = targets[0].to(pa.device)
            if tgt.shape != pa.shape:
                tgt = F.interpolate(tgt[None,None], size=pa.shape, mode='bilinear', align_corners=False)[0,0]
            err = (pa - tgt)**2
            mt = float(lcfg.get('masked_threshold', 0))
            alpha_mse = float(err[(tgt>mt)|(pa>mt)].mean()) if mt > 0 else float(err.mean())
        if png and pred and pred.get('image') is not None:
            d = out_dir / f'ep{ep:03d}'; d.mkdir(parents=True, exist_ok=True)
            save_image_png(d / 'render.png', pred['image'])
            save_image_png(d / 'alpha_primary.png', pa.detach().cpu().numpy())
            save_image_png(d / 'alpha_primary_target.png', tgt.detach().cpu().numpy())
            save_image_png(d / 'alpha_primary_error.png', err.detach().cpu().numpy())

    print(f"  alpha_mse={alpha_mse:.6f}")

    losses = {'ep': ep, 'loss_physics': float(loss_phys), 'alpha_mse': alpha_mse,
              'dx_mean': dx_m, 'dFc_mean': dFc_mean, 'dFc_max': dFc_max}
    for k in ['loss_bce', 'loss_iou', 'loss_dt', 'loss_total',
              'loss_bce_mv', 'loss_iou_mv', 'loss_dt_mv', 'loss_edge_mv', 'loss_rgb', 'loss_rgb_mv',
              'loss_total_mv', 'loss_total_obj_mv', 'loss_weighted_mv',
              'loss_hardmax_mv', 'loss_topk_mv', 'mv_topk_k',
              'n_views', 'dLdx_norm',
              'active', 'diffused_ratio', 'dFp_norm',
              'cohesion_affected_count', 'cohesion_affected_ratio',
              'cohesion_score_mean', 'cohesion_pull_mean']:
        if k in rm: losses[k] = rm[k]

    if bce_list:
        worst_idx = int(np.argmax(np.asarray(bce_list, dtype=np.float32)))
        losses['worst_view_idx'] = worst_idx
        losses['worst_view_bce'] = float(bce_list[worst_idx])
        if view_labels and worst_idx < len(view_labels):
            losses['worst_view_label'] = view_labels[worst_idx]

        if png and alpha_list and len(targets) > worst_idx:
            d = out_dir / f'ep{ep:03d}'; d.mkdir(parents=True, exist_ok=True)
            worst_alpha = np.asarray(alpha_list[worst_idx], dtype=np.float32)
            worst_target = targets[worst_idx]
            if hasattr(worst_target, 'detach'):
                worst_target = worst_target.detach().cpu().numpy()
            else:
                worst_target = np.asarray(worst_target, dtype=np.float32)
            if worst_target.shape != worst_alpha.shape:
                worst_target = F.interpolate(
                    torch.from_numpy(worst_target)[None, None],
                    size=worst_alpha.shape, mode='bilinear', align_corners=False
                )[0, 0].cpu().numpy()
            save_image_png(d / 'alpha_worst.png', worst_alpha)
            save_image_png(d / 'alpha_worst_target.png', worst_target)
            save_image_png(d / 'alpha_worst_error.png', np.abs(worst_alpha - worst_target))

    dLdx_norms = np.linalg.norm(dLdx, axis=1) if dLdx is not None else None
    return losses, dFp, direction, cohesion_imp, dLdx_norms, bce_list, alpha_list, diffused, phys_grad
