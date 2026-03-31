"""
Training Loop — Render-Penalized Physics + Isochoric Plasticity

L_total = EndLayerMassLoss + λ · DiffusedRenderGradient
Fp affects rendering only (Gaussian covariance).
7 cameras (3 low + 3 mid + 1 top).
"""

import numpy as np
import torch
import torch.nn.functional as F_nn
from typing import Dict, List, Tuple, Any, Optional
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

def _alpha_loss(pred, target, cfg):
    tgt = target.to(pred.device)
    if tgt.shape != pred.shape:
        tgt = F_nn.interpolate(tgt[None,None], size=pred.shape, mode='bilinear', align_corners=False)[0,0]
    return combined_alpha_loss(pred, tgt,
        w_bce=float(cfg.get('w_bce',1)), w_iou=float(cfg.get('w_iou',1)),
        mask_threshold=float(cfg.get('masked_threshold',0)))


def compute_view_gradient(x, F_e, sigma0, opacity, target, renderer, campos, rcfg, color, lcfg, Fp=None, target_rgb=None):
    """Single view → dL/dx from alpha + RGB loss. Each call is graph-independent."""
    pred_alpha, pred_dict, x_t, _ = render(x, F_e, sigma0, opacity, renderer, campos, rcfg, color, True, Fp)
    if pred_alpha is None: return None, {}, None
    pred_np = pred_alpha.detach().cpu().numpy().copy()

    # Alpha loss
    loss_alpha, m = _alpha_loss(pred_alpha, target, lcfg)

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
                tgt_img = F_nn.interpolate(
                    tgt_img.permute(2,0,1).unsqueeze(0),
                    size=pred_img.shape[:2], mode='bilinear', align_corners=False
                )[0].permute(1,2,0)
        loss_rgb = w_rgb * ((pred_img - tgt_img) ** 2).mean()
        m['loss_rgb'] = float(loss_rgb.item())

    loss = loss_alpha + loss_rgb
    loss.backward()
    dLdx = x_t.grad.detach().cpu().numpy().astype(np.float32) if x_t.grad is not None else None
    m['dLdx_norm'] = float(np.linalg.norm(dLdx)) if dLdx is not None else 0.0
    return dLdx, m, pred_np


def compute_multiview_gradients(x, F_e, sigma0, opacity, targets, renderers, campos_list, rcfg, color, lcfg, Fp=None, target_rgbs=None):
    """8 views → summed dL/dx from alpha + RGB loss."""
    N = x.shape[0]
    dLdx_sum = np.zeros((N, 3), dtype=np.float32)
    bce_list, alpha_list, first_m = [], [], {}

    for v, (rend, tgt, cam) in enumerate(zip(renderers, targets, campos_list)):
        tgt_rgb = target_rgbs[v] if target_rgbs and v < len(target_rgbs) else None
        dv, m, pa = compute_view_gradient(x, F_e, sigma0, opacity, tgt, rend, cam, rcfg, color, lcfg, Fp, target_rgb=tgt_rgb)
        if dv is not None: dLdx_sum += dv
        bce_list.append(float(m.get('loss_bce', 0)))
        alpha_list.append(pa)
        if v == 0: first_m = m

    n = len(renderers)
    rgb_sum = sum(float(first_m.get('loss_rgb', 0)) for _ in range(1))  # first view only for now
    metrics = {
        'loss_bce': first_m.get('loss_bce', 0),
        'loss_bce_mv': sum(bce_list) / n,
        'loss_rgb': first_m.get('loss_rgb', 0),
        'n_views': n,
        'dLdx_norm': float(np.linalg.norm(dLdx_sum)),
    }
    rgb_str = f", rgb={metrics['loss_rgb']:.4f}" if metrics['loss_rgb'] > 0 else ""
    print(f"  [MV] {n} views, mean_bce={metrics['loss_bce_mv']:.1f}{rgb_str}")
    return dLdx_sum, metrics, bce_list, alpha_list


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

def compute_cohesion(x, k=32, strength=0.1):
    from scipy.spatial import cKDTree
    N = x.shape[0]
    dd, idx = cKDTree(x).query(x, k=min(k+1, N))
    gm = float(dd[:,1].mean())
    density = dd[:,1:8].mean(1) / (gm + 1e-8)
    iso = density * density[idx[:,1:8]].mean(1)
    disp = x[idx[:,1:]].mean(1) - x
    pull = np.clip(iso - 2.0, 0, None) * strength
    imp = (disp / (np.linalg.norm(disp, axis=1, keepdims=True) + 1e-8) * pull[:,None]).astype(np.float32)
    n_aff = int((pull > 0).sum())
    if n_aff: print(f"    [Cohesion] {n_aff:,}/{N:,}")
    return imp, {'floating': n_aff}


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
    render_penalty=None, target_rgbs=None,
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

    # ── Strain softening ──────────────────────────────────────────────────
    softening_rate = float(cfg.get('softening_rate', 0.0))
    if softening_rate > 0:
        F_cur = np.array(pc0.get_def_grads(), dtype=np.float32)
        strain = np.linalg.norm((F_cur - np.eye(3, dtype=np.float32)[None]).reshape(N, -1), axis=1)
        softening = 1.0 / (1.0 + softening_rate * strain)
        lam_base = float(cfg.get('lam_base', 38888.89))
        mu_base = float(cfg.get('mu_base', 58333.3))
        pc0.set_material(
            (lam_base * softening).astype(np.float32),
            (mu_base * softening).astype(np.float32),
        )

    # ── Physics ───────────────────────────────────────────────────────────
    if render_penalty is not None:
        res = cg.run_e2e_pass_batched(opt, render_penalty['dLdF'], render_penalty['dLdx'], has_render_grads=True)
        loss_phys = res['loss_physics']
    else:
        cg.run_optimization(opt)
        loss_phys = cg.end_layer_mass_loss()

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
    dFp = direction = cohesion_imp = diffused = None
    rm, bce_list, alpha_list = {}, [], []

    if len(renderers) > 0 and len(targets) > 0:
        dLdx, rm, bce_list, alpha_list = compute_multiview_gradients(
            x, F_e, sigma0, opacity, targets, renderers, campos_list, rcfg, color, lcfg, Fp,
            target_rgbs=target_rgbs)

        if dLdx is not None:
            dFp, direction, diffused, pm = diffuse_and_compute_plasticity(x, dLdx, cfg)
            rm.update(pm)

        cs = float(cfg.get('cohesion_strength', 0))
        if cs > 0:
            cohesion_imp, cm = compute_cohesion(x, int(cfg.get('cohesion_k', 32)), cs)
            rm.update(cm)

    # ── Alpha metric (primary view) ───────────────────────────────────────
    alpha_mse = 0.0
    if len(renderers) > 0 and len(targets) > 0:
        with torch.no_grad():
            pa, pred, _, _ = render(x, F_e, sigma0, opacity, renderers[0], campos_list[0], rcfg, color, False, Fp)
        if pa is not None:
            tgt = targets[0].to(pa.device)
            if tgt.shape != pa.shape:
                tgt = F_nn.interpolate(tgt[None,None], size=pa.shape, mode='bilinear', align_corners=False)[0,0]
            err = (pa - tgt)**2
            mt = float(lcfg.get('masked_threshold', 0))
            alpha_mse = float(err[(tgt>mt)|(pa>mt)].mean()) if mt > 0 else float(err.mean())
        if png and pred and pred.get('image') is not None:
            d = out_dir / f'ep{ep:03d}'; d.mkdir(parents=True, exist_ok=True)
            save_image_png(d / 'render.png', pred['image'])

    print(f"  alpha_mse={alpha_mse:.6f}")

    losses = {'ep': ep, 'loss_physics': float(loss_phys), 'alpha_mse': alpha_mse,
              'dx_mean': dx_m, 'dFc_mean': dFc_mean, 'dFc_max': dFc_max}
    for k in ['loss_bce', 'loss_bce_mv', 'loss_rgb', 'n_views', 'dLdx_norm',
              'active', 'diffused_ratio', 'dFp_norm', 'floating']:
        if k in rm: losses[k] = rm[k]

    dLdx_norms = np.linalg.norm(dLdx, axis=1) if dLdx is not None else None
    return losses, dFp, direction, cohesion_imp, dLdx_norms, bce_list, alpha_list, diffused
