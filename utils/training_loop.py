"""
Training Loop - Hard-Coupled Physics-to-Render Pipeline

Pipeline per episode:
  1. Physics (C++ MPM): optimization (with render grads from prev ep)
  2. Extract positions + F_elastic
  3. Deterministic bridge: h(x_T, F_T) -> Gaussian params (no learnable render params)
  4. Alpha loss (BCE + IoU) -> backward -> dL/dF, dL/dx
  5. Injection strategy (fixed_norm or option_a) -> scale & store grads for next ep
  6. Metrics + visualization

Injection modes (hard_coupling.mode):
  'fixed_norm' : g_inject = alpha_fixed * normalize(g_render)   [Exp 1]
  'option_a'   : x_target = x_T - eta*clip(normalize(dLdx))     [Phase 2]
                 dLdx_attr = 2*lambda_attr*(x_T - x_target), dLdF=0
"""

import numpy as np
import torch
import torch.nn.functional as F_nn
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

from utils.io_utils import save_image_png, save_depth_png
from utils.alpha_losses import combined_alpha_loss, compute_dt_map, compute_regularization


# ── PCA normals for RGB shading ──────────────────────────────────────────

def _compute_pca_normals(x: np.ndarray, k: int = 16) -> np.ndarray:
    """Estimate surface normals via local PCA (k nearest neighbors)."""
    from scipy.spatial import cKDTree
    N = x.shape[0]
    tree = cKDTree(x)
    _, idx = tree.query(x, k=min(k, N))

    normals = np.zeros_like(x)
    for i in range(N):
        neighbors = x[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]

    centroid = x.mean(axis=0)
    outward = x - centroid
    flip = (normals * outward).sum(axis=1) < 0
    normals[flip] *= -1

    return normals.astype(np.float32)


def _smooth_positions(x_target: np.ndarray, x_ref: np.ndarray, k: int = 8) -> np.ndarray:
    """KNN neighbor averaging to smooth the displacement field (x_target - x_ref).

    Averages displacements rather than absolute positions to avoid introducing
    a position-repulsion term from x_ref[i] - mean(x_ref[neighbors]).
    """
    from scipy.spatial import cKDTree
    delta = x_target - x_ref  # (N, 3) displacements
    tree = cKDTree(x_ref)
    _, idx = tree.query(x_ref, k=min(k + 1, len(x_ref)))
    delta_smoothed = delta[idx].mean(axis=1)  # (N, 3) averaged displacements
    return (x_ref + delta_smoothed).astype(np.float32)


# ── Deterministic bridge: h(x_T, F_T) -> render ──────────────────────────

def render_hard_coupled(
    x: np.ndarray,
    F_e: np.ndarray,
    sigma0_fixed: float,
    opacity_fixed: float,
    renderer: Any,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    training: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[Dict], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Deterministic physics-to-render bridge: g_t = h(x_t, F_t).

    Covariance: Sigma = F * Sigma0 * F^T  (paper hard coupling).
    No learnable parameters.
    """
    from renderer import compute_shading

    N = x.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_t = torch.from_numpy(x).float().to(device)
    F_t = torch.from_numpy(F_e).float().to(device)

    if training:
        x_t = x_t.requires_grad_(True)
        F_t = F_t.requires_grad_(True)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        Sigma0 = (sigma0_fixed ** 2) * torch.eye(3, device=device, dtype=torch.float32)
        F_Sigma0 = torch.matmul(F_t, Sigma0.unsqueeze(0).expand(N, -1, -1))
        cov = torch.bmm(F_Sigma0, F_t.transpose(1, 2))
        cov = cov + 1e-6 * torch.eye(3, device=device).unsqueeze(0)

        opacity = torch.full((N, 1), opacity_fixed, device=device, dtype=torch.float32)

        normals = _compute_pca_normals(x)
        rgb_np = compute_shading(
            x, normals,
            camera_pos=campos,
            light_cfg=render_cfg.get("lighting", {}),
            albedo_color=particle_color,
            model="phong"
        )
        rgb = torch.from_numpy(rgb_np).float().to(device)

        pred = renderer.render(
            x_t, cov, rgb=rgb, opacity=opacity,
            prefer_cov_precomp=True, return_torch=training,
        )

    pred_alpha = pred.get('alpha')
    if pred_alpha is not None:
        if isinstance(pred_alpha, np.ndarray):
            pred_alpha = torch.from_numpy(pred_alpha).to(device)
        if pred_alpha.dim() == 3:
            pred_alpha = pred_alpha[0]

    return pred_alpha, pred, x_t, F_t


# ── Extract raw render gradients ─────────────────────────────────────────

def extract_render_gradients(
    x: np.ndarray,
    F_e: np.ndarray,
    sigma0_fixed: float,
    opacity_fixed: float,
    target_alpha: torch.Tensor,
    renderer: Any,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    loss_cfg: Dict,
    reg_cfg: Optional[Dict] = None,
    dt_map: Optional[torch.Tensor] = None,
    x_prev: Optional[np.ndarray] = None,
    x_prev2: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Render with hard coupling, compute alpha loss + regularization, backward.

    Returns:
        dLdF: (N, 3, 3) numpy or None
        dLdx: (N, 3) numpy or None
        metrics: dict with loss values and gradient stats
    """
    reg_cfg = reg_cfg or {}

    pred_alpha, _, x_t, F_t = render_hard_coupled(
        x, F_e, sigma0_fixed, opacity_fixed,
        renderer, campos, render_cfg, particle_color,
        training=True,
    )

    if pred_alpha is None:
        return None, None, {'loss_total': 0.0}

    tgt = target_alpha.to(pred_alpha.device)
    if tgt.shape != pred_alpha.shape:
        tgt = F_nn.interpolate(
            tgt.unsqueeze(0).unsqueeze(0),
            size=pred_alpha.shape, mode='bilinear', align_corners=False
        )[0, 0]

    if dt_map is not None:
        dt_map_dev = dt_map.to(pred_alpha.device)
        if dt_map_dev.shape != pred_alpha.shape:
            dt_map_dev = F_nn.interpolate(
                dt_map_dev.unsqueeze(0).unsqueeze(0),
                size=pred_alpha.shape, mode='bilinear', align_corners=False
            )[0, 0]
    else:
        dt_map_dev = None

    alpha_loss, metrics = combined_alpha_loss(
        pred_alpha, tgt, dt_map=dt_map_dev,
        w_bce=float(loss_cfg.get('w_bce', 1.0)),
        w_iou=float(loss_cfg.get('w_iou', 1.0)),
        w_dt=float(loss_cfg.get('w_dt', 0.0)),
        mask_threshold=float(loss_cfg.get('masked_threshold', 0.0)),
    )

    x_prev_t = torch.from_numpy(x_prev).float().to(F_t.device) if x_prev is not None else None
    x_prev2_t = torch.from_numpy(x_prev2).float().to(F_t.device) if x_prev2 is not None else None

    reg_loss, reg_metrics = compute_regularization(
        F_t, x_t=x_t, x_prev=x_prev_t, x_prev2=x_prev2_t,
        w_vol=float(reg_cfg.get('w_vol', 1.0)),
        w_def=float(reg_cfg.get('w_def', 0.1)),
        w_temp=float(reg_cfg.get('w_temp', 0.01)),
    )
    metrics.update(reg_metrics)

    loss = alpha_loss + reg_loss
    metrics['loss_alpha'] = metrics.pop('loss_total')
    metrics['loss_total'] = float(loss.item())

    loss.backward()

    dLdF = None
    dLdx = None

    if F_t.grad is not None:
        dLdF = F_t.grad.detach().cpu().numpy().astype(np.float32)
        metrics['dLdF_norm'] = float(np.linalg.norm(dLdF))
        metrics['dLdF_nonzero'] = int((np.abs(dLdF.reshape(-1)) > 1e-10).sum())
    else:
        metrics['dLdF_norm'] = 0.0
        metrics['dLdF_nonzero'] = 0

    if x_t.grad is not None:
        dLdx = x_t.grad.detach().cpu().numpy().astype(np.float32)
        metrics['dLdx_norm'] = float(np.linalg.norm(dLdx))
        metrics['dLdx_nonzero'] = int((np.abs(dLdx.reshape(-1)) > 1e-10).sum())
    else:
        metrics['dLdx_norm'] = 0.0
        metrics['dLdx_nonzero'] = 0

    return dLdF, dLdx, metrics


# ── Injection strategies ──────────────────────────────────────────────────

def compute_inject_fixed_norm(
    dLdF_render: np.ndarray,
    dLdx_render: np.ndarray,
    alpha_fixed: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exp 1: Fixed-norm injection, independent of physics gradient magnitude.

    g_inject = alpha_fixed * normalize(g_render)

    Keeps render supervision alive even when physics gradients are small.
    """
    eps = 1e-8
    F_norm = np.linalg.norm(dLdF_render)
    x_norm = np.linalg.norm(dLdx_render)
    dLdF_scaled = alpha_fixed * dLdF_render / (F_norm + eps)
    dLdx_scaled = alpha_fixed * dLdx_render / (x_norm + eps)
    return dLdF_scaled, dLdx_scaled


def compute_attractor_grads(
    x_T: np.ndarray,
    dLdx_render: np.ndarray,
    eta: float,
    lambda_attr: float,
    tau: float,
    smooth_k: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Option A: Target-state proximal attractor.

    1. Compute x_target = x_T - eta * clip(normalize(dLdx), tau)
    2. Optionally smooth x_target via KNN (on displacements, not absolute positions)
    3. Return attractor gradient: dLdx_attr = 2*lambda*(x_T - x_target)
       dLdF = 0 (no F-space injection)

    Separates 'where to go' (render) from 'how to get there' (physics).
    """
    eps = 1e-8
    x_norm = np.linalg.norm(dLdx_render)
    g_hat = dLdx_render / (x_norm + eps)
    g_hat_clipped = np.clip(g_hat, -tau, tau)
    x_target = x_T - eta * g_hat_clipped

    if smooth_k > 0:
        x_target = _smooth_positions(x_target, x_T, k=smooth_k)

    N = x_T.shape[0]
    dLdx_attr = (2.0 * lambda_attr * (x_T - x_target)).astype(np.float32)
    dLdF_zero = np.zeros((N, 3, 3), dtype=np.float32)

    return dLdF_zero, dLdx_attr


def compute_attractor_grads_v2(
    x_T: np.ndarray,
    dLdx_render: np.ndarray,
    eta: float,
    lambda_attr: float,
    smooth_k: int = 32,
    clip_pct: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Option A v2: Surface-aware target-state proximal attractor.

    Key insight from Exp A: only ~10% of particles have nonzero render gradient
    (visually contributing particles). These have coherence=0.28 raw, 0.64 after
    clip+knn32 — far above the noisy threshold. The other 90% have zero gradient
    and dilute the signal.

    Algorithm:
    1. Identify active particles: ||dLdx_i|| > 0
    2. On active subset: clip p95 outliers, then KNN-smooth displacements
    3. Globally normalize the smoothed field
    4. x_target = x_T - eta * g_smooth  (active particles only)
    5. dLdx_attr = 2*lambda*(x_T - x_target)  (zero for inactive particles)
       dLdF = 0
    """
    from scipy.spatial import cKDTree as _KDTree
    eps = 1e-8
    N = x_T.shape[0]

    # Step 1: identify visually contributing particles
    per_norms = np.linalg.norm(dLdx_render, axis=1)
    mask = per_norms > 1e-10
    n_active = int(mask.sum())

    if n_active < 10:
        return np.zeros((N, 3, 3), dtype=np.float32), np.zeros((N, 3), dtype=np.float32)

    g_active = dLdx_render[mask].copy()  # (n_active, 3)
    x_active = x_T[mask]                 # (n_active, 3)

    # Step 2: clip per-particle outliers
    norms_a = np.linalg.norm(g_active, axis=1)
    thresh = np.percentile(norms_a, clip_pct)
    if thresh > eps:
        scale = np.minimum(1.0, thresh / (norms_a + eps))
        g_active = g_active * scale[:, None]

    # Step 3: KNN-smooth displacement field on active subset
    if smooth_k > 0 and n_active > smooth_k + 1:
        tree = _KDTree(x_active)
        _, idx = tree.query(x_active, k=min(smooth_k + 1, n_active))
        g_active = g_active[idx].mean(axis=1).astype(np.float32)

    # Step 4: global normalize
    g_norm = np.linalg.norm(g_active)
    g_hat = g_active / (g_norm + eps)

    # Step 5: construct x_target and attractor
    x_target_active = x_active - eta * g_hat
    dLdx_attr_active = (2.0 * lambda_attr * (x_active - x_target_active)).astype(np.float32)

    # Assemble: inactive particles get zero
    dLdx_attr = np.zeros((N, 3), dtype=np.float32)
    dLdx_attr[mask] = dLdx_attr_active

    print(f"    [v2] active={n_active:,}/{N:,} ({100*n_active/N:.1f}%), "
          f"inject_x_active={float(np.linalg.norm(dLdx_attr_active)):.4f}")

    return np.zeros((N, 3, 3), dtype=np.float32), dLdx_attr


# ── Episode runner ────────────────────────────────────────────────────────

def run_episode(
    ep: int,
    cg: Any,
    opt: Any,
    sigma0_fixed: float,
    opacity_fixed: float,
    rs_full: Dict,
    renderer: Any,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    out_dir: Path,
    png_enabled: bool,
    target_alpha: Optional[torch.Tensor] = None,
    cam_cfg: Optional[Dict] = None,
    stored_render_grads: Optional[Dict] = None,
    hard_coupling_cfg: Optional[Dict] = None,
    dt_map: Optional[torch.Tensor] = None,
    x_prev: Optional[np.ndarray] = None,
    x_prev2: Optional[np.ndarray] = None,
) -> Tuple[Dict, Optional[Dict]]:
    """
    Run one episode: physics (with render feedback) + render grad extraction.

    Returns:
        losses: dict with all metrics
        new_render_grads: dict {'dLdF': ..., 'dLdx': ...} for next ep, or None
    """
    num_timesteps = int(opt.num_timesteps)
    hard_coupling_cfg = hard_coupling_cfg or {}

    loss_cfg = hard_coupling_cfg.get('loss_weights', {})
    reg_cfg = hard_coupling_cfg.get('regularization', {})
    render_start_ep = int(hard_coupling_cfg.get('render_start_ep', 5))
    mode = hard_coupling_cfg.get('mode', 'fixed_norm')

    # ── 0. Save initial state ─────────────────────────────────────────────
    pc0 = cg.get_point_cloud(0)
    x0 = np.array(pc0.get_positions(), dtype=np.float32).copy()

    # ── 1. Physics pass ───────────────────────────────────────────────────
    inject = (stored_render_grads is not None and ep >= render_start_ep)

    if inject:
        dLdF_in = stored_render_grads['dLdF']
        dLdx_in = stored_render_grads['dLdx']
        result_e2e = cg.run_e2e_pass_batched(opt, dLdF_in, dLdx_in, has_render_grads=True)
        loss_physics = result_e2e['loss_physics']
        print(f"  [Inject] mode={mode}, active")
    else:
        cg.run_optimization(opt)
        loss_physics = cg.end_layer_mass_loss()

    # ── 2. Extract final physics state ────────────────────────────────────
    pc = cg.get_point_cloud(num_timesteps - 1)
    x = np.ascontiguousarray(np.array(pc.get_positions(), dtype=np.float32))
    try:
        F_e = pc.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy()
    except Exception:
        F_e = np.array(pc.get_def_grads(), dtype=np.float32)
    F_e = np.ascontiguousarray(F_e.astype(np.float32))
    N = x.shape[0]

    det_Fe = np.linalg.det(F_e)
    dx_norms = np.linalg.norm(x - x0, axis=1)
    Fe_I = F_e - np.eye(3, dtype=np.float32)[None]
    dFe_norms = np.linalg.norm(Fe_I.reshape(N, -1), axis=1)

    print(f"\n[Ep {ep:03d}] physics_loss={loss_physics:.2f}, N={N:,}")
    print(f"  J(F_e): [{det_Fe.min():.3f}, {det_Fe.max():.3f}]")
    print(f"  dx: mean={dx_norms.mean():.4f}, max={dx_norms.max():.4f}")
    print(f"  ||F_e-I||: mean={dFe_norms.mean():.4f}, max={dFe_norms.max():.4f}")

    # ── 3. Compute render grads + injection for NEXT episode ──────────────
    new_render_grads = None
    render_metrics = {}
    render_active = (ep >= render_start_ep)

    if renderer is not None and target_alpha is not None:
        dLdF_render, dLdx_render, render_metrics = extract_render_gradients(
            x, F_e, sigma0_fixed, opacity_fixed,
            target_alpha, renderer, campos, render_cfg, particle_color,
            loss_cfg, reg_cfg=reg_cfg, dt_map=dt_map,
            x_prev=x_prev, x_prev2=x_prev2,
        )

        if render_active and dLdF_render is not None and dLdx_render is not None:
            dLdF_norm = render_metrics.get('dLdF_norm', 0.0)
            dLdx_norm = render_metrics.get('dLdx_norm', 0.0)

            # Injection strategy dispatch
            if mode == 'fixed_norm':
                alpha_fixed = float(hard_coupling_cfg.get('alpha_fixed', 0.5))
                dLdF_scaled, dLdx_scaled = compute_inject_fixed_norm(
                    dLdF_render, dLdx_render, alpha_fixed
                )
                render_metrics['inject_mode'] = 'fixed_norm'
                render_metrics['alpha_fixed'] = alpha_fixed

            elif mode == 'option_a':
                eta = float(hard_coupling_cfg.get('eta', 0.1))
                lambda_attr = float(hard_coupling_cfg.get('lambda_attr', 1.0))
                tau = float(hard_coupling_cfg.get('tau', 0.5))
                smooth_k = int(hard_coupling_cfg.get('smooth_k', 0))
                dLdF_scaled, dLdx_scaled = compute_attractor_grads(
                    x, dLdx_render, eta, lambda_attr, tau, smooth_k
                )
                render_metrics['inject_mode'] = 'option_a'
                render_metrics['eta'] = eta
                render_metrics['lambda_attr'] = lambda_attr

            elif mode == 'option_a_v2':
                eta = float(hard_coupling_cfg.get('eta', 0.1))
                lambda_attr = float(hard_coupling_cfg.get('lambda_attr', 1.0))
                smooth_k = int(hard_coupling_cfg.get('smooth_k', 32))
                clip_pct = float(hard_coupling_cfg.get('clip_pct', 95.0))
                dLdF_scaled, dLdx_scaled = compute_attractor_grads_v2(
                    x, dLdx_render, eta, lambda_attr, smooth_k, clip_pct
                )
                render_metrics['inject_mode'] = 'option_a_v2'
                render_metrics['eta'] = eta
                render_metrics['lambda_attr'] = lambda_attr
                render_metrics['smooth_k'] = smooth_k

            else:
                raise ValueError(f"Unknown injection mode: '{mode}'. Use 'fixed_norm', 'option_a', or 'option_a_v2'.")

            inject_F_norm = float(np.linalg.norm(dLdF_scaled))
            inject_x_norm = float(np.linalg.norm(dLdx_scaled))
            render_metrics['inject_F_norm'] = inject_F_norm
            render_metrics['inject_x_norm'] = inject_x_norm

            # Cosine similarity: render vs physics gradient (diagnostic)
            try:
                _, dLdx_phys_raw = cg.get_last_layer_phys_gradients()
                dLdx_phys_np = np.array(dLdx_phys_raw, dtype=np.float32).reshape(-1)
                render_flat = dLdx_render.reshape(-1)
                p_norm = np.linalg.norm(dLdx_phys_np)
                r_norm = np.linalg.norm(render_flat)
                if p_norm > 1e-10 and r_norm > 1e-10:
                    cos_sim = float(np.dot(dLdx_phys_np, render_flat) / (p_norm * r_norm))
                    render_metrics['cos_phys_render_x'] = cos_sim
                    print(f"  cos(phys, render)_x = {cos_sim:.4f}")
            except Exception:
                pass

            print(f"  [Render grad] mode={mode}, ||dLdF||={dLdF_norm:.1f}, "
                  f"||dLdx||={dLdx_norm:.2f}, inject_F={inject_F_norm:.3f}, "
                  f"inject_x={inject_x_norm:.3f}")

            new_render_grads = {
                'dLdF': np.ascontiguousarray(dLdF_scaled),
                'dLdx': np.ascontiguousarray(dLdx_scaled),
            }

        elif not render_active:
            print(f"  [Warmup] ep{ep} < render_start_ep={render_start_ep}, "
                  f"render_loss={render_metrics.get('loss_total', 0):.4f} (monitor only)")

    # ── 4. Final render + metrics ─────────────────────────────────────────
    alpha_mse = 0.0
    if renderer is not None:
        with torch.no_grad():
            pred_alpha, pred, _, _ = render_hard_coupled(
                x, F_e, sigma0_fixed, opacity_fixed,
                renderer, campos, render_cfg, particle_color,
                training=False,
            )

        if pred_alpha is not None and target_alpha is not None:
            tgt = target_alpha.to(pred_alpha.device)
            if tgt.shape != pred_alpha.shape:
                tgt = F_nn.interpolate(tgt.unsqueeze(0).unsqueeze(0),
                                       size=pred_alpha.shape, mode='bilinear',
                                       align_corners=False)[0, 0]
            err = (pred_alpha - tgt) ** 2

            mask_thresh = float(loss_cfg.get('masked_threshold', 0.0))
            if mask_thresh > 0:
                mask = (tgt > mask_thresh) | (pred_alpha > mask_thresh)
                alpha_mse = float(err[mask].mean().item()) if mask.any() else float(err.mean().item())
                fg_pix = int(mask.sum())
                print(f"  [Masked metric] fg={fg_pix:,}/{err.numel():,} "
                      f"({100*fg_pix/err.numel():.1f}%), alpha_mse={alpha_mse:.6f}")
            else:
                alpha_mse = float(err.mean().item())

        if png_enabled and pred is not None:
            ep_dir = out_dir / f"ep{ep:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            if pred.get('image') is not None:
                save_image_png(ep_dir / 'render.png', pred['image'])
            if pred.get('alpha') is not None:
                save_image_png(ep_dir / 'alpha.png', pred['alpha'])
            if pred.get('depth') is not None:
                save_depth_png(ep_dir / 'depth.png', pred['depth'])
            if target_alpha is not None and pred.get('alpha') is not None:
                pred_a_np = pred['alpha'] if isinstance(pred['alpha'], np.ndarray) else pred['alpha'].cpu().numpy()
                if pred_a_np.ndim == 3:
                    pred_a_np = pred_a_np[0]
                tgt_np = target_alpha.cpu().numpy()
                if tgt_np.shape != pred_a_np.shape:
                    from PIL import Image
                    tgt_pil = Image.fromarray((tgt_np * 255).astype(np.uint8))
                    tgt_pil = tgt_pil.resize((pred_a_np.shape[1], pred_a_np.shape[0]), Image.BILINEAR)
                    tgt_np = np.array(tgt_pil).astype(np.float32) / 255.0
                err_np = np.abs(pred_a_np - tgt_np)
                heatmap = np.stack([err_np, np.zeros_like(err_np), np.zeros_like(err_np)], axis=-1)
                save_image_png(ep_dir / 'alpha_error.png', heatmap)
            print(f"  Saved -> {ep_dir}/")

    print(f"  alpha_mse={alpha_mse:.6f}, render_active={render_active}")

    losses = {
        'loss_physics': float(loss_physics),
        'alpha_mse': alpha_mse,
        'render_active': render_active,
        'sigma0_fixed': sigma0_fixed,
        'opacity_fixed': opacity_fixed,
        'dx_mean': float(dx_norms.mean()),
        'dx_max': float(dx_norms.max()),
        'dFe_mean': float(dFe_norms.mean()),
        'dFe_max': float(dFe_norms.max()),
        'det_min': float(det_Fe.min()),
        'det_max': float(det_Fe.max()),
        **render_metrics,
    }
    return losses, new_render_grads
