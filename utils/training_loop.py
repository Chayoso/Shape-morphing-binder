"""
Training Loop - Physics + F_plastic Appearance Episode

Architecture:
  1. Physics: cg.run_optimization(opt) → EndLayerMassLoss drives morphing
  2. Extract: x, F_elastic from final physics state
  3. [E2E] Render gradient injection: dL/dF + dL/dx → C++ backward
  4. Appearance: F_plastic analytical update (surface-aligned Gaussians)
  5. Render: F_total = F_elastic @ F_plastic → Σ = F_total F_total^T
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

from utils.plastic_appearance import apply_appearance_update
from utils.io_utils import save_image_png, save_depth_png


def _render_grad_pass(
    x: np.ndarray,                   # (N, 3) particle positions
    F_total: np.ndarray,             # (N, 3, 3) deformation gradient
    target_alpha: torch.Tensor,      # (H, W) target silhouette
    renderer: Any,
    rs_full: Dict,
    ema_state: Dict,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    ep: int,
    render_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Differentiable render pass: compute dL/dx and dL/dF from silhouette loss.

    Returns:
        dLdx:  (N, 3) render gradient w.r.t. positions
        dLdF:  (N, 3, 3) render gradient w.r.t. deformation gradient
        loss:  scalar render loss (float)
    """
    from sampling import upsample
    from utils.rendering_utils import prepare_rendering_inputs
    import torch.nn.functional as F_nn

    N = x.shape[0]
    x_t = torch.from_numpy(x).requires_grad_(True)
    F_t = torch.from_numpy(F_total).requires_grad_(True)

    result = upsample(x_t, F_t, cfg=rs_full, state=ema_state,
                      seed=9000 + ep, return_torch=True, current_episode=ep)

    mu  = result.get('points')
    cov = result.get('cov')

    if mu is None or cov is None:
        print("  [E2E] upsample returned None — skipping render grad")
        return np.zeros_like(x), np.zeros((N, 3, 3), dtype=np.float32), 0.0

    rgb = prepare_rendering_inputs(mu, result, campos, render_cfg, particle_color)
    pred = renderer.render(mu, cov, rgb=rgb, prefer_cov_precomp=True, return_torch=True)

    pred_alpha = pred.get('alpha')
    if pred_alpha is None:
        print("  [E2E] renderer returned no alpha — skipping render grad")
        return np.zeros_like(x), np.zeros((N, 3, 3), dtype=np.float32), 0.0

    if pred_alpha.dim() == 3:
        pred_alpha = pred_alpha[0]

    tgt = target_alpha.to(pred_alpha.device)
    if tgt.shape != pred_alpha.shape:
        tgt = F_nn.interpolate(tgt.unsqueeze(0).unsqueeze(0),
                               size=pred_alpha.shape, mode='bilinear',
                               align_corners=False)[0, 0]

    loss_render = render_weight * torch.mean((pred_alpha - tgt) ** 2)
    loss_render.backward()

    dLdx = x_t.grad.detach().cpu().numpy().astype(np.float32) if x_t.grad is not None else np.zeros_like(x)
    dLdF = F_t.grad.detach().cpu().numpy().astype(np.float32) if F_t.grad is not None else np.zeros((N, 3, 3), dtype=np.float32)
    loss_val = loss_render.item()

    print(f"  [E2E] render_loss={loss_val:.4f}, "
          f"||dLdx||={np.linalg.norm(dLdx):.3e}, "
          f"||dLdF||={np.linalg.norm(dLdF):.3e}")
    return dLdx, dLdF, loss_val


def run_episode(
    ep: int,
    cg: Any,
    opt: Any,
    F_plastic: np.ndarray,           # (N, 3, 3)
    surface_pts: np.ndarray,         # (M, 3)
    surface_nrm: np.ndarray,         # (M, 3)
    appearance_cfg: Dict,
    rs_full: Dict,
    renderer: Any,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    out_dir: Path,
    png_enabled: bool,
    ema_state: Dict,
    target_alpha: Optional[torch.Tensor] = None,
    e2e_cfg: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Run one physics episode + optional E2E render injection + F_plastic update.

    Returns:
        F_plastic: (N, 3, 3) updated plastic state
        losses:    dict with all metrics
    """
    num_timesteps = int(opt.num_timesteps)

    e2e_cfg      = e2e_cfg or {}
    e2e_enabled  = bool(e2e_cfg.get('enabled', False)) and target_alpha is not None
    e2e_weight   = float(e2e_cfg.get('render_weight', 1.0))
    e2e_gain     = float(e2e_cfg.get('render_gain', 0.1))
    e2e_start_ep = int(e2e_cfg.get('start_ep', 0))

    # ── 0. Save initial state (for dx, dF metrics) ──────────────────────────
    pc0 = cg.get_point_cloud(0)
    x0 = np.array(pc0.get_positions(), dtype=np.float32).copy()

    # ── 1. Physics pass ─────────────────────────────────────────────────────
    cg.run_optimization(opt)
    loss_physics = cg.end_layer_mass_loss()

    # ── 2. Extract final physics state ──────────────────────────────────────
    pc = cg.get_point_cloud(num_timesteps - 1)
    x  = np.ascontiguousarray(np.array(pc.get_positions(), dtype=np.float32))
    try:
        F_e = pc.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy()
    except Exception:
        F_e = np.array(pc.get_def_grads(), dtype=np.float32)
    F_e = np.ascontiguousarray(F_e.astype(np.float32))
    N = x.shape[0]

    det_Fe = np.linalg.det(F_e)
    print(f"\n[Ep {ep:03d}] physics_loss={loss_physics:.2f}, N={N:,}")
    print(f"  J(F_e):   [{det_Fe.min():.3f}, {det_Fe.max():.3f}]")

    # ── Displacement & deformation metrics ──────────────────────────────────
    dx_vec = x - x0
    dx_norms = np.linalg.norm(dx_vec, axis=1)
    Fe_I = F_e - np.eye(3, dtype=np.float32)[None]
    dFe_norms = np.linalg.norm(Fe_I.reshape(N, -1), axis=1)
    print(f"  dx:  mean={dx_norms.mean():.4f}, max={dx_norms.max():.4f}")
    print(f"  dFe: mean={dFe_norms.mean():.4f}, max={dFe_norms.max():.4f}")

    # ── 3. E2E render gradient injection (optional) ─────────────────────────
    loss_render = 0.0
    dFc_render_diag = {}

    if e2e_enabled and ep >= e2e_start_ep and renderer is not None:
        x_pass1 = x.copy()
        F_e_pass1 = F_e.copy()

        F_total = np.einsum('nij,njk->nik', F_e, F_plastic).astype(np.float32)

        render_dLdx, render_dLdF, loss_render = _render_grad_pass(
            x, F_total, target_alpha, renderer, rs_full, ema_state,
            campos, render_cfg, particle_color, ep, render_weight=e2e_weight
        )

        cg.set_render_gain(e2e_gain)
        cg.set_render_gradients(render_dLdF, render_dLdx)
        cg.run_optimization(opt, skip_setup=True)
        cg.clear_render_gradients()

        pc = cg.get_point_cloud(num_timesteps - 1)
        x  = np.ascontiguousarray(np.array(pc.get_positions(), dtype=np.float32))
        try:
            F_e = pc.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy()
        except Exception:
            F_e = np.array(pc.get_def_grads(), dtype=np.float32)
        F_e = np.ascontiguousarray(F_e.astype(np.float32))

        # Measure render's contribution to dFc (pass2 - pass1)
        dFc_render = F_e - F_e_pass1
        dFc_norms = np.linalg.norm(dFc_render.reshape(N, -1), axis=1)
        dx_render = x - x_pass1
        dx_render_norms = np.linalg.norm(dx_render, axis=1)
        print(f"  [E2E] Pass 2 injected (dL/dF+dL/dx), gain={e2e_gain}")
        print(f"  [E2E] dFc_render: mean={dFc_norms.mean():.6f}, max={dFc_norms.max():.6f}")
        print(f"  [E2E] dx_render:  mean={dx_render_norms.mean():.6f}, max={dx_render_norms.max():.6f}")
        dFc_render_diag = {
            'dFc_render_mean': float(dFc_norms.mean()),
            'dFc_render_max': float(dFc_norms.max()),
            'dx_render_mean': float(dx_render_norms.mean()),
            'dx_render_max': float(dx_render_norms.max()),
        }

        # Save per-particle data for gradient heatmap
        ep_dir = out_dir / f"ep{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'x': torch.from_numpy(x),
            'dFc_render_norm': torch.from_numpy(dFc_norms),
            'dx_render_norm': torch.from_numpy(dx_render_norms),
        }, ep_dir / 'grad_data.pt')

    # ── 4. Analytical F_plastic update ──────────────────────────────────────
    F_plastic, app_diag = apply_appearance_update(
        x, F_e, F_plastic, surface_pts, surface_nrm, appearance_cfg
    )
    print(f"  F_plastic: dF_mean={app_diag['dF_mean']:.4f}, "
          f"sv=[{app_diag['sv_min']:.3f},{app_diag['sv_max']:.3f}], "
          f"det=[{app_diag['det_min']:.3f},{app_diag['det_mean']:.3f},{app_diag['det_max']:.3f}]")

    # ── 5. Render + metrics ─────────────────────────────────────────────────
    alpha_mse = 0.0
    if renderer is not None:
        from sampling import upsample
        from utils.rendering_utils import prepare_rendering_inputs
        import torch.nn.functional as F_nn

        F_total = np.einsum('nij,njk->nik', F_e, F_plastic).astype(np.float32)
        x_t = torch.from_numpy(x)
        F_t = torch.from_numpy(F_total)

        with torch.no_grad():
            result = upsample(x_t, F_t, cfg=rs_full, state=ema_state,
                              seed=9999 + ep, return_torch=True, current_episode=ep)

        mu  = result.get('points')
        cov = result.get('cov')

        if mu is not None and cov is not None:
            rgb = prepare_rendering_inputs(mu, result, campos, render_cfg, particle_color)
            with torch.no_grad():
                pred = renderer.render(mu, cov, rgb=rgb,
                                       prefer_cov_precomp=True, return_torch=False)

            # Alpha MSE vs target
            if target_alpha is not None and pred.get('alpha') is not None:
                pred_a = pred['alpha']
                if isinstance(pred_a, np.ndarray):
                    pred_a = torch.from_numpy(pred_a)
                if pred_a.dim() == 3:
                    pred_a = pred_a[0]
                tgt = target_alpha
                if tgt.shape != pred_a.shape:
                    tgt = F_nn.interpolate(tgt.unsqueeze(0).unsqueeze(0),
                                           size=pred_a.shape, mode='bilinear',
                                           align_corners=False)[0, 0]
                alpha_mse = float(torch.mean((pred_a - tgt) ** 2).item())

            # Save PNGs
            if png_enabled:
                ep_dir = out_dir / f"ep{ep:03d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                if pred.get('image') is not None:
                    save_image_png(ep_dir / 'render.png', pred['image'])
                if pred.get('alpha') is not None:
                    save_image_png(ep_dir / 'alpha.png', pred['alpha'])
                if pred.get('depth') is not None:
                    save_depth_png(ep_dir / 'depth.png', pred['depth'])
                # Alpha error heatmap
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
                    err = np.abs(pred_a_np - tgt_np)
                    heatmap = np.stack([err, np.zeros_like(err), np.zeros_like(err)], axis=-1)
                    save_image_png(ep_dir / 'alpha_error.png', heatmap)
                print(f"  Saved → {ep_dir}/")

    print(f"  alpha_mse={alpha_mse:.6f}")

    losses = {
        'loss_physics': float(loss_physics),
        'alpha_mse': alpha_mse,
        'dx_mean': float(dx_norms.mean()),
        'dx_max': float(dx_norms.max()),
        'dFe_mean': float(dFe_norms.mean()),
        'dFe_max': float(dFe_norms.max()),
        'det_min': float(det_Fe.min()),
        'det_max': float(det_Fe.max()),
        **app_diag,
    }
    if e2e_enabled:
        losses['loss_render'] = float(loss_render)
    losses.update(dFc_render_diag)
    return F_plastic, losses
