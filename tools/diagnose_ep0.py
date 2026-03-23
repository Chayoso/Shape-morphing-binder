"""
Diagnose ep0: run 1 physics episode, then visualize
  1) dF heatmap — where F_elastic changed from I (physics deformation)
  2) Render loss heatmap — post-physics alpha vs target alpha
  3) Per-region loss breakdown (red/blue/green)

Usage:
    python tools/diagnose_ep0.py -c configs/sphere_to_bunny.yaml
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import math
import numpy as np
import torch
import yaml
from pathlib import Path

from sampling import default_cfg, upsample
from sampling.utils.config_adapter import adapt_config
from utils.physics_utils import (
    build_opt_input, initialize_point_clouds, initialize_grids,
    initialize_comp_graph,
)
from utils.rendering_utils import prepare_rendering_inputs
from utils.io_utils import save_image_png
from renderer import GSRenderer3DGS, make_matrices_from_yaml, compute_shading
import diffmpm_bindings


def auto_frame_camera(positions, cam_cfg, render_cfg, res=256, fill_ratio=0.80):
    """Create a square camera (res x res) framed on positions."""
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    bsphere_r = float(radii.max())

    lookat = cam_cfg.get('lookat', {})
    eye = np.array(lookat.get('eye', [20, -25, 12.5]), dtype=np.float64)
    target = np.array(lookat.get('target', [0, 0, 0]), dtype=np.float64)
    up = np.array(lookat.get('up', [0, 0, 1]), dtype=np.float64)

    direction = eye - target
    direction = direction / np.linalg.norm(direction)

    W, H = res, res
    fx = fy = float(res)
    half_fov = math.atan(0.5)

    dist = bsphere_r / (fill_ratio * math.tan(half_fov))
    new_eye = center + direction * dist

    new_cam_cfg = dict(cam_cfg)
    new_cam_cfg['width'] = W
    new_cam_cfg['height'] = H
    new_cam_cfg['fx'] = fx
    new_cam_cfg['fy'] = fy
    new_cam_cfg['cx'] = W / 2.0
    new_cam_cfg['cy'] = H / 2.0
    new_cam_cfg['lookat'] = {
        'eye': new_eye.tolist(),
        'target': center.tolist(),
        'up': up.tolist(),
    }

    W_out, H_out, tanfovx, tanfovy, view_T, proj_T, campos = make_matrices_from_yaml(new_cam_cfg)

    bg = render_cfg.get("bg", [1.0, 1.0, 1.0])
    renderer = GSRenderer3DGS(
        W_out, H_out, tanfovx, tanfovy, view_T, proj_T, campos,
        bg=tuple(bg), sh_degree=0, scale_modifier=1.0,
        prefiltered=False, debug=False,
        antialiasing=render_cfg.get("antialiasing", False),
        device="cuda"
    )
    return renderer, campos, W_out, H_out


def render_particles(positions, F_field, rs_full, renderer, campos, render_cfg, particle_color, label="", ep=0):
    """Render particle positions with given F field → alpha + image."""
    N = positions.shape[0]
    x_t = torch.from_numpy(positions.astype(np.float32))
    F_t = torch.from_numpy(F_field.astype(np.float32))

    with torch.no_grad():
        result = upsample(x_t, F_t, cfg=rs_full, seed=42, return_torch=True, current_episode=ep)

    mu = result['points']
    cov = result['cov']

    rgb = prepare_rendering_inputs(mu, result, campos, render_cfg, particle_color)

    with torch.no_grad():
        pred = renderer.render(mu, cov, rgb=rgb, prefer_cov_precomp=True, return_torch=False)

    alpha = pred['alpha']
    image = pred.get('image')

    if not torch.is_tensor(alpha):
        alpha = torch.from_numpy(np.asarray(alpha, dtype=np.float32))
    if alpha.dim() == 3:
        alpha = alpha[0]

    print(f"  [{label}] alpha: [{alpha.min():.4f}, {alpha.max():.4f}], coverage={( alpha > 0.5).float().mean().item():.1%}")
    return alpha.detach().cpu(), image


def render_scalar_heatmap(positions, F_field, scalar_values, rs_full, renderer, campos, render_cfg, label="", ep=0, cmap_name='hot'):
    """
    Render per-particle scalar values as colored Gaussians through 3DGS.
    Uses the same camera/renderer as normal renders, so projection matches exactly.

    scalar_values: (N,) per-particle values, will be normalized to [0,1] and colormapped.
    Returns: (H, W, 3) numpy image.
    """
    import matplotlib.cm as cm

    N = positions.shape[0]
    x_t = torch.from_numpy(positions.astype(np.float32))
    F_t = torch.from_numpy(F_field.astype(np.float32))

    with torch.no_grad():
        result = upsample(x_t, F_t, cfg=rs_full, seed=42, return_torch=True, current_episode=ep)

    mu = result['points']
    cov = result['cov']
    M = mu.shape[0]

    # Map scalar to color via colormap
    # If simple pipeline (no subdivision), M == N. Otherwise need to map.
    if M == N:
        vals = scalar_values
    else:
        # Subdivided: repeat parent values (upsample doesn't change ordering)
        ratio = M // N
        vals = np.repeat(scalar_values, ratio)[:M]

    v_max = max(vals.max(), 1e-8)
    v_norm = vals / v_max

    cmap = cm.get_cmap(cmap_name)
    colors = cmap(v_norm)[:, :3].astype(np.float32)  # (M, 3)
    rgb = torch.from_numpy(colors).to(mu.device)

    with torch.no_grad():
        pred = renderer.render(mu, cov, rgb=rgb, prefer_cov_precomp=True, return_torch=False)

    image = pred.get('image')
    alpha = pred.get('alpha')

    if not torch.is_tensor(alpha):
        alpha = torch.from_numpy(np.asarray(alpha, dtype=np.float32))
    if alpha.dim() == 3:
        alpha = alpha[0]

    print(f"  [{label}] scalar range=[0, {v_max:.6f}], Gaussians={M:,}")
    return image, alpha.detach().cpu(), v_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-n', '--num-episodes', type=int, default=40)
    parser.add_argument('--res', type=int, default=512, help='Render resolution (square)')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import json

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg.get('output_dir', 'output/run')) / 'diagnose_physics'
    out_dir.mkdir(parents=True, exist_ok=True)

    num_eps = args.num_episodes

    # ── Physics init ──
    opt = build_opt_input(cfg)
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
    cg = initialize_comp_graph(input_pc, input_grid, target_grid)

    loss0 = cg.end_layer_mass_loss()
    x_input = np.array(input_pc.get_positions(), dtype=np.float32)
    x_target = np.array(target_pc.get_positions(), dtype=np.float32)
    N = x_input.shape[0]
    num_timesteps = int(opt.num_timesteps)

    print(f"[Init] EndLayerMassLoss = {loss0:.2f}")
    print(f"Input: {N:,} particles, Target: {x_target.shape[0]:,} particles")
    print(f"Running {num_eps} episodes, render res={args.res}")

    # ── Upsample config ──
    rs = default_cfg()
    rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))
    sim_cfg = cfg.get('simulation', {})
    rs['physics_grid'] = {
        'grid_min': sim_cfg.get('grid_min_point', [-16, -16, -16]),
        'grid_max': sim_cfg.get('grid_max_point', [16, 16, 16]),
        'grid_dx':  sim_cfg.get('grid_dx', 1.0),
    }
    rs.setdefault('upsample', {}).setdefault('covariance', {})
    rs['upsample']['covariance']['sigma0'] = 0.08
    rs['upsample']['covariance']['sigma_isotropic'] = 0.08

    # ── Auto-frame camera (bunny-based, two-pass) ──
    cam_cfg = cfg.get('camera', {})
    render_cfg = cfg.get('render', {})
    particle_color = render_cfg.get('particle_color', [0.27, 0.51, 0.71])
    F_id = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    F_id_target = np.tile(np.eye(3, dtype=np.float32), (x_target.shape[0], 1, 1))

    # Pass 1: measure at low res
    renderer_tmp, campos_tmp, W_tmp, H_tmp = auto_frame_camera(
        x_target, cam_cfg, render_cfg, res=256, fill_ratio=0.80)
    alpha_tmp, _ = render_particles(
        x_target, F_id_target, rs, renderer_tmp, campos_tmp, render_cfg, particle_color, "measure")
    vis = alpha_tmp > 0.01
    rows_v = vis.any(dim=1)
    cols_v = vis.any(dim=0)
    r0, r1 = torch.where(rows_v)[0][[0, -1]].tolist()
    c0, c1 = torch.where(cols_v)[0][[0, -1]].tolist()
    current_fill = max((r1 - r0) / H_tmp, (c1 - c0) / W_tmp)
    adjusted = 0.75 / current_fill * 0.80 if current_fill > 0 else 0.80
    del renderer_tmp

    # Pass 2: final render
    renderer, campos, W, H = auto_frame_camera(
        x_target, cam_cfg, render_cfg, res=args.res, fill_ratio=adjusted)

    # ── Render target (once) ──
    alpha_target, color_target = render_particles(
        x_target, F_id_target, rs, renderer, campos, render_cfg, particle_color, "Target", ep=0)
    save_image_png(out_dir / 'target.png', color_target)
    save_image_png(out_dir / 'alpha_target.png', alpha_target.numpy())

    # ── Render initial sphere (once) ──
    alpha_init, color_init = render_particles(
        x_input, F_id, rs, renderer, campos, render_cfg, particle_color, "Initial", ep=0)
    mse_init = torch.nn.functional.mse_loss(alpha_init, alpha_target).item()
    (out_dir / 'initial').mkdir(parents=True, exist_ok=True)
    save_image_png(out_dir / 'initial' / 'color.png', color_init)

    # Save initial particle data + target
    torch.save({
        'x': torch.from_numpy(x_input),
        'F_e': torch.from_numpy(F_id),
        'alpha': alpha_init,
    }, out_dir / 'initial' / 'particles.pt')
    torch.save({
        'x': torch.from_numpy(x_target),
        'alpha': alpha_target,
    }, out_dir / 'target_particles.pt')

    # ── Episode loop ──
    loss_history = []
    prev_loss = loss0

    print(f"\n{'=' * 70}")
    print(f"{'ep':>3} | {'phys_loss':>10} | {'alpha_mse':>10} | {'obj_mse':>10} | {'||dF||_max':>10} | {'||dx||_max':>10} | {'det(F)':>12}")
    print(f"{'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    for ep in range(num_eps):
        # ── Physics ──
        cg.run_optimization(opt)
        loss_phys = cg.end_layer_mass_loss()

        pc = cg.get_point_cloud(num_timesteps - 1)
        x_post = np.ascontiguousarray(np.array(pc.get_positions(), dtype=np.float32))
        try:
            F_e = pc.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy()
        except Exception:
            F_e = np.array(pc.get_def_grads(), dtype=np.float32)
        F_e = np.ascontiguousarray(F_e.astype(np.float32))

        # ── Stats ──
        dF = F_e - F_id
        dF_norm = np.linalg.norm(dF.reshape(N, -1), axis=1)
        dx = x_post - x_input
        dx_norm = np.linalg.norm(dx, axis=1)
        det_Fe = np.linalg.det(F_e)

        # ── Render ──
        alpha_post, color_post = render_particles(
            x_post, F_e, rs, renderer, campos, render_cfg, particle_color, f"ep{ep:03d}", ep=ep)

        mse_full = torch.nn.functional.mse_loss(alpha_post, alpha_target).item()
        obj_mask = (alpha_post > 0.01) | (alpha_target > 0.01)
        obj_mse = ((alpha_post[obj_mask] - alpha_target[obj_mask]) ** 2).mean().item()

        print(f"{ep:3d} | {loss_phys:10.2f} | {mse_full:10.6f} | {obj_mse:10.6f} | {dF_norm.max():10.4f} | {dx_norm.max():10.4f} | [{det_Fe.min():.3f},{det_Fe.max():.3f}]")

        # ── Save per-episode ──
        ep_dir = out_dir / f'ep{ep:03d}'
        ep_dir.mkdir(parents=True, exist_ok=True)

        if color_post is not None:
            save_image_png(ep_dir / 'color.png', color_post)

        # Signed diff heatmap
        signed_np = (alpha_post - alpha_target).numpy()
        signed_hm = np.zeros((H, W, 3), dtype=np.float32)
        signed_hm[:, :, 0] = np.clip(-signed_np, 0, 1)
        signed_hm[:, :, 2] = np.clip(signed_np, 0, 1)
        overlap = np.minimum(alpha_post.numpy(), alpha_target.numpy())
        signed_hm[:, :, 1] = overlap * 0.2
        save_image_png(ep_dir / 'signed_diff.png', (signed_hm * 255).astype(np.uint8))

        # dF heatmap via 3DGS
        dF_img, _, dF_max = render_scalar_heatmap(
            x_post, F_e, dF_norm, rs, renderer, campos, render_cfg, f"dF-ep{ep}", ep=ep, cmap_name='hot')
        if dF_img is not None:
            save_image_png(ep_dir / 'heatmap_dF.png', dF_img)

        # dx heatmap via 3DGS
        dx_img, _, dx_max = render_scalar_heatmap(
            x_post, F_e, dx_norm, rs, renderer, campos, render_cfg, f"dx-ep{ep}", ep=ep, cmap_name='hot')
        if dx_img is not None:
            save_image_png(ep_dir / 'heatmap_dx.png', dx_img)

        # ── Save particle data (.pt) ──
        torch.save({
            'x': torch.from_numpy(x_post),           # (N,3) positions
            'F_e': torch.from_numpy(F_e),             # (N,3,3) elastic deformation
            'dx': torch.from_numpy(dx),               # (N,3) displacement from initial
            'dF': torch.from_numpy(dF.astype(np.float32)),  # (N,3,3) F_e - I
            'det_Fe': torch.from_numpy(det_Fe.astype(np.float32)),  # (N,) determinant
            'dF_norm': torch.from_numpy(dF_norm.astype(np.float32)),  # (N,) ||dF||
            'dx_norm': torch.from_numpy(dx_norm.astype(np.float32)),  # (N,) ||dx||
            'alpha': alpha_post,                      # (H,W) rendered alpha
        }, ep_dir / 'particles.pt')

        loss_history.append({
            'ep': ep,
            'phys_loss': float(loss_phys),
            'alpha_mse': float(mse_full),
            'obj_mse': float(obj_mse),
            'dF_max': float(dF_norm.max()),
            'dF_mean': float(dF_norm.mean()),
            'dx_max': float(dx_norm.max()),
            'dx_mean': float(dx_norm.mean()),
            'det_min': float(det_Fe.min()),
            'det_max': float(det_Fe.max()),
        })

        # Save loss log incrementally
        with open(out_dir / 'losses.json', 'w') as f:
            json.dump(loss_history, f, indent=2)

        prev_loss = loss_phys

        # Promote for next episode
        cg.promote_last_as_initial(carry_grid=False)

    # ── Summary plot: loss curves ──
    eps = [h['ep'] for h in loss_history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(eps, [h['phys_loss'] for h in loss_history], 'b-o', markersize=3)
    axes[0, 0].set_title('Physics Loss (EndLayerMassLoss)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].grid(True)

    axes[0, 1].plot(eps, [h['alpha_mse'] for h in loss_history], 'r-o', markersize=3, label='full')
    axes[0, 1].plot(eps, [h['obj_mse'] for h in loss_history], 'r--s', markersize=3, label='obj-only')
    axes[0, 1].axhline(y=mse_init, color='gray', linestyle=':', label=f'initial={mse_init:.4f}')
    axes[0, 1].set_title('Alpha MSE vs Target')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(eps, [h['dF_max'] for h in loss_history], 'g-o', markersize=3, label='max')
    axes[1, 0].plot(eps, [h['dF_mean'] for h in loss_history], 'g--s', markersize=3, label='mean')
    axes[1, 0].set_title('||dF|| (F_e - I)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(eps, [h['det_min'] for h in loss_history], 'purple', label='det min')
    axes[1, 1].plot(eps, [h['det_max'] for h in loss_history], 'orange', label='det max')
    axes[1, 1].axhline(y=1.0, color='gray', linestyle=':')
    axes[1, 1].set_title('det(F_e) range')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle(f'Physics-Only Diagnosis — {num_eps} episodes', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Done. Saved to {out_dir}/")
    print(f"  ep{{NNN}}/color.png, signed_diff.png, heatmap_dF.png, heatmap_dx.png, particles.pt")
    print(f"  initial/particles.pt, target_particles.pt")
    print(f"  losses.json, loss_curves.png")


if __name__ == '__main__':
    main()
