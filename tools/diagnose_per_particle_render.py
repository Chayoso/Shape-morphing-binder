"""
Per-particle render loss 분해 진단

각 입자 p에 대해:
  1. 3D position → 2D screen projection
  2. 2D Gaussian footprint (몇 pixel에 기여하나)
  3. alpha compositing에서의 기여도
  4. 해당 pixel들의 loss 기여
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import yaml

cfg = yaml.safe_load(open('configs/e2e_render.yaml'))
cfg['optimization']['num_animations'] = 1

from sampling import default_cfg, upsample
from sampling.utils.config_adapter import adapt_config
from utils.physics_utils import (
    build_opt_input, initialize_point_clouds,
    initialize_grids, initialize_comp_graph,
)
from utils.rendering_utils import setup_renderer, generate_target_render, prepare_rendering_inputs
from renderer import make_matrices_from_yaml
import torch.nn.functional as F_nn
import diffmpm_bindings

# ── Init ──────────────────────────────────────────────────────────────────
opt = build_opt_input(cfg)
input_pc, target_pc = initialize_point_clouds(opt)
input_grid, target_grid = initialize_grids(opt)
diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
cg = initialize_comp_graph(input_pc, input_grid, target_grid)

cam_cfg    = cfg.get('camera', {})
render_cfg = cfg.get('render', {})
particle_color = render_cfg.get('particle_color', [0.27, 0.51, 0.71])
renderer, view_params = setup_renderer(cam_cfg, render_cfg, training_mode=True)
campos = view_params['campos']
W, H = view_params['W'], view_params['H']

rs = default_cfg()
rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))
sim_cfg = cfg.get('simulation', {})
rs['physics_grid'] = {
    'grid_min': sim_cfg.get('grid_min_point'),
    'grid_max': sim_cfg.get('grid_max_point'),
    'grid_dx':  sim_cfg.get('grid_dx', 1.0),
}

# Target alpha
target_positions = np.array(target_pc.get_positions(), dtype=np.float32)
target_alpha = generate_target_render(
    target_positions, rs, renderer, campos, render_cfg, particle_color
)

# ── Physics Pass 1 ────────────────────────────────────────────────────────
cg.run_optimization(opt)
pc = cg.get_point_cloud(int(opt.num_timesteps) - 1)
x = np.ascontiguousarray(np.array(pc.get_positions(), dtype=np.float32))
F_e = np.ascontiguousarray(np.array(pc.get_def_grads(), dtype=np.float32))
N = x.shape[0]

print(f"\n[Physics] N={N:,}, x range=[{x.min():.3f}, {x.max():.3f}]")

# ── Camera matrices (manual projection) ──────────────────────────────────
W_full, H_full, tanfovx, tanfovy, view_T, proj_T, campos_np = make_matrices_from_yaml(cam_cfg)
# Training resolution
scale = render_cfg.get("training_resolution_scale", 0.5)
W_r, H_r = int(W_full * scale), int(H_full * scale)

view_mat = torch.from_numpy(view_T).float()   # (4, 4)
fx = cam_cfg['fx'] * scale
fy = cam_cfg['fy'] * scale
cx = cam_cfg['cx'] * scale
cy = cam_cfg['cy'] * scale

print(f"[Camera] {W_r}x{H_r}, fx={fx}, fy={fy}, cx={cx}, cy={cy}")
print(f"[Camera] eye={cam_cfg['lookat']['eye']}")

# ── Project all particles to 2D ──────────────────────────────────────────
x_h = np.concatenate([x, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N, 4)
x_cam = (view_T @ x_h.T).T  # (N, 4) in camera space

# Camera space: x_cam[:, :3]
x_c = x_cam[:, 0]
y_c = x_cam[:, 1]
z_c = x_cam[:, 2]

# Perspective projection to pixel coords
u = fx * x_c / z_c + cx
v = fy * y_c / z_c + cy

# Visibility: in front of camera and within image bounds
visible = (z_c > 0.01) & (u >= 0) & (u < W_r) & (v >= 0) & (v < H_r)
print(f"\n[Projection] Visible particles: {visible.sum():,} / {N:,} ({100*visible.sum()/N:.1f}%)")
print(f"  depth z_c: [{z_c[visible].min():.2f}, {z_c[visible].max():.2f}]")

# ── Render (same as training_loop) ────────────────────────────────────────
F_total = F_e.copy()  # F_plastic = I
ema_state = {}

x_t = torch.from_numpy(x).requires_grad_(True)
F_t = torch.from_numpy(F_total)

result = upsample(x_t, F_t, cfg=rs, state=ema_state,
                  seed=9000, return_torch=True, current_episode=0)

mu  = result['points']
cov = result['cov']
rgb = prepare_rendering_inputs(mu, result, campos, render_cfg, particle_color)
pred = renderer.render(mu, cov, rgb=rgb, prefer_cov_precomp=True, return_torch=True)

pred_alpha = pred['alpha']
if pred_alpha.dim() == 3:
    pred_alpha = pred_alpha[0]  # (H, W)

tgt = target_alpha.to(pred_alpha.device)
if tgt.shape != pred_alpha.shape:
    tgt = F_nn.interpolate(tgt.unsqueeze(0).unsqueeze(0),
                           size=pred_alpha.shape, mode='bilinear',
                           align_corners=False)[0, 0]

# Per-pixel error
diff = pred_alpha - tgt                     # (H, W)
per_pixel_sq_error = diff ** 2              # (H, W)
loss_render = per_pixel_sq_error.mean()

print(f"\n[Render]")
print(f"  pred_alpha: shape={pred_alpha.shape}, range=[{pred_alpha.min().item():.4f}, {pred_alpha.max().item():.4f}]")
print(f"  target:     range=[{tgt.min().item():.4f}, {tgt.max().item():.4f}]")
print(f"  loss = {loss_render.item():.6f}")

# Backward for per-particle gradient
loss_render.backward()
dLdx = x_t.grad.detach().cpu().numpy()
grad_norms = np.linalg.norm(dLdx, axis=1)

# ── Per-pixel analysis ────────────────────────────────────────────────────
diff_np = diff.detach().cpu().numpy()
pred_np = pred_alpha.detach().cpu().numpy()
tgt_np  = tgt.detach().cpu().numpy()
sq_err_np = per_pixel_sq_error.detach().cpu().numpy()

print(f"\n{'='*70}")
print(f"PER-PIXEL ERROR MAP")
print(f"{'='*70}")

# Where pred has stuff but target doesn't (sphere sticks out)
pred_only = (pred_np > 0.01) & (tgt_np < 0.01)
# Where target has stuff but pred doesn't (sphere missing bunny region)
tgt_only  = (tgt_np > 0.01) & (pred_np < 0.01)
# Overlap
overlap   = (pred_np > 0.01) & (tgt_np > 0.01)

print(f"  Pred-only pixels (sphere exceeds bunny):  {pred_only.sum():,}  avg error: {sq_err_np[pred_only].mean():.6f}" if pred_only.any() else "  Pred-only: 0")
print(f"  Target-only pixels (bunny exceeds sphere): {tgt_only.sum():,}  avg error: {sq_err_np[tgt_only].mean():.6f}" if tgt_only.any() else "  Target-only: 0")
print(f"  Overlap pixels:                           {overlap.sum():,}  avg error: {sq_err_np[overlap].mean():.6f}" if overlap.any() else "  Overlap: 0")
print(f"  Background (both zero):                   {((pred_np<0.01)&(tgt_np<0.01)).sum():,}")

# Loss contribution by region
total_loss = sq_err_np.sum()
if pred_only.any():
    print(f"\n  Loss from pred-only:  {sq_err_np[pred_only].sum()/total_loss*100:.1f}% of total")
if tgt_only.any():
    print(f"  Loss from tgt-only:   {sq_err_np[tgt_only].sum()/total_loss*100:.1f}% of total")
if overlap.any():
    print(f"  Loss from overlap:    {sq_err_np[overlap].sum()/total_loss*100:.1f}% of total")

# ── Per-particle: screen position + gradient ──────────────────────────────
print(f"\n{'='*70}")
print(f"PER-PARTICLE RENDER CONTRIBUTION (visible particles)")
print(f"{'='*70}")

# For visible particles, show: position, screen coords, depth, gradient
vis_idx = np.where(visible)[0]
vis_grad_norms = grad_norms[vis_idx]
vis_depths = z_c[vis_idx]
vis_u = u[vis_idx]
vis_v = v[vis_idx]

# Sort by gradient norm (descending)
sort_order = np.argsort(vis_grad_norms)[::-1]

print(f"\n  Visible particles: {len(vis_idx):,}")
print(f"  With nonzero grad: {(vis_grad_norms > 1e-10).sum():,}")
print(f"  With zero grad:    {(vis_grad_norms < 1e-10).sum():,} (occluded even though in frustum)")

# Check: how many visible particles are on the silhouette edge?
# A particle is on the edge if its projected pixel is near the boundary of pred_alpha
edge_count = 0
for i in sort_order[:min(1000, len(sort_order))]:
    pi, pj = int(vis_v[i]), int(vis_u[i])
    if 0 <= pi < H_r and 0 <= pj < W_r:
        alpha_at_particle = pred_np[pi, pj]
        # Check if any neighboring pixel has significantly different alpha
        neighbors_alpha = []
        for di in [-2, -1, 0, 1, 2]:
            for dj in [-2, -1, 0, 1, 2]:
                ni, nj = pi + di, pj + dj
                if 0 <= ni < H_r and 0 <= nj < W_r:
                    neighbors_alpha.append(pred_np[ni, nj])
        if neighbors_alpha:
            alpha_range = max(neighbors_alpha) - min(neighbors_alpha)
            if alpha_range > 0.3:
                edge_count += 1

print(f"  Silhouette edge particles (top 1000): {edge_count} ({100*edge_count/min(1000,len(sort_order)):.1f}%)")

# Top 20 by gradient
print(f"\n  Top 20 particles by ||dLdx||:")
print(f"  {'rank':>4s}  {'idx':>6s}  {'||dLdx||':>12s}  {'x':>7s} {'y':>7s} {'z':>7s}  "
      f"{'u_px':>6s} {'v_px':>6s} {'depth':>6s}  {'α@px':>6s} {'tgt@px':>6s} {'err@px':>8s}")
for rank, i in enumerate(sort_order[:20]):
    idx = vis_idx[i]
    pi, pj = int(vis_v[i]), int(vis_u[i])
    pi = min(max(pi, 0), H_r - 1)
    pj = min(max(pj, 0), W_r - 1)
    alpha_here = pred_np[pi, pj]
    tgt_here   = tgt_np[pi, pj]
    err_here   = sq_err_np[pi, pj]
    print(f"  {rank+1:4d}  {idx:6d}  {vis_grad_norms[i]:12.6e}  "
          f"{x[idx,0]:7.2f} {x[idx,1]:7.2f} {x[idx,2]:7.2f}  "
          f"{vis_u[i]:6.1f} {vis_v[i]:6.1f} {vis_depths[i]:6.2f}  "
          f"{alpha_here:6.3f} {tgt_here:6.3f} {err_here:8.6f}")

# Bottom 20 (visible but zero gradient = occluded)
zero_grad_vis = sort_order[vis_grad_norms[sort_order] < 1e-10]
if len(zero_grad_vis) > 0:
    print(f"\n  Sample visible-but-zero-gradient particles (occluded):")
    print(f"  {'idx':>6s}  {'x':>7s} {'y':>7s} {'z':>7s}  {'u_px':>6s} {'v_px':>6s} {'depth':>6s}  {'α@px':>6s}")
    for i in zero_grad_vis[:10]:
        idx = vis_idx[i]
        pi, pj = int(vis_v[i]), int(vis_u[i])
        pi = min(max(pi, 0), H_r - 1)
        pj = min(max(pj, 0), W_r - 1)
        alpha_here = pred_np[pi, pj]
        print(f"  {idx:6d}  {x[idx,0]:7.2f} {x[idx,1]:7.2f} {x[idx,2]:7.2f}  "
              f"{vis_u[i]:6.1f} {vis_v[i]:6.1f} {vis_depths[i]:6.2f}  {alpha_here:6.3f}")

# ── Gradient direction analysis ───────────────────────────────────────────
print(f"\n{'='*70}")
print(f"GRADIENT DIRECTION ANALYSIS (top 100 active particles)")
print(f"{'='*70}")

active_vis = sort_order[:min(100, (vis_grad_norms > 1e-10).sum())]
if len(active_vis) > 0:
    active_idx = vis_idx[active_vis]
    active_dLdx = dLdx[active_idx]  # (K, 3)

    # What direction does the gradient point?
    # Normalize per-particle
    active_dirs = active_dLdx / (np.linalg.norm(active_dLdx, axis=1, keepdims=True) + 1e-10)

    # Mean direction
    mean_dir = active_dirs.mean(axis=0)
    mean_dir_norm = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)
    print(f"  Mean gradient direction (normalized): [{mean_dir_norm[0]:.3f}, {mean_dir_norm[1]:.3f}, {mean_dir_norm[2]:.3f}]")

    # Camera direction
    eye = np.array(cam_cfg['lookat']['eye'])
    target_pt = np.array(cam_cfg['lookat']['target'])
    cam_dir = target_pt - eye
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    print(f"  Camera look direction:               [{cam_dir[0]:.3f}, {cam_dir[1]:.3f}, {cam_dir[2]:.3f}]")

    # Are gradients mostly radial (towards/away from camera)?
    radial_component = np.abs(active_dirs @ cam_dir)  # dot product with camera direction
    print(f"  |dot(grad_dir, cam_dir)|: mean={radial_component.mean():.3f}, "
          f"std={radial_component.std():.3f}")
    print(f"  → 1.0 = gradient points along camera ray (depth direction)")
    print(f"  → 0.0 = gradient points perpendicular to camera (tangent to image plane)")

    # Tangential vs radial decomposition
    radial = (active_dirs @ cam_dir)[:, None] * cam_dir[None, :]  # (K, 3)
    tangential = active_dirs - radial  # (K, 3)
    rad_mag = np.linalg.norm(radial, axis=1)
    tan_mag = np.linalg.norm(tangential, axis=1)
    print(f"\n  Radial magnitude:     mean={rad_mag.mean():.3f}")
    print(f"  Tangential magnitude: mean={tan_mag.mean():.3f}")
    print(f"  → Gradient is {'mostly radial (depth)' if rad_mag.mean() > tan_mag.mean() else 'mostly tangential (image plane)'}")
