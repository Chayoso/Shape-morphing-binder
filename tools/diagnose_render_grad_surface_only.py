"""
Surface-only render gradient 진단:
88.5% 내부 입자 제거 → 표면 ~11.5%만으로 렌더링 + gradient 비교
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import yaml
from pathlib import Path

cfg = yaml.safe_load(open('configs/e2e_render.yaml'))
cfg['optimization']['num_animations'] = 1

from sampling import default_cfg, upsample
from sampling.utils.config_adapter import adapt_config
from utils.physics_utils import (
    build_opt_input, initialize_point_clouds,
    initialize_grids, initialize_comp_graph,
)
from utils.rendering_utils import setup_renderer, generate_target_render, prepare_rendering_inputs
import torch.nn.functional as F_nn
import diffmpm_bindings

# ── Init (same as before) ────────────────────────────────────────────────
opt = build_opt_input(cfg)
input_pc, target_pc = initialize_point_clouds(opt)
input_grid, target_grid = initialize_grids(opt)
diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
cg = initialize_comp_graph(input_pc, input_grid, target_grid)

N = np.array(input_pc.get_positions()).shape[0]

cam_cfg    = cfg.get('camera', {})
render_cfg = cfg.get('render', {})
particle_color = render_cfg.get('particle_color', [0.27, 0.51, 0.71])
renderer, view_params = setup_renderer(cam_cfg, render_cfg, training_mode=True)
campos = view_params['campos']

rs = default_cfg()
rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))
sim_cfg = cfg.get('simulation', {})
rs['physics_grid'] = {
    'grid_min': sim_cfg.get('grid_min_point', [-16, -16, -16]),
    'grid_max': sim_cfg.get('grid_max_point', [16, 16, 16]),
    'grid_dx':  sim_cfg.get('grid_dx', 1.0),
}

target_positions = np.array(target_pc.get_positions(), dtype=np.float32)
target_alpha = generate_target_render(
    target_positions, rs, renderer, campos, render_cfg, particle_color
)

# ── Physics Pass ──────────────────────────────────────────────────────────
cg.run_optimization(opt)
num_timesteps = int(opt.num_timesteps)
pc = cg.get_point_cloud(num_timesteps - 1)
x_all = np.ascontiguousarray(np.array(pc.get_positions(), dtype=np.float32))
F_e_all = np.ascontiguousarray(np.array(pc.get_def_grads(), dtype=np.float32))


def run_render_pass(x_np, F_np, label, rs, ema_state=None):
    """Run render pass and return per-particle gradient stats."""
    if ema_state is None:
        ema_state = {}

    N_local = x_np.shape[0]
    F_total = F_np.copy()  # F_plastic = I at ep0

    x_t = torch.from_numpy(x_np).requires_grad_(True)
    F_t = torch.from_numpy(F_total)

    result = upsample(x_t, F_t, cfg=rs, state=ema_state,
                      seed=9000, return_torch=True, current_episode=0)

    mu  = result.get('points')
    cov = result.get('cov')

    rgb = prepare_rendering_inputs(mu, result, campos, render_cfg, particle_color)
    pred = renderer.render(mu, cov, rgb=rgb, prefer_cov_precomp=True, return_torch=True)

    pred_alpha = pred.get('alpha')
    if pred_alpha.dim() == 3:
        pred_alpha = pred_alpha[0]

    tgt = target_alpha.to(pred_alpha.device)
    if tgt.shape != pred_alpha.shape:
        tgt = F_nn.interpolate(tgt.unsqueeze(0).unsqueeze(0),
                               size=pred_alpha.shape, mode='bilinear',
                               align_corners=False)[0, 0]

    loss = torch.mean((pred_alpha - tgt) ** 2)
    loss.backward()

    dLdx = x_t.grad.detach().cpu().numpy() if x_t.grad is not None else np.zeros_like(x_np)
    grad_norms = np.linalg.norm(dLdx, axis=1)

    zero_count = (grad_norms < 1e-10).sum()
    active_count = N_local - zero_count

    # Active particles only stats
    active_mask = grad_norms > 1e-10
    active_norms = grad_norms[active_mask] if active_mask.any() else np.array([0.0])

    nonzero_pred = (pred_alpha > 0.01).sum().item()
    nonzero_tgt  = (tgt > 0.01).sum().item()

    print(f"\n{'='*70}")
    print(f"[{label}]  N = {N_local:,}")
    print(f"{'='*70}")
    print(f"  Render:")
    print(f"    pred_alpha range: [{pred_alpha.min().item():.4f}, {pred_alpha.max().item():.4f}]")
    print(f"    pred nonzero px:  {nonzero_pred:,} / {pred_alpha.numel():,} ({100*nonzero_pred/pred_alpha.numel():.1f}%)")
    print(f"    tgt  nonzero px:  {nonzero_tgt:,}")
    print(f"    loss = {loss.item():.6f}")
    print(f"\n  Gradient (all {N_local:,} particles):")
    print(f"    mean ||dLdx|| = {grad_norms.mean():.6e}")
    print(f"    max  ||dLdx|| = {grad_norms.max():.6e}")
    print(f"    zero count:     {zero_count:,} ({100*zero_count/N_local:.1f}%)")
    print(f"    active count:   {active_count:,} ({100*active_count/N_local:.1f}%)")
    print(f"\n  Gradient (active {active_count:,} particles only):")
    print(f"    mean ||dLdx|| = {active_norms.mean():.6e}")
    print(f"    std  ||dLdx|| = {active_norms.std():.6e}")
    print(f"    min  ||dLdx|| = {active_norms.min():.6e}")
    print(f"    max  ||dLdx|| = {active_norms.max():.6e}")
    print(f"    median        = {np.median(active_norms):.6e}")
    for p in [5, 25, 50, 75, 95, 99]:
        print(f"    p{p:02d}           = {np.percentile(active_norms, p):.6e}")

    print(f"\n  ||dLdx|| total Frobenius = {np.linalg.norm(dLdx):.6e}")

    return {
        'loss': loss.item(),
        'grad_norms': grad_norms,
        'active_mean': active_norms.mean(),
        'active_count': active_count,
        'total_frobenius': np.linalg.norm(dLdx),
    }


# ── A. ALL particles (baseline) ──────────────────────────────────────────
print("\n" + "#"*70)
print("# A. ALL 89K particles")
print("#"*70)
res_all = run_render_pass(x_all, F_e_all, "ALL 89K", rs)

# ── B. Surface-only: distance from centroid ───────────────────────────────
# Sphere: particles near the surface (outer shell)
centroid = x_all.mean(axis=0)
dists_from_center = np.linalg.norm(x_all - centroid, axis=1)
radius = dists_from_center.max()

# Take outer 15% shell (by distance percentile)
for shell_pct in [15, 25, 50]:
    threshold = np.percentile(dists_from_center, 100 - shell_pct)
    surface_mask = dists_from_center >= threshold

    x_surface = np.ascontiguousarray(x_all[surface_mask])
    F_surface = np.ascontiguousarray(F_e_all[surface_mask])

    print(f"\n" + "#"*70)
    print(f"# B. Surface only (outer {shell_pct}% shell, threshold={threshold:.2f})")
    print(f"#    {surface_mask.sum():,} / {N:,} particles")
    print("#"*70)

    res_surf = run_render_pass(x_surface, F_surface, f"Surface {shell_pct}%", rs)

# ── C. Camera-visible: based on previous gradient ────────────────────────
# Use the "all particles" gradient to identify visible ones, then re-render
prev_active = res_all['grad_norms'] > 1e-10
if prev_active.sum() > 0:
    x_visible = np.ascontiguousarray(x_all[prev_active])
    F_visible = np.ascontiguousarray(F_e_all[prev_active])

    print(f"\n" + "#"*70)
    print(f"# C. Camera-visible only (from prev gradient != 0)")
    print(f"#    {prev_active.sum():,} / {N:,} particles")
    print("#"*70)

    res_vis = run_render_pass(x_visible, F_visible, f"Visible {prev_active.sum():,}", rs)

# ── Summary comparison ────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"SUMMARY COMPARISON")
print(f"{'='*70}")
print(f"{'Config':<30s} {'N':>8s} {'Loss':>10s} {'Active':>8s} {'Active mean':>14s} {'Total ||g||':>14s}")
print(f"{'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*14} {'-'*14}")
