"""
1 episode render gradient 진단: per-particle dLdx 분포 확인

Physics forward → render loss 계산 → per-particle gradient 출력
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import yaml
from pathlib import Path

# ── Config (e2e_render.yaml 기반, 1 episode, start_ep=0) ─────────────────
cfg = yaml.safe_load(open('configs/e2e_render.yaml'))
cfg['optimization']['num_animations'] = 1  # 1 episode만

from sampling import default_cfg
from sampling.utils.config_adapter import adapt_config
from utils.physics_utils import (
    build_opt_input, initialize_point_clouds,
    initialize_grids, initialize_comp_graph,
)
from utils.rendering_utils import setup_renderer, generate_target_render

import diffmpm_bindings

# ── Init ──────────────────────────────────────────────────────────────────
opt = build_opt_input(cfg)
input_pc, target_pc = initialize_point_clouds(opt)
input_grid, target_grid = initialize_grids(opt)
diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
cg = initialize_comp_graph(input_pc, input_grid, target_grid)

N = np.array(input_pc.get_positions()).shape[0]
print(f"[Init] N = {N:,} particles")

# ── Renderer + target alpha ───────────────────────────────────────────────
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
print(f"[Target] alpha shape={target_alpha.shape}, range=[{target_alpha.min():.3f}, {target_alpha.max():.3f}]")

# ── 1. Physics Pass ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("[Pass 1] Physics forward + optimization")
print("="*70)
cg.run_optimization(opt)
loss_phys = cg.end_layer_mass_loss()

num_timesteps = int(opt.num_timesteps)
pc = cg.get_point_cloud(num_timesteps - 1)
x = np.ascontiguousarray(np.array(pc.get_positions(), dtype=np.float32))
F_e = np.ascontiguousarray(np.array(pc.get_def_grads(), dtype=np.float32))

print(f"  physics_loss = {loss_phys:.4f}")
print(f"  x:   shape={x.shape}, range=[{x.min():.3f}, {x.max():.3f}]")
print(f"  F_e: det range=[{np.linalg.det(F_e).min():.3f}, {np.linalg.det(F_e).max():.3f}]")

# ── 2. Render Gradient Computation ────────────────────────────────────────
print("\n" + "="*70)
print("[Render] Computing render loss & per-particle gradient")
print("="*70)

F_plastic = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
F_total = np.einsum('nij,njk->nik', F_e, F_plastic).astype(np.float32)

from sampling import upsample
from utils.rendering_utils import prepare_rendering_inputs
import torch.nn.functional as F_nn

ema_state = {}

# Leaf tensors
x_t = torch.from_numpy(x).requires_grad_(True)
F_t = torch.from_numpy(F_total)

print(f"\n  [Step 1] x_t: shape={x_t.shape}, requires_grad={x_t.requires_grad}")
print(f"  [Step 1] F_t: shape={F_t.shape}")

# Upsample
result = upsample(x_t, F_t, cfg=rs, state=ema_state,
                  seed=9000, return_torch=True, current_episode=0)

mu  = result.get('points')
cov = result.get('cov')
print(f"\n  [Step 2] Upsample:")
print(f"    mu:  shape={mu.shape}, requires_grad={mu.requires_grad}")
print(f"    cov: shape={cov.shape}")
print(f"    mu range: [{mu.min().item():.3f}, {mu.max().item():.3f}]")

# Render
rgb = prepare_rendering_inputs(mu, result, campos, render_cfg, particle_color)
pred = renderer.render(mu, cov, rgb=rgb, prefer_cov_precomp=True, return_torch=True)

pred_alpha = pred.get('alpha')
if pred_alpha.dim() == 3:
    pred_alpha = pred_alpha[0]

print(f"\n  [Step 3] Render:")
print(f"    pred_alpha: shape={pred_alpha.shape}, requires_grad={pred_alpha.requires_grad}")
print(f"    pred_alpha range: [{pred_alpha.min().item():.3f}, {pred_alpha.max().item():.3f}]")

# Target resize
tgt = target_alpha.to(pred_alpha.device)
if tgt.shape != pred_alpha.shape:
    tgt = F_nn.interpolate(tgt.unsqueeze(0).unsqueeze(0),
                           size=pred_alpha.shape, mode='bilinear',
                           align_corners=False)[0, 0]

# Loss
diff = pred_alpha - tgt
loss_render = torch.mean(diff ** 2)

print(f"\n  [Step 4] Loss:")
print(f"    target alpha range: [{tgt.min().item():.3f}, {tgt.max().item():.3f}]")
print(f"    diff: mean={diff.mean().item():.6f}, std={diff.std().item():.6f}")
print(f"    loss_render = {loss_render.item():.6f}")

# Non-zero pixel stats
nonzero_pred = (pred_alpha > 0.01).sum().item()
nonzero_tgt  = (tgt > 0.01).sum().item()
total_pixels = pred_alpha.numel()
print(f"    pred nonzero pixels: {nonzero_pred:,} / {total_pixels:,} ({100*nonzero_pred/total_pixels:.1f}%)")
print(f"    tgt  nonzero pixels: {nonzero_tgt:,} / {total_pixels:,} ({100*nonzero_tgt/total_pixels:.1f}%)")

# Backward
loss_render.backward()

print(f"\n  [Step 5] Backward:")
print(f"    x_t.grad is None? {x_t.grad is None}")

if x_t.grad is not None:
    dLdx = x_t.grad.detach().cpu().numpy()

    # Per-particle stats
    grad_norms = np.linalg.norm(dLdx, axis=1)  # (N,)

    print(f"\n{'='*70}")
    print(f"[RESULT] Per-particle render gradient dLdx")
    print(f"{'='*70}")
    print(f"  Shape: {dLdx.shape}")
    print(f"  ||dLdx|| per particle:")
    print(f"    mean   = {grad_norms.mean():.6e}")
    print(f"    std    = {grad_norms.std():.6e}")
    print(f"    min    = {grad_norms.min():.6e}")
    print(f"    max    = {grad_norms.max():.6e}")
    print(f"    median = {np.median(grad_norms):.6e}")

    # Percentiles
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    p{p:02d}    = {np.percentile(grad_norms, p):.6e}")

    # Zero vs nonzero
    zero_count = (grad_norms < 1e-10).sum()
    print(f"\n  Zero gradient particles: {zero_count:,} / {N:,} ({100*zero_count/N:.1f}%)")
    print(f"  Active particles:       {N - zero_count:,} / {N:,} ({100*(N-zero_count)/N:.1f}%)")

    # Component-wise
    print(f"\n  Component-wise:")
    for d, label in enumerate(['x', 'y', 'z']):
        comp = dLdx[:, d]
        print(f"    dLd{label}: mean={comp.mean():.6e}, std={comp.std():.6e}, "
              f"range=[{comp.min():.6e}, {comp.max():.6e}]")

    # Top 10 particles by gradient norm
    top_idx = np.argsort(grad_norms)[-10:][::-1]
    print(f"\n  Top 10 particles by ||dLdx||:")
    print(f"  {'idx':>6s}  {'||dLdx||':>12s}  {'x':>8s}  {'y':>8s}  {'z':>8s}  {'dx':>10s}  {'dy':>10s}  {'dz':>10s}")
    for idx in top_idx:
        print(f"  {idx:6d}  {grad_norms[idx]:12.6e}  "
              f"{x[idx,0]:8.3f}  {x[idx,1]:8.3f}  {x[idx,2]:8.3f}  "
              f"{dLdx[idx,0]:10.6e}  {dLdx[idx,1]:10.6e}  {dLdx[idx,2]:10.6e}")

    # Bottom 10 nonzero particles
    nonzero_mask = grad_norms > 1e-10
    if nonzero_mask.sum() > 0:
        nonzero_idx = np.where(nonzero_mask)[0]
        nonzero_norms = grad_norms[nonzero_idx]
        bottom_idx = nonzero_idx[np.argsort(nonzero_norms)[:10]]
        print(f"\n  Bottom 10 nonzero particles by ||dLdx||:")
        print(f"  {'idx':>6s}  {'||dLdx||':>12s}  {'x':>8s}  {'y':>8s}  {'z':>8s}")
        for idx in bottom_idx:
            print(f"  {idx:6d}  {grad_norms[idx]:12.6e}  "
                  f"{x[idx,0]:8.3f}  {x[idx,1]:8.3f}  {x[idx,2]:8.3f}")

    # Compare with physics gradient magnitude
    print(f"\n{'='*70}")
    print(f"[COMPARISON] Physics vs Render gradient scale")
    print(f"{'='*70}")
    print(f"  physics_loss     = {loss_phys:.4f}")
    print(f"  render_loss      = {loss_render.item():.6f}")
    print(f"  ||render_dLdx||  = {np.linalg.norm(dLdx):.6e}  (total Frobenius)")
    print(f"  ||render_dLdx||  = {grad_norms.mean():.6e}  (per-particle mean)")

    e2e_gain = float(cfg.get('e2e', {}).get('render_gain', 0.05))
    print(f"\n  e2e_gain = {e2e_gain}")
    print(f"  After injection: gain * ||dLdx|| per particle = {e2e_gain * grad_norms.mean():.6e}")

else:
    print("  [ERROR] x_t.grad is None — gradient did not flow back!")
    print("  Check: does upsample maintain the computation graph to x_t?")
