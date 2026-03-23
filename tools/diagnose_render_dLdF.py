"""
Diagnostic: Measure render dL/dF and dL/dx magnitudes.

Bypasses upsample pipeline — builds covariance directly from F,
renders, computes alpha L2 loss, then extracts gradients.

Usage:
    python tools/diagnose_render_dLdF.py -c configs/sphere_to_bunny.yaml
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import torch
import torch.nn.functional as F_nn
import yaml
from pathlib import Path

from sampling.geometry.deformation_covariance import build_deformation_covariance, polar_with_sv_soft_clamp
from sampling.analysis.knn import HybridFAISSKNN, FAISS_AVAILABLE
from utils.physics_utils import (
    build_opt_input,
    initialize_point_clouds,
    initialize_grids,
    initialize_comp_graph,
)
from utils.rendering_utils import setup_renderer, generate_target_render, prepare_rendering_inputs
from sampling import default_cfg, upsample
from sampling.utils.config_adapter import adapt_config


def build_cov_direct(x_t, F_t, sigma0=0.2, k_F=32, sv_min=0.3):
    """
    F → covariance, fully differentiable, no upsample pipeline.

    Σ = σ₀² · H · H   where F = R·H (polar decomp)
    """
    knn = HybridFAISSKNN(use_faiss=FAISS_AVAILABLE, tau=0.15)
    cov_cfg = {
        "sigma0": sigma0, "k_F": k_F,
        "use_polar_decomposition": True,
        "use_F_smoothing": False,
        "sv_min": sv_min,
    }
    cov, F_interp, idx = build_deformation_covariance(
        points=x_t, x_low=x_t, F_low=F_t, knn=knn, cfg=cov_cfg
    )
    return cov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # ── Physics init ──
    import diffmpm_bindings
    opt = build_opt_input(cfg)
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
    cg = initialize_comp_graph(input_pc, input_grid, target_grid)

    loss0 = cg.end_layer_mass_loss()
    print(f"[Init] EndLayerMassLoss = {loss0:.2f}")

    # ── Renderer ──
    cam_cfg = cfg.get('camera', {})
    render_cfg = cfg.get('render', {})
    particle_color = render_cfg.get('particle_color', [0.27, 0.51, 0.71])
    renderer, view_params = setup_renderer(cam_cfg, render_cfg, training_mode=True)
    campos = view_params.get('campos')

    if renderer is None:
        print("ERROR: Renderer failed to init")
        return

    # ── Upsample config (for target render only) ──
    rs = default_cfg()
    rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))
    sim_cfg = cfg.get('simulation', {})
    rs['physics_grid'] = {
        'grid_min': sim_cfg.get('grid_min_point', [-16, -16, -16]),
        'grid_max': sim_cfg.get('grid_max_point', [16, 16, 16]),
        'grid_dx':  sim_cfg.get('grid_dx', 1.0),
    }

    # ── Target alpha ──
    target_positions = np.array(target_pc.get_positions(), dtype=np.float32)
    target_alpha = generate_target_render(
        target_positions, rs, renderer, campos, render_cfg, particle_color
    )
    if target_alpha is None:
        print("ERROR: Target alpha generation failed")
        return

    # ── Run 1 physics episode ──
    print("\n" + "=" * 70)
    print("Running 1 physics episode...")
    print("=" * 70)
    cg.run_optimization(opt)
    loss_phys = cg.end_layer_mass_loss()
    print(f"[Ep 0] physics_loss = {loss_phys:.2f}")

    # ── Extract final state ──
    num_timesteps = int(opt.num_timesteps)
    pc = cg.get_point_cloud(num_timesteps - 1)
    x = np.ascontiguousarray(np.array(pc.get_positions(), dtype=np.float32))
    try:
        F_e = pc.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy()
    except Exception:
        F_e = np.array(pc.get_def_grads(), dtype=np.float32)
    F_e = np.ascontiguousarray(F_e.astype(np.float32))
    N = x.shape[0]

    print(f"\nN = {N:,}")
    det_Fe = np.linalg.det(F_e)
    print(f"det(F_e): [{det_Fe.min():.4f}, {det_Fe.max():.4f}]")

    # ── Covariance config ──
    cov_cfg = cfg.get('upsample', {}).get('covariance', {})
    sigma0 = float(cov_cfg.get('sigma0', 0.2))
    sv_min = float(cov_cfg.get('sv_min', 0.3))

    # ── Differentiable render: F → cov → render → alpha loss ──
    print("\n" + "=" * 70)
    print(f"Differentiable render pass (sigma0={sigma0}, sv_min={sv_min})")
    print("=" * 70)

    x_t = torch.from_numpy(x).requires_grad_(True)
    F_t = torch.from_numpy(F_e).requires_grad_(True)

    # Build covariance directly (skip upsample pipeline)
    cov = build_cov_direct(x_t, F_t, sigma0=sigma0, sv_min=sv_min)

    # Normals for shading (detached, just for RGB)
    with torch.no_grad():
        # Simple: use F columns as proxy normals (won't affect gradients)
        normals_approx = torch.nn.functional.normalize(
            F_t.detach()[:, :, 2], dim=-1  # 3rd column as approximate normal
        )

    # Prepare rendering inputs manually
    from renderer import compute_shading
    rgb_np = compute_shading(
        x_t.detach().cpu().numpy(),
        normals_approx.cpu().numpy(),
        camera_pos=campos,
        light_cfg=render_cfg.get("lighting", {}),
        albedo_color=particle_color,
        model="phong"
    )
    rgb = torch.from_numpy(rgb_np).to(x_t.device)

    # Render
    pred = renderer.render(x_t, cov, rgb=rgb, prefer_cov_precomp=True, return_torch=True)
    pred_alpha = pred.get('alpha')
    if pred_alpha is None:
        print("ERROR: renderer returned no alpha")
        return
    if pred_alpha.dim() == 3:
        pred_alpha = pred_alpha[0]

    tgt = target_alpha.to(pred_alpha.device)
    if tgt.shape != pred_alpha.shape:
        tgt = F_nn.interpolate(tgt.unsqueeze(0).unsqueeze(0),
                               size=pred_alpha.shape, mode='bilinear',
                               align_corners=False)[0, 0]

    loss_render = torch.mean((pred_alpha - tgt) ** 2)
    print(f"\nrender_loss (alpha L2) = {loss_render.item():.6f}")
    print(f"pred_alpha: [{pred_alpha.min().item():.4f}, {pred_alpha.max().item():.4f}]")

    loss_render.backward()

    # ── Gradient analysis ──
    print("\n" + "=" * 70)
    print("GRADIENT ANALYSIS")
    print("=" * 70)

    for name, tensor in [("dL/dx", x_t), ("dL/dF", F_t)]:
        g = tensor.grad
        if g is None:
            print(f"\n{name}: None (gradient not computed)")
            continue

        g_np = g.detach().cpu().numpy()
        if g_np.ndim == 2:
            per_particle = np.linalg.norm(g_np, axis=1)
        else:
            per_particle = np.linalg.norm(g_np.reshape(N, -1), axis=1)

        global_norm = np.linalg.norm(g_np)
        print(f"\n{name}:")
        print(f"  Global Frobenius norm: {global_norm:.6e}")
        print(f"  Per-particle: mean={per_particle.mean():.6e}, "
              f"median={np.median(per_particle):.6e}, max={per_particle.max():.6e}")
        print(f"  Non-zero (>1e-10): {(per_particle > 1e-10).sum()}/{N}")
        print(f"  Percentiles: p50={np.percentile(per_particle, 50):.6e}, "
              f"p90={np.percentile(per_particle, 90):.6e}, "
              f"p99={np.percentile(per_particle, 99):.6e}")

        nan_count = np.isnan(g_np).sum()
        inf_count = np.isinf(g_np).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  WARNING: NaN={nan_count}, Inf={inf_count}")

    # ── Scale comparison ──
    if x_t.grad is not None and F_t.grad is not None:
        dLdx_norm = np.linalg.norm(x_t.grad.detach().cpu().numpy())
        dLdF_norm = np.linalg.norm(F_t.grad.detach().cpu().numpy())

        print(f"\n{'=' * 70}")
        print("SCALE COMPARISON")
        print(f"{'=' * 70}")
        print(f"  ||dL/dF|| / ||dL/dx||       = {dLdF_norm / (dLdx_norm + 1e-12):.4f}")
        print(f"  ||dL/dx|| global             = {dLdx_norm:.6e}")
        print(f"  ||dL/dF|| global             = {dLdF_norm:.6e}")

        # What C++ physics backward typically produces
        print(f"\n  C++ context:")
        print(f"    adaptive_alpha_target      = 2500.0")
        print(f"    If render dL/dF injected directly at final layer,")
        print(f"    it would be amplified ~T×Hessian through backprop.")
        print(f"    Recommend render_gain = physics_scale / render_scale")

    print("\nDone.")


if __name__ == '__main__':
    main()
