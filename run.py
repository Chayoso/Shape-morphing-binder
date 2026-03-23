"""
run.py - PhysMorph-GS: Physics + F_plastic Appearance Morphing

Architecture:
  [Physics]   EndLayerMassLoss drives shape morphing
  [Appearance] F_plastic = F_elastic^{-1} @ Σ_target^{1/2}  (analytical)
  [Render]    F_total = F_elastic @ F_plastic → Σ = F_total F_total^T

Usage:
    python run.py -c configs/sphere_to_bunny.yaml [--png]
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import numpy as np
import torch
import yaml
from pathlib import Path

from sampling import default_cfg
from sampling.utils.config_adapter import adapt_config
from utils.physics_utils import (
    build_opt_input,
    initialize_point_clouds,
    initialize_grids,
    initialize_comp_graph,
)
from utils.rendering_utils import setup_renderer, generate_target_render
from utils.training_loop import run_episode


def load_surface_samples(mesh_path: str, n_samples: int = 50000) -> tuple:
    """Sample n_samples points + normals uniformly from mesh surface."""
    import trimesh
    mesh = trimesh.load(mesh_path, force='mesh')
    pts, face_idxs = trimesh.sample.sample_surface(mesh, n_samples)
    nrm = mesh.face_normals[face_idxs]
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)
    return pts.astype(np.float32), nrm.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='PhysMorph-GS Shape Morphing')
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--png', action='store_true', help='Save PNG renders each episode')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg.get('output_dir', 'output/run'))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PhysMorph-GS: Physics + F_plastic Appearance")
    print("=" * 70)

    # ── Physics Init ──────────────────────────────────────────────────────────
    import diffmpm_bindings

    opt = build_opt_input(cfg)
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
    cg = initialize_comp_graph(input_pc, input_grid, target_grid)

    loss0 = cg.end_layer_mass_loss()
    print(f"[Init] EndLayerMassLoss (initial): {loss0:.2f}")

    # ── Target surface samples ────────────────────────────────────────────────
    appearance_cfg = cfg.get('appearance', {})
    n_surface = int(appearance_cfg.get('n_surface_samples', 50000))
    print(f"\n[Init] Sampling {n_surface:,} points from: {cfg['target_mesh_path']}")
    surface_pts, surface_nrm = load_surface_samples(cfg['target_mesh_path'], n_surface)
    print(f"  surface_pts: {surface_pts.shape}, normal range [{surface_nrm.min():.3f}, {surface_nrm.max():.3f}]")

    # ── F_plastic initialization ───────────────────────────────────────────────
    N = np.array(input_pc.get_positions()).shape[0]
    F_plastic = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))  # (N, 3, 3) = I
    print(f"\n[Init] F_plastic = I, N={N:,}")

    # ── Bilevel X₀ setup ──────────────────────────────────────────────────────
    bilevel_cfg     = cfg.get('bilevel', {})
    bilevel_enabled = bool(bilevel_cfg.get('enabled', False))
    bilevel_lr      = float(bilevel_cfg.get('lr', 0.05))
    bilevel_clip    = float(bilevel_cfg.get('grad_clip', 0.5))
    bilevel_start   = int(bilevel_cfg.get('start_ep', 2))
    bilevel_mom     = float(bilevel_cfg.get('momentum', 0.9))

    X0         = np.array(input_pc.get_positions(), dtype=np.float32)  # (N, 3)
    X0_vel     = np.zeros_like(X0)                                      # SGD momentum
    F_identity = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))       # (N,3,3)=I

    if bilevel_enabled:
        print(f"[Bilevel] ENABLED — lr={bilevel_lr}, grad_clip={bilevel_clip}, start_ep={bilevel_start}, momentum={bilevel_mom}")

    # ── Split Control (Δx) setup ──────────────────────────────────────────────
    sc_cfg = cfg.get('split_control', {})
    sc_enabled = bool(sc_cfg.get('enabled', False))
    split_control_state = None

    if sc_enabled:
        sc_lr = float(sc_cfg.get('lr', 0.01))
        # Keep delta_x on CPU — upsample/PCA pipeline operates on CPU tensors,
        # renderer internally moves to GPU as needed
        delta_x = torch.zeros(N, 3, dtype=torch.float32, requires_grad=True)
        sc_optimizer = torch.optim.Adam([delta_x], lr=sc_lr)
        split_control_state = {
            'delta_x': delta_x,
            'optimizer': sc_optimizer,
            'target_dt': None,   # filled after target_alpha generation
            'cfg': sc_cfg,
        }
        print(f"[SplitControl] ENABLED — lr={sc_lr}, decay={sc_cfg.get('decay', 0.95)}")

    # ── E2E config ────────────────────────────────────────────────────────────
    e2e_cfg = cfg.get('e2e', {})
    # Split control replaces E2E injection
    if sc_enabled:
        e2e_cfg = {'enabled': False}

    # ── Renderer ──────────────────────────────────────────────────────────────
    cam_cfg        = cfg.get('camera', {})
    render_cfg     = cfg.get('render', {})
    particle_color = render_cfg.get('particle_color', [0.27, 0.51, 0.71])
    renderer, view_params = setup_renderer(cam_cfg, render_cfg, training_mode=True)
    campos = view_params.get('campos')

    if renderer is None:
        print("[WARN] Renderer not available — PNG output disabled")
        args.png = False

    # ── Upsample config ───────────────────────────────────────────────────────
    rs = default_cfg()
    rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))
    sim_cfg = cfg.get('simulation', {})
    rs['physics_grid'] = {
        'grid_min': sim_cfg.get('grid_min_point', [-16, -16, -16]),
        'grid_max': sim_cfg.get('grid_max_point', [16, 16, 16]),
        'grid_dx':  sim_cfg.get('grid_dx', 1.0),
    }

    # ── Target render (always generate for alpha_mse metric) ────────────────
    target_alpha = None
    if renderer is not None:
        target_positions = np.array(target_pc.get_positions(), dtype=np.float32)
        target_alpha = generate_target_render(
            target_positions, rs, renderer, campos, render_cfg, particle_color,
            target_mesh_path=cfg.get('target_mesh_path'),
            cam_cfg=cam_cfg,
        )
        if target_alpha is not None:
            from utils.io_utils import save_image_png
            save_image_png(out_dir / 'target_alpha.png', target_alpha.numpy())
            print(f"  Saved → {out_dir}/target_alpha.png")
            # Precompute boundary DT for split control
            if split_control_state is not None:
                from utils.training_loop import _compute_boundary_dt
                target_dt = _compute_boundary_dt(target_alpha)
                split_control_state['target_dt'] = target_dt
                dt_viz = ((target_dt.numpy() + 1.0) * 0.5).clip(0, 1)
                save_image_png(out_dir / 'target_dt.png', dt_viz)
                print(f"  Saved → {out_dir}/target_dt.png")
        else:
            print("[WARN] Target alpha generation failed — E2E disabled for this run")
            e2e_cfg = {}

    # ── Training Loop ─────────────────────────────────────────────────────────
    ema_state    = {}
    num_eps      = int(opt.num_animations)
    loss_history = []

    print(f"\n{'=' * 70}")
    print(f"Training: {num_eps} episodes, {'--png' if args.png else 'no png'}")
    print(f"{'=' * 70}\n")

    # eta schedule: warmup (fast convergence) → steady
    eta_warmup     = float(appearance_cfg.get('eta_warmup',     appearance_cfg.get('eta', 0.1)))
    eta_warmup_eps = int(appearance_cfg.get('eta_warmup_eps',   5))
    eta_steady     = float(appearance_cfg.get('eta',            0.1))

    for ep in range(num_eps):
        # Per-episode appearance config with scheduled eta
        ep_appearance_cfg = dict(appearance_cfg)
        ep_appearance_cfg['eta'] = eta_warmup if ep < eta_warmup_eps else eta_steady

        # Bilevel: reset initial state to X₀ before each episode
        if bilevel_enabled:
            cg.promote_last_as_initial(carry_grid=False)  # resize layers to 1
            pc0 = cg.get_point_cloud(0)
            pc0.set_positions(X0)       # override: x ← X₀
            pc0.set_def_grads(F_identity)  # override: F ← I
            pc0.reset_kinematics()      # zero v, C

        F_plastic, losses, dLdx = run_episode(
            ep, cg, opt, F_plastic,
            surface_pts, surface_nrm, ep_appearance_cfg,
            rs, renderer, campos, render_cfg, particle_color,
            out_dir, args.png, ema_state,
            target_alpha=target_alpha,
            e2e_cfg=e2e_cfg,
            split_control_state=split_control_state,
        )

        if not bilevel_enabled:
            cg.promote_last_as_initial(carry_grid=False)

        # Split Control: feed Δx back into physics initial positions
        if split_control_state is not None:
            sc_decay = float(sc_cfg.get('decay', 0.95))
            dx_np = split_control_state['delta_x'].detach().cpu().numpy()
            dx_norm = np.linalg.norm(dx_np, axis=1)

            # Inject Δx into next episode's initial positions
            if dx_norm.max() > 1e-8:
                pc0 = cg.get_point_cloud(0)
                x0_current = np.array(pc0.get_positions(), dtype=np.float32)
                pc0.set_positions(x0_current + dx_np)
                print(f"  [SplitCtrl] Δx → physics init, ||Δx||: mean={dx_norm.mean():.4f}, max={dx_norm.max():.4f}")

            # Decay + keep for next episode
            with torch.no_grad():
                split_control_state['delta_x'] *= sc_decay

        # Bilevel outer loop: update X₀ using Chamfer gradient
        if bilevel_enabled and ep >= bilevel_start:
            grad_norms = np.linalg.norm(dLdx, axis=1, keepdims=True)   # (N,1)
            scale      = np.minimum(1.0, bilevel_clip / (grad_norms + 1e-8))
            X0_vel     = bilevel_mom * X0_vel + bilevel_lr * (scale * dLdx)
            X0         = X0 - X0_vel

            x0_gnorm = float(np.linalg.norm(dLdx))
            x0_unorm = float(np.linalg.norm(X0_vel))
            losses['x0_grad_norm']   = x0_gnorm
            losses['x0_update_norm'] = x0_unorm
            print(f"  [Bilevel] grad_norm={x0_gnorm:.4f}  update_norm={x0_unorm:.4f}")

        loss_history.append({'ep': ep, **losses})

        # Save loss log each episode
        with open(out_dir / 'losses.json', 'w') as f:
            json.dump(loss_history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Done. Output: {out_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
