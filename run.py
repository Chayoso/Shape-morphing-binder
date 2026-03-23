"""
run.py - PhysMorph-GS: Physics + Render-Guided Shape Morphing

Architecture:
  [Physics]    EndLayerMassLoss drives shape morphing
  [E2E]        Render gradient injection: dL/dF + dL/dx → C++ backward
  [Appearance] F_plastic analytical update (surface-aligned Gaussians)
  [Render]     F_total = F_elastic @ F_plastic → Σ = F_total F_total^T

Usage:
    python run.py -c configs/sphere_to_bunny.yaml [--png]
    python run.py -c configs/e2e_dLdF_test.yaml --png
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
    print("PhysMorph-GS: Physics + Render-Guided Shape Morphing")
    print("=" * 70)

    # ── Physics init ───────────────────────────────────────────────────────
    import diffmpm_bindings

    opt = build_opt_input(cfg)
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
    cg = initialize_comp_graph(input_pc, input_grid, target_grid)

    loss0 = cg.end_layer_mass_loss()
    print(f"[Init] EndLayerMassLoss (initial): {loss0:.2f}")

    # ── Target surface samples ─────────────────────────────────────────────
    appearance_cfg = cfg.get('appearance', {})
    n_surface = int(appearance_cfg.get('n_surface_samples', 50000))
    print(f"\n[Init] Sampling {n_surface:,} points from: {cfg['target_mesh_path']}")
    surface_pts, surface_nrm = load_surface_samples(cfg['target_mesh_path'], n_surface)
    print(f"  surface_pts: {surface_pts.shape}, normal range [{surface_nrm.min():.3f}, {surface_nrm.max():.3f}]")

    # ── F_plastic initialization ──────────────────────────────────────────
    N = np.array(input_pc.get_positions()).shape[0]
    F_plastic = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    print(f"\n[Init] F_plastic = I, N={N:,}")

    # ── E2E config ─────────────────────────────────────────────────────────
    e2e_cfg = cfg.get('e2e', {})

    # ── Renderer ───────────────────────────────────────────────────────────
    cam_cfg        = cfg.get('camera', {})
    render_cfg     = cfg.get('render', {})
    particle_color = render_cfg.get('particle_color', [0.27, 0.51, 0.71])
    renderer, view_params = setup_renderer(cam_cfg, render_cfg, training_mode=True)
    campos = view_params.get('campos')

    if renderer is None:
        print("[WARN] Renderer not available — PNG output disabled")
        args.png = False

    # ── Upsample config ────────────────────────────────────────────────────
    rs = default_cfg()
    rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))
    sim_cfg = cfg.get('simulation', {})
    rs['physics_grid'] = {
        'grid_min': sim_cfg.get('grid_min_point', [-16, -16, -16]),
        'grid_max': sim_cfg.get('grid_max_point', [16, 16, 16]),
        'grid_dx':  sim_cfg.get('grid_dx', 1.0),
    }

    # ── Target render (for alpha_mse metric + E2E) ────────────────────────
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
        else:
            print("[WARN] Target alpha generation failed — E2E disabled")
            e2e_cfg = {}

    # ── Training loop ──────────────────────────────────────────────────────
    ema_state    = {}
    num_eps      = int(opt.num_animations)
    loss_history = []

    # eta schedule: warmup → steady
    eta_warmup     = float(appearance_cfg.get('eta_warmup', appearance_cfg.get('eta', 0.1)))
    eta_warmup_eps = int(appearance_cfg.get('eta_warmup_eps', 5))
    eta_steady     = float(appearance_cfg.get('eta', 0.1))

    print(f"\n{'=' * 70}")
    print(f"Training: {num_eps} episodes, {'--png' if args.png else 'no png'}")
    if bool(e2e_cfg.get('enabled', False)):
        print(f"E2E: start_ep={e2e_cfg.get('start_ep', 0)}, gain={e2e_cfg.get('render_gain', 0.1)}")
    print(f"{'=' * 70}\n")

    for ep in range(num_eps):
        ep_appearance_cfg = dict(appearance_cfg)
        ep_appearance_cfg['eta'] = eta_warmup if ep < eta_warmup_eps else eta_steady

        F_plastic, losses = run_episode(
            ep, cg, opt, F_plastic,
            surface_pts, surface_nrm, ep_appearance_cfg,
            rs, renderer, campos, render_cfg, particle_color,
            out_dir, args.png, ema_state,
            target_alpha=target_alpha,
            e2e_cfg=e2e_cfg,
        )

        cg.promote_last_as_initial(carry_grid=False)

        loss_history.append({'ep': ep, **losses})
        with open(out_dir / 'losses.json', 'w') as f:
            json.dump(loss_history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Done. Output: {out_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
