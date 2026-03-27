"""
run.py - PhysMorph-GS: Hard-Coupled Physics-to-Render Pipeline

Pipeline:
  [Physics]  C++ MPM optimization (with render grads from prev episode)
  [Bridge]   Deterministic h(x_T, F_T) -> Gaussian params (Sigma = F*Sigma0*F^T)
  [Render]   Alpha loss (BCE + IoU) -> dL/dF, dL/dx -> inject next ep

Injection modes (hard_coupling.mode):
  'fixed_norm' : g_inject = alpha_fixed * normalize(g_render)
  'option_a'   : proximal attractor via x_target

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
from scipy.spatial import cKDTree

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
from utils.alpha_losses import compute_dt_map


def compute_sigma0_from_particles(positions: np.ndarray, scale: float = 0.5) -> float:
    """Compute fixed sigma0 from mean nearest-neighbor distance."""
    tree = cKDTree(positions)
    dd, _ = tree.query(positions, k=2)
    mean_nn_dist = float(dd[:, 1].mean())
    return mean_nn_dist * scale


def main():
    parser = argparse.ArgumentParser(description='PhysMorph-GS Shape Morphing')
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--png', action='store_true', help='Save PNG renders each episode')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg.get('output_dir', 'output/run'))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PhysMorph-GS: Hard-Coupled Physics-to-Render Pipeline")
    print("=" * 70)

    # ── Physics init ───────────────────────────────────────────────────────
    import diffmpm_bindings

    opt = build_opt_input(cfg)
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
    cg = initialize_comp_graph(input_pc, input_grid, target_grid)

    x0 = np.array(input_pc.get_positions(), dtype=np.float32)
    N = x0.shape[0]
    loss0 = cg.end_layer_mass_loss()
    print(f"[Init] N={N:,}, EndLayerMassLoss (initial): {loss0:.2f}")

    # ── Hard coupling config ──────────────────────────────────────────────
    hc_cfg = cfg.get('hard_coupling', {})
    mode = hc_cfg.get('mode', 'fixed_norm')

    sigma0_cfg = hc_cfg.get('sigma0', 'auto')
    if sigma0_cfg == 'auto':
        sigma0_scale = float(hc_cfg.get('sigma0_scale', 0.5))
        sigma0_fixed = compute_sigma0_from_particles(x0, scale=sigma0_scale)
    else:
        sigma0_fixed = float(sigma0_cfg)

    opacity_fixed = float(hc_cfg.get('opacity', 0.95))
    render_start_ep = int(hc_cfg.get('render_start_ep', 5))

    print(f"\n[Hard Coupling] mode={mode}")
    print(f"  sigma0_fixed={sigma0_fixed:.6f}, opacity_fixed={opacity_fixed:.3f}")
    print(f"  render_start_ep={render_start_ep}")

    if mode == 'fixed_norm':
        print(f"  alpha_fixed={hc_cfg.get('alpha_fixed', 0.5)}")
    elif mode == 'option_a':
        print(f"  eta={hc_cfg.get('eta', 0.1)}, lambda_attr={hc_cfg.get('lambda_attr', 1.0)}, "
              f"tau={hc_cfg.get('tau', 0.5)}, smooth_k={hc_cfg.get('smooth_k', 0)}")

    loss_w = hc_cfg.get('loss_weights', {})
    print(f"  loss: w_bce={loss_w.get('w_bce', 1.0)}, w_iou={loss_w.get('w_iou', 1.0)}, "
          f"w_dt={loss_w.get('w_dt', 0.0)}, mask={loss_w.get('masked_threshold', 0.0)}")

    # ── Renderer ───────────────────────────────────────────────────────────
    cam_cfg = cfg.get('camera', {})
    render_cfg = cfg.get('render', {})
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
        'grid_dx': sim_cfg.get('grid_dx', 1.0),
    }

    # ── Target render ──────────────────────────────────────────────────────
    target_alpha = None
    dt_map = None
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
            print(f"  Saved -> {out_dir}/target_alpha.png")
            dt_map = compute_dt_map(target_alpha)
            print(f"  DT map: range=[{dt_map.min():.3f}, {dt_map.max():.3f}]")
        else:
            print("[WARN] Target alpha generation failed")

    # ── Training loop ──────────────────────────────────────────────────────
    num_eps = int(opt.num_animations)
    loss_history = []
    stored_render_grads = None
    x_prev = None
    x_prev2 = None

    print(f"\n{'=' * 70}")
    print(f"Training: {num_eps} episodes, {'--png' if args.png else 'no png'}")
    print(f"{'=' * 70}\n")

    for ep in range(num_eps):
        losses, new_render_grads = run_episode(
            ep, cg, opt,
            sigma0_fixed, opacity_fixed,
            rs, renderer, campos, render_cfg, particle_color,
            out_dir, args.png,
            target_alpha=target_alpha,
            cam_cfg=cam_cfg,
            stored_render_grads=stored_render_grads,
            hard_coupling_cfg=hc_cfg,
            dt_map=dt_map,
            x_prev=x_prev,
            x_prev2=x_prev2,
        )

        cg.promote_last_as_initial(carry_grid=False)

        pc_final = cg.get_point_cloud(0)
        x_final = np.array(pc_final.get_positions(), dtype=np.float32).copy()
        x_prev2 = x_prev
        x_prev = x_final

        if new_render_grads is not None:
            stored_render_grads = new_render_grads

        loss_history.append({'ep': ep, **losses})
        with open(out_dir / 'losses.json', 'w') as f:
            json.dump(loss_history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Done. Output: {out_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
