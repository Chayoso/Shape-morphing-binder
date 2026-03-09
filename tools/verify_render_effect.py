"""
Verify whether render gradients actually change F.

Runs two experiments with identical physics setup:
  A) Physics-only (no render loss)
  B) Physics + Render (render loss enabled)

Then compares the F (deformation gradient) tensors after each episode.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sampling import default_cfg
from sampling.utils.config_adapter import adapt_config
from loss import E2ELossManager
from utils.physics_utils import (
    build_opt_input, initialize_point_clouds, initialize_grids,
    initialize_comp_graph, extract_target_point_cloud, extract_point_cloud_state
)
from utils.rendering_utils import setup_renderer, create_target_render
from utils.training_loop import run_e2e_episode


def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(cfg, label, num_episodes=2):
    """Run physics simulation and return F after each episode."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {label}")
    print(f"{'='*70}")

    out_dir = Path(cfg.get("output_dir", f"output/{label}/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build runtime config
    from run import build_runtime_config, _safe_deepcopy
    cfg_runtime, rs = build_runtime_config(cfg, out_dir)

    # Init physics
    opt = build_opt_input(cfg_runtime)
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)

    import diffmpm_bindings
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)

    cg = initialize_comp_graph(input_pc, input_grid, target_grid)
    tgt = extract_target_point_cloud(target_pc)[0]

    # Check if E2E mode
    enable_e2e = cfg_runtime.get("optimization", {}).get("loss", {}).get("enabled", False)

    cam_cfg = cfg_runtime.get("camera", {}) or {}
    render_cfg = cfg_runtime.get("render", {}) or {}
    particle_color = render_cfg.get("particle_color", [0.27, 0.51, 0.71])
    renderer, view_params = setup_renderer(cam_cfg, render_cfg, training_mode=enable_e2e)

    loss_manager = None
    target_render = None

    if enable_e2e and renderer is not None:
        loss_cfg = cfg.get("optimization", {}).get("loss", {})
        loss_manager = E2ELossManager(loss_cfg)
        campos = view_params.get('campos')
        target_render = create_target_render(
            target_pc, renderer, rs, campos, render_cfg, particle_color, out_dir
        )
        if target_render is None:
            enable_e2e = False

    # Collect F after each episode
    F_history = []
    x_history = []
    ema_state = {}

    num_passes = cfg.get("optimization", {}).get("num_passes", 2)
    num_timesteps = int(opt.num_timesteps)
    control_stride = int(opt.control_stride)

    for ep in range(num_episodes):
        print(f"\n--- {label}: Episode {ep} ---")

        rs_ep = default_cfg()
        rs_ep_user = cfg.get("upsample", {}) or {}
        adapted = adapt_config({'upsample': rs_ep_user})
        rs_ep.update(adapted)
        if "covariance" not in rs_ep:
            rs_ep["covariance"] = {}
        rs_ep["covariance"]["episode"] = ep
        rs_ep["optimization"] = cfg.get("optimization", {})

        ep_dir = out_dir / f"ep{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        if enable_e2e and loss_manager is not None and target_render is not None:
            campos = view_params.get('campos')
            ema_state, episode_losses = run_e2e_episode(
                ep, cg, opt, num_timesteps, control_stride, num_passes,
                rs_ep, ema_state, renderer, loss_manager, target_render,
                view_params, campos, render_cfg, particle_color,
                ep_dir, False, tgt
            )
        else:
            opt.current_episodes = ep
            cg.run_optimization(opt)

        # Extract F BEFORE promote (this is the final state of this episode)
        num_layers = cg.get_num_layers()
        pc_final = cg.get_point_cloud(num_layers - 1)
        x, v, F = extract_point_cloud_state(pc_final, requires_grad=False)
        F_np = F.detach().cpu().numpy().copy()
        x_np = x.detach().cpu().numpy().copy()
        F_history.append(F_np)
        x_history.append(x_np)

        print(f"  [{label}] ep{ep}: F shape={F_np.shape}, "
              f"F range=[{F_np.min():.6f}, {F_np.max():.6f}], "
              f"||F||={np.linalg.norm(F_np):.4f}")

        # Promote for next episode
        cg.promote_last_as_initial(carry_grid=True)

    return F_history, x_history


def compare_results(F_phys, F_render, x_phys, x_render, num_episodes):
    """Compare F between physics-only and physics+render."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Physics-only vs Physics+Render")
    print(f"{'='*70}")

    for ep in range(num_episodes):
        Fp = F_phys[ep]
        Fr = F_render[ep]
        xp = x_phys[ep]
        xr = x_render[ep]

        # F difference
        F_diff = Fr - Fp
        F_diff_norm = np.linalg.norm(F_diff)
        F_diff_max = np.abs(F_diff).max()
        F_diff_mean = np.abs(F_diff).mean()
        Fp_norm = np.linalg.norm(Fp)

        # x difference
        x_diff = xr - xp
        x_diff_norm = np.linalg.norm(x_diff)
        x_diff_max = np.abs(x_diff).max()
        x_diff_mean = np.abs(x_diff).mean()
        xp_norm = np.linalg.norm(xp)

        # Per-particle F change
        per_particle_F_diff = np.linalg.norm(F_diff.reshape(-1, 9), axis=1)  # N particles
        pct_changed = (per_particle_F_diff > 1e-8).mean() * 100

        print(f"\n  Episode {ep}:")
        print(f"  ┌─ F (deformation gradient):")
        print(f"  │  ||F_phys||     = {Fp_norm:.6f}")
        print(f"  │  ||F_render||   = {np.linalg.norm(Fr):.6f}")
        print(f"  │  ||F_diff||     = {F_diff_norm:.6e}")
        print(f"  │  |F_diff|_max   = {F_diff_max:.6e}")
        print(f"  │  |F_diff|_mean  = {F_diff_mean:.6e}")
        print(f"  │  Relative diff  = {F_diff_norm/Fp_norm*100:.4f}%")
        print(f"  │  Particles with F changed: {pct_changed:.1f}%")
        print(f"  │")
        print(f"  └─ x (positions):")
        print(f"     ||x_phys||     = {xp_norm:.6f}")
        print(f"     ||x_render||   = {np.linalg.norm(xr):.6f}")
        print(f"     ||x_diff||     = {x_diff_norm:.6e}")
        print(f"     |x_diff|_max   = {x_diff_max:.6e}")
        print(f"     |x_diff|_mean  = {x_diff_mean:.6e}")
        print(f"     Relative diff  = {x_diff_norm/xp_norm*100:.4f}%")

        if F_diff_norm > 1e-6:
            print(f"\n  ✅ Render gradients ARE changing F (diff={F_diff_norm:.6e})")
        else:
            print(f"\n  ❌ Render gradients NOT changing F (diff={F_diff_norm:.6e})")

        if x_diff_norm > 1e-6:
            print(f"  ✅ Render gradients ARE changing x (diff={x_diff_norm:.6e})")
        else:
            print(f"  ❌ Render gradients NOT changing x (diff={x_diff_norm:.6e})")


def main():
    num_episodes = 2

    # Load configs
    cfg_phys = load_config("configs/test_physics_only.yaml")
    cfg_render = load_config("configs/test_gradient_flow.yaml")

    # Run physics-only
    F_phys, x_phys = run_experiment(cfg_phys, "Physics-only", num_episodes)

    # Run physics+render
    F_render, x_render = run_experiment(cfg_render, "Physics+Render", num_episodes)

    # Compare
    compare_results(F_phys, F_render, x_phys, x_render, num_episodes)


if __name__ == "__main__":
    main()
