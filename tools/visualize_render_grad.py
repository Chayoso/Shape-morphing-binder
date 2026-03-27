"""
Exp 4: Render gradient field visualization.

Loads particle positions from a given episode and overlays dL/dx vectors
to check if render gradients point toward missing silhouette regions.

Usage:
    python tools/visualize_render_grad.py -c configs/sphere_to_bunny.yaml --ep 10

Output:
    output/<run>/ep010/render_grad_field.png   -- 3D scatter with gradient arrows
    output/<run>/ep010/render_grad_stats.json  -- norm, cos_sim, coherence metrics
"""

import argparse
import json
import numpy as np
import yaml
import torch
from pathlib import Path


def load_episode_particles(cg, ep_idx: int, num_timesteps: int):
    """Run physics up to ep_idx and return final positions + F_e."""
    import diffmpm_bindings
    # After promote_last_as_initial(), the final state is at layer 0
    pc = cg.get_point_cloud(0)
    x = np.array(pc.get_positions(), dtype=np.float32)
    try:
        F_e = pc.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy()
    except Exception:
        F_e = np.array(pc.get_def_grads(), dtype=np.float32)
    return x, F_e.astype(np.float32)


def compute_gradient_field(x, F_e, sigma0_fixed, opacity_fixed,
                           target_alpha, renderer, campos, render_cfg,
                           particle_color, loss_cfg, dt_map=None):
    """Compute dLdF and dLdx for given physics state."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.training_loop import extract_render_gradients

    dLdF, dLdx, metrics = extract_render_gradients(
        x, F_e, sigma0_fixed, opacity_fixed,
        target_alpha, renderer, campos, render_cfg, particle_color,
        loss_cfg, dt_map=dt_map,
    )
    return dLdF, dLdx, metrics


def visualize_grad_field_3d(x, dLdx, out_path: Path, subsample: int = 2000,
                             title: str = "dL/dx render gradient field"):
    """3D scatter plot with gradient arrows (matplotlib)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    N = x.shape[0]
    if N > subsample:
        idx = np.random.choice(N, subsample, replace=False)
        x_vis = x[idx]
        g_vis = dLdx[idx]
    else:
        x_vis = x
        g_vis = dLdx

    # Normalize for display
    norms = np.linalg.norm(g_vis, axis=1, keepdims=True)
    g_norm = g_vis / (norms.max() + 1e-8)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter: colored by gradient magnitude
    sc = ax.scatter(x_vis[:, 0], x_vis[:, 1], x_vis[:, 2],
                    c=norms.squeeze(), cmap='hot', s=2, alpha=0.4)
    plt.colorbar(sc, ax=ax, label='||dLdx||', shrink=0.6)

    # Quiver: gradient direction
    scale = (x_vis.max() - x_vis.min()) * 0.08
    ax.quiver(x_vis[:, 0], x_vis[:, 1], x_vis[:, 2],
              g_norm[:, 0], g_norm[:, 1], g_norm[:, 2],
              length=scale, normalize=False, color='cyan', alpha=0.6, linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out_path}")


def compute_coherence(dLdx, x, k: int = 16) -> float:
    """
    Measure spatial coherence of gradient field:
    average cosine similarity between neighboring particle gradients.
    High coherence (>0.5) = smooth, semantically meaningful field.
    Low coherence (<0.1) = noisy, likely not useful.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(x)
    _, idx = tree.query(x, k=min(k + 1, len(x)))

    norms = np.linalg.norm(dLdx, axis=1, keepdims=True)
    g_hat = dLdx / (norms + 1e-8)

    cos_sims = []
    for i in range(len(x)):
        neighbors = idx[i, 1:]  # exclude self
        g_self = g_hat[i]
        g_neigh = g_hat[neighbors]
        cos = (g_self * g_neigh).sum(axis=1)
        cos_sims.append(cos.mean())

    return float(np.mean(cos_sims))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--ep', type=int, default=10, help='Episode to visualize')
    parser.add_argument('--subsample', type=int, default=2000)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg.get('output_dir', 'output/run'))
    ep_dir = out_dir / f"ep{args.ep:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Physics init
    import diffmpm_bindings
    from utils.physics_utils import (
        build_opt_input, initialize_point_clouds,
        initialize_grids, initialize_comp_graph,
    )
    from utils.rendering_utils import setup_renderer, generate_target_render
    from utils.alpha_losses import compute_dt_map
    from scipy.spatial import cKDTree

    opt = build_opt_input(cfg)
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
    cg = initialize_comp_graph(input_pc, input_grid, target_grid)

    # Run physics to target episode
    print(f"Running physics to ep{args.ep}...")
    for ep in range(args.ep + 1):
        cg.run_optimization(opt)
        cg.promote_last_as_initial(carry_grid=False)
        opt.current_episodes = ep + 1
        if ep % 5 == 0:
            print(f"  ep{ep}: phys={cg.end_layer_mass_loss():.1f}")

    # Extract state
    num_timesteps = int(opt.num_timesteps)
    x, F_e = load_episode_particles(cg, args.ep, num_timesteps)
    print(f"N={len(x):,} particles")

    # Renderer + target
    hc_cfg = cfg.get('hard_coupling', {})
    cam_cfg = cfg.get('camera', {})
    render_cfg = cfg.get('render', {})
    particle_color = render_cfg.get('particle_color', [0.27, 0.51, 0.71])
    renderer, view_params = setup_renderer(cam_cfg, render_cfg, training_mode=True)
    campos = view_params.get('campos')

    from sampling import default_cfg
    from sampling.utils.config_adapter import adapt_config
    rs = default_cfg()
    rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))
    sim_cfg = cfg.get('simulation', {})
    rs['physics_grid'] = {
        'grid_min': sim_cfg.get('grid_min_point', [-16, -16, -16]),
        'grid_max': sim_cfg.get('grid_max_point', [16, 16, 16]),
        'grid_dx': sim_cfg.get('grid_dx', 1.0),
    }

    target_positions = np.array(target_pc.get_positions(), dtype=np.float32)
    target_alpha = generate_target_render(
        target_positions, rs, renderer, campos, render_cfg, particle_color,
        target_mesh_path=cfg.get('target_mesh_path'), cam_cfg=cam_cfg,
    )
    dt_map = compute_dt_map(target_alpha) if target_alpha is not None else None

    # Sigma0
    tree = cKDTree(x)
    dd, _ = tree.query(x, k=2)
    sigma0_fixed = float(dd[:, 1].mean()) * float(hc_cfg.get('sigma0_scale', 0.5))
    opacity_fixed = float(hc_cfg.get('opacity', 0.95))
    loss_cfg = hc_cfg.get('loss_weights', {})

    # Compute gradient field
    print("Computing render gradient field...")
    dLdF, dLdx, metrics = compute_gradient_field(
        x, F_e, sigma0_fixed, opacity_fixed,
        target_alpha, renderer, campos, render_cfg, particle_color,
        loss_cfg, dt_map=dt_map,
    )

    if dLdx is None:
        print("ERROR: dLdx is None — renderer may have failed")
        return

    # Save raw arrays for offline analysis
    np.save(ep_dir / 'dLdx.npy', dLdx)
    np.save(ep_dir / 'x_particles.npy', x)
    print(f"  Saved dLdx.npy, x_particles.npy -> {ep_dir}")

    # ── Raw coherence ─────────────────────────────────────────────────────
    coherence_raw = compute_coherence(dLdx, x)
    dLdx_norm = float(np.linalg.norm(dLdx))
    dLdF_norm = float(np.linalg.norm(dLdF)) if dLdF is not None else 0.0
    per_particle_norms = np.linalg.norm(dLdx, axis=1)

    print(f"\n--- Gradient Field Stats (ep{args.ep}) ---")
    print(f"  N={len(x):,}")
    print(f"  ||dLdx|| = {dLdx_norm:.2f},  ||dLdF|| = {dLdF_norm:.2f}")
    print(f"  per-particle: mean={per_particle_norms.mean():.5f}, max={per_particle_norms.max():.3f}")
    print(f"  spatial_coherence (raw) = {coherence_raw:.4f}  (>0.5=coherent, <0.1=noisy)")
    print(f"  loss_total = {metrics.get('loss_total', 0):.4f}")

    # ── Exp B: Smoothing variants ──────────────────────────────────────────
    from scipy.spatial import cKDTree as _KDTree
    print("\n--- Exp B: Smoothing variants ---")

    smoothing_results = {}

    def _smooth_displacement(g, x_pos, k):
        """Average displacement vectors over k+1 neighbors (including self)."""
        tree = _KDTree(x_pos)
        _, idx = tree.query(x_pos, k=min(k + 1, len(x_pos)))
        return g[idx].mean(axis=1).astype(np.float32)

    def _clip_normalize(g, clip_pct=95):
        """Clip per-particle norms at clip_pct percentile, then renormalize."""
        norms = np.linalg.norm(g, axis=1)
        thresh = np.percentile(norms, clip_pct)
        scale = np.minimum(1.0, thresh / (norms + 1e-8))
        return g * scale[:, None]

    # Variant 1: raw
    c = compute_coherence(dLdx, x)
    smoothing_results['raw'] = {'coherence': c, 'norm': dLdx_norm}
    print(f"  raw:                   coherence={c:.4f}, ||g||={dLdx_norm:.2f}")

    # Variant 2: clip outliers (p95)
    g_clipped = _clip_normalize(dLdx, clip_pct=95)
    c = compute_coherence(g_clipped, x)
    smoothing_results['clip_p95'] = {'coherence': c, 'norm': float(np.linalg.norm(g_clipped))}
    print(f"  clip_p95:              coherence={c:.4f}, ||g||={np.linalg.norm(g_clipped):.2f}")

    # Variant 3: KNN smooth k=8
    g_s8 = _smooth_displacement(dLdx, x, k=8)
    c = compute_coherence(g_s8, x)
    smoothing_results['knn_k8'] = {'coherence': c, 'norm': float(np.linalg.norm(g_s8))}
    print(f"  knn_k8:                coherence={c:.4f}, ||g||={np.linalg.norm(g_s8):.2f}")

    # Variant 4: KNN smooth k=32
    g_s32 = _smooth_displacement(dLdx, x, k=32)
    c = compute_coherence(g_s32, x)
    smoothing_results['knn_k32'] = {'coherence': c, 'norm': float(np.linalg.norm(g_s32))}
    print(f"  knn_k32:               coherence={c:.4f}, ||g||={np.linalg.norm(g_s32):.2f}")

    # Variant 5: clip + k=8
    g_ck8 = _smooth_displacement(g_clipped, x, k=8)
    c = compute_coherence(g_ck8, x)
    smoothing_results['clip_p95_knn_k8'] = {'coherence': c, 'norm': float(np.linalg.norm(g_ck8))}
    print(f"  clip_p95 + knn_k8:     coherence={c:.4f}, ||g||={np.linalg.norm(g_ck8):.2f}")

    # Variant 6: clip + k=32
    g_ck32 = _smooth_displacement(g_clipped, x, k=32)
    c = compute_coherence(g_ck32, x)
    smoothing_results['clip_p95_knn_k32'] = {'coherence': c, 'norm': float(np.linalg.norm(g_ck32))}
    print(f"  clip_p95 + knn_k32:    coherence={c:.4f}, ||g||={np.linalg.norm(g_ck32):.2f}")

    # ── Build stats dict ──────────────────────────────────────────────────
    stats = {
        'ep': args.ep,
        'N': len(x),
        'dLdx_norm': dLdx_norm,
        'dLdF_norm': dLdF_norm,
        'dLdx_per_particle_mean': float(per_particle_norms.mean()),
        'dLdx_per_particle_max': float(per_particle_norms.max()),
        'spatial_coherence': coherence_raw,
        'smoothing_study': smoothing_results,
        **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
    }

    # Save stats
    stats_path = ep_dir / 'render_grad_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved -> {stats_path}")

    # Visualize raw and best smoothed variant
    visualize_grad_field_3d(
        x, dLdx,
        out_path=ep_dir / 'render_grad_field.png',
        subsample=args.subsample,
        title=f"dL/dx raw (ep{args.ep}, coherence={coherence_raw:.3f})",
    )
    best_smooth = g_ck32  # clip_p95 + knn_k32
    best_c = smoothing_results['clip_p95_knn_k32']['coherence']
    visualize_grad_field_3d(
        x, best_smooth,
        out_path=ep_dir / 'render_grad_field_smoothed.png',
        subsample=args.subsample,
        title=f"dL/dx clip+knn_k32 (ep{args.ep}, coherence={best_c:.3f})",
    )


if __name__ == '__main__':
    main()
