"""
run.py — Surface-Aware Render-Guided Physics Morphing

Current pipeline:
  A. Physics rollout with optional render-guided control-space penalty
  B. Multi-view observation on a fixed surface shell / optional proxy shell
  C. Plastic update (Fp) + logging / checkpointing

Usage:
  Single:  python run.py -c configs/experiment.yaml [--png]
  Batch:   python run.py -c configs/batch.yaml [--png] [--skip-existing]
"""

import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse, copy, json, traceback, numpy as np, yaml, torch
from pathlib import Path
from scipy.spatial import cKDTree

from sampling import default_cfg
from sampling.utils.config_adapter import adapt_config
from utils.physics_utils import build_opt_input, initialize_point_clouds, initialize_grids, initialize_comp_graph
from utils.rendering_utils import setup_renderer, generate_target_observation
from utils.training_loop import run_episode, isochoric_project
from utils.visualize import create_episode_visualization, create_per_view_visualization
from utils.surface_utils import build_fixed_surface_mask, build_frozen_surface_graph, load_triangle_mesh
from utils.control_guidance import build_control_guidance_penalty, premerge_surface_obs_gradient
from utils.chamfer_plasticity import compute_chamfer_plasticity
from utils.covariance_opt import CovarianceOptimizer


def compute_sigma0(pos, scale=0.5):
    dd, _ = cKDTree(pos).query(pos, k=2)
    return float(dd[:,1].mean()) * scale


def _deep_update(dst, src):
    """Recursively update nested dictionaries."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def setup_cameras(base_cam, multi_cfg=None):
    """Build camera layouts from a small preset vocabulary."""
    if isinstance(multi_cfg, int):
        multi_cfg = {'num_cameras': multi_cfg}
    multi_cfg = multi_cfg or {}

    lookat = base_cam.get('lookat', {})
    eye = np.array(lookat.get('eye', [20, -25, 12.5]))
    target = np.array(lookat.get('target', [0, 0, 0]))
    offset = eye - target
    dist = float(np.linalg.norm(offset))
    base_elev = np.degrees(np.arctan2(offset[2], np.sqrt(offset[0]**2 + offset[1]**2)))
    base_azim = np.degrees(np.arctan2(offset[1], offset[0]))
    preset = str(multi_cfg.get('preset', 'ring')).lower()
    num_cameras = int(multi_cfg.get('num_cameras', 8))
    low_elev_offset = float(multi_cfg.get('low_elev_offset', 0.0))
    high_elev_offset = float(multi_cfg.get('high_elev_offset', 30.0))
    top_elev = float(multi_cfg.get('top_elev', 85.0))
    high_azim_offset = float(multi_cfg.get('high_azim_offset', 0.0))
    top_azim = float(multi_cfg.get('top_azim', base_azim))
    num_low = int(multi_cfg.get('num_low', max(1, num_cameras // 2)))
    num_high = int(multi_cfg.get('num_high', max(1, num_cameras - num_low)))
    ring_view_weight = float(multi_cfg.get('ring_view_weight', 1.0))
    low_view_weight = float(multi_cfg.get('low_view_weight', 1.0))
    high_view_weight = float(multi_cfg.get('high_view_weight', 1.0))
    top_view_weight = float(multi_cfg.get('top_view_weight', 1.0))

    def cam(elev, azim):
        e, a = np.radians(elev), np.radians(azim)
        pos = target + dist * np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
        c = base_cam.copy()
        c['lookat'] = {'eye': pos.tolist(), 'target': target.tolist(), 'up': [0, 0, 1]}
        return c, pos

    def add_ring(elev, count, prefix, weight, azim_offset=0.0):
        step = 360.0 / max(count, 1)
        for i in range(count):
            c, e = cam(elev, base_azim + azim_offset + step * i)
            cams.append(c)
            eyes.append(e)
            labels.append(f'{prefix}-{i}')
            weights.append(weight)

    cams, eyes, labels, weights = [], [], [], []
    if preset == 'ring':
        add_ring(base_elev + low_elev_offset, num_cameras, 'Ring', ring_view_weight)
        summary = f"{num_cameras} ring"
    elif preset == 'ring_top':
        add_ring(base_elev + low_elev_offset, num_cameras, 'Ring', ring_view_weight)
        c, e = cam(top_elev, top_azim)
        cams.append(c); eyes.append(e); labels.append('Top')
        weights.append(top_view_weight)
        summary = f"{num_cameras} ring + top"
    elif preset == 'dual_ring':
        add_ring(base_elev + low_elev_offset, num_low, 'Low', low_view_weight)
        add_ring(base_elev + high_elev_offset, num_high, 'High', high_view_weight, azim_offset=high_azim_offset)
        summary = f"{num_low} low + {num_high} high"
    elif preset == 'dual_ring_top':
        add_ring(base_elev + low_elev_offset, num_low, 'Low', low_view_weight)
        add_ring(base_elev + high_elev_offset, num_high, 'High', high_view_weight, azim_offset=high_azim_offset)
        c, e = cam(top_elev, top_azim)
        cams.append(c); eyes.append(e); labels.append('Top')
        weights.append(top_view_weight)
        summary = f"{num_low} low + {num_high} high + top"
    else:
        raise ValueError(f"Unknown multi_view preset: {preset}")

    print(f"  Cameras: {summary} (preset={preset}, base_elev={base_elev:.0f}°)")
    return cams, np.array(eyes), labels, np.array(weights, dtype=np.float32)



def _build_ear_focus_mask(
    x,
    active_mask=None,
    lateral_axis=0,
    height_axis=2,
    height_frac=0.58,
    side_frac=0.18,
    side_max_frac=0.50,
    center=None,
):
    """
    Build a bilateral upper-side mask that biases guidance toward two-ear splitting.

    The intent is to amplify gradients only on particles that already lie in the
    plausible ear support region: upper part of the bunny and away from the centerline.
    """
    N = x.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=bool), {'ear_focus_count': 0, 'ear_focus_frac': 0.0}

    if active_mask is None:
        active_mask = np.ones((N,), dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    lateral_axis = int(lateral_axis)
    height_axis = int(height_axis)
    bbox_min = x.min(axis=0)
    bbox_max = x.max(axis=0)
    span = np.maximum(bbox_max - bbox_min, 1e-6)
    z_norm = (x[:, height_axis] - bbox_min[height_axis]) / span[height_axis]

    center = float(center) if center is not None else float(0.5 * (bbox_min[lateral_axis] + bbox_max[lateral_axis]))
    side_dist = np.abs(x[:, lateral_axis] - center) / span[lateral_axis]

    mask = (
        active_mask
        & (z_norm >= float(height_frac))
        & (side_dist >= float(side_frac))
        & (side_dist <= float(side_max_frac))
    )

    return mask, {
        'ear_focus_count': int(mask.sum()),
        'ear_focus_frac': float(mask.mean()),
    }

# ─── Main Loop ───────────────────────────────────────────────────────────

def run_single(cfg, png=False):
    """Run a single experiment with the current render-guided physics pipeline."""
    out = Path(cfg.get('output_dir', 'output/run')); out.mkdir(parents=True, exist_ok=True)

    # ── Physics init ──────────────────────────────────────────────────────
    import diffmpm_bindings
    opt = build_opt_input(cfg)
    in_pc, tgt_pc = initialize_point_clouds(opt, cfg=cfg)
    in_grid, tgt_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(in_pc, in_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(tgt_pc, tgt_grid)
    cg = initialize_comp_graph(in_pc, in_grid, tgt_grid)

    resume_cfg = cfg.get('resume', {}) or {}
    resume_Fp = None
    if resume_cfg.get('enabled', False):
        ckpt_path = resume_cfg.get('checkpoint_path')
        if not ckpt_path:
            raise ValueError("resume.enabled=true but resume.checkpoint_path is missing")
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = np.load(ckpt_path)
        pc0 = cg.get_point_cloud(0)
        resume_pos = np.asarray(ckpt['positions'], dtype=np.float32)
        resume_vel = np.asarray(ckpt['velocities'], dtype=np.float32) if 'velocities' in ckpt.files else None
        resume_dFc = np.asarray(ckpt['dFc'], dtype=np.float32) if 'dFc' in ckpt.files else None
        resume_Fp = np.asarray(ckpt['Fp'], dtype=np.float32) if 'Fp' in ckpt.files else None

        if resume_pos.shape != np.asarray(pc0.get_positions(), dtype=np.float32).shape:
            raise ValueError(
                f"Resume checkpoint position shape mismatch: {resume_pos.shape} "
                f"vs {np.asarray(pc0.get_positions(), dtype=np.float32).shape}"
            )

        pc0.set_positions(np.ascontiguousarray(resume_pos))
        if resume_vel is not None:
            v = pc0.get_velocities_view()
            if v.shape != resume_vel.shape:
                raise ValueError(f"Resume velocity shape mismatch: {resume_vel.shape} vs {v.shape}")
            v[:] = resume_vel
        if resume_dFc is not None:
            pc0.set_dFc(np.ascontiguousarray(resume_dFc))

        print(
            f"[Resume] Loaded {ckpt_path} "
            f"(N={resume_pos.shape[0]:,}, has_vel={resume_vel is not None}, "
            f"has_dFc={resume_dFc is not None}, has_Fp={resume_Fp is not None})"
        )

    x0 = np.array(in_pc.get_positions(), dtype=np.float32)
    N = x0.shape[0]
    if resume_cfg.get('enabled', False):
        x0 = np.array(cg.get_point_cloud(0).get_positions(), dtype=np.float32)
        N = x0.shape[0]

    # ── Config ────────────────────────────────────────────────────────────
    rcfg = cfg.get('render', {})
    sigma0 = compute_sigma0(x0, float(rcfg.get('sigma0_scale', 0.5))) if rcfg.get('sigma0', 'auto') == 'auto' else float(rcfg['sigma0'])
    opacity = float(rcfg.get('opacity', 0.95))
    color = rcfg.get('particle_color', [0.27, 0.51, 0.71])

    pcfg = cfg.get('plasticity', {})
    lcfg = cfg.get('loss_weights', {})
    cam_cfg = cfg.get('camera', {})
    surface_cfg = cfg.get('surface_aware', {}) or {}
    control_guidance_cfg = cfg.get('control_guidance', {}) or {}
    freeze_fp = bool(pcfg.get('freeze_Fp', pcfg.get('freeze_fp', False)))

    eta_v = float(pcfg.get('eta_v', 0.5))
    max_v = float(pcfg.get('max_v', 5.0))
    max_aniso = float(pcfg.get('max_anisotropy', 1.5))
    damping = float(pcfg.get('damping', 0.0))
    debug_gradient_eps = set()
    control_guidance_enabled = bool(control_guidance_cfg.get('enabled', False))
    control_guidance_start_ep = int(control_guidance_cfg.get('start_ep', 1))
    same_step_merge_enabled = bool(control_guidance_cfg.get('same_step_merge', False))

    print(f"[Init] N={N:,}, loss={cg.end_layer_mass_loss():.1f}")
    print(
        f"[Config] eta_v={eta_v}, control_guidance={'ON' if control_guidance_enabled else 'OFF'}, "
        f"Fp={'I' if freeze_fp else 'adaptive'}"
    )

    surface_mask = None
    render_surface_mask = None
    surface_proxy = None
    surface_meta = {
        'surface_particle_count': int(N),
        'surface_particle_frac': 1.0,
        'render_surface_particle_count': int(N),
        'render_surface_particle_frac': 1.0,
        'surface_recon_resolution': int(surface_cfg.get('recon_resolution', 64)),
    }
    if bool(surface_cfg.get('enabled', False)):
        recon = build_fixed_surface_mask(
            x0,
            resolution=int(surface_cfg.get('recon_resolution', 64)),
            padding=float(surface_cfg.get('recon_padding', 2.0)),
            sigma=float(surface_cfg.get('recon_sigma', 1.5)),
            level_ratio=float(surface_cfg.get('recon_level_ratio', 0.02)),
            threshold_mult=float(surface_cfg.get('surface_threshold_mult', 1.75)),
            min_surface_frac=float(surface_cfg.get('min_surface_frac', 0.08)),
            max_surface_frac=float(surface_cfg.get('max_surface_frac', 0.40)),
        )
        surface_mask = np.asarray(recon['surface_mask'], dtype=bool)
        render_recon = build_fixed_surface_mask(
            x0,
            resolution=int(surface_cfg.get('recon_resolution', 64)),
            padding=float(surface_cfg.get('recon_padding', 2.0)),
            sigma=float(surface_cfg.get('recon_sigma', 1.5)),
            level_ratio=float(surface_cfg.get('recon_level_ratio', 0.02)),
            threshold_mult=float(surface_cfg.get('render_surface_threshold_mult', surface_cfg.get('surface_threshold_mult', 1.75))),
            min_surface_frac=float(surface_cfg.get('render_min_surface_frac', surface_cfg.get('min_surface_frac', 0.08))),
            max_surface_frac=float(surface_cfg.get('render_max_surface_frac', surface_cfg.get('max_surface_frac', 0.40))),
        )
        render_surface_mask = np.asarray(render_recon['surface_mask'], dtype=bool)
        surface_meta.update({
            'surface_particle_count': int(surface_mask.sum()),
            'surface_particle_frac': float(surface_mask.mean()),
            'render_surface_particle_count': int(render_surface_mask.sum()),
            'render_surface_particle_frac': float(render_surface_mask.mean()),
            'surface_distance_threshold': float(recon.get('surface_distance_threshold', 0.0)),
            'surface_distance_mean': float(recon.get('surface_distance_mean', 0.0)),
            'render_surface_distance_threshold': float(render_recon.get('surface_distance_threshold', 0.0)),
        })
        np.savez_compressed(
            out / 'surface_recon.npz',
            surface_mask=surface_mask.astype(np.uint8),
            render_surface_mask=render_surface_mask.astype(np.uint8),
            verts=np.zeros((0, 3), dtype=np.float32) if recon.get('verts') is None else recon['verts'],
            faces=np.zeros((0, 3), dtype=np.int32) if recon.get('faces') is None else recon['faces'],
            origin=recon.get('origin', np.zeros((3,), dtype=np.float32)),
            spacing=np.asarray([recon.get('spacing', 0.0)], dtype=np.float32),
            level=np.asarray([recon.get('level', 0.0)], dtype=np.float32),
            density_max=np.asarray([recon.get('density_max', 0.0)], dtype=np.float32),
            surface_distance_threshold=np.asarray([recon.get('surface_distance_threshold', 0.0)], dtype=np.float32),
            surface_fraction=np.asarray([recon.get('surface_fraction', 1.0)], dtype=np.float32),
            render_surface_distance_threshold=np.asarray([render_recon.get('surface_distance_threshold', 0.0)], dtype=np.float32),
            render_surface_fraction=np.asarray([render_recon.get('surface_fraction', 1.0)], dtype=np.float32),
        )
        print(
            f"[Surface] enabled, recon={surface_meta['surface_recon_resolution']}^3, "
            f"mask={surface_meta['surface_particle_count']:,}/{N:,} "
            f"({100.0 * surface_meta['surface_particle_frac']:.1f}%), "
            f"render_mask={surface_meta['render_surface_particle_count']:,}/{N:,} "
            f"({100.0 * surface_meta['render_surface_particle_frac']:.1f}%)"
        )
    else:
        render_surface_mask = surface_mask

    surface_proxy_cfg = cfg.get('surface_proxy', {}) or {}
    if bool(surface_proxy_cfg.get('enabled', False)) and render_surface_mask is not None:
        mesh_vertices = None
        mesh_faces = None
        graph_mode_req = str(surface_proxy_cfg.get('graph_mode', 'mesh_geodesic')).lower()
        if graph_mode_req != 'euclidean_knn':
            mesh_path = cfg.get('input_mesh_path')
            if mesh_path:
                mesh_path = Path(mesh_path)
                try:
                    mesh_vertices, mesh_faces = load_triangle_mesh(str(mesh_path))
                except Exception as exc:
                    print(f"[SurfaceProxy] mesh graph fallback -> euclidean kNN ({exc})")
        surface_graph = build_frozen_surface_graph(
            x0,
            particle_mask=render_surface_mask,
            k=int(surface_proxy_cfg.get('graph_k', 16)),
            sigma_scale=float(surface_proxy_cfg.get('graph_sigma_scale', 1.0)),
            num_patches=int(surface_proxy_cfg.get('num_patches', control_guidance_cfg.get('num_patches', 96))),
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            max_hops=int(surface_proxy_cfg.get('graph_max_hops', 4)),
            hop_weight=float(surface_proxy_cfg.get('hop_weight', 0.75)),
            separation_enabled=bool(surface_proxy_cfg.get('separation_enabled', False)),
            separation_partner_k=int(surface_proxy_cfg.get('separation_partner_k', 4)),
            separation_patch_min_hops=int(surface_proxy_cfg.get('separation_patch_min_hops', 6)),
        )
        surface_proxy = {
            'enabled': surface_graph is not None,
            'graph': surface_graph,
            'cfg': surface_proxy_cfg,
        }
        if surface_graph is not None:
            surface_meta['proxy_graph_count'] = int(surface_graph['source_positions'].shape[0])
            surface_meta['proxy_graph_mode'] = str(surface_graph.get('graph_mode', 'unknown'))
            print(
                f"[SurfaceProxy] enabled, nodes={surface_graph['source_positions'].shape[0]:,}, "
                f"k={surface_graph['neighbors'].shape[1]}, mode={surface_graph.get('graph_mode', 'unknown')}"
            )

    # ── Cameras + targets ─────────────────────────────────────────────────
    rs = default_cfg()
    rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))
    sim = cfg.get('simulation', {})
    rs['physics_grid'] = {
        'grid_min': sim.get('grid_min_point', [-16]*3),
        'grid_max': sim.get('grid_max_point', [16]*3),
        'grid_dx': sim.get('grid_dx', 1.0),
    }

    multi_cfg = cfg.get('multi_view', {})
    multi = multi_cfg.get('enabled', False)
    renderers, campos_list, targets, target_depths, cam_eyes, cam_labels = [], [], [], [], np.array([]), []
    view_weights = []

    if multi:
        all_cams, cam_eyes, cam_labels, all_view_weights = setup_cameras(cam_cfg, multi_cfg)
        tpos = np.array(tgt_pc.get_positions(), dtype=np.float32)
        active_eyes, active_labels = [], []
        for cam, view_w, eye, label in zip(all_cams, all_view_weights, cam_eyes, cam_labels):
            r, p = setup_renderer(cam, rcfg, training_mode=True)
            if r:
                obs = generate_target_observation(
                    tpos, rs, r, p['campos'], rcfg, color,
                    target_mesh_path=cfg.get('target_mesh_path'), cam_cfg=cam
                )
                if obs is not None and obs.get('alpha') is not None:
                    t = obs['alpha']
                    renderers.append(r); campos_list.append(p['campos']); targets.append(t)
                    target_depths.append(obs.get('depth'))
                    view_weights.append(float(view_w))
                    active_eyes.append(eye)
                    active_labels.append(label)
        cam_eyes = np.asarray(active_eyes, dtype=np.float32) if active_eyes else np.array([])
        cam_labels = active_labels
        print(f"[Cameras] {len(renderers)} views")
    else:
        r, p = setup_renderer(cam_cfg, rcfg, training_mode=True)
        if r:
            tpos = np.array(tgt_pc.get_positions(), dtype=np.float32)
            obs = generate_target_observation(
                tpos, rs, r, p['campos'], rcfg, color,
                target_mesh_path=cfg.get('target_mesh_path'), cam_cfg=cam_cfg
            )
            if obs is not None and obs.get('alpha') is not None:
                t = obs['alpha']
                renderers.append(r); campos_list.append(p['campos']); targets.append(t)
                target_depths.append(obs.get('depth'))
                cam_eyes = np.array([p['campos']]); cam_labels = ['Primary']
                view_weights = [1.0]

    if targets:
        from utils.io_utils import save_image_png, save_depth_png
        save_image_png(out / 'target_alpha.png', targets[0].numpy())
        if target_depths and target_depths[0] is not None:
            td = target_depths[0]
            td_np = td.cpu().numpy() if hasattr(td, 'cpu') else np.asarray(td, dtype=np.float32)
            if (td_np > 0).any():
                save_depth_png(out / 'target_depth.png', td_np, bits=16)
    # ── Train ─────────────────────────────────────────────────────────────
    Fp = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)) if resume_Fp is None else resume_Fp.copy()
    if freeze_fp:
        Fp[:] = np.eye(3, dtype=np.float32)[None]
    num_eps = int(opt.num_animations)
    episode_offset = int(resume_cfg.get('episode_offset', 0))
    total_sched_eps = int(resume_cfg.get('total_episodes', episode_offset + num_eps))
    history = []

    alpha_start = float(cfg.get('optimization', {}).get('initial_alpha', 0.01))
    alpha_end = float(cfg.get('optimization', {}).get('final_alpha', alpha_start * 0.1))

    if episode_offset > 0:
        print(
            f"\nTraining: {num_eps} episodes (global ep {episode_offset}..{episode_offset + num_eps - 1})\n"
        )
    else:
        print(f"\nTraining: {num_eps} episodes\n")

    prev_surface_obs_x = None
    prev_surface_obs_grad = None
    prev_phys_grad = None
    prev_dLdF_render = None  # render gradient on F (for covariance guidance)
    prev_dLdx_render = None  # render gradient on x (for position-injection ablation)

    # ── Covariance optimizer ─────────────────────────────────────────────
    cov_cfg = cfg.get('covariance_opt', {})
    cov_opt_enabled = bool(cov_cfg.get('enabled', False))
    cov_optimizer = None
    if cov_opt_enabled:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cov_optimizer = CovarianceOptimizer(N, sigma0, opacity, device, cov_cfg)
        cov_optimizer.init_from_positions(x0)
        if targets:
            cov_optimizer.set_targets(targets)
        print(f"[CovOpt] Enabled: {cov_cfg.get('cov_iters_per_ep', 5)} iters/ep, "
              f"lr={cov_cfg.get('cov_lr', 0.003)}, rotation={cov_cfg.get('cov_optimize_rotation', True)}")

    for ep_local in range(num_eps):
        ep = ep_local + episode_offset
        # LR decay
        t = ep / max(total_sched_eps - 1, 1)
        opt.initial_alpha = alpha_end + 0.5 * (alpha_start - alpha_end) * (1 + np.cos(np.pi * t))
        if ep % 10 == 0:
            print(f"  [LR] alpha={opt.initial_alpha:.6f}")

        # ── Chamfer-based surface mask update (union) ─────────────────
        chamfer_mask_threshold = float(surface_cfg.get('chamfer_mask_threshold', 0.0))
        if chamfer_mask_threshold > 0 and render_surface_mask is not None:
            x_current_mask = np.array(cg.get_point_cloud(0).get_positions(), dtype=np.float32)
            tgt_tree = cKDTree(tpos)
            cd_dist, _ = tgt_tree.query(x_current_mask, k=1)
            chamfer_new_mask = cd_dist < chamfer_mask_threshold
            render_surface_mask = render_surface_mask | chamfer_new_mask  # union: never shrink
            losses_mask = {
                'chamfer_mask_count': int(render_surface_mask.sum()),
                'chamfer_mask_frac': float(render_surface_mask.mean()),
                'chamfer_mask_new': int((chamfer_new_mask & ~render_surface_mask).sum()) if ep > 0 else 0,
            }

        capture_debug_viz = ep in debug_gradient_eps
        capture_images = png or capture_debug_viz
        render_penalty = None

        # Render-gradient injection from previous episode.
        # render_F_gain → inject dL/dF (F-space). render_x_gain → inject dL/dx (position-space).
        # Both can be 0 (physics-only), one nonzero (ablation), or both nonzero (combined).
        render_F_gain = float(cfg.get('render_F_gain', 0.0))
        render_x_gain = float(cfg.get('render_x_gain', 0.0))
        F_inject_active = (render_F_gain > 0 and prev_dLdF_render is not None)
        x_inject_active = (render_x_gain > 0 and prev_dLdx_render is not None)
        if F_inject_active or x_inject_active:
            dLdF_inj = (render_F_gain * prev_dLdF_render).astype(np.float32) \
                if F_inject_active else np.zeros((N, 3, 3), dtype=np.float32)
            dLdx_inj = (render_x_gain * prev_dLdx_render).astype(np.float32) \
                if x_inject_active else np.zeros((N, 3), dtype=np.float32)
            render_penalty = {'dLdF': dLdF_inj, 'dLdx': dLdx_inj}
            losses_F_inject = {
                'render_F_inject_applied': int(F_inject_active),
                'render_F_inject_norm': float(np.linalg.norm(dLdF_inj)),
                'render_x_inject_applied': int(x_inject_active),
                'render_x_inject_norm': float(np.linalg.norm(dLdx_inj)),
            }
        control_guidance_meta = {
            'control_guidance_applied': 0,
            'control_guidance_render_gain': 0.0,
            'control_guidance_physics_weight': 1.0,
            'control_guidance_dLdx_norm': 0.0,
            'control_guidance_dLdF_norm': 0.0,
            'control_guidance_dLdx_max': 0.0,
            'control_guidance_dLdF_max': 0.0,
            'control_guidance_active_count': 0,
            'control_guidance_active_frac': 0.0,
            'control_guidance_focus_count': 0,
            'control_guidance_focus_frac': 0.0,
            'control_guidance_partner_k': int(control_guidance_cfg.get('smooth_k', 64)),
            'control_guidance_diffusion_iters': int(control_guidance_cfg.get('diffusion_iters', 2)),
            'control_guidance_patch_balanced': 0,
            'control_guidance_patch_count': 0,
            'control_guidance_patch_scale_mean': 1.0,
            'control_guidance_patch_scale_min': 1.0,
            'control_guidance_patch_scale_max': 1.0,
            'control_guidance_patch_conflict_count': 0,
        }
        same_step_merge_meta = {
            'same_step_merge_applied': 0,
            'same_step_merge_support_count': 0,
            'same_step_merge_patch_count': 0,
            'same_step_merge_scale_mean': 1.0,
            'same_step_merge_scale_min': 1.0,
            'same_step_merge_scale_max': 1.0,
            'same_step_merge_conflict_count': 0,
        }
        try:
            cg.set_render_gain(1.0)
            cg.set_physics_weight(1.0)
        except Exception:
            pass

        if control_guidance_enabled and ep >= control_guidance_start_ep and prev_surface_obs_grad is not None:
            focus_mask = None
            if float(control_guidance_cfg.get('ear_focus_boost', 0.0)) > 0.0:
                focus_mask, focus_meta = _build_ear_focus_mask(
                    prev_surface_obs_x,
                    active_mask=render_surface_mask,
                    lateral_axis=int(control_guidance_cfg.get('ear_focus_lateral_axis', 0)),
                    height_axis=int(control_guidance_cfg.get('ear_focus_height_axis', 2)),
                    height_frac=float(control_guidance_cfg.get('ear_focus_height_frac', 0.58)),
                    side_frac=float(control_guidance_cfg.get('ear_focus_side_frac', 0.18)),
                    side_max_frac=float(control_guidance_cfg.get('ear_focus_side_max_frac', 0.50)),
                    center=control_guidance_cfg.get('ear_focus_center', None),
                )
                control_guidance_meta['control_guidance_focus_count'] = int(focus_meta['ear_focus_count'])
                control_guidance_meta['control_guidance_focus_frac'] = float(focus_meta['ear_focus_frac'])

            render_penalty, cg_meta = build_control_guidance_penalty(
                prev_surface_obs_x,
                prev_surface_obs_grad,
                control_guidance_cfg,
                focus_mask=focus_mask,
                core_mask=surface_mask,
                support_mask=render_surface_mask,
                surface_graph=None if surface_proxy is None else surface_proxy.get('graph', None),
                phys_grad_x=None if same_step_merge_enabled else prev_phys_grad,
            )
            control_guidance_meta.update(cg_meta)
            if render_penalty is not None:
                render_gain = float(control_guidance_cfg.get('render_gain', 1.0))
                render_gain_start = float(control_guidance_cfg.get('render_gain_start', render_gain))
                render_gain_warmup_eps = int(control_guidance_cfg.get('render_gain_warmup_eps', 0))
                if render_gain_warmup_eps > 0:
                    rel_ep = max(ep - control_guidance_start_ep, 0)
                    a = min(rel_ep / max(render_gain_warmup_eps, 1), 1.0)
                    render_gain = render_gain_start + a * (render_gain - render_gain_start)
                physics_weight = float(control_guidance_cfg.get('physics_weight', 1.0))
                control_guidance_meta['control_guidance_render_gain'] = render_gain
                control_guidance_meta['control_guidance_physics_weight'] = physics_weight
                try:
                    cg.set_render_gain(render_gain)
                    cg.set_physics_weight(physics_weight)
                except Exception:
                    pass

        # ── Stage A: Physics Rollout ──────────────────────────────────────
        losses, dFp, direction, cohesion, dLdx_norms, bce_list, alpha_list, diffused, phys_grad, x_episode, dLdF_render, dLdx_render = run_episode(
            ep, cg, opt, sigma0, opacity,
            renderers, campos_list, targets, rcfg, color,
            out, capture_images, Fp, pcfg, lcfg,
            render_penalty=render_penalty,
            target_depths=target_depths,
            view_weights=view_weights, particle_mask=render_surface_mask, view_labels=cam_labels,
            surface_proxy=surface_proxy,
        )
        losses.update(control_guidance_meta)
        if render_F_gain > 0 or render_x_gain > 0:
            losses.update(locals().get('losses_F_inject', {
                'render_F_inject_applied': 0, 'render_F_inject_norm': 0.0,
                'render_x_inject_applied': 0, 'render_x_inject_norm': 0.0,
            }))
        losses.update(surface_meta)
        if chamfer_mask_threshold > 0:
            losses.update(locals().get('losses_mask', {}))
        prev_surface_obs_x = None if x_episode is None else np.asarray(x_episode, dtype=np.float32).copy()
        merged_surface_obs_grad = None if diffused is None else np.asarray(diffused, dtype=np.float32).copy()
        if same_step_merge_enabled and merged_surface_obs_grad is not None and phys_grad is not None:
            merged_surface_obs_grad, same_step_merge_meta = premerge_surface_obs_gradient(
                merged_surface_obs_grad,
                phys_grad,
                control_guidance_cfg,
                support_mask=render_surface_mask,
                surface_graph=None if surface_proxy is None else surface_proxy.get('graph', None),
            )
        # EMA on render gradient to reduce oscillation
        ema_beta = float(control_guidance_cfg.get('ema_beta', 0.0))
        if ema_beta > 0 and prev_surface_obs_grad is not None and merged_surface_obs_grad is not None:
            prev_surface_obs_grad = ema_beta * prev_surface_obs_grad + (1.0 - ema_beta) * merged_surface_obs_grad
        else:
            prev_surface_obs_grad = merged_surface_obs_grad
        prev_phys_grad = None if phys_grad is None else np.asarray(phys_grad, dtype=np.float32).copy()
        prev_dLdF_render = dLdF_render  # store for next episode F-injection
        prev_dLdx_render = dLdx_render  # store for next episode position-injection
        losses.update(same_step_merge_meta)

        # Save dFc before promote
        dFc_raw = np.array(cg.get_point_cloud(0).get_dFc(), dtype=np.float32)
        dFc_norms_viz = np.linalg.norm(dFc_raw.reshape(N, -1), axis=1)

        # Promote
        cg.promote_last_as_initial(carry_grid=False)
        pc = cg.get_point_cloud(0)

        # Impulse (from render direction, if available)
        if eta_v > 0 or cohesion is not None:
            v = pc.get_velocities_view()
            imp = np.zeros((N, 3), dtype=np.float32)
            if direction is not None and eta_v > 0:
                imp -= (eta_v * direction).astype(np.float32)
            if cohesion is not None:
                imp += cohesion
            v[:] = np.clip(v + imp, -max_v, max_v)
            losses['impulse_mean'] = float(np.linalg.norm(imp, axis=1).mean())

        # ── Fp update ─────────────────────────────────────────────────────
        if (dFp is not None) and (not freeze_fp):
            Fp = np.matmul(np.eye(3)[None] + dFp, Fp).astype(np.float32)
            if damping > 0:
                Fp = ((1 - damping) * Fp + damping * np.eye(3)[None]).astype(np.float32)
            Fp = isochoric_project(Fp, max_aniso)
            losses['dFp_applied'] = 1
        else:
            losses['dFp_applied'] = 0
            if freeze_fp:
                losses['Fp_dev'] = 0.0
                losses['Fp_frozen'] = 1

        # ── Chamfer plasticity ────────────────────────────────────────────
        chamfer_eta = float(pcfg.get('chamfer_eta', 0.0))
        chamfer_start_ep = int(pcfg.get('chamfer_start_ep', 0))
        if chamfer_eta > 0 and not freeze_fp and ep >= chamfer_start_ep:
            x_current = np.array(pc.get_positions(), dtype=np.float32)
            dFp_chamfer, chamfer_meta = compute_chamfer_plasticity(
                x_current, tpos, Fp,
                eta=chamfer_eta,
                adaptive_eta_max=float(pcfg.get('adaptive_eta_max', 5.0)),
                smooth_k=int(pcfg.get('smooth_k', 64)),
                diffusion_iters=int(pcfg.get('diffusion_iters', 3)),
                clip_pct=float(pcfg.get('clip_pct', 95.0)),
                damping=0.0,
            )
            Fp = np.matmul(np.eye(3)[None] + dFp_chamfer, Fp).astype(np.float32)
            if damping > 0:
                Fp = ((1 - damping) * Fp + damping * np.eye(3)[None]).astype(np.float32)
            Fp = isochoric_project(Fp, max_aniso)
            losses.update(chamfer_meta)

        norms = np.linalg.norm((Fp - np.eye(3)[None]).reshape(N, -1), axis=1)
        losses['Fp_dev'] = float(norms.mean())
        if (ep + 1) % 10 == 0 or ep == num_eps - 1:
            np.save(out / f'Fp_ep{ep:03d}.npy', Fp)

        # ── Covariance optimization (position frozen) ─────────────────────
        if cov_opt_enabled and cov_optimizer is not None and renderers:
            x_current_cov = np.array(pc.get_positions(), dtype=np.float32)
            cov_meta = cov_optimizer.step(
                x_current_cov, renderers, targets, campos_list, color, rcfg,
                particle_mask=render_surface_mask,
            )
            losses.update(cov_meta)

        # ── Save ──────────────────────────────────────────────────────────
        if bce_list: losses['per_view_bce'] = bce_list
        history.append(losses)
        with open(out / 'losses.json', 'w') as f:
            json.dump(history, f, indent=2)

        # ── Checkpoint ────────────────────────────────────────────────────
        is_last_ep = (ep == episode_offset + num_eps - 1)
        if True:  # save every episode for post-processing flexibility
            ckpt_dir = out / 'checkpoints'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            x_ckpt = np.array(pc.get_positions(), dtype=np.float32)
            try:
                F_e_ckpt = pc.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy().astype(np.float32)
            except:
                F_e_ckpt = np.array(pc.get_def_grads(), dtype=np.float32)
            ckpt_data = dict(
                positions=x_ckpt,
                velocities=np.array(pc.get_velocities_view(), dtype=np.float32),
                dFc=np.array(pc.get_dFc(), dtype=np.float32),
                Fp=Fp.copy(),
                F_elastic=np.ascontiguousarray(F_e_ckpt),
            )
            if cov_opt_enabled and cov_optimizer is not None:
                cov_params = cov_optimizer.get_render_params()
                ckpt_data['cov_scale'] = cov_params['scale']
                ckpt_data['cov_opacity'] = cov_params['opacity']
                ckpt_data['cov_rotation'] = cov_params['rotation']
            np.savez_compressed(ckpt_dir / f'ckpt_ep{ep:03d}.npz', **ckpt_data)
            print(f"  [Checkpoint] ep{ep:03d}")

        # ── Viz ───────────────────────────────────────────────────────────
        if capture_images and ((png and (ep % 5 == 0 or ep == num_eps - 1)) or capture_debug_viz):
            x_viz = np.array(pc.get_positions(), dtype=np.float32)
            dLdx_dirs = None
            if dLdx_norms is not None and diffused is not None:
                d_n = np.linalg.norm(diffused, axis=1, keepdims=True)
                dLdx_dirs = np.zeros_like(diffused, dtype=np.float32)
                np.divide(diffused, d_n, out=dLdx_dirs, where=d_n > 1e-8)
            try:
                fp_n = np.linalg.norm((Fp - np.eye(3)[None]).reshape(N, -1), axis=1)
                phys_n = np.linalg.norm(phys_grad, axis=1) if phys_grad is not None else None
                create_episode_visualization(ep, x_viz, dLdx_norms, dFc_norms_viz, out,
                    cam_eye=cam_cfg.get('lookat', {}).get('eye'),
                    cam_target=cam_cfg.get('lookat', {}).get('target'),
                    cam_positions=cam_eyes,
                    dLdx_directions=dLdx_dirs,
                    Fp_norms=fp_n,
                    phys_norms=phys_n)
                if alpha_list:
                    tgt_nps = [t.cpu().numpy() if hasattr(t, 'cpu') else np.array(t) for t in targets]
                    create_per_view_visualization(ep, alpha_list, tgt_nps, bce_list, cam_labels, out)
                print(f"  [Viz] ep{ep:03d}")
            except Exception as e:
                print(f"  [Viz] {e}"); traceback.print_exc()

    np.save(out / 'Fp_final.npy', Fp)
    print(f"\nDone. {out}")


# ─── Batch / Main ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--png', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if 'experiments' in cfg:
        # copy already imported at top
        defaults = cfg.get('defaults', {})
        experiments = cfg['experiments']
        base_out = cfg.get('output_base', 'output')

        print(f"=== Batch mode: {len(experiments)} experiments ===\n")
        for i, exp in enumerate(experiments):
            ecfg = copy.deepcopy(defaults)
            ecfg['target_mesh_path'] = exp.get('target_mesh_path', defaults.get('target_mesh_path'))
            ecfg['input_mesh_path'] = defaults.get('input_mesh_path', 'assets/isosphere.obj')
            ecfg['output_dir'] = f"{base_out}/{exp['name']}"

            # Per-experiment overrides
            if 'num_cameras' in exp:
                ecfg.setdefault('multi_view', {})['num_cameras'] = exp['num_cameras']
            if 'num_animations' in exp:
                ecfg.setdefault('optimization', {})['num_animations'] = exp['num_animations']
            for section in [
                'plasticity', 'multi_view', 'optimization',
                'loss_weights', 'render', 'camera', 'simulation', 'upsample',
                'surface_aware', 'surface_proxy', 'control_guidance', 'debug', 'resume'
            ]:
                if section in exp:
                    base_section = ecfg.get(section, {})
                    if isinstance(base_section, dict) and isinstance(exp[section], dict):
                        ecfg[section] = _deep_update(copy.deepcopy(base_section), exp[section])
                    else:
                        ecfg[section] = copy.deepcopy(exp[section])

            # Flat top-level key overrides (render_F_gain, render_x_gain, etc.)
            for flat_key in ['render_F_gain', 'render_x_gain', 'covariance_opt']:
                if flat_key in exp:
                    ecfg[flat_key] = exp[flat_key]

            if args.skip_existing and (Path(ecfg['output_dir']) / 'losses.json').exists():
                print(f"[{i+1}/{len(experiments)}] SKIP {exp['name']}")
                continue

            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(experiments)}] {exp['name']}")
            print(f"  target: {ecfg.get('target_mesh_path')}")
            if 'multi_view' in ecfg:
                print(f"  cameras: {ecfg['multi_view'].get('preset', 'ring')} / {ecfg['multi_view'].get('num_cameras', 'n/a')}")
            print(f"{'='*60}\n")

            run_single(ecfg, png=args.png)

        print(f"\n=== All {len(experiments)} experiments complete ===")
    else:
        run_single(cfg, png=args.png)


if __name__ == '__main__':
    main()
