"""
run.py — Render-Penalized Physics + Isochoric Plasticity

L_physics + λ · diffused_render_gradient → dFc optimization
Fp for Gaussian shape (rendering only).
7 cameras: 3 low(20°) + 3 mid(50°) + 1 top(85°).

Usage: python run.py -c configs/experiment.yaml [--png]
"""

import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse, json, numpy as np, torch, yaml
from pathlib import Path
from scipy.spatial import cKDTree

from sampling import default_cfg
from sampling.utils.config_adapter import adapt_config
from utils.physics_utils import build_opt_input, initialize_point_clouds, initialize_grids, initialize_comp_graph
from utils.rendering_utils import setup_renderer, generate_target_render
from utils.training_loop import run_episode, isochoric_project
from utils.visualize import create_episode_visualization, create_per_view_visualization


def compute_sigma0(pos, scale=0.5):
    dd, _ = cKDTree(pos).query(pos, k=2)
    return float(dd[:,1].mean()) * scale


def setup_cameras(base_cam):
    """
    9 cameras: 8 ring (base elevation, 45° apart) + 1 ear-top.
    Ring uses the original camera's distance and elevation.
    """
    lookat = base_cam.get('lookat', {})
    eye = np.array(lookat.get('eye', [20, -25, 12.5]))
    target = np.array(lookat.get('target', [0, 0, 0]))
    offset = eye - target
    dist = float(np.linalg.norm(offset))
    base_elev = np.degrees(np.arctan2(offset[2], np.sqrt(offset[0]**2 + offset[1]**2)))
    base_azim = np.degrees(np.arctan2(offset[1], offset[0]))

    def cam(elev, azim, up_vec=[0, 0, 1]):
        e, a = np.radians(elev), np.radians(azim)
        pos = target + dist * np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
        c = base_cam.copy()
        c['lookat'] = {'eye': pos.tolist(), 'target': target.tolist(), 'up': up_vec}
        return c, pos

    cams, eyes, labels = [], [], []
    for i in range(8):
        c, e = cam(base_elev, base_azim + 45 * i)
        cams.append(c); eyes.append(e); labels.append(f'Ring-{i}')

    print(f"  Cameras: 8 ring (elev={base_elev:.0f}°, 45° apart)")
    return cams, np.array(eyes), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--png', action='store_true')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg.get('output_dir', 'output/run')); out.mkdir(parents=True, exist_ok=True)

    # ── Physics ───────────────────────────────────────────────────────────
    import diffmpm_bindings
    opt = build_opt_input(cfg)
    in_pc, tgt_pc = initialize_point_clouds(opt)
    in_grid, tgt_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(in_pc, in_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(tgt_pc, tgt_grid)
    cg = initialize_comp_graph(in_pc, in_grid, tgt_grid)

    x0 = np.array(in_pc.get_positions(), dtype=np.float32)
    N = x0.shape[0]

    # ── Render config ─────────────────────────────────────────────────────
    rcfg = cfg.get('render', {})
    sigma0 = compute_sigma0(x0, float(rcfg.get('sigma0_scale',0.5))) if rcfg.get('sigma0','auto')=='auto' else float(rcfg['sigma0'])
    opacity = float(rcfg.get('opacity', 0.95))
    color = rcfg.get('particle_color', [0.27, 0.51, 0.71])

    pcfg = cfg.get('plasticity', {})
    lcfg = cfg.get('loss_weights', {})
    cam_cfg = cfg.get('camera', {})

    eta_v = float(pcfg.get('eta_v', 0.5))
    max_v = float(pcfg.get('max_v', 5.0))
    max_aniso = float(pcfg.get('max_anisotropy', 1.5))
    damping = float(pcfg.get('damping', 0.0))
    lambda_render = float(pcfg.get('lambda_render', 0.0))

    print(f"[Init] N={N:,}, loss={cg.end_layer_mass_loss():.1f}")
    print(f"[Config] eta_v={eta_v}, lambda_render={lambda_render}")

    # ── Cameras + targets ─────────────────────────────────────────────────
    rs = default_cfg()
    rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))
    sim = cfg.get('simulation', {})
    rs['physics_grid'] = {
        'grid_min': sim.get('grid_min_point', [-16]*3),
        'grid_max': sim.get('grid_max_point', [16]*3),
        'grid_dx': sim.get('grid_dx', 1.0),
    }

    dist = float(np.linalg.norm(np.array(cam_cfg.get('lookat',{}).get('eye',[20,-25,12.5]))))
    multi = cfg.get('multi_view', {}).get('enabled', False)

    renderers, campos_list, targets, cam_eyes, cam_labels = [], [], [], np.array([]), []

    if multi:
        all_cams, cam_eyes, cam_labels = setup_cameras(cam_cfg)
        tpos = np.array(tgt_pc.get_positions(), dtype=np.float32)
        for cam in all_cams:
            r, p = setup_renderer(cam, rcfg, training_mode=True)
            if r:
                t = generate_target_render(tpos, rs, r, p['campos'], rcfg, color,
                                           target_mesh_path=cfg.get('target_mesh_path'), cam_cfg=cam)
                if t is not None:
                    renderers.append(r); campos_list.append(p['campos']); targets.append(t)
        print(f"[Cameras] {len(renderers)} views: {', '.join(cam_labels[:len(renderers)])}")
    else:
        r, p = setup_renderer(cam_cfg, rcfg, training_mode=True)
        if r:
            tpos = np.array(tgt_pc.get_positions(), dtype=np.float32)
            t = generate_target_render(tpos, rs, r, p['campos'], rcfg, color,
                                       target_mesh_path=cfg.get('target_mesh_path'), cam_cfg=cam_cfg)
            if t is not None:
                renderers.append(r); campos_list.append(p['campos']); targets.append(t)
                cam_eyes = np.array([p['campos']]); cam_labels = ['Primary']

    # Generate target RGB for each view (same Phong shader)
    target_rgbs = []
    w_rgb = float(lcfg.get('w_rgb', 0.0))
    if w_rgb > 0 and len(renderers) > 0:
        tpos = np.array(tgt_pc.get_positions(), dtype=np.float32)
        F_I_tgt = np.tile(np.eye(3, dtype=np.float32), (len(tpos), 1, 1))
        tgt_sigma0 = compute_sigma0(tpos, float(rcfg.get('sigma0_scale', 0.5)))
        from utils.training_loop import render as tl_render
        for r, cam in zip(renderers, campos_list):
            with torch.no_grad():
                _, pred_dict, _, _ = tl_render(tpos, F_I_tgt, tgt_sigma0, opacity, r, cam, rcfg, color, False)
            img = pred_dict.get('image')
            if img is not None:
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                target_rgbs.append(img)
            else:
                target_rgbs.append(None)
        print(f"[Target RGB] {len(target_rgbs)} views generated (w_rgb={w_rgb})")

    if targets:
        from utils.io_utils import save_image_png
        save_image_png(out / 'target_alpha.png', targets[0].numpy())

    # ── Train ─────────────────────────────────────────────────────────────
    Fp = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    num_eps = int(opt.num_animations)
    history = []
    stored_penalty = None

    print(f"\nTraining: {num_eps} episodes\n")

    for ep in range(num_eps):
        losses, dFp, direction, cohesion, dLdx_norms, bce_list, alpha_list, diffused = run_episode(
            ep, cg, opt, sigma0, opacity,
            renderers, campos_list, targets, rcfg, color,
            out, args.png, Fp, pcfg, lcfg,
            render_penalty=stored_penalty,
            target_rgbs=target_rgbs if len(target_rgbs) > 0 else None,
        )

        # ── Save dFc before promote (promote overwrites layer 0) ──────────
        dFc_raw = np.array(cg.get_point_cloud(0).get_dFc(), dtype=np.float32)
        dFc_norms_viz = np.linalg.norm(dFc_raw.reshape(N, -1), axis=1)

        # ── Promote + impulse ─────────────────────────────────────────────
        cg.promote_last_as_initial(carry_grid=False)
        pc = cg.get_point_cloud(0)

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
        if dFp is not None:
            Fp = np.matmul(np.eye(3)[None] + dFp, Fp).astype(np.float32)
            if damping > 0:
                Fp = ((1-damping)*Fp + damping*np.eye(3)[None]).astype(np.float32)
            Fp = isochoric_project(Fp, max_aniso)

            norms = np.linalg.norm((Fp - np.eye(3)[None]).reshape(N,-1), axis=1)
            sv = np.linalg.svd(Fp, compute_uv=False)
            losses['Fp_dev'] = float(norms.mean())
            losses['Fp_aniso'] = float((sv.max(1)/(sv.min(1)+1e-10)).mean())
            print(f"  [Fp] ||Fp-I||={norms.mean():.6f}")

            if (ep+1) % 10 == 0 or ep == num_eps-1:
                np.save(out / f'Fp_ep{ep:03d}.npy', Fp)

        # ── Store render penalty for next episode (gradient-normalized) ────
        if lambda_render > 0 and diffused is not None:
            render_norm = float(np.linalg.norm(diffused))

            # Get physics gradient norm for normalization
            try:
                phys_F_norm, phys_x_norm = cg.get_last_layer_phys_grad_norm()
                phys_norm = float(phys_x_norm)
            except:
                phys_norm = 0.0

            # Normalize: scale render gradient to match physics gradient magnitude
            if render_norm > 1e-8 and phys_norm > 1e-8:
                gain = lambda_render * phys_norm / render_norm
                pen_x = (gain * diffused).astype(np.float32)
            else:
                pen_x = (lambda_render * diffused).astype(np.float32)
                gain = lambda_render

            stored_penalty = {
                'dLdF': np.zeros((N, 3, 3), dtype=np.float32),
                'dLdx': np.ascontiguousarray(pen_x),
            }
            losses['penalty_norm'] = float(np.linalg.norm(pen_x))
            losses['phys_grad_norm'] = phys_norm
            losses['render_grad_norm'] = render_norm
            losses['render_gain'] = float(gain)
            print(f"  [Penalty] gain={gain:.2f} (phys={phys_norm:.1f}/render={render_norm:.1f}), ||pen||={losses['penalty_norm']:.2f}")

        # ── Save ──────────────────────────────────────────────────────────
        if bce_list: losses['per_view_bce'] = bce_list
        history.append(losses)
        with open(out / 'losses.json', 'w') as f:
            json.dump(history, f, indent=2)

        # ── Viz ───────────────────────────────────────────────────────────
        if args.png and ep % 5 == 0:
            x_viz = np.array(pc.get_positions(), dtype=np.float32)
            dFc_n = dFc_norms_viz  # saved before promote
            # Gradient direction (normalized dLdx per particle)
            dLdx_dirs = None
            if dLdx_norms is not None and diffused is not None:
                d_n = np.linalg.norm(diffused, axis=1, keepdims=True)
                dLdx_dirs = np.where(d_n > 1e-8, diffused / d_n, 0).astype(np.float32)

            try:
                create_episode_visualization(ep, x_viz, dLdx_norms, dFc_n, out,
                    cam_eye=cam_cfg.get('lookat',{}).get('eye'),
                    cam_target=cam_cfg.get('lookat',{}).get('target'),
                    cam_positions=cam_eyes,
                    dLdx_directions=dLdx_dirs)
                if alpha_list:
                    tgt_nps = [t.cpu().numpy() if hasattr(t,'cpu') else np.array(t) for t in targets]
                    create_per_view_visualization(ep, alpha_list, tgt_nps, bce_list, cam_labels, out)
                print(f"  [Viz] ep{ep:03d}")
            except Exception as e:
                import traceback; print(f"  [Viz] {e}"); traceback.print_exc()

    np.save(out / 'Fp_final.npy', Fp)
    print(f"\nDone. {out}")


if __name__ == '__main__':
    main()
