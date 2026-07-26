"""
Phase 3: Surface upsampling + render-guided covariance optimization.

Takes a physics checkpoint (coarse ~120K particles), extracts surface,
upsamples to dense surface Gaussians (~300K), then optimizes
scale/opacity/rotation via multi-view render loss.

Position is FIXED — only Gaussian appearance parameters are optimized.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import json
import sys
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import setup_cameras, setup_renderer, compute_sigma0
from utils.io_utils import save_image_png


def extract_surface_particles(x, k=16, surface_frac=0.35):
    """Extract surface particles using local density / PCA flatness."""
    tree = cKDTree(x)
    dd, idx = tree.query(x, k=k + 1)
    avg_dist = dd[:, 1:].mean(axis=1)

    # Surface = particles with larger avg neighbor distance (less dense = surface)
    threshold = np.percentile(avg_dist, (1.0 - surface_frac) * 100)
    surface_mask = avg_dist >= threshold
    return surface_mask


def upsample_surface(x_surface, target_count, k=8):
    """
    Upsample surface particles by inserting midpoints between neighbors.
    """
    N_surf = len(x_surface)
    if N_surf >= target_count:
        return x_surface.copy()

    tree = cKDTree(x_surface)
    _, idx = tree.query(x_surface, k=k + 1)
    idx = idx[:, 1:]  # exclude self

    # Generate midpoints
    new_points = [x_surface.copy()]
    needed = target_count - N_surf

    # Randomly sample particle-neighbor pairs for midpoints
    rng = np.random.default_rng(42)
    pair_i = rng.integers(0, N_surf, size=needed)
    pair_j_local = rng.integers(0, k, size=needed)
    pair_j = idx[pair_i, pair_j_local]

    # Midpoints with small jitter
    midpts = 0.5 * (x_surface[pair_i] + x_surface[pair_j])
    jitter = rng.normal(0, 0.001, size=midpts.shape).astype(np.float32)
    midpts += jitter

    new_points.append(midpts.astype(np.float32))
    result = np.concatenate(new_points, axis=0)
    print(f"  Upsampled: {N_surf} → {len(result)} surface Gaussians")
    return result


def compute_pca_normals(x, k=16):
    """PCA normals for surface orientation."""
    tree = cKDTree(x)
    _, idx = tree.query(x, k=k + 1)
    idx = idx[:, 1:]
    normals = np.zeros_like(x)
    for i in range(len(x)):
        neighbors = x[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / len(neighbors)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]
    return normals


def normals_to_quaternions(normals):
    """Convert normal vectors to quaternions aligning z-axis to normal."""
    N = len(normals)
    z = np.array([0, 0, 1], dtype=np.float32)
    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    for i in range(N):
        n = normals[i]
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n = n / nn
        dot = np.clip(np.dot(z, n), -1.0, 1.0)
        if dot > 0.9999:
            continue
        elif dot < -0.9999:
            quats[i] = [0, 1, 0, 0]
        else:
            axis = np.cross(z, n)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            angle = np.arccos(dot)
            s = np.sin(angle / 2)
            quats[i] = [np.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s]
    return quats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--ckpt', required=True, help='Checkpoint path')
    ap.add_argument('--target-count', type=int, default=300000)
    ap.add_argument('--surface-frac', type=float, default=0.35)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if 'defaults' in cfg:
        cfg = {**cfg['defaults'], **{k: v for k, v in cfg.items() if k not in ('defaults', 'experiments')}}

    rcfg = cfg.get('render', {})
    cam_cfg = cfg.get('camera', {})
    multi_cfg = cfg.get('multi_view', {})
    color = rcfg.get('particle_color', [0.27, 0.51, 0.71])
    sigma0_scale = float(rcfg.get('sigma0_scale', 0.7))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 1. Load checkpoint ===
    ckpt = np.load(args.ckpt)
    x_all = ckpt['positions']
    print(f"Loaded: {len(x_all)} particles")

    # === 2. Extract surface ===
    surface_mask = extract_surface_particles(x_all, surface_frac=args.surface_frac)
    x_surface = x_all[surface_mask]
    print(f"Surface: {len(x_surface)} particles ({surface_mask.sum()/len(x_all)*100:.1f}%)")

    # === 3. Upsample ===
    x_dense = upsample_surface(x_surface, args.target_count)
    N = len(x_dense)
    sigma0 = compute_sigma0(x_dense, sigma0_scale)
    print(f"sigma0={sigma0:.5f}")

    # === 3.5. Prune particles far from target ===
    target_mesh_path_prune = cfg.get('target_mesh_path')
    if target_mesh_path_prune is None:
        exps = yaml.safe_load(open(args.config)).get('experiments', [])
        if exps:
            target_mesh_path_prune = exps[0].get('target_mesh_path')
    if target_mesh_path_prune:
        import trimesh as _tm
        _tgt = _tm.load(target_mesh_path_prune, force='mesh')
        _tgt_pts = np.array(_tgt.vertices, dtype=np.float32)
        _tree = cKDTree(_tgt_pts)
        _dd, _ = _tree.query(x_dense, k=1)
        _thresh = np.percentile(_dd, 95)  # keep 95% closest
        _keep = _dd < _thresh
        x_dense = x_dense[_keep]
        N = len(x_dense)
        sigma0 = compute_sigma0(x_dense, sigma0_scale)
        print(f"Pruned: kept {N} particles (removed {(~_keep).sum()} far outliers)")

    # === 4. Initialize Gaussian params ===
    # PCA normals → rotation init
    print("Computing PCA normals...")
    normals = compute_pca_normals(x_dense, k=12)
    quats = normals_to_quaternions(normals)

    x_t = torch.from_numpy(x_dense).float().to(device)
    log_scale = torch.full((N, 3), np.log(sigma0), device=device, requires_grad=True)
    logit_opacity = torch.full((N,), 2.944, device=device, requires_grad=True)  # sigmoid → 0.95
    rotation = torch.from_numpy(quats).float().to(device)
    rotation.requires_grad = True

    log_scale_min = np.log(sigma0 * 0.3)
    log_scale_max = np.log(sigma0 * 1.5)

    optimizer = torch.optim.Adam([log_scale, logit_opacity, rotation], lr=args.lr)

    # === 5. Setup cameras & targets ===
    all_cams, cam_eyes, cam_labels, _ = setup_cameras(cam_cfg, multi_cfg)
    renderers, targets, campos_list = [], [], []

    # Render target from mesh
    target_mesh_path = cfg.get('target_mesh_path')
    if target_mesh_path is None:
        # Try experiments list
        exps = yaml.safe_load(open(args.config)).get('experiments', [])
        if exps:
            target_mesh_path = exps[0].get('target_mesh_path')
    import trimesh
    tgt_mesh = trimesh.load(target_mesh_path, force='mesh')
    tgt_verts = np.array(tgt_mesh.vertices, dtype=np.float32)
    tgt_sigma = compute_sigma0(tgt_verts, sigma0_scale)
    tgt_x = torch.from_numpy(tgt_verts).float().to(device)
    tgt_scale = torch.full((len(tgt_verts), 3), tgt_sigma, device=device)
    tgt_rot = torch.zeros((len(tgt_verts), 4), device=device)
    tgt_rot[:, 0] = 1.0
    tgt_opa = torch.full((len(tgt_verts),), 0.95, device=device)
    tgt_rgb = torch.tensor(color, device=device).float().unsqueeze(0).expand(len(tgt_verts), -1)
    tgt_norm = torch.zeros((len(tgt_verts), 3), device=device)
    tgt_norm[:, 2] = 1.0

    dt_maps = []
    for cam, eye, label in zip(all_cams, cam_eyes, cam_labels):
        r, p = setup_renderer(cam, rcfg, training_mode=True)
        if not r:
            continue
        with torch.no_grad():
            means2D = torch.zeros((len(tgt_verts), 3), device=device)
            tgt_out = r.rasterizer(
                means3D=tgt_x, means2D=means2D,
                opacities=tgt_opa.unsqueeze(-1), colors_precomp=tgt_rgb,
                scales=tgt_scale, rotations=tgt_rot, norm3Ds_precomp=tgt_norm,
            )
            tgt_alpha = tgt_out[3]
            if tgt_alpha.dim() == 3:
                tgt_alpha = tgt_alpha[0]

        renderers.append(r)
        campos_list.append(p['campos'])
        targets.append(tgt_alpha.detach())

        # Pre-compute DT map
        tgt_np = (tgt_alpha.detach().cpu().numpy() > 0.5).astype(np.float64)
        dt_out = distance_transform_edt(1.0 - tgt_np)
        dt_in = distance_transform_edt(tgt_np)
        dt_maps.append(torch.from_numpy((dt_out - dt_in).astype(np.float32)).to(device))

    print(f"Views: {len(renderers)}")

    # Free target mesh GPU memory
    del tgt_x, tgt_scale, tgt_rot, tgt_opa, tgt_rgb, tgt_norm
    torch.cuda.empty_cache()

    # === 6. Optimize ===
    color_t = torch.tensor(color, device=device).float().unsqueeze(0).expand(N, -1)
    norm3d = torch.zeros((N, 3), device=device)
    norm3d[:, 2] = 1.0

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent.parent / 'phase3_upsample'
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    view_indices = list(range(0, len(renderers), 2))  # every other view

    for it in range(args.iters):
        optimizer.zero_grad()

        for v in view_indices:
            rend = renderers[v]
            scale = torch.exp(log_scale)
            opa = torch.sigmoid(logit_opacity)
            rot = F.normalize(rotation, p=2, dim=1)
            means2D = torch.zeros((N, 3), device=device, requires_grad=True)

            color_out, depth_out, norm_out, alpha_raw, radii, extra = rend.rasterizer(
                means3D=x_t, means2D=means2D,
                opacities=opa.unsqueeze(-1), colors_precomp=color_t,
                scales=scale, rotations=rot, norm3Ds_precomp=norm3d,
            )
            alpha = alpha_raw[0] if alpha_raw.dim() == 3 else alpha_raw

            dt_map = dt_maps[v]
            if dt_map.shape != alpha.shape:
                dt_map = F.interpolate(dt_map[None, None], size=alpha.shape,
                                       mode='bilinear', align_corners=False)[0, 0]
            l_dt = (alpha * dt_map).mean()

            tgt = targets[v]
            if tgt.shape != alpha.shape:
                tgt = F.interpolate(tgt[None, None], size=alpha.shape,
                                    mode='bilinear', align_corners=False)[0, 0]
            intersection = (alpha * tgt).sum()
            union = alpha.sum() + tgt.sum() - intersection
            l_iou = 1.0 - intersection / (union + 1e-7)

            view_loss = (l_dt + 0.5 * l_iou) / len(view_indices)
            if v == view_indices[0]:
                s_reg = 0.001 * ((log_scale - np.log(sigma0)) ** 2).mean() / len(view_indices)
                view_loss = view_loss + s_reg

            view_loss.backward()

        # NaN guard
        valid = True
        for p in [log_scale, logit_opacity, rotation]:
            if p.grad is not None and torch.isnan(p.grad).any():
                valid = False
                p.grad.zero_()
        if valid:
            optimizer.step()

        with torch.no_grad():
            nan_s = torch.isnan(log_scale)
            if nan_s.any():
                log_scale[nan_s] = np.log(sigma0)
            log_scale.clamp_(log_scale_min, log_scale_max)

            nan_o = torch.isnan(logit_opacity)
            if nan_o.any():
                logit_opacity[nan_o] = 2.944
            # Opacity floor: clamp logit so sigmoid >= 0.85 (logit >= 1.735)
            logit_opacity.clamp_min_(1.735)

            nan_r = torch.isnan(rotation).any(dim=1)
            if nan_r.any():
                rotation.data[nan_r] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device)
            rotation.data = F.normalize(rotation.data, p=2, dim=1)

        if it % 20 == 0 or it == args.iters - 1:
            with torch.no_grad():
                sc = torch.exp(log_scale).cpu().numpy()
                op = torch.sigmoid(logit_opacity).cpu().numpy()
                # Quick alpha MSE on first view
                rend0 = renderers[0]
                s_ = torch.exp(log_scale)
                o_ = torch.sigmoid(logit_opacity)
                r_ = F.normalize(rotation, p=2, dim=1)
                m2d = torch.zeros((N, 3), device=device)
                c_o, d_o, n_o, a_o, _, _ = rend0.rasterizer(
                    means3D=x_t, means2D=m2d, opacities=o_.unsqueeze(-1),
                    colors_precomp=color_t, scales=s_, rotations=r_, norm3Ds_precomp=norm3d,
                )
                a_ = a_o[0] if a_o.dim() == 3 else a_o
                t_ = targets[0]
                if t_.shape != a_.shape:
                    t_ = F.interpolate(t_[None, None], size=a_.shape, mode='bilinear', align_corners=False)[0, 0]
                amse = float(((a_ - t_) ** 2).mean().item())

            print(f"  iter {it:4d}: alpha_mse={amse:.5f}  scale=[{sc.min():.4f},{sc.mean():.4f},{sc.max():.4f}]  opa=[{op.min():.3f},{op.mean():.3f}]")
            history.append({'iter': it, 'alpha_mse': amse, 'scale_mean': float(sc.mean()), 'opacity_mean': float(op.mean())})

    # === 7. Save & Render ===
    with open(out_dir / 'phase3_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    np.savez_compressed(out_dir / 'phase3_params.npz',
        positions=x_dense, log_scale=log_scale.detach().cpu().numpy(),
        logit_opacity=logit_opacity.detach().cpu().numpy(),
        rotation=rotation.detach().cpu().numpy())

    print("\nRendering final views...")
    with torch.no_grad():
        s_final = torch.exp(log_scale)
        o_final = torch.sigmoid(logit_opacity)
        r_final = F.normalize(rotation, p=2, dim=1)

        # Compute PCA normals for Phong shading
        try:
            from utils.training_loop import _pca_normals
            render_normals = _pca_normals(x_dense)
        except:
            render_normals = None

        for v, (rend, label, cam_pos) in enumerate(zip(renderers, cam_labels, campos_list)):
            r_eval, _ = setup_renderer(all_cams[v], rcfg, training_mode=False)
            if not r_eval:
                continue

            # Phong shading per view
            if render_normals is not None:
                from renderer import compute_shading
                rgb_shaded = torch.from_numpy(compute_shading(
                    x_dense, render_normals, camera_pos=cam_pos,
                    light_cfg=rcfg.get('lighting', {}), albedo_color=color, model='phong'
                )).float().to(device)
            else:
                rgb_shaded = color_t

            m2d = torch.zeros((N, 3), device=device)
            c_o, d_o, n_o, a_o, _, _ = r_eval.rasterizer(
                means3D=x_t, means2D=m2d, opacities=o_final.unsqueeze(-1),
                colors_precomp=rgb_shaded, scales=s_final, rotations=r_final, norm3Ds_precomp=norm3d,
            )
            img = c_o.permute(1, 2, 0).cpu().numpy()
            alpha = a_o[0].cpu().numpy() if a_o.dim() == 3 else a_o.cpu().numpy()
            save_image_png(out_dir / f'view{v:02d}_{label}_rgb.png', img)
            save_image_png(out_dir / f'view{v:02d}_{label}_alpha.png', alpha)

    print(f"\nDone. Saved to {out_dir}")


if __name__ == '__main__':
    main()
