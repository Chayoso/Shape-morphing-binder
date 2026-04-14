"""
Target-adaptive post-processing densification.

Given a physics checkpoint and a target mesh (e.g., bob.obj), identifies
gap/tearing regions where current particles poorly cover the target surface,
then inserts new Gaussians concentrated in those regions.

Positions of NEW particles come from the target surface (gap-filling only).
Original particle positions are UNCHANGED.
Only Gaussian appearance (scale, opacity, rotation) is optimized via multi-view
render loss — consistent with the paper's "positions governed by physics" claim.

Usage:
    python tools/target_adaptive_densify.py \
        -c configs/morph_from_isosphere.yaml \
        --ckpt output/morph_from_isosphere/isosphere_to_bob/checkpoints/ckpt_ep059.npz \
        --target assets/bob.obj \
        --gap-threshold 0.3 \
        --max-new 50000 \
        --iters 300
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as TF
import yaml
import json
import sys
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import setup_cameras, setup_renderer, compute_sigma0
from utils.io_utils import save_image_png


# ---------------------------------------------------------------------------
# 1. Gap detection: find target surface regions with poor particle coverage
# ---------------------------------------------------------------------------

def detect_gaps(x_current, target_verts, target_faces,
                gap_threshold, n_surface_samples=500_000):
    """
    Densely sample the target surface, then find regions where the nearest
    current particle is farther than gap_threshold.

    Returns:
        gap_points  (M, 3)  — target surface points in gap regions
        gap_dists   (M,)    — distance to nearest current particle
    """
    import trimesh
    mesh = trimesh.Trimesh(vertices=target_verts, faces=target_faces,
                           process=False)
    surface_pts, _ = trimesh.sample.sample_surface(mesh, n_surface_samples)
    surface_pts = np.asarray(surface_pts, dtype=np.float32)

    tree = cKDTree(x_current)
    dists, _ = tree.query(surface_pts, k=1)

    gap_mask = dists > gap_threshold
    print(f"  Target surface samples: {len(surface_pts)}")
    print(f"  Gap points (dist > {gap_threshold}): {gap_mask.sum()} "
          f"({gap_mask.sum()/len(surface_pts)*100:.1f}%)")
    return surface_pts[gap_mask], dists[gap_mask]


# ---------------------------------------------------------------------------
# 2. Adaptive densification: place more particles where gaps are larger
# ---------------------------------------------------------------------------

def adaptive_densify(gap_points, gap_dists, max_new, min_spacing):
    """
    Sub-sample gap points with density proportional to gap distance
    (bigger gap → more particles), respecting max_new budget and min_spacing.
    """
    if len(gap_points) == 0:
        print("  No gap points — nothing to densify.")
        return np.zeros((0, 3), dtype=np.float32)

    # Probability proportional to gap distance (larger gap = higher priority)
    weights = gap_dists / gap_dists.sum()

    # Over-sample then prune by min_spacing
    n_candidates = min(max_new * 3, len(gap_points))
    rng = np.random.default_rng(42)
    chosen_idx = rng.choice(len(gap_points), size=n_candidates,
                            replace=(n_candidates > len(gap_points)),
                            p=weights)
    candidates = gap_points[chosen_idx]

    # Farthest-point-style pruning to enforce min_spacing
    selected = [candidates[0]]
    sel_tree = None
    for i in range(1, len(candidates)):
        if len(selected) >= max_new:
            break
        pt = candidates[i]
        if sel_tree is None or len(selected) % 500 == 0:
            sel_tree = cKDTree(np.array(selected))
        d, _ = sel_tree.query(pt, k=1)
        if d >= min_spacing:
            selected.append(pt)

    new_pts = np.array(selected, dtype=np.float32)
    print(f"  New particles placed: {len(new_pts)} (budget: {max_new})")
    return new_pts


# ---------------------------------------------------------------------------
# 3. Merge original surface + new particles, then optimize Gaussians
# ---------------------------------------------------------------------------

def extract_surface(x, k=16, surface_frac=0.35):
    """Same as surface_upsample_render.py — density-based surface extraction."""
    tree = cKDTree(x)
    dd, _ = tree.query(x, k=k + 1)
    avg_dist = dd[:, 1:].mean(axis=1)
    threshold = np.percentile(avg_dist, (1.0 - surface_frac) * 100)
    return avg_dist >= threshold


def compute_pca_normals(x, k=16):
    """PCA normals for Gaussian rotation initialization."""
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
            quats[i] = [np.cos(angle / 2), axis[0]*s, axis[1]*s, axis[2]*s]
    return quats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Target-adaptive post-processing densification")
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--ckpt', required=True, help='Physics checkpoint (.npz)')
    ap.add_argument('--target', required=True, help='Target mesh (.obj)')
    ap.add_argument('--gap-threshold', type=float, default=None,
                    help='Distance threshold for gap detection '
                         '(default: auto from particle spacing)')
    ap.add_argument('--max-new', type=int, default=50000,
                    help='Max new particles to add')
    ap.add_argument('--surface-frac', type=float, default=0.35)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    # --- Load config ---
    cfg = yaml.safe_load(open(args.config))
    if 'defaults' in cfg:
        cfg = {**cfg['defaults'],
               **{k: v for k, v in cfg.items()
                  if k not in ('defaults', 'experiments')}}
    rcfg = cfg.get('render', {})
    cam_cfg = cfg.get('camera', {})
    multi_cfg = cfg.get('multi_view', {})
    color = rcfg.get('particle_color', [0.27, 0.51, 0.71])
    sigma0_scale = float(rcfg.get('sigma0_scale', 0.7))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load checkpoint ---
    ckpt = np.load(args.ckpt)
    x_all = ckpt['positions']  # (534658, 3)
    print(f"Loaded checkpoint: {len(x_all)} particles")

    # --- Load target mesh ---
    import trimesh
    tgt_mesh = trimesh.load(args.target, force='mesh')
    tgt_verts = np.array(tgt_mesh.vertices, dtype=np.float32)
    tgt_faces = np.array(tgt_mesh.faces, dtype=np.int64)
    print(f"Target mesh: {len(tgt_verts)} verts, {len(tgt_faces)} faces")

    # --- Auto gap threshold from particle spacing ---
    if args.gap_threshold is None:
        tree_tmp = cKDTree(x_all[::10])  # subsample for speed
        dd_tmp, _ = tree_tmp.query(x_all[::10], k=2)
        mean_spacing = float(dd_tmp[:, 1].mean())
        gap_threshold = mean_spacing * 2.5
        print(f"Auto gap threshold: {gap_threshold:.4f} "
              f"(2.5 × mean_spacing={mean_spacing:.4f})")
    else:
        gap_threshold = args.gap_threshold
        mean_spacing = gap_threshold / 2.5  # approximate

    # --- Step 1: Detect gaps ---
    print("\n=== Step 1: Gap detection ===")
    gap_pts, gap_dists = detect_gaps(
        x_all, tgt_verts, tgt_faces, gap_threshold)

    # --- Step 2: Adaptive densification ---
    print("\n=== Step 2: Adaptive densification ===")
    min_spacing = mean_spacing * 0.8
    new_pts = adaptive_densify(gap_pts, gap_dists, args.max_new, min_spacing)

    if len(new_pts) == 0:
        print("No gaps found — try lowering --gap-threshold")
        return

    # --- Step 3: Build combined particle set ---
    print("\n=== Step 3: Build rendering set ===")
    print(f"Original particles (ALL): {len(x_all)}")

    x_combined = np.concatenate([x_all, new_pts], axis=0)
    n_orig = len(x_all)
    n_new = len(new_pts)
    N = len(x_combined)
    print(f"Combined: {N} ({n_orig} original + {n_new} new)")

    sigma0 = compute_sigma0(x_combined, sigma0_scale)
    print(f"sigma0 = {sigma0:.5f}")

    # --- Step 4: Initialize Gaussian params from F_elastic ---
    print("\n=== Step 4: Initialize Gaussians ===")
    F_elastic = ckpt['F_elastic']  # (n_orig, 3, 3)
    print(f"Using F_elastic from checkpoint for {n_orig} original particles")

    # Polar decomposition: F = R * S, then cov = S * Sigma0 * S^T
    # Eigendecompose cov → scale, rotation
    F_t = torch.from_numpy(F_elastic).float().to(device)
    U, S_vals, Vh = torch.linalg.svd(F_t)
    R_mat = U @ Vh  # rotation
    # Stretch eigenvalues = singular values of F
    orig_scales = S_vals * sigma0  # (n_orig, 3) — scale = sv * sigma0

    # Convert R to quaternions
    from renderer.utils.covariance import _rotation_matrix_to_quaternion_torch
    orig_quats = _rotation_matrix_to_quaternion_torch(R_mat)  # (n_orig, 4)

    # New particles: isotropic, PCA normals for rotation
    print(f"Computing PCA normals for {n_new} new particles...")
    new_normals = compute_pca_normals(new_pts, k=min(12, n_new - 1))
    new_quats = normals_to_quaternions(new_normals)
    new_scales = np.full((n_new, 3), sigma0 * 1.3, dtype=np.float32)

    # Combine
    x_t = torch.from_numpy(x_combined).float().to(device)

    all_scales = torch.cat([
        orig_scales,
        torch.from_numpy(new_scales).float().to(device)
    ], dim=0)  # (N, 3)
    log_scale = torch.log(all_scales.clamp(min=1e-7)).requires_grad_(True)

    all_quats = torch.cat([
        orig_quats,
        torch.from_numpy(new_quats).float().to(device)
    ], dim=0)  # (N, 4)
    rotation = all_quats.clone().requires_grad_(True)

    logit_opacity = torch.full((N,), 2.944, device=device, requires_grad=True)

    log_scale_min = np.log(sigma0 * 0.3)
    log_scale_max = np.log(sigma0 * 3.0)  # wider range to preserve F-derived scales

    del F_t, U, S_vals, Vh, R_mat, orig_scales, orig_quats
    torch.cuda.empty_cache()

    optimizer = torch.optim.Adam([log_scale, logit_opacity, rotation], lr=args.lr)

    # --- Step 5: Setup cameras & targets ---
    all_cams, cam_eyes, cam_labels, _ = setup_cameras(cam_cfg, multi_cfg)
    renderers, targets, campos_list, dt_maps = [], [], [], []

    # Render target silhouettes from mesh
    tgt_x = torch.from_numpy(tgt_verts).float().to(device)
    tgt_sigma = compute_sigma0(tgt_verts, sigma0_scale)
    tgt_scale = torch.full((len(tgt_verts), 3), tgt_sigma, device=device)
    tgt_rot = torch.zeros((len(tgt_verts), 4), device=device)
    tgt_rot[:, 0] = 1.0
    tgt_opa = torch.full((len(tgt_verts),), 0.95, device=device)
    tgt_rgb = torch.tensor(color, device=device).float().unsqueeze(0).expand(
        len(tgt_verts), -1)
    tgt_norm = torch.zeros((len(tgt_verts), 3), device=device)
    tgt_norm[:, 2] = 1.0

    for cam, eye, label in zip(all_cams, cam_eyes, cam_labels):
        r, p = setup_renderer(cam, rcfg, training_mode=True)
        if not r:
            continue
        with torch.no_grad():
            m2d = torch.zeros((len(tgt_verts), 3), device=device)
            tgt_out = r.rasterizer(
                means3D=tgt_x, means2D=m2d,
                opacities=tgt_opa.unsqueeze(-1), colors_precomp=tgt_rgb,
                scales=tgt_scale, rotations=tgt_rot,
                norm3Ds_precomp=tgt_norm,
            )
            tgt_alpha = tgt_out[3]
            if tgt_alpha.dim() == 3:
                tgt_alpha = tgt_alpha[0]

        renderers.append(r)
        campos_list.append(p['campos'])
        targets.append(tgt_alpha.detach())

        tgt_np = (tgt_alpha.detach().cpu().numpy() > 0.5).astype(np.float64)
        dt_out = distance_transform_edt(1.0 - tgt_np)
        dt_in = distance_transform_edt(tgt_np)
        dt_maps.append(torch.from_numpy(
            (dt_out - dt_in).astype(np.float32)).to(device))

    print(f"Views: {len(renderers)}")

    del tgt_x, tgt_scale, tgt_rot, tgt_opa, tgt_rgb, tgt_norm
    torch.cuda.empty_cache()

    # --- Step 6: Optimize ---
    color_t = torch.tensor(color, device=device).float().unsqueeze(0).expand(N, -1)
    norm3d = torch.zeros((N, 3), device=device)
    norm3d[:, 2] = 1.0

    out_dir = Path(args.out) if args.out else \
        Path(args.ckpt).parent.parent / 'phase3_adaptive'
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    view_indices = list(range(0, len(renderers), 2))  # every other view

    print(f"\n=== Step 6: Optimizing ({args.iters} iters) ===")
    for it in range(args.iters):
        optimizer.zero_grad()

        for v in view_indices:
            rend = renderers[v]
            scale = torch.exp(log_scale)
            opa = torch.sigmoid(logit_opacity)
            rot = TF.normalize(rotation, p=2, dim=1)
            m2d = torch.zeros((N, 3), device=device, requires_grad=True)

            color_out, depth_out, norm_out, alpha_raw, radii, extra = \
                rend.rasterizer(
                    means3D=x_t, means2D=m2d,
                    opacities=opa.unsqueeze(-1), colors_precomp=color_t,
                    scales=scale, rotations=rot, norm3Ds_precomp=norm3d,
                )
            alpha = alpha_raw[0] if alpha_raw.dim() == 3 else alpha_raw

            dt_map = dt_maps[v]
            if dt_map.shape != alpha.shape:
                dt_map = TF.interpolate(
                    dt_map[None, None], size=alpha.shape,
                    mode='bilinear', align_corners=False)[0, 0]
            l_dt = (alpha * dt_map).mean()

            tgt = targets[v]
            if tgt.shape != alpha.shape:
                tgt = TF.interpolate(
                    tgt[None, None], size=alpha.shape,
                    mode='bilinear', align_corners=False)[0, 0]
            intersection = (alpha * tgt).sum()
            union = alpha.sum() + tgt.sum() - intersection
            l_iou = 1.0 - intersection / (union + 1e-7)

            view_loss = (l_dt + 0.5 * l_iou) / len(view_indices)
            if v == view_indices[0]:
                s_reg = 0.001 * ((log_scale - np.log(sigma0)) ** 2).mean() \
                    / len(view_indices)
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
            logit_opacity.clamp_min_(1.735)

            nan_r = torch.isnan(rotation).any(dim=1)
            if nan_r.any():
                rotation.data[nan_r] = torch.tensor(
                    [1, 0, 0, 0], dtype=torch.float32, device=device)
            rotation.data = TF.normalize(rotation.data, p=2, dim=1)

        if it % 20 == 0 or it == args.iters - 1:
            with torch.no_grad():
                sc = torch.exp(log_scale).cpu().numpy()
                op = torch.sigmoid(logit_opacity).cpu().numpy()
                rend0 = renderers[0]
                s_ = torch.exp(log_scale)
                o_ = torch.sigmoid(logit_opacity)
                r_ = TF.normalize(rotation, p=2, dim=1)
                m2d = torch.zeros((N, 3), device=device)
                _, _, _, a_o, _, _ = rend0.rasterizer(
                    means3D=x_t, means2D=m2d, opacities=o_.unsqueeze(-1),
                    colors_precomp=color_t, scales=s_, rotations=r_,
                    norm3Ds_precomp=norm3d,
                )
                a_ = a_o[0] if a_o.dim() == 3 else a_o
                t_ = targets[0]
                if t_.shape != a_.shape:
                    t_ = TF.interpolate(
                        t_[None, None], size=a_.shape,
                        mode='bilinear', align_corners=False)[0, 0]
                amse = float(((a_ - t_) ** 2).mean().item())

            print(f"  iter {it:4d}: alpha_mse={amse:.5f}  "
                  f"scale=[{sc.min():.4f},{sc.mean():.4f},{sc.max():.4f}]  "
                  f"opa=[{op.min():.3f},{op.mean():.3f}]")
            history.append({
                'iter': it, 'alpha_mse': amse,
                'scale_mean': float(sc.mean()),
                'opacity_mean': float(op.mean()),
                'n_original': n_orig, 'n_new': n_new,
            })

    # --- Step 7: Save & Render ---
    with open(out_dir / 'densify_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    np.savez_compressed(
        out_dir / 'densified_params.npz',
        positions=x_combined,
        log_scale=log_scale.detach().cpu().numpy(),
        logit_opacity=logit_opacity.detach().cpu().numpy(),
        rotation=rotation.detach().cpu().numpy(),
        n_original=np.array(n_orig),
        n_new=np.array(n_new),
    )

    print("\nRendering final views...")
    with torch.no_grad():
        s_final = torch.exp(log_scale)
        o_final = torch.sigmoid(logit_opacity)
        r_final = TF.normalize(rotation, p=2, dim=1)

        try:
            from utils.training_loop import _pca_normals
            render_normals = _pca_normals(x_combined)
        except Exception:
            render_normals = None

        for v, (rend, label, cam_pos) in enumerate(
                zip(renderers, cam_labels, campos_list)):
            r_eval, _ = setup_renderer(all_cams[v], rcfg, training_mode=False)
            if not r_eval:
                continue

            if render_normals is not None:
                from renderer import compute_shading
                rgb_shaded = torch.from_numpy(compute_shading(
                    x_combined, render_normals, camera_pos=cam_pos,
                    light_cfg=rcfg.get('lighting', {}),
                    albedo_color=color, model='phong'
                )).float().to(device)
            else:
                rgb_shaded = color_t

            m2d = torch.zeros((N, 3), device=device)
            c_o, d_o, n_o, a_o, _, _ = r_eval.rasterizer(
                means3D=x_t, means2D=m2d, opacities=o_final.unsqueeze(-1),
                colors_precomp=rgb_shaded, scales=s_final,
                rotations=r_final, norm3Ds_precomp=norm3d,
            )
            img = c_o.permute(1, 2, 0).cpu().numpy()
            alpha = a_o[0].cpu().numpy() if a_o.dim() == 3 else a_o.cpu().numpy()
            save_image_png(out_dir / f'view{v:02d}_{label}_rgb.png', img)
            save_image_png(out_dir / f'view{v:02d}_{label}_alpha.png', alpha)

    # Also save a gap visualization
    print("Saving gap visualization...")
    gap_viz = np.zeros((len(x_combined), 3), dtype=np.float32)
    gap_viz[:n_orig] = np.array(color)  # original = blue
    gap_viz[n_orig:] = np.array([0.9, 0.2, 0.2])  # new = red
    np.save(out_dir / 'particle_colors.npy', gap_viz)

    print(f"\nDone. Output: {out_dir}")
    print(f"  Original surface particles: {n_orig}")
    print(f"  New gap-filling particles:  {n_new}")
    print(f"  Total:                      {N}")


if __name__ == '__main__':
    main()
