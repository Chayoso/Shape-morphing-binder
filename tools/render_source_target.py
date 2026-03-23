"""
Render initial (sphere) and target (bunny) states, compute alpha/RGB loss.
Camera auto-framed so bunny fills ~80% of the image.

Usage:
    python tools/render_source_target.py -c configs/sphere_to_bunny.yaml
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import math
import numpy as np
import torch
import yaml
from pathlib import Path

from sampling import default_cfg, upsample
from sampling.utils.config_adapter import adapt_config
from utils.physics_utils import (
    build_opt_input, initialize_point_clouds, initialize_grids,
)
from utils.rendering_utils import prepare_rendering_inputs
from utils.io_utils import save_image_png
from renderer import GSRenderer3DGS, make_matrices_from_yaml, compute_shading
import diffmpm_bindings


def auto_frame_camera(positions, cam_cfg, render_cfg, res=256, fill_ratio=0.80):
    """
    Create a square camera (res x res) that frames `positions` tightly.
    fill_ratio=0.8 means the bounding sphere occupies 80% of the image.
    """
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    bsphere_r = float(radii.max())

    lookat = cam_cfg.get('lookat', {})
    eye = np.array(lookat.get('eye', [20, -25, 12.5]), dtype=np.float64)
    target = np.array(lookat.get('target', [0, 0, 0]), dtype=np.float64)
    up = np.array(lookat.get('up', [0, 0, 1]), dtype=np.float64)

    direction = eye - target
    direction = direction / np.linalg.norm(direction)

    # Square image: fx = fy = res (→ FOV ≈ 53°)
    W, H = res, res
    fx = fy = float(res)
    half_fov = math.atan(0.5)  # atan(W / 2*fx) = atan(0.5) ≈ 26.6°

    dist = bsphere_r / (fill_ratio * math.tan(half_fov))
    new_eye = center + direction * dist

    print(f"[AutoFrame] center={center}, bsphere_r={bsphere_r:.2f}")
    print(f"  res={W}x{H}, dist={dist:.2f}, fill={fill_ratio}")

    # Build camera config
    tanfovx = W / (2 * fx)
    tanfovy = H / (2 * fy)

    # View matrix (look-at)
    fwd = center - new_eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, fwd)

    # 4x4 view matrix (world→camera, OpenGL convention)
    R = np.stack([right, true_up, -fwd], axis=0)  # (3,3)
    t = -R @ new_eye
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = R
    view[:3, 3] = t

    # Use make_matrices_from_yaml for projection
    new_cam_cfg = dict(cam_cfg)
    new_cam_cfg['width'] = W
    new_cam_cfg['height'] = H
    new_cam_cfg['fx'] = fx
    new_cam_cfg['fy'] = fy
    new_cam_cfg['cx'] = W / 2.0
    new_cam_cfg['cy'] = H / 2.0
    new_cam_cfg['lookat'] = {
        'eye': new_eye.tolist(),
        'target': center.tolist(),
        'up': up.tolist(),
    }

    W_out, H_out, tanfovx, tanfovy, view_T, proj_T, campos = make_matrices_from_yaml(new_cam_cfg)

    bg = render_cfg.get("bg", [1.0, 1.0, 1.0])
    renderer = GSRenderer3DGS(
        W_out, H_out, tanfovx, tanfovy, view_T, proj_T, campos,
        bg=tuple(bg), sh_degree=0, scale_modifier=1.0,
        prefiltered=False, debug=False,
        antialiasing=render_cfg.get("antialiasing", False),
        device="cuda"
    )

    print(f"  Output: {W_out}x{H_out}")
    return renderer, campos, W_out, H_out


def render_particles(positions, rs_full, renderer, campos, render_cfg, particle_color, label=""):
    """Render particle positions with identity F → alpha + image."""
    N = positions.shape[0]
    x_t = torch.from_numpy(positions.astype(np.float32))
    F_id = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1)

    print(f"[{label}] Upsampling {N:,} particles...")
    with torch.no_grad():
        result = upsample(x_t, F_id, cfg=rs_full, seed=42, return_torch=True, current_episode=0)

    mu = result['points']
    cov = result['cov']
    print(f"  After upsample: {mu.shape[0]:,} Gaussians")

    rgb = prepare_rendering_inputs(mu, result, campos, render_cfg, particle_color)

    with torch.no_grad():
        pred = renderer.render(mu, cov, rgb=rgb, prefer_cov_precomp=True, return_torch=False)

    alpha = pred['alpha']
    image = pred.get('image')

    if not torch.is_tensor(alpha):
        alpha = torch.from_numpy(np.asarray(alpha, dtype=np.float32))
    if alpha.dim() == 3:
        alpha = alpha[0]

    print(f"  Alpha: shape={alpha.shape}, range=[{alpha.min():.4f}, {alpha.max():.4f}]")
    coverage = (alpha > 0.5).float().mean().item()
    print(f"  Coverage (alpha>0.5): {coverage:.1%}")
    return alpha.detach().cpu(), image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg.get('output_dir', 'output/run')) / 'diagnostic_framed'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Physics init ──
    opt = build_opt_input(cfg)
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)

    x_input = np.array(input_pc.get_positions(), dtype=np.float32)
    x_target = np.array(target_pc.get_positions(), dtype=np.float32)
    print(f"\nInput (sphere):  {x_input.shape[0]:,} particles")
    print(f"Target (bunny):  {x_target.shape[0]:,} particles")

    # ── Upsample config ──
    rs = default_cfg()
    rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))
    sim_cfg = cfg.get('simulation', {})
    rs['physics_grid'] = {
        'grid_min': sim_cfg.get('grid_min_point', [-16, -16, -16]),
        'grid_max': sim_cfg.get('grid_max_point', [16, 16, 16]),
        'grid_dx':  sim_cfg.get('grid_dx', 1.0),
    }
    # Override: larger Gaussians for solid rendering
    rs.setdefault('upsample', {}).setdefault('covariance', {})
    # Larger Gaussians: both sigma0 (F-based path) and sigma_isotropic (target path)
    rs['upsample']['covariance']['sigma0'] = 0.08
    rs['upsample']['covariance']['sigma_isotropic'] = 0.08
    print(f"[Config] sigma0={rs['upsample']['covariance']['sigma0']}, sigma_iso={rs['upsample']['covariance']['sigma_isotropic']}")

    # ── Auto-frame camera to bunny (two-pass) ──
    cam_cfg = cfg.get('camera', {})
    render_cfg = cfg.get('render', {})
    particle_color = render_cfg.get('particle_color', [0.27, 0.51, 0.71])

    # Pass 1: measure silhouette at low res
    renderer_tmp, campos_tmp, W_tmp, H_tmp = auto_frame_camera(
        x_target, cam_cfg, render_cfg, res=256, fill_ratio=0.80)
    alpha_tmp, _ = render_particles(
        x_target, rs, renderer_tmp, campos_tmp, render_cfg, particle_color, "Target/measure")

    vis = alpha_tmp > 0.01
    rows_v = vis.any(dim=1)
    cols_v = vis.any(dim=0)
    r0t, r1t = torch.where(rows_v)[0][[0, -1]].tolist()
    c0t, c1t = torch.where(cols_v)[0][[0, -1]].tolist()
    sil_h, sil_w = r1t - r0t, c1t - c0t
    current_fill = max(sil_h / H_tmp, sil_w / W_tmp)
    target_fill = 0.75
    adjusted = target_fill / current_fill * 0.80 if current_fill > 0 else 0.80
    print(f"[Measure] Silhouette {sil_w}x{sil_h}px in {W_tmp}x{H_tmp} = {current_fill:.1%} → adjusted fill={adjusted:.2f}")
    del renderer_tmp

    # Pass 2: final render at 1080x1080
    renderer, campos, W, H = auto_frame_camera(
        x_target, cam_cfg, render_cfg, res=1080, fill_ratio=adjusted)

    # ── Render both ──
    print("\n" + "=" * 60)
    alpha_input, color_input = render_particles(
        x_input, rs, renderer, campos, render_cfg, particle_color, "Input/Sphere"
    )
    print()
    alpha_target, color_target = render_particles(
        x_target, rs, renderer, campos, render_cfg, particle_color, "Target/Bunny"
    )

    # ── Alpha loss ──
    diff = (alpha_input - alpha_target).abs()
    alpha_mse = torch.nn.functional.mse_loss(alpha_input, alpha_target).item()
    alpha_l1 = torch.nn.functional.l1_loss(alpha_input, alpha_target).item()

    print(f"\n{'=' * 60}")
    print(f"Rendering Loss (bunny-framed, {W}x{H})")
    print(f"{'=' * 60}")
    print(f"Alpha MSE:   {alpha_mse:.6f}")
    print(f"Alpha MAE:   {alpha_l1:.6f}")
    print(f"Pixels |diff|>0.01: {(diff > 0.01).sum().item()} / {diff.numel()} ({100*(diff > 0.01).float().mean().item():.1f}%)")
    print(f"Max pixel diff:     {diff.max().item():.4f}")
    print(f"Input  coverage:    {(alpha_input > 0.5).float().mean().item():.1%}")
    print(f"Target coverage:    {(alpha_target > 0.5).float().mean().item():.1%}")

    # ── Per-region loss (red / blue / green) ──
    signed_diff = alpha_input - alpha_target  # positive = sphere-only (blue), negative = bunny-only (red)
    thresh = 0.01

    # Red region: bunny-only (target > input)
    red_mask = signed_diff < -thresh
    red_n = red_mask.sum().item()
    red_mse = (signed_diff[red_mask] ** 2).mean().item() if red_n > 0 else 0
    red_l1 = (-signed_diff[red_mask]).mean().item() if red_n > 0 else 0
    red_total = (signed_diff[red_mask] ** 2).sum().item() if red_n > 0 else 0

    # Blue region: sphere-only (input > target)
    blue_mask = signed_diff > thresh
    blue_n = blue_mask.sum().item()
    blue_mse = (signed_diff[blue_mask] ** 2).mean().item() if blue_n > 0 else 0
    blue_l1 = signed_diff[blue_mask].mean().item() if blue_n > 0 else 0
    blue_total = (signed_diff[blue_mask] ** 2).sum().item() if blue_n > 0 else 0

    # Green region: overlap
    green_mask = signed_diff.abs() <= thresh
    green_n = green_mask.sum().item()

    total_sq = (signed_diff ** 2).sum().item()

    print(f"\n--- Per-region breakdown ---")
    print(f"Red  (bunny-only):  {red_n:,} px ({100*red_n/diff.numel():.1f}%)  MSE={red_mse:.6f}  MAE={red_l1:.6f}  contribution={100*red_total/total_sq:.1f}%")
    print(f"Blue (sphere-only): {blue_n:,} px ({100*blue_n/diff.numel():.1f}%)  MSE={blue_mse:.6f}  MAE={blue_l1:.6f}  contribution={100*blue_total/total_sq:.1f}%")
    print(f"Green (overlap):    {green_n:,} px ({100*green_n/diff.numel():.1f}%)  loss≈0")

    # ── Object-only loss (exclude background) ──
    obj_mask = (alpha_input > 0.01) | (alpha_target > 0.01)  # union of both silhouettes
    obj_n = obj_mask.sum().item()
    obj_mse = ((alpha_input[obj_mask] - alpha_target[obj_mask]) ** 2).mean().item()
    obj_l1 = (alpha_input[obj_mask] - alpha_target[obj_mask]).abs().mean().item()

    bg_mask = ~obj_mask
    bg_n = bg_mask.sum().item()

    print(f"\n--- Object-only vs Full-image ---")
    print(f"Full image:   {diff.numel():,} px  Alpha MSE={alpha_mse:.6f}  MAE={alpha_l1:.6f}")
    print(f"Object only:  {obj_n:,} px ({100*obj_n/diff.numel():.1f}%)  Alpha MSE={obj_mse:.6f}  MAE={obj_l1:.6f}")
    print(f"Background:   {bg_n:,} px ({100*bg_n/diff.numel():.1f}%)  loss=0")

    # ── RGB loss ──
    if color_input is not None and color_target is not None:
        img_in = torch.from_numpy(np.asarray(color_input, dtype=np.float32))
        img_tgt = torch.from_numpy(np.asarray(color_target, dtype=np.float32))
        if img_in.max() > 1.5:
            img_in = img_in / 255.0
            img_tgt = img_tgt / 255.0

        rgb_mse = torch.nn.functional.mse_loss(img_in, img_tgt).item()
        rgb_l1 = torch.nn.functional.l1_loss(img_in, img_tgt).item()
        print(f"\nRGB MSE:     {rgb_mse:.6f}")
        print(f"RGB MAE:     {rgb_l1:.6f}")

        # Save RGB diff
        img_diff = np.abs(np.asarray(color_input, dtype=np.float32) - np.asarray(color_target, dtype=np.float32))
        if img_diff.max() > 1.5:
            img_diff = img_diff / 255.0
        save_image_png(out_dir / 'rgb_diff.png', img_diff)

    # ── Save ──
    save_image_png(out_dir / 'alpha_input.png', alpha_input.numpy())
    save_image_png(out_dir / 'alpha_target.png', alpha_target.numpy())
    save_image_png(out_dir / 'alpha_diff.png', diff.numpy())
    if color_input is not None:
        save_image_png(out_dir / 'color_input.png', color_input)
    if color_target is not None:
        save_image_png(out_dir / 'color_target.png', color_target)

    # ── Heatmaps ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Alpha diff heatmap (jet colormap)
    diff_np = diff.numpy()
    heatmap_alpha = cm.jet(diff_np / max(diff_np.max(), 1e-8))[:, :, :3]  # (H,W,3) float [0,1]
    # Overlay: blend heatmap with grayscale target silhouette for context
    tgt_gray = alpha_target.numpy()[:, :, None] * 0.3  # dim target as background
    overlay = np.where(diff_np[:, :, None] > 0.01, heatmap_alpha, tgt_gray * np.array([[[1, 1, 1]]]))
    save_image_png(out_dir / 'heatmap_alpha.png', (overlay * 255).astype(np.uint8))

    # Signed diff heatmap: blue = input > target, red = target > input
    signed_diff = (alpha_input - alpha_target).numpy()
    signed_hm = np.zeros((signed_diff.shape[0], signed_diff.shape[1], 3), dtype=np.float32)
    # Red channel: target has more (input missing → need to grow)
    signed_hm[:, :, 0] = np.clip(-signed_diff, 0, 1)
    # Blue channel: input has more (input excess → need to shrink)
    signed_hm[:, :, 2] = np.clip(signed_diff, 0, 1)
    # Green = overlap region (dim)
    overlap = np.minimum(alpha_input.numpy(), alpha_target.numpy())
    signed_hm[:, :, 1] = overlap * 0.2
    save_image_png(out_dir / 'heatmap_signed.png', (signed_hm * 255).astype(np.uint8))

    # RGB per-pixel L2 heatmap
    if color_input is not None and color_target is not None:
        rgb_diff_perpixel = np.sqrt(((img_in.numpy() - img_tgt.numpy()) ** 2).sum(axis=-1))  # (H,W)
        rgb_hm = cm.hot(rgb_diff_perpixel / max(rgb_diff_perpixel.max(), 1e-8))[:, :, :3]
        save_image_png(out_dir / 'heatmap_rgb.png', (rgb_hm * 255).astype(np.uint8))

    print(f"\nSaved to {out_dir}/")
    print("  heatmap_alpha.png  — alpha diff (jet colormap)")
    print("  heatmap_signed.png — red=target>input, blue=input>target")
    print("  heatmap_rgb.png    — RGB L2 per-pixel (hot colormap)")


if __name__ == '__main__':
    main()
