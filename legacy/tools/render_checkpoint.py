"""Render a saved checkpoint with optional render-time covariance smoothing."""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import _deep_update, setup_cameras, setup_renderer, compute_sigma0
from utils.io_utils import save_image_png
from utils.training_loop import _pca_normals
from utils.surface_utils import build_fixed_surface_mask
from renderer import compute_shading


def _resolve_config(cfg: dict, experiment_name: str | None) -> dict:
    if 'experiments' not in cfg:
        return cfg

    experiments = cfg.get('experiments', [])
    if not experiments:
        raise ValueError("Batch config has no experiments")

    if experiment_name is None:
        raise ValueError("Batch config requires --experiment")

    exp = next((e for e in experiments if e.get('name') == experiment_name), None)
    if exp is None:
        names = ', '.join(e.get('name', '<unnamed>') for e in experiments)
        raise ValueError(f"Experiment '{experiment_name}' not found. Available: {names}")

    defaults = copy.deepcopy(cfg.get('defaults', {}))
    out_base = cfg.get('output_base', 'output')
    ecfg = defaults
    ecfg['target_mesh_path'] = exp.get('target_mesh_path', defaults.get('target_mesh_path'))
    ecfg['input_mesh_path'] = defaults.get('input_mesh_path', exp.get('input_mesh_path'))
    ecfg['output_dir'] = f"{out_base}/{exp['name']}"

    for section in [
        'plasticity', 'multi_view', 'optimization', 'loss_weights', 'render', 'camera',
        'simulation', 'upsample', 'surface_aware', 'surface_proxy', 'control_guidance',
        'debug', 'resume', 'covariance_opt',
    ]:
        if section in exp:
            base_section = ecfg.get(section, {})
            if isinstance(base_section, dict) and isinstance(exp[section], dict):
                ecfg[section] = _deep_update(copy.deepcopy(base_section), exp[section])
            else:
                ecfg[section] = copy.deepcopy(exp[section])
    return ecfg


def _compute_covariances(F_elastic: np.ndarray, Fp: np.ndarray | None, sigma0: float) -> np.ndarray:
    F_e = np.asarray(F_elastic, dtype=np.float32)
    if Fp is not None:
        F_r = np.matmul(F_e, np.asarray(Fp, dtype=np.float32)).astype(np.float32)
    else:
        F_r = F_e
    s0 = (float(sigma0) ** 2) * np.eye(3, dtype=np.float32)[None]
    cov = np.matmul(np.matmul(F_r, s0), np.transpose(F_r, (0, 2, 1))).astype(np.float32)
    cov += 1e-6 * np.eye(3, dtype=np.float32)[None]
    return cov


def _build_render_mask(x: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    surface_cfg = cfg.get('surface_aware', {}) or {}
    if not bool(surface_cfg.get('enabled', False)):
        mask = np.ones((x.shape[0],), dtype=bool)
        return mask, {
            'render_mask_applied': 0,
            'render_mask_count': int(mask.sum()),
            'render_mask_frac': 1.0,
        }

    recon = build_fixed_surface_mask(
        x,
        resolution=int(surface_cfg.get('recon_resolution', 64)),
        padding=float(surface_cfg.get('recon_padding', 2.0)),
        sigma=float(surface_cfg.get('recon_sigma', 1.5)),
        level_ratio=float(surface_cfg.get('recon_level_ratio', 0.02)),
        threshold_mult=float(surface_cfg.get('render_surface_threshold_mult', surface_cfg.get('surface_threshold_mult', 1.75))),
        min_surface_frac=float(surface_cfg.get('render_min_surface_frac', surface_cfg.get('min_surface_frac', 0.08))),
        max_surface_frac=float(surface_cfg.get('render_max_surface_frac', surface_cfg.get('max_surface_frac', 0.40))),
    )
    mask = np.asarray(recon['surface_mask'], dtype=bool).reshape(-1)
    return mask, {
        'render_mask_applied': 1,
        'render_mask_count': int(mask.sum()),
        'render_mask_frac': float(mask.mean()),
        'render_mask_threshold': float(recon.get('surface_distance_threshold', 0.0)),
    }


def _smooth_covariances(
    x: np.ndarray,
    cov: np.ndarray,
    normals: np.ndarray | None,
    k: int,
    iters: int,
    blend: float,
    sigma_scale: float,
    normal_sigma: float,
) -> tuple[np.ndarray, dict]:
    n = int(x.shape[0])
    zero = {
        'cov_smoothing_applied': 0,
        'cov_smoothing_k': int(k),
        'cov_smoothing_iters': int(iters),
        'cov_smoothing_blend': float(blend),
        'cov_smoothing_sigma': 0.0,
        'cov_smoothing_normal_sigma': float(normal_sigma),
    }
    if n < 4 or k <= 0 or iters <= 0 or blend <= 0:
        return cov, zero

    k_eff = min(max(int(k) + 1, 2), n)
    dd, idx = cKDTree(x).query(x, k=k_eff)
    if idx.ndim != 2 or idx.shape[1] <= 1:
        return cov, zero

    nbr = np.asarray(idx[:, 1:], dtype=np.int32)
    dist = np.asarray(dd[:, 1:], dtype=np.float32)
    nn = dist[:, 0]
    sigma = float(np.median(nn[nn > 1e-8])) if np.any(nn > 1e-8) else 0.0
    sigma *= float(sigma_scale)
    if sigma <= 1e-8:
        return cov, zero

    w = np.exp(-(dist ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    if normals is not None and normals.shape == x.shape:
        cos_sim = np.clip((normals[:, None, :] * normals[nbr]).sum(axis=2), -1.0, 1.0)
        sigma_n = max(float(normal_sigma), 1e-3)
        w *= np.exp(-((1.0 - cos_sim) ** 2) / (2.0 * sigma_n * sigma_n)).astype(np.float32)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-8)

    cur = np.asarray(cov, dtype=np.float32).copy()
    eye = np.eye(3, dtype=np.float32)[None]
    b = float(np.clip(blend, 0.0, 1.0))
    for _ in range(int(iters)):
        nbr_cov = cur[nbr]
        smooth = np.einsum('nk,nkij->nij', w, nbr_cov).astype(np.float32)
        cur = ((1.0 - b) * cur + b * smooth).astype(np.float32)
        cur = 0.5 * (cur + np.transpose(cur, (0, 2, 1)))
        cur += 1e-6 * eye

    meta = {
        'cov_smoothing_applied': 1,
        'cov_smoothing_k': int(k_eff - 1),
        'cov_smoothing_iters': int(iters),
        'cov_smoothing_blend': float(b),
        'cov_smoothing_sigma': float(sigma),
        'cov_smoothing_normal_sigma': float(normal_sigma),
    }
    return cur, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--experiment', type=str, default=None, help='Experiment name for batch configs')
    ap.add_argument('--ep', type=int, default=-1, help='Episode to render (-1 = last)')
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--primary-only', action='store_true')
    ap.add_argument('--cov-smooth-blend', type=float, default=0.0)
    ap.add_argument('--cov-smooth-iters', type=int, default=1)
    ap.add_argument('--cov-smooth-k', type=int, default=12)
    ap.add_argument('--cov-smooth-sigma-scale', type=float, default=1.0)
    ap.add_argument('--cov-smooth-normal-sigma', type=float, default=0.35)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg = _resolve_config(cfg, args.experiment)

    rcfg = cfg.get('render', {})
    cam_cfg = cfg.get('camera', {})
    multi_cfg = cfg.get('multi_view', {})
    color = rcfg.get('particle_color', [0.27, 0.51, 0.71])
    sigma0_scale = float(rcfg.get('sigma0_scale', 0.7))
    opacity_val = float(rcfg.get('opacity', 0.95))

    out_dir = Path(cfg.get('output_dir', 'output'))
    ckpt_dir = out_dir / 'checkpoints'
    if args.ep >= 0:
        ckpt_path = ckpt_dir / f'ckpt_ep{args.ep:03d}.npz'
    else:
        ckpts = sorted(ckpt_dir.glob('ckpt_ep*.npz'))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        ckpt_path = ckpts[-1]

    ep_num = int(ckpt_path.stem.split('ep')[1])
    print(f"Rendering checkpoint: {ckpt_path}")

    ckpt = np.load(ckpt_path)
    x_full = np.asarray(ckpt['positions'], dtype=np.float32)
    F_elastic_full = np.asarray(ckpt['F_elastic'], dtype=np.float32) if 'F_elastic' in ckpt.files else None
    Fp_full = np.asarray(ckpt['Fp'], dtype=np.float32) if 'Fp' in ckpt.files else None
    render_mask, render_meta = _build_render_mask(x_full, cfg)
    x = np.ascontiguousarray(x_full[render_mask])
    F_elastic = np.ascontiguousarray(F_elastic_full[render_mask]) if F_elastic_full is not None else None
    Fp = np.ascontiguousarray(Fp_full[render_mask]) if Fp_full is not None else None
    if F_elastic is None:
        raise ValueError("Checkpoint does not contain F_elastic; cannot render covariance faithfully")

    sigma0 = compute_sigma0(x, sigma0_scale)
    cov = _compute_covariances(F_elastic, Fp, sigma0)
    normals = _pca_normals(x, k=32)
    cov, cov_meta = _smooth_covariances(
        x, cov,
        normals=normals,
        k=args.cov_smooth_k,
        iters=args.cov_smooth_iters,
        blend=args.cov_smooth_blend,
        sigma_scale=args.cov_smooth_sigma_scale,
        normal_sigma=args.cov_smooth_normal_sigma,
    )

    print(
        f"  N={len(x):,}/{len(x_full):,}, sigma0={sigma0:.6f}, "
        f"render_mask={render_meta['render_mask_frac']:.3f}"
    )
    if cov_meta['cov_smoothing_applied']:
        print(
            f"  covariance smoothing: blend={cov_meta['cov_smoothing_blend']:.3f}, "
            f"iters={cov_meta['cov_smoothing_iters']}, k={cov_meta['cov_smoothing_k']}, "
            f"sigma={cov_meta['cov_smoothing_sigma']:.6f}"
        )
    else:
        print("  covariance smoothing: off")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_t = torch.from_numpy(x).float().to(device)
    cov_t = torch.from_numpy(cov).float().to(device)
    opacity = torch.full((len(x), 1), opacity_val, device=device)

    all_cams, cam_eyes, cam_labels, _ = setup_cameras(cam_cfg, multi_cfg)
    if args.primary_only:
        all_cams = all_cams[:1]
        cam_eyes = cam_eyes[:1]
        cam_labels = cam_labels[:1]

    suffix = (
        f"_covsmooth_b{args.cov_smooth_blend:.2f}_k{args.cov_smooth_k}_i{args.cov_smooth_iters}"
        if args.cov_smooth_blend > 0 and args.cov_smooth_iters > 0 and args.cov_smooth_k > 0
        else "_rawcov"
    )
    render_out = Path(args.out) if args.out else out_dir / f'renders_ep{ep_num:03d}{suffix}'
    render_out.mkdir(parents=True, exist_ok=True)

    for v, (cam, eye, label) in enumerate(zip(all_cams, cam_eyes, cam_labels)):
        r, _ = setup_renderer(cam, rcfg, training_mode=True)
        if not r:
            continue

        # Orient normals toward camera for correct shading
        view_dir = eye - x
        flip = (normals * view_dir).sum(axis=1) < 0
        normals_view = normals.copy()
        normals_view[flip] *= -1

        rgb = torch.from_numpy(compute_shading(
            x, normals_view, camera_pos=eye,
            light_cfg=rcfg.get('lighting', {}), albedo_color=color, model='phong'
        )).float().to(device)

        pred = r.render(
            x_t, cov_t, rgb=rgb, opacity=opacity,
            prefer_cov_precomp=True, return_torch=False
        )

        if pred.get('image') is not None:
            save_image_png(render_out / f'view{v:02d}_{label}_rgb.png', pred['image'])
        if pred.get('alpha') is not None:
            alpha = pred['alpha']
            if hasattr(alpha, 'numpy'):
                alpha = alpha.numpy()
            save_image_png(render_out / f'view{v:02d}_{label}_alpha.png', alpha)
        if v == 0 and pred.get('image') is not None:
            save_image_png(render_out / 'render.png', pred['image'])
        print(f"  View {v} ({label}) done")

    print(f"\nSaved to {render_out}")


if __name__ == '__main__':
    main()
