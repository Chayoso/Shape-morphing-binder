"""
Laplacian smoothing + Phong render for paper figures.
Smooths surface particle positions to reduce wrinkling artifacts.
"""
import argparse
import numpy as np
import torch
import sys
from pathlib import Path
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import setup_cameras, setup_renderer, compute_sigma0
from utils.io_utils import save_image_png
from utils.training_loop import _pca_normals
from renderer import compute_shading


def laplacian_smooth(x, iters=1, k=16, lamb=0.5):
    """Laplacian smoothing: x_i ← (1-λ)x_i + λ * mean(neighbors)"""
    tree = cKDTree(x)
    _, idx = tree.query(x, k=k + 1)
    idx = idx[:, 1:]  # exclude self

    x_smooth = x.copy()
    for _ in range(iters):
        neighbor_mean = x_smooth[idx].mean(axis=1)
        x_smooth = (1 - lamb) * x_smooth + lamb * neighbor_mean

    return x_smooth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', '-c', required=True)
    ap.add_argument('--iters', type=int, default=2)
    ap.add_argument('--lamb', type=float, default=0.3)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--sigma-scale', type=float, default=0.7)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    if 'defaults' in cfg:
        cfg = {**cfg['defaults'], **{k: v for k, v in cfg.items() if k not in ('defaults', 'experiments')}}

    rcfg = cfg.get('render', {})
    cam_cfg = cfg.get('camera', {})
    multi_cfg = cfg.get('multi_view', {})
    color = rcfg.get('particle_color', [0.27, 0.51, 0.71])

    ckpt = np.load(args.ckpt)
    x_orig = ckpt['positions']
    N = len(x_orig)
    print(f'Loaded: N={N}')

    # Laplacian smoothing
    x_smooth = laplacian_smooth(x_orig, iters=args.iters, k=args.k, lamb=args.lamb)
    displacement = np.linalg.norm(x_smooth - x_orig, axis=1)
    print(f'Smoothing: {args.iters} iters, lambda={args.lamb}, k={args.k}')
    print(f'  displacement: mean={displacement.mean():.5f}, max={displacement.max():.5f}')

    device = torch.device('cuda')
    sigma0 = compute_sigma0(x_smooth, args.sigma_scale)
    print(f'  sigma0={sigma0:.5f} (scale={args.sigma_scale})')

    x_t = torch.from_numpy(x_smooth).float().to(device)
    opacity = torch.full((N,), 0.95, device=device)
    cov = torch.zeros((N, 6), device=device)
    cov[:, 0] = sigma0 ** 2
    cov[:, 3] = sigma0 ** 2
    cov[:, 5] = sigma0 ** 2

    normals = _pca_normals(x_smooth)
    light_cfg = {'model': 'phong', 'type': 'directional', 'direction': [0.3, -0.5, 0.8],
                 'two_sided': True, 'ambient': 0.18, 'diffuse': 0.85, 'specular': 0.10, 'shininess': 32}

    all_cams, cam_eyes, cam_labels, _ = setup_cameras(cam_cfg, multi_cfg)

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent.parent / f'render_laplacian_i{args.iters}'
    out_dir.mkdir(parents=True, exist_ok=True)

    for v, (cam, eye, label) in enumerate(zip(all_cams, cam_eyes, cam_labels)):
        r, p = setup_renderer(cam, rcfg, training_mode=False)
        if not r:
            continue
        rgb = torch.from_numpy(compute_shading(
            x_smooth, normals, camera_pos=eye,
            light_cfg=light_cfg, albedo_color=color, model='phong'
        )).float().to(device)

        pred = r.render(x_t, cov, rgb=rgb, opacity=opacity,
                       prefer_cov_precomp=True, return_torch=False)
        if pred.get('image') is not None:
            save_image_png(out_dir / f'view{v:02d}_{label}_rgb.png', pred['image'])
        if pred.get('alpha') is not None:
            a = pred['alpha']
            if hasattr(a, 'numpy'):
                a = a.numpy()
            save_image_png(out_dir / f'view{v:02d}_{label}_alpha.png', a)

    print(f'Done → {out_dir}')


if __name__ == '__main__':
    main()
