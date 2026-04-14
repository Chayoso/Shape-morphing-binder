"""Compare isotropic (F=I) vs anisotropic (F=F_elastic) rendering of bob ep059."""
import numpy as np
import torch
import sys
from pathlib import Path
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import setup_renderer, compute_sigma0
from utils.io_utils import save_image_png
from renderer.shading.compute import compute_shading
from renderer.utils.covariance import _rotation_matrix_to_quaternion_torch
import torch.nn.functional as TF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

device = torch.device('cuda')

ckpt = np.load('output/morph_from_isosphere/isosphere_to_bob/checkpoints/ckpt_ep059.npz')
x = ckpt['positions']
F_el = ckpt['F_elastic']
N = len(x)
color = [0.27, 0.51, 0.71]
sigma0 = compute_sigma0(x, 0.7)

# F -> anisotropic scale, rotation
F_t = torch.from_numpy(F_el).float().to(device)
U, Sv, Vh = torch.linalg.svd(F_t)
scales_aniso = (Sv * sigma0).cpu().numpy()
quats_aniso = _rotation_matrix_to_quaternion_torch(U @ Vh).cpu().numpy()
del F_t, U, Sv, Vh
torch.cuda.empty_cache()

# Isotropic: same positions, F=I
scales_iso = np.full((N, 3), sigma0, dtype=np.float32)
quats_iso = np.zeros((N, 4), dtype=np.float32)
quats_iso[:, 0] = 1.0

# PCA normals
tree = cKDTree(x)
_, idx = tree.query(x, k=17)
idx = idx[:, 1:]
normals = np.zeros_like(x)
for i in range(N):
    nb = x[idx[i]]
    c = nb - nb.mean(0)
    _, ev = np.linalg.eigh(c.T @ c / len(nb))
    normals[i] = ev[:, 0]

cam_eye = np.array([34.0, 0.0, 0.0])
light_cfg = {
    'type': 'directional', 'direction': [0.5, -0.3, 0.8],
    'ambient': 0.3, 'diffuse': 0.6, 'specular': 0.3, 'shininess': 32,
}
rgb = compute_shading(x, normals, cam_eye, light_cfg, color, 'phong')

cam = {
    'width': 960, 'height': 540,
    'fx': 356.25, 'fy': 356.25, 'cx': 480.0, 'cy': 270.0,
    'znear': 0.01, 'zfar': 100.0,
    'lookat': {'eye': cam_eye.tolist(), 'target': [0, 0, 0], 'up': [0, 0, 1]},
}

out = Path('output/morph_from_isosphere/isosphere_to_bob')

for name, scales, quats in [('isotropic', scales_iso, quats_iso),
                              ('anisotropic', scales_aniso, quats_aniso)]:
    r, _ = setup_renderer(cam, {'bg_color': [1, 1, 1], 'opacity': 0.95}, training_mode=False)
    with torch.no_grad():
        co, _, _, _, _, _ = r.rasterizer(
            means3D=torch.from_numpy(x).float().to(device),
            means2D=torch.zeros((N, 3), device=device),
            opacities=torch.full((N, 1), 0.95, device=device),
            colors_precomp=torch.from_numpy(rgb).float().to(device),
            scales=torch.from_numpy(scales).float().to(device),
            rotations=TF.normalize(torch.from_numpy(quats).float().to(device), p=2, dim=1),
            norm3Ds_precomp=torch.zeros((N, 3), device=device).scatter_(
                1, torch.full((N, 1), 2, dtype=torch.long, device=device), 1.0),
        )
    save_image_png(out / f'fig_cov_{name}.png', co.permute(1, 2, 0).cpu().numpy())
    print(f'{name} done')

# Compose side-by-side figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 8,
})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, bottom=0.02, top=0.85)

for ax, name, title in [
    (axes[0], 'isotropic', r'(a) $\mathbf{F} = \mathbf{I}$ (isotropic)'),
    (axes[1], 'anisotropic', r'(b) $\boldsymbol{\Sigma} = \mathbf{S}\,\boldsymbol{\Sigma}_0\,\mathbf{S}^\top$ (ours)'),
]:
    img = mpimg.imread(str(out / f'fig_cov_{name}.png'))
    # Crop whitespace
    gray = img.mean(axis=2)
    rows = np.any(gray < 0.98, axis=1)
    cols = np.any(gray < 0.98, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 15
        rmin = max(0, rmin - pad)
        rmax = min(img.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(img.shape[1], cmax + pad)
        img = img[rmin:rmax, cmin:cmax]
    ax.imshow(img)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.axis('off')

fig.suptitle('Effect of $\\mathbf{F}$-derived covariance on rendering (Bob, frame 59)',
             fontsize=9, y=0.95)

plt.savefig(str(out / 'fig_iso_vs_aniso.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(str(out / 'fig_iso_vs_aniso.pdf'), bbox_inches='tight', facecolor='white')
print('Done: comparison figure')
