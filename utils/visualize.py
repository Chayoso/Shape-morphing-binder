"""
Visualization per episode:
  Main: 3-panel — render gradient mag | gradient direction (RGB) | dFc magnitude
  Per-view: N cameras × (target / predicted / error)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


def particles_to_density(x, values=None, resolution=64, padding=2.0, sigma=1.5):
    mins = x.min(axis=0) - padding
    maxs = x.max(axis=0) + padding
    spacing = (maxs - mins).max() / resolution
    idx = np.clip(((x - mins) / spacing).astype(int), 0, resolution - 1)

    density = np.zeros((resolution,)*3, dtype=np.float32)
    vfield = np.zeros_like(density) if values is not None and values.ndim == 1 else None
    vfield_rgb = np.zeros((resolution,resolution,resolution,3), dtype=np.float32) if values is not None and values.ndim == 2 else None
    count = np.zeros_like(density)

    for i in range(len(x)):
        ix, iy, iz = idx[i]
        density[ix, iy, iz] += 1.0
        count[ix, iy, iz] += 1.0
        if vfield is not None:
            vfield[ix, iy, iz] += values[i]
        if vfield_rgb is not None:
            vfield_rgb[ix, iy, iz] += values[i]

    mask = count > 0
    if vfield is not None:
        vfield[mask] /= count[mask]
    if vfield_rgb is not None:
        vfield_rgb[mask] /= count[mask, None]

    density = gaussian_filter(density, sigma=sigma)
    if vfield is not None:
        vfield = gaussian_filter(vfield, sigma=sigma)
    if vfield_rgb is not None:
        for c in range(3):
            vfield_rgb[:,:,:,c] = gaussian_filter(vfield_rgb[:,:,:,c], sigma=sigma)

    return density, vfield if vfield is not None else vfield_rgb, mins, spacing


def extract_surface(x, resolution=64, sigma=1.5):
    density, _, origin, spacing = particles_to_density(x, resolution=resolution, sigma=sigma)
    level = density.max() * 0.02
    try:
        verts, faces, _, _ = marching_cubes(density, level=level)
    except:
        return None, None, origin, spacing
    return verts * spacing + origin, faces, origin, spacing


def interpolate_on_surface(verts, x, values, origin, spacing, resolution=64, sigma=1.5):
    """Interpolate per-particle values onto surface vertices."""
    _, vfield, _, _ = particles_to_density(x, values, resolution=resolution, sigma=sigma)
    if vfield is None:
        return None
    vi = np.clip(((verts - origin) / spacing).astype(int), 0, resolution - 1)
    if vfield.ndim == 3:
        return vfield[vi[:,0], vi[:,1], vi[:,2]]
    elif vfield.ndim == 4:
        return vfield[vi[:,0], vi[:,1], vi[:,2], :]
    return None


def render_surface_heatmap(verts, faces, values, ax, title='',
                           cmap='inferno', eye=None, target=None, vmin=None, vmax=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if verts is None:
        ax.text2D(0.5, 0.5, 'No surface', transform=ax.transAxes, ha='center')
        ax.set_title(title); return

    if vmin is None: vmin = values.min() if values is not None else 0
    if vmax is None: vmax = values.max() if values is not None else 1
    if vmax <= vmin: vmax = vmin + 1e-8

    cmap_fn = cm.get_cmap(cmap)
    fc = cmap_fn((values[faces].mean(axis=1) - vmin) / (vmax - vmin + 1e-8)) if values is not None else cmap_fn(np.ones(len(faces)) * 0.5)

    poly = Poly3DCollection(verts[faces], alpha=0.9)
    poly.set_facecolor(fc); poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    center = verts.mean(axis=0)
    hr = max(verts.max(axis=0) - verts.min(axis=0)) / 2 + 3
    for fn, c in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
        fn(center[c] - hr, center[c] + hr)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=-60)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_axis_off()


def render_surface_rgb(verts, faces, rgb_values, ax, title=''):
    """Render surface with per-vertex RGB colors (for gradient direction)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if verts is None:
        ax.text2D(0.5, 0.5, 'No surface', transform=ax.transAxes, ha='center')
        ax.set_title(title); return

    # Per-face color = mean of vertex RGB
    face_rgb = rgb_values[faces].mean(axis=1)  # (F, 3)
    face_rgb = np.clip(face_rgb, 0, 1)
    face_rgba = np.concatenate([face_rgb, np.ones((len(faces), 1))], axis=1)

    poly = Poly3DCollection(verts[faces], alpha=0.9)
    poly.set_facecolor(face_rgba); poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    center = verts.mean(axis=0)
    hr = max(verts.max(axis=0) - verts.min(axis=0)) / 2 + 3
    for fn, c in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
        fn(center[c] - hr, center[c] + hr)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=-60)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_axis_off()


def create_episode_visualization(
    ep, x, dLdx_norms, dFc_norms, out_dir,
    cam_eye=None, cam_target=None, cam_positions=None,
    dLdx_directions=None, **kwargs,
):
    """3-panel: render gradient mag | gradient direction RGB | dFc magnitude."""
    fig = plt.figure(figsize=(21, 6))

    verts, faces, origin, spacing = extract_surface(x, resolution=80, sigma=1.2)

    # Panel 1: Render gradient magnitude
    ax1 = fig.add_subplot(131, projection='3d')
    if dLdx_norms is not None and verts is not None:
        g_viz = np.log1p(dLdx_norms * 1000)
        vv = interpolate_on_surface(verts, x, g_viz, origin, spacing, 80, 1.2)
        vmax_g = np.percentile(vv[vv > 0], 95) if (vv is not None and (vv > 0).any()) else 1.0
        render_surface_heatmap(verts, faces, vv, ax1, title='Gradient Magnitude',
                               cmap='hot', vmin=0, vmax=vmax_g)
    else:
        ax1.set_title('Gradient Magnitude', fontsize=12, fontweight='bold')
    if cam_positions is not None and len(cam_positions) > 0:
        cps = np.array(cam_positions)
        center = x.mean(axis=0)
        dirs = cps - center
        sr = np.linalg.norm(x - center, axis=1).max()
        scaled = center + dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8) * (sr + 2)
        ax1.scatter(scaled[:, 0], scaled[:, 1], scaled[:, 2],
                    c='cyan', s=60, marker='^', zorder=10, edgecolors='black', linewidths=0.8)

    # Panel 2: Gradient direction (RGB: x=R, y=G, z=B)
    ax2 = fig.add_subplot(132, projection='3d')
    if dLdx_directions is not None and verts is not None:
        # Normalize direction to [0,1] for RGB: (d+1)/2
        d_rgb = (dLdx_directions + 1.0) / 2.0  # [-1,1] → [0,1]
        d_rgb = np.clip(d_rgb, 0, 1).astype(np.float32)
        vv_rgb = interpolate_on_surface(verts, x, d_rgb, origin, spacing, 80, 1.2)
        if vv_rgb is not None:
            render_surface_rgb(verts, faces, vv_rgb, ax2, title='Gradient Direction (RGB=XYZ)')
        else:
            ax2.text2D(0.5, 0.5, 'No data', transform=ax2.transAxes, ha='center')
            ax2.set_title('Gradient Direction', fontsize=12, fontweight='bold')
    else:
        ax2.text2D(0.5, 0.5, 'No direction', transform=ax2.transAxes, ha='center')
        ax2.set_title('Gradient Direction', fontsize=12, fontweight='bold')
    ax2.set_axis_off()

    # Panel 3: dFc magnitude
    ax3 = fig.add_subplot(133, projection='3d')
    if dFc_norms is not None and verts is not None and dFc_norms.max() > 1e-8:
        dfc_viz = np.log1p(dFc_norms * 1000)
        vv2 = interpolate_on_surface(verts, x, dfc_viz, origin, spacing, 80, 1.2)
        vmax_f = np.percentile(vv2[vv2 > 0], 95) if (vv2 is not None and (vv2 > 0).any()) else 1.0
        render_surface_heatmap(verts, faces, vv2, ax3, title='||dFc|| (log)',
                               cmap='plasma', vmin=0, vmax=vmax_f)
    else:
        ax3.text2D(0.5, 0.5, 'dFc = 0', transform=ax3.transAxes, ha='center', fontsize=13)
        ax3.set_title('||dFc||', fontsize=12, fontweight='bold'); ax3.set_axis_off()

    plt.suptitle(f'Episode {ep:03d}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    viz_dir = Path(out_dir) / 'viz'
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(viz_dir / f'ep{ep:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_per_view_visualization(ep, pred_alphas, target_alphas, per_view_bce, labels, out_dir):
    n = len(pred_alphas)
    if n == 0: return

    fig, axes = plt.subplots(3, n, figsize=(3*n, 9))
    if n == 1: axes = axes[:, None]

    for v in range(n):
        label = labels[v] if labels and v < len(labels) else f'V{v}'
        bce = per_view_bce[v] if per_view_bce and v < len(per_view_bce) else 0.0

        if v < len(target_alphas) and target_alphas[v] is not None:
            ta = target_alphas[v]
            if ta.ndim == 3: ta = ta[0] if ta.shape[0] == 1 else ta.mean(axis=2)
            axes[0, v].imshow(ta, cmap='gray', vmin=0, vmax=1)
        axes[0, v].set_title(f'{label}\nTarget', fontsize=9); axes[0, v].set_axis_off()

        if pred_alphas[v] is not None:
            pa = pred_alphas[v]
            if pa.ndim == 3: pa = pa[0] if pa.shape[0] == 1 else pa.mean(axis=2)
            axes[1, v].imshow(pa, cmap='gray', vmin=0, vmax=1)
        axes[1, v].set_title('Predicted', fontsize=9); axes[1, v].set_axis_off()

        if pred_alphas[v] is not None and v < len(target_alphas) and target_alphas[v] is not None:
            ta = target_alphas[v]
            pa = pred_alphas[v]
            if ta.ndim == 3: ta = ta[0] if ta.shape[0] == 1 else ta.mean(axis=2)
            if pa.ndim == 3: pa = pa[0] if pa.shape[0] == 1 else pa.mean(axis=2)
            if ta.shape != pa.shape:
                from PIL import Image
                ta = np.array(Image.fromarray((ta*255).astype(np.uint8)).resize(
                    (pa.shape[1], pa.shape[0]))).astype(np.float32) / 255
            axes[2, v].imshow(np.abs(pa - ta), cmap='hot', vmin=0, vmax=0.5)
            axes[2, v].set_title(f'Error (BCE={bce:.0f})', fontsize=9)
        else:
            axes[2, v].set_title('Error', fontsize=9)
        axes[2, v].set_axis_off()

    plt.suptitle(f'Per-View Alpha — Episode {ep:03d}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    viz_dir = Path(out_dir) / 'viz'
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(viz_dir / f'ep{ep:03d}_views.png', dpi=120, bbox_inches='tight')
    plt.close()
