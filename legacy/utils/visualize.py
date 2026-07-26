"""
Visualization per episode:
  Main: up to 6 panels — render grad | physics grad | cosine sim | grad dir | dFc | Fp
  Per-view: N cameras × (target / predicted / error)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.ndimage import gaussian_filter



def _heatmap(verts, faces, values, ax, title, cmap='hot', vmin=None, vmax=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if verts is None:
        ax.text2D(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
        ax.set_title(title, fontsize=11, fontweight='bold'); ax.set_axis_off(); return

    if vmin is None: vmin = values.min() if values is not None else 0
    if vmax is None: vmax = values.max() if values is not None else 1
    if vmax <= vmin: vmax = vmin + 1e-8

    cmap_obj = cm.get_cmap(cmap)
    normalized = (values[faces].mean(1) - vmin) / (vmax - vmin + 1e-8)
    fc = cmap_obj(normalized)
    poly = Poly3DCollection(verts[faces], alpha=0.9)
    poly.set_facecolor(fc); poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    center = verts.mean(0)
    hr = max(verts.max(0) - verts.min(0)) / 2 + 3
    for fn, c in [(ax.set_xlim,0),(ax.set_ylim,1),(ax.set_zlim,2)]:
        fn(center[c]-hr, center[c]+hr)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=30, azim=-60)
    ax.set_title(title, fontsize=11, fontweight='bold'); ax.set_axis_off()

    # Colorbar text
    ax.text2D(0.5, -0.02, f'[{vmin:.2f}, {vmax:.2f}]', transform=ax.transAxes,
              ha='center', fontsize=8, color='gray')


def _rgb_surface(verts, faces, rgb, ax, title):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if verts is None:
        ax.text2D(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
        ax.set_title(title, fontsize=11, fontweight='bold'); ax.set_axis_off(); return

    fc = np.clip(rgb[faces].mean(1), 0, 1)
    fc = np.concatenate([fc, np.ones((len(faces),1))], axis=1)
    poly = Poly3DCollection(verts[faces], alpha=0.9)
    poly.set_facecolor(fc); poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    center = verts.mean(0)
    hr = max(verts.max(0) - verts.min(0)) / 2 + 3
    for fn, c in [(ax.set_xlim,0),(ax.set_ylim,1),(ax.set_zlim,2)]:
        fn(center[c]-hr, center[c]+hr)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=30, azim=-60)
    ax.set_title(title, fontsize=11, fontweight='bold'); ax.set_axis_off()


def create_episode_visualization(
    ep, x, dLdx_norms, dFc_norms, out_dir,
    cam_eye=None, cam_target=None, cam_positions=None,
    dLdx_directions=None, Fp_norms=None, phys_norms=None, cos_sim=None,
    **kwargs,
):
    """Up to 6 panels: render grad | physics grad | cos_sim | grad dir | dFc | Fp."""
    res, sig = 80, 1.2
    verts, faces, origin, spacing = extract_surface(x, resolution=res, sigma=sig)

    # Determine panels
    panels = ['render_grad', 'phys_grad', 'cos_sim']
    panels.append('grad_dir')
    panels.append('dFc')
    if Fp_norms is not None and Fp_norms.max() > 1e-6:
        panels.append('Fp')
    n_panels = len(panels)

    fig = plt.figure(figsize=(5.5 * n_panels, 5.5))

    for idx, panel in enumerate(panels):
        ax = fig.add_subplot(1, n_panels, idx + 1, projection='3d')

        if panel == 'render_grad':
            if dLdx_norms is not None and verts is not None:
                vv = interpolate_on_surface(verts, x, np.log1p(dLdx_norms * 1000), origin, spacing, res, sig)
                vm = np.percentile(vv[vv>0], 95) if vv is not None and (vv>0).any() else 1.0
                _heatmap(verts, faces, vv, ax, 'Render Grad', 'hot', 0, vm)
            else:
                ax.set_title('Render Grad', fontsize=11, fontweight='bold')
            if cam_positions is not None and len(cam_positions) > 0:
                cps = np.array(cam_positions)
                center = x.mean(0)
                dirs = cps - center
                sr = np.linalg.norm(x - center, axis=1).max()
                scaled = center + dirs / (np.linalg.norm(dirs, axis=1, keepdims=True)+1e-8) * (sr+2)
                ax.scatter(scaled[:,0], scaled[:,1], scaled[:,2], c='cyan', s=50, marker='^', zorder=10, edgecolors='k', linewidths=0.5)

        elif panel == 'phys_grad':
            if phys_norms is not None and verts is not None:
                vv = interpolate_on_surface(verts, x, np.log1p(phys_norms * 1000), origin, spacing, res, sig)
                vm = np.percentile(vv[vv>0], 95) if vv is not None and (vv>0).any() else 1.0
                _heatmap(verts, faces, vv, ax, 'Physics Grad', 'cool', 0, vm)
            else:
                ax.text2D(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
                ax.set_title('Physics Grad', fontsize=11, fontweight='bold')
            ax.set_axis_off()

        elif panel == 'cos_sim':
            if cos_sim is not None and verts is not None:
                vv = interpolate_on_surface(verts, x, cos_sim, origin, spacing, res, sig)
                _heatmap(verts, faces, vv, ax, 'Cos(phys,render)', 'RdBu', -1, 1)
            else:
                ax.text2D(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
                ax.set_title('Cosine Sim', fontsize=11, fontweight='bold')
            ax.set_axis_off()

        elif panel == 'grad_dir':
            if dLdx_directions is not None and verts is not None:
                from matplotlib.colors import hsv_to_rgb
                hue = (np.arctan2(dLdx_directions[:,1], dLdx_directions[:,0]) / (2*np.pi) + 0.5) % 1.0
                sat = np.ones_like(hue)
                val = np.clip(np.abs(dLdx_directions[:,2]) + 0.5, 0, 1)
                hsv = np.stack([hue, sat, val], axis=1)
                d_rgb = np.array([hsv_to_rgb(h) for h in hsv]).astype(np.float32)
                vv = interpolate_on_surface(verts, x, d_rgb, origin, spacing, res, sig)
                if vv is not None:
                    _rgb_surface(verts, faces, vv, ax, 'Grad Dir (HSV)')
                else:
                    ax.text2D(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
                    ax.set_title('Grad Dir', fontsize=11, fontweight='bold')
            else:
                ax.text2D(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
                ax.set_title('Grad Dir', fontsize=11, fontweight='bold')
            ax.set_axis_off()

        elif panel == 'dFc':
            if dFc_norms is not None and verts is not None and dFc_norms.max() > 1e-8:
                vv = interpolate_on_surface(verts, x, np.log1p(dFc_norms*1000), origin, spacing, res, sig)
                vm = np.percentile(vv[vv>0], 95) if vv is not None and (vv>0).any() else 1.0
                _heatmap(verts, faces, vv, ax, '||dFc||', 'plasma', 0, vm)
            else:
                ax.text2D(0.5, 0.5, 'dFc=0', transform=ax.transAxes, ha='center')
                ax.set_title('||dFc||', fontsize=11, fontweight='bold')
            ax.set_axis_off()

        elif panel == 'Fp':
            vv = interpolate_on_surface(verts, x, np.log1p(Fp_norms*1000), origin, spacing, res, sig)
            vm = np.percentile(vv[vv>0], 95) if vv is not None and (vv>0).any() else 1.0
            _heatmap(verts, faces, vv, ax, '||Fp-I||', 'inferno', 0, vm)

    plt.suptitle(f'Episode {ep:03d}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    viz_dir = Path(out_dir) / 'viz'
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(viz_dir / f'ep{ep:03d}.png', dpi=120, bbox_inches='tight')
    plt.close()


def create_per_view_visualization(ep, pred_alphas, target_alphas, per_view_bce, labels, out_dir):
    n = len(pred_alphas)
    if n == 0: return

    fig, axes = plt.subplots(3, n, figsize=(3*n, 9))
    if n == 1: axes = axes[:, None]

    for v in range(n):
        label = labels[v] if labels and v < len(labels) else f'V{v}'
        bce = per_view_bce[v] if per_view_bce and v < len(per_view_bce) else 0

        if v < len(target_alphas) and target_alphas[v] is not None:
            ta = target_alphas[v]
            if ta.ndim == 3: ta = ta[0] if ta.shape[0]==1 else ta.mean(2)
            axes[0,v].imshow(ta, cmap='gray', vmin=0, vmax=1)
        axes[0,v].set_title(f'{label}\nTarget', fontsize=9); axes[0,v].set_axis_off()

        if pred_alphas[v] is not None:
            pa = pred_alphas[v]
            if pa.ndim == 3: pa = pa[0] if pa.shape[0]==1 else pa.mean(2)
            axes[1,v].imshow(pa, cmap='gray', vmin=0, vmax=1)
        axes[1,v].set_title('Predicted', fontsize=9); axes[1,v].set_axis_off()

        if pred_alphas[v] is not None and v < len(target_alphas) and target_alphas[v] is not None:
            ta = target_alphas[v]; pa = pred_alphas[v]
            if ta.ndim == 3: ta = ta[0] if ta.shape[0]==1 else ta.mean(2)
            if pa.ndim == 3: pa = pa[0] if pa.shape[0]==1 else pa.mean(2)
            if ta.shape != pa.shape:
                from PIL import Image
                ta = np.array(Image.fromarray((ta*255).astype(np.uint8)).resize((pa.shape[1],pa.shape[0]))).astype(np.float32)/255
            axes[2,v].imshow(np.abs(pa-ta), cmap='hot', vmin=0, vmax=0.5)
            axes[2,v].set_title(f'Error (BCE={bce:.0f})', fontsize=9)
        else:
            axes[2,v].set_title('Error', fontsize=9)
        axes[2,v].set_axis_off()

    plt.suptitle(f'Per-View Alpha — Episode {ep:03d}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    viz_dir = Path(out_dir) / 'viz'
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(viz_dir / f'ep{ep:03d}_views.png', dpi=100, bbox_inches='tight')
    plt.close()
