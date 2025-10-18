"""File export utilities (PNG, PLY, NPZ)."""

import numpy as np
from pathlib import Path
from ..utils.utils import as_numpy


def setup_matplotlib():
    """Setup matplotlib for non-interactive mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def compute_plot_bounds(points_list):
    """Compute unified plot bounds for multiple point clouds."""
    all_points = np.concatenate([p for p in points_list if p is not None and p.size > 0], axis=0)
    if all_points.size == 0:
        return None
    
    mins, maxs = all_points.min(0), all_points.max(0)
    rng = (maxs - mins).max() * 0.5
    mid = (maxs + mins) * 0.5
    return mid, rng


def set_axis_limits(ax, mid, rng):
    """Set 3D axis limits with uniform scaling."""
    ax.set_xlim(mid[0]-rng, mid[0]+rng)
    ax.set_ylim(mid[1]-rng, mid[1]+rng)
    ax.set_zlim(mid[2]-rng, mid[2]+rng)
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def save_comparison_png(path, current_before=None, current_after=None, radial_after=None, target_before=None, dpi=160, ptsize=0.5):
    """Save 3-panel comparison visualization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    X = as_numpy(current_before) if current_before is not None else None
    PU = as_numpy(current_after) if current_after is not None else None
    
    plt = setup_matplotlib()
    
    if PU is None or PU.size == 0:
        fig = plt.figure(figsize=(3,3))
        fig.text(0.5, 0.5, "No points", ha="center")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return
    
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    
    if X is not None and X.size > 0:
        ax1.scatter(X[:,0], X[:,1], X[:,2], s=ptsize, alpha=0.6)
    ax1.set_title("Current (before upsampling)")
    
    ax2.scatter(PU[:,0], PU[:,1], PU[:,2], s=ptsize, alpha=0.6)
    ax2.set_title("Current (after upsampling)")
    
    c = PU.mean(0)
    r = np.linalg.norm(PU - c[None,:], axis=1)
    sc = ax3.scatter(PU[:,0], PU[:,1], PU[:,2], c=r, s=ptsize, alpha=0.6, cmap="viridis")
    ax3.set_title("Radial color (after upsampling)")
    fig.colorbar(sc, ax=ax3, shrink=0.6)
    
    points_list = [X, PU]
    bounds = compute_plot_bounds(points_list)
    if bounds is not None:
        mid, rng = bounds
        for ax in (ax1, ax2, ax3):
            set_axis_limits(ax, mid, rng)
    
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_axis_hist_png(path, pts, dpi=160):
    """Save axis histogram visualization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    PU = as_numpy(pts)
    plt = setup_matplotlib()
    
    if PU is None or PU.size == 0:
        fig = plt.figure(figsize=(3,3))
        fig.text(0.5, 0.5, "No points", ha="center")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    titles = ["X-Axis Distribution", "Y-Axis Distribution", "Z-Axis Distribution"]
    labels = ["X", "Y", "Z"]
    
    for j, (ax, title, label) in enumerate(zip(axes, titles, labels)):
        ax.hist(PU[:, j], bins=48, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(f"{label} Coordinate")
        ax.set_ylabel("Count")
    
    fig.suptitle("Runtime Surface (Axis Distribution)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_ply_xyz(path: Path, xyz: np.ndarray):
    """Save point cloud in PLY format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open('w', encoding='utf-8') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        np.savetxt(f, xyz, fmt="%.6f")


def save_gaussians_npz(path: Path, xyz: np.ndarray, cov: np.ndarray, rgb=None, opacity=None):
    """Save Gaussian splatting data in NPZ format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if rgb is None:
        rgb = np.ones_like(xyz, dtype=np.float32)
    if opacity is None:
        opacity = np.ones((len(xyz), 1), dtype=np.float32)
    
    np.savez_compressed(
        path, 
        xyz=xyz.astype(np.float32),
        cov=cov.astype(np.float32),
        rgb=rgb.astype(np.float32),
        opacity=opacity.astype(np.float32)
    )