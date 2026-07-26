"""
I/O Utilities - File Operations & Data Export

Handles saving/loading of images, point clouds, and data files.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Image I/O
try:
    import imageio.v2 as iio
except ImportError:
    try:
        import imageio as iio
    except ImportError:
        iio = None
        print("[WARN] imageio not available, PNG export disabled")


# ============================================================================
# Image Saving
# ============================================================================

def save_image_png(path: Path, image: np.ndarray) -> None:
    """Save RGB image as PNG."""
    if iio is None:
        return
    
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    iio.imwrite(str(path), img_uint8)


def save_depth_png(path: Path, depth: np.ndarray, bits: int = 16) -> None:
    """
    Save depth map as 16-bit PNG.
    
    Args:
        path: Output path
        depth: Depth array in meters
        bits: 8 or 16
    """
    if iio is None:
        return
    
    if bits == 16:
        depth_norm = np.clip(depth / (depth.max() + 1e-8), 0, 1)
        depth_uint16 = (depth_norm * 65535).astype(np.uint16)
        iio.imwrite(str(path), depth_uint16)
    else:
        depth_norm = np.clip(depth / (depth.max() + 1e-8), 0, 1)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        iio.imwrite(str(path), depth_uint8)


# ============================================================================
# SDF Visualization
# ============================================================================

def visualize_sdf_result(
    target_dir: Path,
    sdf_grid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    points: Optional[np.ndarray] = None
) -> None:
    """
    Visualize SDF grid and surface anchors with matplotlib.
    
    Creates:
    - sdf_slices.png: XY, XZ, YZ slices through center
    - sdf_surface.png: 3D visualization of surface anchors
    
    Args:
        target_dir: Output directory
        sdf_grid: (R, R, R) SDF values
        bbox_min: (3,) bounding box min
        bbox_max: (3,) bounding box max
        points: (N, 3) surface anchor points (optional)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("[WARN] matplotlib not available, skipping SDF visualization")
        return
    
    R = sdf_grid.shape[0]
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(16, 14))
    
    # ========== Panel 1: XY slice (Z = middle) ==========
    ax1 = fig.add_subplot(2, 2, 1)
    z_mid = R // 2
    xy_slice = sdf_grid[:, :, z_mid].T
    
    im1 = ax1.imshow(xy_slice, cmap='RdBu', origin='lower', vmin=-0.2, vmax=0.2)
    ax1.contour(xy_slice, levels=[0], colors='yellow', linewidths=2)
    ax1.set_title(f'SDF XY Slice (Z={z_mid}/{R})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='SDF value')
    
    # ========== Panel 2: XZ slice (Y = middle) ==========
    ax2 = fig.add_subplot(2, 2, 2)
    y_mid = R // 2
    xz_slice = sdf_grid[:, y_mid, :].T
    
    im2 = ax2.imshow(xz_slice, cmap='RdBu', origin='lower', vmin=-0.2, vmax=0.2)
    ax2.contour(xz_slice, levels=[0], colors='yellow', linewidths=2)
    ax2.set_title(f'SDF XZ Slice (Y={y_mid}/{R})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    plt.colorbar(im2, ax=ax2, label='SDF value')
    
    # ========== Panel 3: YZ slice (X = middle) ==========
    ax3 = fig.add_subplot(2, 2, 3)
    x_mid = R // 2
    yz_slice = sdf_grid[x_mid, :, :].T
    
    im3 = ax3.imshow(yz_slice, cmap='RdBu', origin='lower', vmin=-0.2, vmax=0.2)
    ax3.contour(yz_slice, levels=[0], colors='yellow', linewidths=2)
    ax3.set_title(f'SDF YZ Slice (X={x_mid}/{R})', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    plt.colorbar(im3, ax=ax3, label='SDF value')
    
    # ========== Panel 4: Statistics ==========
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    sdf_min, sdf_max = sdf_grid.min(), sdf_grid.max()
    sdf_mean = sdf_grid.mean()
    surface_voxels = (np.abs(sdf_grid) < 0.05).sum()
    total_voxels = R ** 3
    
    stats_text = f"""
SDF Grid Statistics
{'='*40}

Grid Size: {R}³ = {total_voxels:,} voxels
Bounding Box: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}] × 
              [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}] × 
              [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]

SDF Range: [{sdf_min:.4f}, {sdf_max:.4f}]
SDF Mean:  {sdf_mean:.4f}

Surface Voxels (|SDF| < 0.05): {surface_voxels:,}
Surface Ratio: {surface_voxels/total_voxels:.2%}
"""
    
    if points is not None:
        stats_text += f"\nSurface Anchors: {len(points):,} points"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(target_dir / "sdf_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # ========== 3D Visualization of Surface Anchors ==========
    if points is not None and len(points) > 0:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample points if too many
        if len(points) > 10000:
            idx = np.random.choice(len(points), 10000, replace=False)
            pts = points[idx]
        else:
            pts = points
        
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                  c=pts[:, 2], cmap='viridis', s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Surface Anchors (N={len(points):,})', 
                    fontsize=14, fontweight='bold')
        
        # Equal aspect ratio
        max_range = np.array([
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min(),
            pts[:, 2].max() - pts[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
        mid_y = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
        mid_z = (pts[:, 2].max() + pts[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig(target_dir / "sdf_surface.png", dpi=150, bbox_inches='tight')
        plt.close(fig)


# ============================================================================
# Target Rendering Export
# ============================================================================

def save_target_renders(
    out_dir: Path,
    img_tgt: Optional[np.ndarray],
    alpha_tgt: Optional[np.ndarray],
    depth_tgt: Optional[np.ndarray],
    normal_map_tgt: Optional[np.ndarray],
    result_tgt: Dict
) -> None:
    """
    Save target rendering outputs to output/target/.
    
    Args:
        out_dir: Output directory
        img_tgt: RGB image
        alpha_tgt: Alpha channel
        depth_tgt: Depth map
        normal_map_tgt: Normal map
        result_tgt: Full upsampling result
    """
    target_dir = out_dir / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Target] Saving to {target_dir}/")
    
    # Save images
    if img_tgt is not None:
        save_image_png(target_dir / "target_image.png", img_tgt)
        print(f"  ✅ target_image.png")
    
    if alpha_tgt is not None:
        alpha_vis = np.stack([alpha_tgt]*3, axis=-1)
        save_image_png(target_dir / "target_alpha.png", alpha_vis)
        print(f"  ✅ target_alpha.png")
    
    if depth_tgt is not None:
        save_depth_png(target_dir / "target_depth.png", depth_tgt, bits=16)
        print(f"  ✅ target_depth.png (16-bit)")
    
    if normal_map_tgt is not None:
        save_image_png(target_dir / "target_normal.png", normal_map_tgt)
        print(f"  ✅ target_normal.png")
    
    # Save stage outputs if available
    stage_outputs = result_tgt.get("stage_outputs")
    if stage_outputs:
        from sampling.io.export import save_stage_progression
        save_stage_progression(target_dir, episode=-1, stage_data=stage_outputs)
    
    # Visualize SDF grid if available
    if "sdf_grid" in result_tgt:
        sdf_grid = result_tgt["sdf_grid"]
        bbox_min = result_tgt["bbox_min"]
        bbox_max = result_tgt["bbox_max"]
        
        # Convert to numpy if needed
        if hasattr(sdf_grid, 'cpu'):
            sdf_grid = sdf_grid.detach().cpu().numpy()
        if hasattr(bbox_min, 'cpu'):
            bbox_min = bbox_min.detach().cpu().numpy()
        if hasattr(bbox_max, 'cpu'):
            bbox_max = bbox_max.detach().cpu().numpy()
        
        # Get points for visualization
        points = result_tgt.get("points")
        if points is not None and hasattr(points, 'cpu'):
            points = points.detach().cpu().numpy()
        
        # Visualize SDF with matplotlib
        visualize_sdf_result(target_dir, sdf_grid, bbox_min, bbox_max, points)
        print(f"  ✅ SDF visualization saved")


# ============================================================================
# Episode Data Export
# ============================================================================

def save_gaussians_npz(
    path: Path,
    mu: np.ndarray,
    cov: np.ndarray,
    rgb: np.ndarray
) -> None:
    """
    Save Gaussian parameters as NPZ.
    
    Args:
        path: Output path
        mu: Positions (N, 3)
        cov: Covariances (N, 3, 3)
        rgb: Colors (N, 3)
    """
    np.savez_compressed(
        str(path),
        mu=mu,
        cov=cov,
        rgb=rgb
    )


def save_point_cloud_ply(
    path: Path,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None
) -> None:
    """
    Save point cloud as PLY file.
    
    Args:
        path: Output path
        points: (N, 3) positions
        colors: (N, 3) RGB colors in [0, 1]
    """
    with open(path, 'w') as f:
        N = len(points)
        
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # Data
        for i in range(N):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = (colors[i] * 255).astype(np.uint8)
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
            else:
                f.write(f"{x} {y} {z}\n")


def save_episode_summary(
    ep: int,
    ep_dir: Path,
    F: Any,
    mu_np: np.ndarray,
    loss_physics: float,
    render_losses=None
) -> None:
    """
    Save episode summary JSON.

    Args:
        ep: Episode number
        ep_dir: Episode directory
        F: Deformation gradients
        mu_np: Final positions
        loss_physics: Physics loss
        render_losses: Optional dict of render loss components
    """
    # Compute Jacobian determinant
    if hasattr(F, 'detach'):
        F_np = F.detach().cpu().numpy()
    else:
        F_np = np.asarray(F)

    J = np.linalg.det(F_np)

    summary = {
        "episode": ep + 1,
        "J_min": float(J.min()),
        "J_mean": float(J.mean()),
        "loss_physics_final": float(loss_physics),
        "num_surface_points": len(mu_np),
    }

    # Add render losses if available
    if render_losses is not None:
        summary["render_losses"] = {}
        for k, v in render_losses.items():
            try:
                # Handle torch tensors
                if hasattr(v, 'item'):
                    summary["render_losses"][k] = float(v.item())
                elif hasattr(v, 'detach'):
                    summary["render_losses"][k] = float(v.detach().cpu().numpy())
                elif isinstance(v, (int, float, np.number)):
                    summary["render_losses"][k] = float(v)
                else:
                    # Skip non-numeric values
                    continue
            except Exception as e:
                print(f"  ⚠️ Failed to convert {k}={v}: {e}")
                continue

    with (ep_dir / f"ep{ep:03d}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


__all__ = [
    'save_image_png',
    'save_depth_png',
    'save_target_renders',
    'save_gaussians_npz',
    'save_point_cloud_ply',
    'save_episode_summary',
]

