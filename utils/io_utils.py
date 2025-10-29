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
    loss_physics: float
) -> None:
    """
    Save episode summary JSON.
    
    Args:
        ep: Episode number
        ep_dir: Episode directory
        F: Deformation gradients
        mu_np: Final positions
        loss_physics: Physics loss
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
    
    with (ep_dir / f"ep{ep:03d}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_episode_data(
    ep: int,
    ep_dir: Path,
    mu_np: np.ndarray,
    cov_np: np.ndarray,
    particle_color: list
) -> None:
    """
    Save episode data files (NPZ, PLY).
    
    Args:
        ep: Episode number
        ep_dir: Episode directory
        mu_np: Positions
        cov_np: Covariances
        particle_color: RGB color
    """
    rgb_mu = np.tile(np.array(particle_color, dtype=np.float32), (len(mu_np), 1))
    
    # Save Gaussians
    save_gaussians_npz(
        ep_dir / f"ep{ep:03d}_gaussians.npz",
        mu_np, cov_np, rgb_mu
    )
    
    # Save point cloud
    save_point_cloud_ply(
        ep_dir / f"ep{ep:03d}_surface_{len(mu_np)}.ply",
        mu_np, rgb_mu
    )


__all__ = [
    'save_image_png',
    'save_depth_png',
    'save_target_renders',
    'save_gaussians_npz',
    'save_point_cloud_ply',
    'save_episode_summary',
    'save_episode_data',
]

