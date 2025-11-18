#!/usr/bin/env python3
"""
Re-render Gaussians with enlarged covariances using diff-gaussian-rasterization.
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import yaml
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.io_utils import save_image_png


def create_camera_settings(camera_config, render_config):
    """Create camera and rendering settings."""

    # Camera parameters
    width = camera_config['width']
    height = camera_config['height']
    fx = camera_config['fx']
    fy = camera_config['fy']
    cx = camera_config['cx']
    cy = camera_config['cy']

    # Create view matrix (camera pose)
    eye = torch.tensor(camera_config['lookat']['eye'], dtype=torch.float32)
    target = torch.tensor(camera_config['lookat']['target'], dtype=torch.float32)
    up = torch.tensor(camera_config['lookat']['up'], dtype=torch.float32)

    # Compute camera coordinate system
    forward = target - eye
    forward = forward / torch.norm(forward)
    right = torch.cross(forward, up)
    right = right / torch.norm(right)
    up_new = torch.cross(right, forward)

    # View matrix (world to camera)
    view_matrix = torch.eye(4, dtype=torch.float32)
    view_matrix[:3, 0] = right
    view_matrix[:3, 1] = up_new
    view_matrix[:3, 2] = -forward
    view_matrix[:3, 3] = eye
    view_matrix = torch.inverse(view_matrix)

    # Projection matrix
    znear = camera_config['znear']
    zfar = camera_config['zfar']

    proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
    proj_matrix[0, 0] = 2.0 * fx / width
    proj_matrix[1, 1] = 2.0 * fy / height
    proj_matrix[0, 2] = (2.0 * cx - width) / width
    proj_matrix[1, 2] = (2.0 * cy - height) / height
    proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
    proj_matrix[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    proj_matrix[3, 2] = 1.0

    # Background color
    bg_color = torch.tensor(render_config['bg'], dtype=torch.float32)

    return width, height, view_matrix, proj_matrix, bg_color, eye


def render_gaussians_enlarged(gaussians_npz, camera_config, render_config, scale_factor=1.5):
    """
    Render Gaussians with enlarged covariances.

    Args:
        gaussians_npz: Path to Gaussians NPZ file or dict with mu, cov, rgb
        camera_config: Camera configuration
        render_config: Render configuration
        scale_factor: Factor to scale covariances (default: 1.5)

    Returns:
        Rendered image as numpy array (H, W, 3)
    """

    # Load Gaussians
    if isinstance(gaussians_npz, (str, Path)):
        data = np.load(gaussians_npz)
    else:
        data = gaussians_npz

    mu = data['mu']  # (N, 3)
    cov = data['cov']  # (N, 3, 3)
    rgb = data['rgb']  # (N, 3)

    # Enlarge covariances
    cov_enlarged = cov * (scale_factor ** 2)

    # Convert to torch tensors
    means3D = torch.from_numpy(mu).cuda()

    # Convert covariance to quaternion + scale representation
    # Eigendecomposition: cov = R @ S @ R^T
    cov_torch = torch.from_numpy(cov_enlarged).cuda()

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_torch)

    # Scales are sqrt of eigenvalues
    scales = torch.sqrt(torch.clamp(eigenvalues, min=1e-7))

    # Convert rotation matrix to quaternion
    def rotation_matrix_to_quaternion(R):
        """Convert batch of 3x3 rotation matrices to quaternions."""
        batch_size = R.shape[0]
        q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)

        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        # Case 1: trace > 0
        mask1 = trace > 0
        s = torch.sqrt(trace[mask1] + 1.0) * 2
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

        # Case 2: R[0,0] is max
        mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

        # Case 3: R[1,1] is max
        mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

        # Case 4: R[2,2] is max
        mask4 = (~mask1) & (~mask2) & (~mask3)
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s

        return q

    rotations = rotation_matrix_to_quaternion(eigenvectors)

    # Colors (SH coefficients - just use RGB for DC term)
    shs = torch.from_numpy(rgb).cuda().unsqueeze(1)  # (N, 1, 3)

    # Opacity (all opaque)
    opacities = torch.ones(means3D.shape[0], 1, device='cuda')

    # Create camera settings
    width, height, view_matrix, proj_matrix, bg_color, camera_center = create_camera_settings(
        camera_config, render_config
    )

    # Move to GPU
    view_matrix = view_matrix.cuda()
    proj_matrix = proj_matrix.cuda()
    bg_color = bg_color.cuda()
    camera_center = camera_center.cuda()

    # Create rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=width / (2.0 * camera_config['fx']),
        tanfovy=height / (2.0 * camera_config['fy']),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=proj_matrix,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    # Create rasterizer
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Render
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=torch.zeros_like(means3D[:, :2]),
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )

    # Convert to numpy
    rendered_image = rendered_image.permute(1, 2, 0).cpu().numpy()
    rendered_image = np.clip(rendered_image, 0, 1)

    return rendered_image


def render_all_episodes(output_dir, config_path, scale_factor=1.5, start_ep=None, end_ep=None):
    """
    Re-render all episodes with enlarged Gaussians.

    Args:
        output_dir: Directory containing episode folders
        config_path: Path to config YAML
        scale_factor: Factor to enlarge Gaussians
        start_ep: Starting episode (optional)
        end_ep: Ending episode (optional)
    """
    output_dir = Path(output_dir)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    camera_config = config['camera']
    render_config = config['render']

    # Find all episode directories
    ep_dirs = sorted([d for d in output_dir.iterdir()
                     if d.is_dir() and d.name.startswith('ep')])

    # Filter by range
    if start_ep is not None or end_ep is not None:
        filtered_dirs = []
        for ep_dir in ep_dirs:
            ep_num = int(ep_dir.name[2:])
            if start_ep is not None and ep_num < start_ep:
                continue
            if end_ep is not None and ep_num > end_ep:
                continue
            filtered_dirs.append(ep_dir)
        ep_dirs = filtered_dirs

    print(f"Found {len(ep_dirs)} episodes")
    print(f"Scale factor: {scale_factor}x")
    print(f"Output resolution: {camera_config['width']}x{camera_config['height']}")

    success_count = 0

    for ep_dir in tqdm(ep_dirs, desc="Re-rendering with enlarged Gaussians"):
        ep_num = int(ep_dir.name[2:])

        # Find Gaussians file
        gaussians_path = ep_dir / f"ep{ep_num:03d}_gaussians.npz"

        if not gaussians_path.exists():
            continue

        # Render
        try:
            rendered_image = render_gaussians_enlarged(
                gaussians_path,
                camera_config,
                render_config,
                scale_factor=scale_factor
            )

            # Save
            output_path = ep_dir / f"ep{ep_num:03d}_render_enlarged.png"
            save_image_png(output_path, rendered_image)

            success_count += 1

        except Exception as e:
            print(f"\n  Error rendering ep{ep_num:03d}: {e}")
            continue

    print(f"\n✅ Successfully rendered {success_count}/{len(ep_dirs)} episodes")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Re-render Gaussians with enlarged covariances')
    parser.add_argument('--output_dir', type=str, default='output_copy',
                        help='Directory containing episode folders')
    parser.add_argument('--config', type=str, default='configs/bob.yaml',
                        help='Config file for camera/render settings')
    parser.add_argument('--scale_factor', type=float, default=1.5,
                        help='Factor to enlarge Gaussians (default: 1.5)')
    parser.add_argument('--start_ep', type=int, default=None,
                        help='Starting episode')
    parser.add_argument('--end_ep', type=int, default=None,
                        help='Ending episode')

    args = parser.parse_args()

    render_all_episodes(args.output_dir, args.config, args.scale_factor, args.start_ep, args.end_ep)


if __name__ == "__main__":
    main()
