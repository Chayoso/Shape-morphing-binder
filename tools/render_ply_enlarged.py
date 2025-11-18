#!/usr/bin/env python3
"""
Render PLY files with enlarged Gaussian visualizations for all episodes.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def render_ply_enlarged(ply_path, output_path, point_size=3.0, camera_pos=None):
    """
    Render a PLY file with enlarged points.

    Args:
        ply_path: Path to PLY file
        output_path: Where to save the rendered image
        point_size: Size multiplier for points
        camera_pos: Camera position [eye, target, up]
    """
    # Load PLY
    pcd = o3d.io.read_point_cloud(str(ply_path))

    if not pcd.has_points():
        print(f"  ⚠ No points found in {ply_path}")
        return False

    points = np.asarray(pcd.points)

    # Get colors if available
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(points) * 0.7  # Gray

    # Create figure
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # Plot points with increased size
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=point_size, alpha=0.8, edgecolors='none')

    # Set camera view
    if camera_pos is None:
        # Default camera matching the config
        ax.view_init(elev=20, azim=45)
    else:
        eye, target, up = camera_pos
        ax.view_init(elev=20, azim=45)

    # Set equal aspect ratio
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Remove axes
    ax.set_axis_off()

    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


def render_all_episodes(output_dir, point_size=5.0, suffix='_enlarged'):
    """
    Render PLY files for all episodes with enlarged points.

    Args:
        output_dir: Directory containing episode folders
        point_size: Size multiplier for Gaussian visualization
        suffix: Suffix to add to output filename
    """
    output_dir = Path(output_dir)

    # Find all episode directories
    ep_dirs = sorted([d for d in output_dir.iterdir()
                     if d.is_dir() and d.name.startswith('ep')])

    if len(ep_dirs) == 0:
        print(f"❌ No episodes found in {output_dir}")
        return

    print(f"Found {len(ep_dirs)} episodes")
    print(f"Point size: {point_size}")

    success_count = 0

    for ep_dir in tqdm(ep_dirs, desc="Rendering episodes"):
        ep_num = int(ep_dir.name[2:])

        # Find PLY file
        ply_files = list(ep_dir.glob("*.ply"))

        if len(ply_files) == 0:
            continue

        # Use the first PLY file found
        ply_path = ply_files[0]

        # Output path
        output_path = ep_dir / f"ep{ep_num:03d}{suffix}.png"

        # Render
        success = render_ply_enlarged(ply_path, output_path, point_size=point_size)

        if success:
            success_count += 1

    print(f"\n✅ Successfully rendered {success_count}/{len(ep_dirs)} episodes")
    print(f"   Output files: ep*{suffix}.png")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Render PLY files with enlarged Gaussians')
    parser.add_argument('--output_dir', type=str, default='output_copy',
                        help='Directory containing episode folders')
    parser.add_argument('--point_size', type=float, default=5.0,
                        help='Point size for rendering (default: 5.0)')
    parser.add_argument('--suffix', type=str, default='_enlarged',
                        help='Suffix for output files (default: _enlarged)')

    args = parser.parse_args()

    render_all_episodes(args.output_dir, args.point_size, args.suffix)


if __name__ == "__main__":
    main()
