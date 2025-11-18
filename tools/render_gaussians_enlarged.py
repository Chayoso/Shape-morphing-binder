#!/usr/bin/env python3
"""
Re-render Gaussians with enlarged covariances for better visualization.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.rendering_utils import render_gaussians_with_config


def enlarge_gaussians(gaussians_path, scale_factor=2.0):
    """
    Load Gaussians and enlarge their covariances.

    Args:
        gaussians_path: Path to gaussians .npz file
        scale_factor: Factor to scale covariance by (default: 2.0)

    Returns:
        Dictionary with enlarged Gaussian parameters
    """
    data = np.load(gaussians_path)

    # Copy all data
    enlarged = {key: data[key] for key in data.keys()}

    # Scale covariances if they exist
    if 'cov' in enlarged:
        # Covariance matrices: scale by factor^2 (variance scales with square)
        enlarged['cov'] = enlarged['cov'] * (scale_factor ** 2)

    if 'scale' in enlarged:
        # Scale parameters: multiply by factor
        enlarged['scale'] = enlarged['scale'] * scale_factor

    return enlarged


def render_enlarged_episode(ep_dir, config, scale_factor=2.0, suffix='_enlarged_gs'):
    """
    Render an episode with enlarged Gaussians.

    Args:
        ep_dir: Episode directory
        config: Rendering configuration
        scale_factor: Covariance scale factor
        suffix: Output filename suffix
    """
    ep_num = int(ep_dir.name[2:])

    # Find Gaussians file
    gaussians_path = ep_dir / f"ep{ep_num:03d}_gaussians.npz"

    if not gaussians_path.exists():
        return False

    # Load and enlarge Gaussians
    gaussians = enlarge_gaussians(gaussians_path, scale_factor=scale_factor)

    # Render
    output_path = ep_dir / f"ep{ep_num:03d}{suffix}.png"

    try:
        render_gaussians_with_config(
            gaussians,
            config['camera'],
            config['render'],
            output_path
        )
        return True
    except Exception as e:
        print(f"  Error rendering ep{ep_num:03d}: {e}")
        return False


def render_all_episodes_enlarged(output_dir, config_path, scale_factor=2.0, suffix='_enlarged_gs'):
    """
    Re-render all episodes with enlarged Gaussians.

    Args:
        output_dir: Directory containing episode folders
        config_path: Path to config file for camera/render settings
        scale_factor: Factor to enlarge Gaussians by
        suffix: Output filename suffix
    """
    output_dir = Path(output_dir)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Find all episode directories
    ep_dirs = sorted([d for d in output_dir.iterdir()
                     if d.is_dir() and d.name.startswith('ep')])

    if len(ep_dirs) == 0:
        print(f"❌ No episodes found in {output_dir}")
        return

    print(f"Found {len(ep_dirs)} episodes")
    print(f"Scale factor: {scale_factor}x")
    print(f"Config: {config_path}")

    success_count = 0

    for ep_dir in tqdm(ep_dirs, desc="Re-rendering with enlarged Gaussians"):
        success = render_enlarged_episode(ep_dir, config, scale_factor, suffix)
        if success:
            success_count += 1

    print(f"\n✅ Successfully rendered {success_count}/{len(ep_dirs)} episodes")
    print(f"   Output files: ep*{suffix}.png")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Re-render Gaussians with enlarged covariances')
    parser.add_argument('--output_dir', type=str, default='output_copy',
                        help='Directory containing episode folders')
    parser.add_argument('--config', type=str, required=True,
                        help='Config file for camera/render settings')
    parser.add_argument('--scale_factor', type=float, default=2.0,
                        help='Factor to enlarge Gaussians by (default: 2.0)')
    parser.add_argument('--suffix', type=str, default='_enlarged_gs',
                        help='Output filename suffix')

    args = parser.parse_args()

    render_all_episodes_enlarged(args.output_dir, args.config, args.scale_factor, args.suffix)


if __name__ == "__main__":
    main()
