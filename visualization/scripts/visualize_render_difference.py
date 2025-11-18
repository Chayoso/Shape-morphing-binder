#!/usr/bin/env python3
"""
Visualize the difference between rendered images and target image across episodes.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

def load_image(path):
    """Load image and convert to numpy array."""
    img = Image.open(path)
    return np.array(img)

def compute_difference(rendered, target):
    """Compute pixel-wise difference between rendered and target images."""
    # Convert to float for computation
    rendered_float = rendered.astype(float) / 255.0
    target_float = target.astype(float) / 255.0

    # Compute absolute difference
    if len(rendered_float.shape) == 3 and rendered_float.shape[2] >= 3:
        # Use RGB channels only
        diff = np.abs(rendered_float[:, :, :3] - target_float[:, :, :3])
        # Average across channels for visualization
        diff_gray = np.mean(diff, axis=2)
    else:
        diff_gray = np.abs(rendered_float - target_float)

    return diff_gray

def visualize_difference(episode_dir, target_path, output_path):
    """Create a 3-panel visualization: rendered, target, difference."""

    # Find rendered image
    render_path = os.path.join(episode_dir, f"{os.path.basename(episode_dir)}_render.png")

    if not os.path.exists(render_path):
        print(f"  ⚠️  Render not found: {render_path}")
        return False

    # Load images
    rendered = load_image(render_path)
    target = load_image(target_path)

    # Compute difference
    diff = compute_difference(rendered, target)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Rendered image
    axes[0].imshow(rendered)
    axes[0].set_title('Rendered', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Target image
    axes[1].imshow(target)
    axes[1].set_title('Target', fontsize=16, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Difference heatmap
    im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Difference (L1)', fontsize=16, fontweight='bold')
    axes[2].axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Pixel Difference', fontsize=12)

    # Compute and display metrics
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)

    fig.suptitle(f'Episode {os.path.basename(episode_dir)} - Mean Diff: {mean_diff:.4f}, Max Diff: {max_diff:.4f}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_render_difference.py <output_directory>")
        sys.exit(1)

    output_dir = sys.argv[1]
    target_path = os.path.join(output_dir, 'target', 'target_image.png')

    if not os.path.exists(target_path):
        print(f"❌ Target image not found: {target_path}")
        sys.exit(1)

    # Find all episode directories
    episode_dirs = sorted([
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith('ep') and os.path.isdir(os.path.join(output_dir, d))
    ])

    if not episode_dirs:
        print(f"❌ No episode directories found in {output_dir}")
        sys.exit(1)

    print(f"Found {len(episode_dirs)} episodes in {output_dir}")
    print("=" * 70)

    success_count = 0
    for ep_dir in episode_dirs:
        ep_name = os.path.basename(ep_dir)
        print(f"Processing {ep_name}...")

        output_path = os.path.join(ep_dir, f"{ep_name}_difference.png")

        if visualize_difference(ep_dir, target_path, output_path):
            print(f"  ✅ Saved: {output_path}")
            success_count += 1

    print("=" * 70)
    print(f"✅ Successfully generated: {success_count}/{len(episode_dirs)}")
    print(f"\nDifference visualizations saved in each episode directory:")
    print(f"  {output_dir}/ep*/*_difference.png")

if __name__ == "__main__":
    main()
