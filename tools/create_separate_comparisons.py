#!/usr/bin/env python3
"""
Create separate comparison images for each episode.
- ep{N}_alpha_comparison.png: render | alpha | target alpha | alpha diff
- ep{N}_depth_comparison.png: render | depth | target depth | depth diff
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v2 as iio


def create_alpha_comparison(ep_num, ep_dir, target_dir, output_path):
    """Create alpha comparison: render | alpha | target alpha | alpha diff"""

    # Load predicted render and alpha
    render_path = ep_dir / f"ep{ep_num:03d}_render.png"
    alpha_path = ep_dir / f"ep{ep_num:03d}_alpha.png"

    if not render_path.exists() or not alpha_path.exists():
        print(f"  ⚠️  Missing render or alpha for ep{ep_num:03d}")
        return False

    render_pred = iio.imread(render_path) / 255.0
    alpha_pred = iio.imread(alpha_path)
    if alpha_pred.ndim == 3:
        alpha_pred = alpha_pred[..., 0]
    alpha_pred = alpha_pred / 255.0

    # Load target
    render_tgt_path = target_dir / "target_image.png"
    alpha_tgt_path = target_dir / "target_alpha.png"

    if not render_tgt_path.exists() or not alpha_tgt_path.exists():
        print(f"  ⚠️  Missing target render or alpha")
        return False

    render_tgt = iio.imread(render_tgt_path) / 255.0
    alpha_tgt = iio.imread(alpha_tgt_path)
    if alpha_tgt.ndim == 3:
        alpha_tgt = alpha_tgt[..., 0]
    alpha_tgt = alpha_tgt / 255.0

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(f'Episode {ep_num} - Alpha Comparison', fontsize=16, fontweight='bold')

    # Row 1: Renders
    axes[0, 0].imshow(render_pred)
    axes[0, 0].set_title('Predicted Render', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(render_tgt)
    axes[0, 1].set_title('Target Render', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Row 2: Alpha
    axes[1, 0].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Predicted Alpha', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(alpha_tgt, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Target Alpha', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # Add alpha difference overlay on row 1, col 1
    alpha_diff = np.abs(alpha_pred - alpha_tgt)
    alpha_mae = alpha_diff.mean()

    # Create overlay on target render
    axes[0, 1].imshow(alpha_diff, cmap='hot', alpha=0.3, vmin=0, vmax=1)

    # Add text box with metrics
    metrics_text = f"Alpha MAE: {alpha_mae:.4f}"
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return True


def create_depth_comparison(ep_num, ep_dir, target_dir, output_path):
    """Create depth comparison: render | depth | target depth | depth diff"""

    # Load predicted render and depth
    render_path = ep_dir / f"ep{ep_num:03d}_render.png"
    depth_path = ep_dir / f"ep{ep_num:03d}_depth.png"

    if not render_path.exists() or not depth_path.exists():
        print(f"  ⚠️  Missing render or depth for ep{ep_num:03d}")
        return False

    render_pred = iio.imread(render_path) / 255.0
    depth_pred = iio.imread(depth_path).astype(np.float32) / 65535.0

    # Load target
    render_tgt_path = target_dir / "target_image.png"
    depth_tgt_path = target_dir / "target_depth.png"

    if not render_tgt_path.exists() or not depth_tgt_path.exists():
        print(f"  ⚠️  Missing target render or depth")
        return False

    render_tgt = iio.imread(render_tgt_path) / 255.0
    depth_tgt = iio.imread(depth_tgt_path).astype(np.float32) / 65535.0

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(f'Episode {ep_num} - Depth Comparison', fontsize=16, fontweight='bold')

    # Row 1: Renders
    axes[0, 0].imshow(render_pred)
    axes[0, 0].set_title('Predicted Render', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(render_tgt)
    axes[0, 1].set_title('Target Render', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Row 2: Depth
    im1 = axes[1, 0].imshow(depth_pred, cmap='viridis')
    axes[1, 0].set_title('Predicted Depth', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(depth_tgt, cmap='viridis')
    axes[1, 1].set_title('Target Depth', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    # Compute depth difference
    valid_mask = (depth_pred > 0) & (depth_tgt > 0)
    depth_diff = np.zeros_like(depth_pred)

    if valid_mask.sum() > 0:
        depth_diff[valid_mask] = np.abs(depth_pred[valid_mask] - depth_tgt[valid_mask])
        depth_mae = depth_diff[valid_mask].mean()
        valid_pct = valid_mask.sum() / valid_mask.size * 100
    else:
        depth_mae = 0.0
        valid_pct = 0.0

    # Add depth difference overlay on target render
    if valid_mask.sum() > 0:
        axes[0, 1].imshow(depth_diff, cmap='hot', alpha=0.3, vmin=0, vmax=depth_diff.max())

    # Add text box with metrics
    metrics_text = f"Depth MAE: {depth_mae:.4f} ({valid_pct:.1f}% valid pixels)"
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output_copy/output/spot')
    parser.add_argument('--episodes', type=str, default='0,10,20,44')
    parser.add_argument('--save_dir', type=str, default='comparisons',
                        help='Directory to save comparison images')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    episodes = [int(x.strip()) for x in args.episodes.split(',')]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    target_dir = output_dir / "target"

    if not target_dir.exists():
        print(f"❌ Target directory not found: {target_dir}")
        return

    print(f"Output directory: {output_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Episodes: {episodes}")
    print("="*70)

    for ep_num in episodes:
        ep_dir = output_dir / f"ep{ep_num:03d}"

        if not ep_dir.exists():
            print(f"⚠️  Episode {ep_num} not found")
            continue

        print(f"\nProcessing Episode {ep_num}:")

        # Create alpha comparison
        alpha_output = save_dir / f"ep{ep_num:03d}_alpha_comparison.png"
        if create_alpha_comparison(ep_num, ep_dir, target_dir, alpha_output):
            print(f"  ✅ Alpha comparison: {alpha_output}")

        # Create depth comparison
        depth_output = save_dir / f"ep{ep_num:03d}_depth_comparison.png"
        if create_depth_comparison(ep_num, ep_dir, target_dir, depth_output):
            print(f"  ✅ Depth comparison: {depth_output}")

    print("\n" + "="*70)
    print(f"✅ All comparisons saved to: {save_dir}/")


if __name__ == "__main__":
    main()
