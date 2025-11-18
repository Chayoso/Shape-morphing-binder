#!/usr/bin/env python3
"""Regenerate all episode comparisons with depth difference."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
import sys

def save_matplotlib_comparison(
    ep_dir: Path,
    ep: int,
    img_pred: np.ndarray,
    img_tgt: np.ndarray,
    alpha_pred: np.ndarray,
    alpha_tgt: np.ndarray,
    depth_pred: Optional[np.ndarray],
    render_losses: Optional[Dict] = None
) -> None:
    """Create and save matplotlib comparison visualization with depth difference."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Episode {ep+1} - Rendering Comparison', fontsize=16, fontweight='bold')

    # Row 1: RGB comparisons
    axes[0, 0].imshow(img_pred)
    axes[0, 0].set_title('Predicted RGB', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_tgt)
    axes[0, 1].set_title('Target RGB', fontsize=12)
    axes[0, 1].axis('off')

    # RGB difference
    rgb_diff = np.abs(img_pred - img_tgt)
    im_diff = axes[0, 2].imshow(rgb_diff, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f'RGB Difference (MAE: {rgb_diff.mean():.4f})', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046)

    # Row 2: Alpha comparisons
    axes[1, 0].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Predicted Alpha', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(alpha_tgt, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Target Alpha', fontsize=12)
    axes[1, 1].axis('off')

    # Depth difference
    if depth_pred is not None:
        target_dir = ep_dir.parent / "target"
        target_depth_path = target_dir / "target_depth.png"

        if target_depth_path.exists():
            try:
                import imageio.v2 as iio
                depth_tgt = iio.imread(target_depth_path).astype(np.float32) / 65535.0
                valid_mask = (depth_pred > 0) & (depth_tgt > 0)

                if valid_mask.sum() > 0:
                    depth_diff = np.zeros_like(depth_pred)
                    depth_diff[valid_mask] = np.abs(depth_pred[valid_mask] - depth_tgt[valid_mask])
                    mae = depth_diff[valid_mask].mean()
                    im_depth_diff = axes[1, 2].imshow(depth_diff, cmap='hot', vmin=0, vmax=depth_diff.max())
                    axes[1, 2].set_title(f'Depth Difference (MAE: {mae:.4f})', fontsize=12)
                    axes[1, 2].axis('off')
                    plt.colorbar(im_depth_diff, ax=axes[1, 2], fraction=0.046)
                else:
                    im_depth = axes[1, 2].imshow(depth_pred, cmap='viridis')
                    axes[1, 2].set_title('Predicted Depth (no valid overlap)', fontsize=12)
                    axes[1, 2].axis('off')
                    plt.colorbar(im_depth, ax=axes[1, 2], fraction=0.046)
            except Exception as e:
                im_depth = axes[1, 2].imshow(depth_pred, cmap='viridis')
                axes[1, 2].set_title('Predicted Depth', fontsize=12)
                axes[1, 2].axis('off')
                plt.colorbar(im_depth, ax=axes[1, 2], fraction=0.046)
        else:
            im_depth = axes[1, 2].imshow(depth_pred, cmap='viridis')
            axes[1, 2].set_title('Predicted Depth (no target)', fontsize=12)
            axes[1, 2].axis('off')
            plt.colorbar(im_depth, ax=axes[1, 2], fraction=0.046)
    else:
        alpha_diff = np.abs(alpha_pred - alpha_tgt)
        im_alpha_diff = axes[1, 2].imshow(alpha_diff, cmap='hot', vmin=0, vmax=1)
        axes[1, 2].set_title(f'Alpha Difference (MAE: {alpha_diff.mean():.4f})', fontsize=12)
        axes[1, 2].axis('off')
        plt.colorbar(im_alpha_diff, ax=axes[1, 2], fraction=0.046)

    if render_losses is not None:
        loss_text = "Render Losses:\n"
        for key, val in render_losses.items():
            if isinstance(val, (int, float)):
                loss_text += f"  {key}: {val:.4f}\n"
        fig.text(0.02, 0.02, loss_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = ep_dir / f"ep{ep:03d}_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_episode(output_dir: Path, ep_dir: Path):
    """Process a single episode."""
    try:
        import imageio.v2 as iio
    except ImportError:
        import imageio as iio

    ep_name = ep_dir.name
    ep_num = int(ep_name.replace("ep", ""))

    # Check if required files exist
    render_path = ep_dir / f"{ep_name}_render.png"
    depth_path = ep_dir / f"{ep_name}_depth.png"

    if not render_path.exists():
        print(f"  ⚠️  Skipping {ep_name}: no render file")
        return False

    # Load predicted images
    img_pred = iio.imread(render_path) / 255.0

    depth_pred = None
    if depth_path.exists():
        depth_pred = iio.imread(depth_path).astype(np.float32) / 65535.0

    # Load target images
    target_dir = output_dir / "target"
    img_tgt = iio.imread(target_dir / "target_image.png") / 255.0
    alpha_tgt = iio.imread(target_dir / "target_alpha.png")[..., 0] / 255.0

    # Load predicted alpha
    alpha_pred_path = ep_dir / f"{ep_name}_alpha.png"
    if alpha_pred_path.exists():
        alpha_pred = iio.imread(alpha_pred_path)[..., 0] / 255.0
    else:
        alpha_pred = np.ones_like(img_pred[..., 0])

    # Generate comparison
    save_matplotlib_comparison(
        ep_dir=ep_dir,
        ep=ep_num,
        img_pred=img_pred,
        img_tgt=img_tgt,
        alpha_pred=alpha_pred,
        alpha_tgt=alpha_tgt,
        depth_pred=depth_pred,
        render_losses=None
    )

    return True


def main():
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("output/bob")

    if not output_dir.exists():
        print(f"❌ Directory not found: {output_dir}")
        return

    # Find all episode directories
    ep_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("ep")])

    print(f"Regenerating comparisons for {len(ep_dirs)} episodes in {output_dir}")
    print("=" * 70)

    success_count = 0
    for ep_dir in ep_dirs:
        print(f"Processing {ep_dir.name}...", end=" ")
        if process_episode(output_dir, ep_dir):
            print(f"✅")
            success_count += 1
        else:
            print()

    print("=" * 70)
    print(f"✅ Successfully regenerated {success_count}/{len(ep_dirs)} comparisons")

if __name__ == "__main__":
    main()
