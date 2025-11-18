#!/usr/bin/env python3
"""
Clean render loss visualization:
- Alpha comparison with 100% alpha loss heatmap
- Depth comparison with depth loss heatmap
- All images same size
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v2 as iio


def resize_to_match(img, target_shape):
    """Resize image to match target shape if needed."""
    from scipy.ndimage import zoom

    if img.shape[:2] == target_shape[:2]:
        return img

    if img.ndim == 3:
        zoom_factors = (target_shape[0] / img.shape[0],
                       target_shape[1] / img.shape[1], 1)
    else:
        zoom_factors = (target_shape[0] / img.shape[0],
                       target_shape[1] / img.shape[1])

    return zoom(img, zoom_factors, order=1)


def create_alpha_loss_visualization(ep_num, ep_dir, target_dir, output_path):
    """
    Create 2×3 grid with 100% alpha loss:
    Row 1: Predicted Render | Target Render | Alpha Loss Heatmap
    Row 2: Predicted Alpha  | Target Alpha  | Alpha Loss Heatmap (same as above)
    """

    # Load predicted
    render_pred_path = ep_dir / f"ep{ep_num:03d}_render.png"
    alpha_pred_path = ep_dir / f"ep{ep_num:03d}_alpha.png"

    if not render_pred_path.exists() or not alpha_pred_path.exists():
        print(f"  ❌ Missing data for ep{ep_num:03d}")
        return False

    render_pred = iio.imread(render_pred_path) / 255.0
    alpha_pred = iio.imread(alpha_pred_path)
    if alpha_pred.ndim == 3:
        alpha_pred = alpha_pred[..., 0]
    alpha_pred = alpha_pred / 255.0

    # Load target
    render_tgt_path = target_dir / "target_image.png"
    alpha_tgt_path = target_dir / "target_alpha.png"

    if not render_tgt_path.exists() or not alpha_tgt_path.exists():
        print(f"  ❌ Missing target data")
        return False

    render_tgt = iio.imread(render_tgt_path) / 255.0
    alpha_tgt = iio.imread(alpha_tgt_path)
    if alpha_tgt.ndim == 3:
        alpha_tgt = alpha_tgt[..., 0]
    alpha_tgt = alpha_tgt / 255.0

    # Ensure all images are same size
    target_shape = render_tgt.shape
    render_pred = resize_to_match(render_pred, target_shape)
    alpha_pred = resize_to_match(alpha_pred, target_shape[:2])
    alpha_tgt = resize_to_match(alpha_tgt, target_shape[:2])

    # Compute alpha loss (100%)
    alpha_loss = np.abs(alpha_pred - alpha_tgt)
    alpha_mae = alpha_loss.mean()

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Episode {ep_num} - Alpha Loss Visualization (100% Alpha)',
                 fontsize=18, fontweight='bold')

    # ========================================================================
    # Row 1: Renders and Alpha Loss Heatmap
    # ========================================================================

    # Predicted Render
    axes[0, 0].imshow(render_pred)
    axes[0, 0].set_title('Predicted Render', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Target Render
    axes[0, 1].imshow(render_tgt)
    axes[0, 1].set_title('Target Render', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Alpha Loss Heatmap
    im_loss = axes[0, 2].imshow(alpha_loss, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Alpha Loss Heatmap', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    cbar1 = plt.colorbar(im_loss, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar1.set_label('Loss Magnitude', fontsize=11)

    # ========================================================================
    # Row 2: Alpha Channels and Alpha Loss Heatmap (repeated)
    # ========================================================================

    # Predicted Alpha
    axes[1, 0].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Predicted Alpha', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Target Alpha
    axes[1, 1].imshow(alpha_tgt, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Target Alpha', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Alpha Loss Heatmap (same as above)
    im_loss2 = axes[1, 2].imshow(alpha_loss, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title('Alpha Loss Heatmap', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    cbar2 = plt.colorbar(im_loss2, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar2.set_label('|Pred - Target|', fontsize=11)

    # Add metrics
    metrics_text = f"Episode {ep_num} | Alpha MAE: {alpha_mae:.6f}"
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12, family='monospace',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Alpha: {output_path.name} (MAE: {alpha_mae:.6f})")
    return True


def create_depth_loss_visualization(ep_num, ep_dir, target_dir, output_path):
    """
    Create 2×3 grid with depth loss:
    Row 1: Predicted Render | Target Render | Depth Loss Heatmap
    Row 2: Predicted Depth  | Target Depth  | Depth Loss Heatmap (same as above)
    """

    # Load predicted
    render_pred_path = ep_dir / f"ep{ep_num:03d}_render.png"
    depth_pred_path = ep_dir / f"ep{ep_num:03d}_depth.png"

    if not render_pred_path.exists() or not depth_pred_path.exists():
        print(f"  ❌ Missing depth data for ep{ep_num:03d}")
        return False

    render_pred = iio.imread(render_pred_path) / 255.0
    depth_pred = iio.imread(depth_pred_path).astype(np.float32) / 65535.0

    # Load target
    render_tgt_path = target_dir / "target_image.png"
    depth_tgt_path = target_dir / "target_depth.png"

    if not render_tgt_path.exists() or not depth_tgt_path.exists():
        print(f"  ❌ Missing target depth data")
        return False

    render_tgt = iio.imread(render_tgt_path) / 255.0
    depth_tgt = iio.imread(depth_tgt_path).astype(np.float32) / 65535.0

    # Ensure all images are same size
    target_shape = render_tgt.shape
    render_pred = resize_to_match(render_pred, target_shape)
    depth_pred = resize_to_match(depth_pred, target_shape[:2])
    depth_tgt = resize_to_match(depth_tgt, target_shape[:2])

    # Compute depth loss
    valid_mask = (depth_pred > 0) & (depth_tgt > 0)
    depth_loss = np.zeros_like(depth_pred)

    if valid_mask.sum() > 0:
        depth_loss[valid_mask] = np.abs(depth_pred[valid_mask] - depth_tgt[valid_mask])
        depth_mae = depth_loss[valid_mask].mean()
        valid_pct = valid_mask.sum() / valid_mask.size * 100
    else:
        depth_mae = 0.0
        valid_pct = 0.0

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Episode {ep_num} - Depth Loss Visualization',
                 fontsize=18, fontweight='bold')

    # ========================================================================
    # Row 1: Renders and Depth Loss Heatmap
    # ========================================================================

    # Predicted Render
    axes[0, 0].imshow(render_pred)
    axes[0, 0].set_title('Predicted Render', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Target Render
    axes[0, 1].imshow(render_tgt)
    axes[0, 1].set_title('Target Render', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Depth Loss Heatmap
    vmax_loss = depth_loss.max() if depth_loss.max() > 0 else 1.0
    im_loss = axes[0, 2].imshow(depth_loss, cmap='hot', vmin=0, vmax=vmax_loss)
    axes[0, 2].set_title('Depth Loss Heatmap', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    cbar1 = plt.colorbar(im_loss, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar1.set_label('Depth Difference', fontsize=11)

    # ========================================================================
    # Row 2: Depth Maps and Depth Loss Heatmap (repeated)
    # ========================================================================

    # Predicted Depth
    im_depth_pred = axes[1, 0].imshow(depth_pred, cmap='viridis')
    axes[1, 0].set_title('Predicted Depth', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im_depth_pred, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Target Depth
    im_depth_tgt = axes[1, 1].imshow(depth_tgt, cmap='viridis')
    axes[1, 1].set_title('Target Depth', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im_depth_tgt, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Depth Loss Heatmap (same as above)
    im_loss2 = axes[1, 2].imshow(depth_loss, cmap='hot', vmin=0, vmax=vmax_loss)
    axes[1, 2].set_title('Depth Loss Heatmap', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    cbar2 = plt.colorbar(im_loss2, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar2.set_label('|Pred - Target|', fontsize=11)

    # Add metrics
    metrics_text = f"Episode {ep_num} | Depth MAE: {depth_mae:.6f} ({valid_pct:.1f}% valid pixels)"
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12, family='monospace',
             bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Depth: {output_path.name} (MAE: {depth_mae:.6f})")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output_copy/output/spot')
    parser.add_argument('--episodes', type=str, default='0,10,20,44')
    parser.add_argument('--save_dir', type=str, default='loss_viz_clean')
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

        print(f"\nEpisode {ep_num}:")

        # Create alpha visualization
        alpha_output = save_dir / f"ep{ep_num:03d}_alpha_loss.png"
        create_alpha_loss_visualization(ep_num, ep_dir, target_dir, alpha_output)

        # Create depth visualization
        depth_output = save_dir / f"ep{ep_num:03d}_depth_loss.png"
        create_depth_loss_visualization(ep_num, ep_dir, target_dir, depth_output)

    print("\n" + "="*70)
    print(f"✅ All visualizations saved to: {save_dir}/")


if __name__ == "__main__":
    main()
