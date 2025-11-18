#!/usr/bin/env python3
"""
Visualize alpha loss comparison.
Layout: Predicted Render | Target Render | Loss Heatmap
        Predicted Alpha  | Target Alpha  | Alpha Difference
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v2 as iio


def create_alpha_loss_visualization(ep_num, ep_dir, target_dir, output_path):
    """
    Create 2×3 grid:
    Row 1: Predicted Render | Target Render | Combined Loss Heatmap
    Row 2: Predicted Alpha  | Target Alpha  | Alpha Difference
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

    # Compute losses
    alpha_diff = np.abs(alpha_pred - alpha_tgt)
    alpha_mae = alpha_diff.mean()

    render_diff = np.abs(render_pred - render_tgt)
    render_mae = render_diff.mean(axis=-1)  # Average over RGB

    # Combined loss (weighted)
    combined_loss = 0.5 * render_mae + 0.5 * alpha_diff

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Episode {ep_num} - Render Loss Visualization', fontsize=18, fontweight='bold')

    # ========================================================================
    # Row 1: Renders and Combined Loss
    # ========================================================================

    # Predicted Render
    axes[0, 0].imshow(render_pred)
    axes[0, 0].set_title('Predicted Render', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Target Render
    axes[0, 1].imshow(render_tgt)
    axes[0, 1].set_title('Target Render', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Combined Loss Heatmap
    im_combined = axes[0, 2].imshow(combined_loss, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Combined Loss Heatmap\n(50% Render + 50% Alpha)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    cbar1 = plt.colorbar(im_combined, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar1.set_label('Loss Magnitude', fontsize=11)

    # ========================================================================
    # Row 2: Alpha and Alpha Difference
    # ========================================================================

    # Predicted Alpha
    axes[1, 0].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Predicted Alpha', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Target Alpha
    axes[1, 1].imshow(alpha_tgt, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Target Alpha', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Alpha Difference (Loss)
    im_alpha_loss = axes[1, 2].imshow(alpha_diff, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title('Alpha Loss (|Pred - Target|)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    cbar2 = plt.colorbar(im_alpha_loss, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar2.set_label('Alpha Difference', fontsize=11)

    # Add metrics text
    metrics_text = f"Episode {ep_num} Metrics:\n"
    metrics_text += f"  • Alpha MAE: {alpha_mae:.6f}\n"
    metrics_text += f"  • Render MAE: {render_mae.mean():.6f}\n"
    metrics_text += f"  • Combined Loss: {combined_loss.mean():.6f}"

    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12, family='monospace',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Saved: {output_path.name} (Alpha MAE: {alpha_mae:.6f})")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output_copy/output/spot')
    parser.add_argument('--episodes', type=str, default='0,10,20,44')
    parser.add_argument('--save_dir', type=str, default='loss_visualizations')
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
        output_path = save_dir / f"ep{ep_num:03d}_loss_visualization.png"
        create_alpha_loss_visualization(ep_num, ep_dir, target_dir, output_path)

    print("\n" + "="*70)
    print(f"✅ All visualizations saved to: {save_dir}/")


if __name__ == "__main__":
    main()
