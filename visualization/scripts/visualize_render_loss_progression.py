#!/usr/bin/env python3
"""
Visualize render loss progression across episodes.
Shows depth and alpha comparisons for multiple episodes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import imageio.v2 as iio


def load_episode_renders(ep_dir: Path):
    """Load rendered depth and alpha from episode directory."""
    data = {}

    # Load alpha
    alpha_path = ep_dir / f"{ep_dir.name}_alpha.png"
    if alpha_path.exists():
        alpha = iio.imread(alpha_path)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]  # Take first channel
        data['alpha'] = alpha / 255.0

    # Load depth (16-bit)
    depth_path = ep_dir / f"{ep_dir.name}_depth.png"
    if depth_path.exists():
        data['depth'] = iio.imread(depth_path).astype(np.float32) / 65535.0

    return data


def load_target_renders(target_dir: Path):
    """Load target depth and alpha."""
    data = {}

    # Target alpha
    alpha_path = target_dir / "target_alpha.png"
    if alpha_path.exists():
        alpha = iio.imread(alpha_path)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        data['alpha'] = alpha / 255.0

    # Target depth
    depth_path = target_dir / "target_depth.png"
    if depth_path.exists():
        data['depth'] = iio.imread(depth_path).astype(np.float32) / 65535.0

    return data


def visualize_4_episodes(
    output_dir: Path,
    episodes: list,
    output_path: Path
):
    """
    Create comprehensive render loss visualization for 4 episodes.

    Args:
        output_dir: Base output directory
        episodes: List of episode numbers [0, 10, 20, 44]
        output_path: Where to save visualization
    """
    # Load target
    target_dir = output_dir / "target"
    target = load_target_renders(target_dir)

    if 'alpha' not in target or 'depth' not in target:
        print("❌ Target data incomplete")
        return

    alpha_tgt = target['alpha']
    depth_tgt = target['depth']

    # Load episodes
    episode_data = []
    for ep_num in episodes:
        ep_dir = output_dir / f"ep{ep_num:03d}"
        if not ep_dir.exists():
            print(f"⚠️  Episode {ep_num} not found")
            episode_data.append(None)
            continue

        data = load_episode_renders(ep_dir)
        if 'alpha' not in data or 'depth' not in data:
            print(f"⚠️  Episode {ep_num} data incomplete")
            episode_data.append(None)
            continue

        episode_data.append(data)
        print(f"✅ Loaded episode {ep_num}")

    # Create figure: 4 episodes × (alpha + depth) × (pred, target, diff) = 4×6 grid
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 7, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('Render Loss Progression: Alpha & Depth Comparison',
                 fontsize=18, fontweight='bold')

    # Process each episode
    for row, (ep_num, data) in enumerate(zip(episodes, episode_data)):
        if data is None:
            # Skip missing episodes
            continue

        alpha_pred = data['alpha']
        depth_pred = data['depth']

        # ====================================================================
        # Alpha comparison
        # ====================================================================

        # Predicted alpha
        ax_alpha_pred = fig.add_subplot(gs[row, 0])
        ax_alpha_pred.imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
        if row == 0:
            ax_alpha_pred.set_title('Predicted Alpha', fontsize=11, fontweight='bold')
        ax_alpha_pred.set_ylabel(f'Episode {ep_num}', fontsize=11, fontweight='bold')
        ax_alpha_pred.axis('off')

        # Target alpha (only show in first row)
        if row == 0:
            ax_alpha_tgt = fig.add_subplot(gs[row, 1])
            ax_alpha_tgt.imshow(alpha_tgt, cmap='gray', vmin=0, vmax=1)
            ax_alpha_tgt.set_title('Target Alpha', fontsize=11, fontweight='bold')
            ax_alpha_tgt.axis('off')

        # Alpha difference
        alpha_diff = np.abs(alpha_pred - alpha_tgt)
        alpha_mae = alpha_diff.mean()

        ax_alpha_diff = fig.add_subplot(gs[row, 2])
        im_alpha = ax_alpha_diff.imshow(alpha_diff, cmap='hot', vmin=0, vmax=1)
        if row == 0:
            ax_alpha_diff.set_title('Alpha Difference', fontsize=11, fontweight='bold')
        ax_alpha_diff.text(0.5, -0.1, f'MAE: {alpha_mae:.4f}',
                          transform=ax_alpha_diff.transAxes,
                          ha='center', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax_alpha_diff.axis('off')

        # Add colorbar only to first row
        if row == 0:
            cbar_ax_alpha = fig.add_axes([0.40, 0.77, 0.01, 0.15])
            plt.colorbar(im_alpha, cax=cbar_ax_alpha)

        # ====================================================================
        # Depth comparison
        # ====================================================================

        # Predicted depth
        ax_depth_pred = fig.add_subplot(gs[row, 3])
        im_depth_pred = ax_depth_pred.imshow(depth_pred, cmap='viridis')
        if row == 0:
            ax_depth_pred.set_title('Predicted Depth', fontsize=11, fontweight='bold')
        ax_depth_pred.axis('off')

        # Target depth (only show in first row)
        if row == 0:
            ax_depth_tgt = fig.add_subplot(gs[row, 4])
            im_depth_tgt = ax_depth_tgt.imshow(depth_tgt, cmap='viridis')
            ax_depth_tgt.set_title('Target Depth', fontsize=11, fontweight='bold')
            ax_depth_tgt.axis('off')

        # Depth difference
        valid_mask = (depth_pred > 0) & (depth_tgt > 0)
        depth_diff = np.zeros_like(depth_pred)

        if valid_mask.sum() > 0:
            depth_diff[valid_mask] = np.abs(depth_pred[valid_mask] - depth_tgt[valid_mask])
            depth_mae = depth_diff[valid_mask].mean()
            valid_pct = valid_mask.sum() / valid_mask.size * 100
        else:
            depth_mae = 0.0
            valid_pct = 0.0

        ax_depth_diff = fig.add_subplot(gs[row, 5])
        im_depth = ax_depth_diff.imshow(depth_diff, cmap='hot', vmin=0, vmax=depth_diff.max() if depth_diff.max() > 0 else 1.0)
        if row == 0:
            ax_depth_diff.set_title('Depth Difference', fontsize=11, fontweight='bold')
        ax_depth_diff.text(0.5, -0.1, f'MAE: {depth_mae:.4f} ({valid_pct:.0f}% valid)',
                          transform=ax_depth_diff.transAxes,
                          ha='center', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax_depth_diff.axis('off')

        # Add colorbar only to first row
        if row == 0:
            cbar_ax_depth = fig.add_axes([0.82, 0.77, 0.01, 0.15])
            plt.colorbar(im_depth, cax=cbar_ax_depth)

        # ====================================================================
        # Statistics panel
        # ====================================================================
        ax_stats = fig.add_subplot(gs[row, 6])
        ax_stats.axis('off')

        stats_text = f"Episode {ep_num}\n"
        stats_text += "="*25 + "\n\n"
        stats_text += f"Alpha MAE: {alpha_mae:.6f}\n"
        stats_text += f"Depth MAE: {depth_mae:.6f}\n"
        stats_text += f"Valid pixels: {valid_pct:.1f}%\n\n"

        # Compute improvement from ep0
        if row > 0 and episode_data[0] is not None:
            alpha_pred_0 = episode_data[0]['alpha']
            alpha_diff_0 = np.abs(alpha_pred_0 - alpha_tgt).mean()
            alpha_improve = (alpha_diff_0 - alpha_mae) / alpha_diff_0 * 100

            depth_pred_0 = episode_data[0]['depth']
            valid_0 = (depth_pred_0 > 0) & (depth_tgt > 0)
            if valid_0.sum() > 0:
                depth_diff_0 = np.abs(depth_pred_0[valid_0] - depth_tgt[valid_0]).mean()
                depth_improve = (depth_diff_0 - depth_mae) / depth_diff_0 * 100
            else:
                depth_improve = 0.0

            stats_text += f"vs Episode 0:\n"
            stats_text += f"  α: {alpha_improve:+.1f}%\n"
            stats_text += f"  depth: {depth_improve:+.1f}%\n"

        ax_stats.text(0.1, 0.5, stats_text, fontsize=8, family='monospace',
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved render loss visualization to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output_copy/output/spot',
                        help='Output directory')
    parser.add_argument('--episodes', type=str, default='0,10,20,44',
                        help='Comma-separated episode numbers')
    parser.add_argument('--output', type=str, default='render_loss_progression.png',
                        help='Output visualization path')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    episodes = [int(x.strip()) for x in args.episodes.split(',')]
    output_path = Path(args.output)

    print(f"Output directory: {output_dir}")
    print(f"Episodes: {episodes}")
    print(f"Output path: {output_path}")
    print("="*70)

    visualize_4_episodes(output_dir, episodes, output_path)


if __name__ == "__main__":
    main()
