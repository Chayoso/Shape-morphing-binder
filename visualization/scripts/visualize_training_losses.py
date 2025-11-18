#!/usr/bin/env python3
"""
Visualize training losses from episode summary files.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path


def load_episode_losses(output_dir):
    """Load all episode summary JSON files."""
    output_dir = Path(output_dir)

    # Find all episode summary files
    summary_files = sorted(output_dir.glob("ep*/ep*_summary.json"))

    if len(summary_files) == 0:
        print(f"❌ No summary files found in {output_dir}")
        return None

    episodes = []
    data = {
        'episode': [],
        'loss_physics': [],
        'loss_render_total': [],
        'loss_alpha': [],
        'loss_depth': [],
        'loss_photo': [],
        'loss_edge': [],
        'loss_cov_align': [],
        'loss_cov_reg': [],
        'J_mean': [],
    }

    print(f"Loading {len(summary_files)} episode summaries...")
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r') as f:
                episode_data = json.load(f)

            ep_num = episode_data.get('episode', 0)
            data['episode'].append(ep_num)

            # Physics loss
            data['loss_physics'].append(episode_data.get('loss_physics_final', 0))

            # Deformation gradient metric
            data['J_mean'].append(episode_data.get('J_mean', 1.0))

            # Render losses
            render_losses = episode_data.get('render_losses', {})
            data['loss_render_total'].append(render_losses.get('loss_render_total', 0))
            data['loss_alpha'].append(render_losses.get('loss_alpha', 0))
            data['loss_depth'].append(render_losses.get('loss_depth', 0))
            data['loss_photo'].append(render_losses.get('loss_photo', 0))
            data['loss_edge'].append(render_losses.get('loss_edge', 0))
            data['loss_cov_align'].append(render_losses.get('loss_cov_align', 0))
            data['loss_cov_reg'].append(render_losses.get('loss_cov_reg', 0))

        except Exception as e:
            print(f"  ⚠️  Failed to load {summary_file.name}: {e}")
            continue

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    print(f"  Loaded {len(data['episode'])} episodes")
    return data


def plot_training_losses(data, output_path):
    """Create comprehensive loss visualization."""

    episodes = data['episode']

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('Training Loss Progression', fontsize=18, fontweight='bold')

    # ========================================================================
    # Plot 1: Total Losses (Physics + Render)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(episodes, data['loss_physics'], 'o-', linewidth=2, markersize=5,
             label='Physics Loss', color='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(episodes, data['loss_render_total'], 's-', linewidth=2, markersize=5,
                  label='Render Loss (Total)', color='red')

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Physics Loss', fontsize=12, color='blue')
    ax1_twin.set_ylabel('Render Loss', fontsize=12, color='red')
    ax1.set_title('Physics vs Render Loss', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # ========================================================================
    # Plot 2: Render Loss Components
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(episodes, data['loss_alpha'], 'o-', label='Alpha', linewidth=2, markersize=4)
    ax2.plot(episodes, data['loss_depth'], 's-', label='Depth', linewidth=2, markersize=4)
    ax2.plot(episodes, data['loss_photo'], '^-', label='Photo', linewidth=2, markersize=4)
    ax2.plot(episodes, data['loss_edge'], 'v-', label='Edge', linewidth=2, markersize=4)
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Loss Value', fontsize=11)
    ax2.set_title('Render Loss Components', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # ========================================================================
    # Plot 3: Covariance Losses
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(episodes, data['loss_cov_align'], 'o-', label='Cov Align', linewidth=2, markersize=4)
    ax3.plot(episodes, data['loss_cov_reg'], 's-', label='Cov Reg', linewidth=2, markersize=4)
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Loss Value', fontsize=11)
    ax3.set_title('Covariance Losses', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # ========================================================================
    # Plot 4: Deformation Gradient (J)
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(episodes, data['J_mean'], 'o-', linewidth=2, markersize=5, color='purple')
    ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='J=1 (incompressible)')
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Mean J (det(F))', fontsize=11)
    ax4.set_title('Deformation Gradient', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 5: Alpha Loss Progression
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(episodes, data['loss_alpha'], 'o-', linewidth=2, markersize=5, color='green')
    ax5.fill_between(episodes, 0, data['loss_alpha'], alpha=0.3, color='green')
    ax5.set_xlabel('Episode', fontsize=11)
    ax5.set_ylabel('Alpha Loss', fontsize=11)
    ax5.set_title('Alpha Loss (Silhouette)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 6: Depth Loss Progression
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(episodes, data['loss_depth'], 'o-', linewidth=2, markersize=5, color='orange')
    ax6.fill_between(episodes, 0, data['loss_depth'], alpha=0.3, color='orange')
    ax6.set_xlabel('Episode', fontsize=11)
    ax6.set_ylabel('Depth Loss', fontsize=11)
    ax6.set_title('Depth Loss', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 7: Loss Statistics Table
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    # Compute statistics
    stats_text = "LOSS STATISTICS\n"
    stats_text += "="*40 + "\n\n"

    # Initial vs Final
    stats_text += f"Physics Loss:\n"
    stats_text += f"  Initial (ep{episodes[0]}): {data['loss_physics'][0]:.2f}\n"
    stats_text += f"  Final (ep{episodes[-1]}): {data['loss_physics'][-1]:.2f}\n"
    stats_text += f"  Reduction: {(1 - data['loss_physics'][-1]/data['loss_physics'][0])*100:.1f}%\n\n"

    stats_text += f"Render Loss (Total):\n"
    stats_text += f"  Initial: {data['loss_render_total'][0]:.4f}\n"
    stats_text += f"  Final: {data['loss_render_total'][-1]:.4f}\n"
    stats_text += f"  Reduction: {(1 - data['loss_render_total'][-1]/data['loss_render_total'][0])*100:.1f}%\n\n"

    stats_text += f"Alpha Loss:\n"
    stats_text += f"  Initial: {data['loss_alpha'][0]:.6f}\n"
    stats_text += f"  Final: {data['loss_alpha'][-1]:.6f}\n"
    stats_text += f"  Reduction: {(1 - data['loss_alpha'][-1]/data['loss_alpha'][0])*100:.1f}%\n\n"

    stats_text += f"Total Episodes: {len(episodes)}\n"

    ax7.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Loss plot saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize training losses')
    parser.add_argument('--output_dir', type=str, default='output/bunny',
                        help='Output directory with episode summaries')
    parser.add_argument('--output_plot', type=str, default='training_losses.png',
                        help='Output plot path')
    args = parser.parse_args()

    # Load losses
    data = load_episode_losses(args.output_dir)

    if data is None:
        return

    # Create plot
    plot_training_losses(data, args.output_plot)


if __name__ == "__main__":
    main()
