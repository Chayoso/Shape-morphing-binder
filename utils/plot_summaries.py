"""
Plot Training Results from Episode Summaries

Collects all ep###_summary.json files and creates comprehensive visualizations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import glob


def collect_summaries(output_dir: str) -> Tuple[List[int], Dict]:
    """
    Collect all episode summary files.

    Args:
        output_dir: Output directory (e.g., 'output/spot')

    Returns:
        Tuple of (episodes, data) where data contains all metrics
    """
    output_path = Path(output_dir)

    # Find all summary files
    summary_files = sorted(glob.glob(str(output_path / "ep*" / "ep*_summary.json")))

    if not summary_files:
        print(f"No summary files found in {output_dir}")
        return [], {}

    print(f"Found {len(summary_files)} summary files")

    # Collect data
    episodes = []
    data = {
        'J_min': [],
        'J_mean': [],
        'loss_physics_final': [],
        'num_surface_points': [],
        'render_losses': {}
    }

    for summary_file in summary_files:
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            ep = summary.get('episode', 0) - 1  # Convert to 0-indexed
            episodes.append(ep)

            data['J_min'].append(summary.get('J_min', np.nan))
            data['J_mean'].append(summary.get('J_mean', np.nan))
            data['loss_physics_final'].append(summary.get('loss_physics_final', np.nan))
            data['num_surface_points'].append(summary.get('num_surface_points', np.nan))

            # Collect render losses
            if 'render_losses' in summary:
                for key, val in summary['render_losses'].items():
                    if key not in data['render_losses']:
                        data['render_losses'][key] = []
                    data['render_losses'][key].append(val)
        except Exception as e:
            print(f"  ⚠️ Failed to load {summary_file}: {e}")

    return episodes, data


def plot_training_summary(output_dir: str, save_path: str = None, figsize=(15, 5)):
    """
    Create focused training visualization from episode summaries.

    Args:
        output_dir: Output directory containing ep### folders
        save_path: Path to save the plot (default: output_dir/training_summary.png)
        figsize: Figure size (width, height)
    """
    episodes, data = collect_summaries(output_dir)

    if not episodes:
        print("No data to plot!")
        return

    if save_path is None:
        save_path = Path(output_dir) / "training_summary.png"
    else:
        save_path = Path(save_path)

    # Create figure with 1x3 grid
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Training Summary - {output_dir}', fontsize=16, fontweight='bold')

    # ============================================================================
    # Plot 1: Physics Loss
    # ============================================================================
    ax = axes[0]
    physics_loss = np.array(data['loss_physics_final'])

    if not np.all(np.isnan(physics_loss)):
        ax.plot(episodes, physics_loss, 'o-', linewidth=2.5, markersize=6, color='#2E86AB', label='Physics Loss')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (Grid Mass Matching)', fontsize=12, fontweight='bold')
        ax.set_title('Physics Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics
        valid_idx = ~np.isnan(physics_loss)
        if np.any(valid_idx):
            initial = physics_loss[valid_idx][0]
            final = physics_loss[valid_idx][-1]
            reduction = initial - final
            reduction_pct = 100 * reduction / max(initial, 1e-6)

            stats_text = f'Initial: {initial:.1f}\nFinal: {final:.1f}\nReduction: {reduction_pct:.1f}%'
            ax.text(0.98, 0.98, stats_text,
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='#2E86AB'))
    else:
        ax.text(0.5, 0.5, 'No physics loss data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)

    # ============================================================================
    # Plot 2: Total Render Loss
    # ============================================================================
    ax = axes[1]

    render_losses = data['render_losses']

    if 'loss_render_total' in render_losses:
        total_render = np.array(render_losses['loss_render_total'])
        if not np.all(np.isnan(total_render)):
            ax.plot(episodes[:len(total_render)], total_render, 'o-',
                   linewidth=2.5, markersize=6, color='#A23B72', label='Total Render Loss')
            ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
            ax.set_ylabel('Total Render Loss', fontsize=12, fontweight='bold')
            ax.set_title('Render Loss (Total)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

            # Add statistics
            valid_idx = ~np.isnan(total_render)
            if np.any(valid_idx):
                initial = total_render[valid_idx][0]
                final = total_render[valid_idx][-1]
                reduction = initial - final
                reduction_pct = 100 * reduction / max(initial, 1e-6)

                stats_text = f'Initial: {initial:.1f}\nFinal: {final:.1f}\nReduction: {reduction_pct:.1f}%'
                ax.text(0.98, 0.98, stats_text,
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='#F8D7DA', alpha=0.7, edgecolor='#A23B72'))
        else:
            ax.text(0.5, 0.5, 'No total render loss', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No total render loss', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)

    # ============================================================================
    # Plot 3: Render Loss Components
    # ============================================================================
    ax = axes[2]

    component_colors = {
        'loss_alpha': '#E63946',
        'loss_depth': '#F77F00',
        'loss_photo': '#06A77D',
        'loss_edge': '#118AB2',
        'loss_cov_align': '#073B4C',
        'loss_det_barrier': '#6A4C93',
        'loss_coverage': '#C9ADA7',
    }

    has_render_data = False
    for key, values in render_losses.items():
        if key == 'loss_render_total':  # Skip total, we already plotted it
            continue

        values_arr = np.array(values)
        if not np.all(np.isnan(values_arr)):
            color = component_colors.get(key, '#999999')
            # Clean up label name
            label = key.replace('loss_', '').replace('_', ' ').title()
            ax.plot(episodes[:len(values)], values_arr, 'o-',
                   linewidth=2, markersize=4, label=label, color=color, alpha=0.85)
            has_render_data = True

    if has_render_data:
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        ax.set_title('Render Loss Components', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
    else:
        ax.text(0.5, 0.5, 'No render component data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Training summary plot saved to: {save_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_summaries.py <output_dir> [save_path]")
        print("Example: python plot_summaries.py output/spot")
        print("         python plot_summaries.py output/spot custom_plot.png")
        sys.exit(1)

    output_dir = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None

    plot_training_summary(output_dir, save_path)
