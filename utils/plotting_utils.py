"""
Plotting Utilities - Loss Visualization

Utilities for plotting training losses and metrics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def plot_training_losses(json_path: Path, output_path: Path, figsize=(14, 10)):
    """
    Plot physics and render losses from training JSON log.

    Creates a comprehensive visualization with:
    - Physics loss (EndLayerMassLoss) over episodes
    - Render loss components over episodes
    - Combined loss view

    Args:
        json_path: Path to training_losses.json
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    # Load loss history
    with open(json_path, 'r') as f:
        data = json.load(f)

    episodes_data = data['episodes']

    if not episodes_data:
        print("[WARN] No episode data to plot")
        return

    # Extract data
    episodes = [ep['episode'] for ep in episodes_data]

    # Physics losses
    loss_physics = [ep.get('loss_physics', None) for ep in episodes_data]

    # Render losses
    loss_render_total = [ep.get('loss_render_total', None) for ep in episodes_data]
    loss_alpha = [ep.get('loss_alpha', None) for ep in episodes_data]
    loss_depth = [ep.get('loss_depth', None) for ep in episodes_data]
    loss_photo = [ep.get('loss_photo', None) for ep in episodes_data]
    loss_edge = [ep.get('loss_edge', None) for ep in episodes_data]
    loss_cov_align = [ep.get('loss_cov_align', None) for ep in episodes_data]
    loss_coverage = [ep.get('loss_coverage', None) for ep in episodes_data]
    loss_det_barrier = [ep.get('loss_det_barrier', None) for ep in episodes_data]

    # Filter out None values
    def filter_none(lst):
        return [x if x is not None else np.nan for x in lst]

    # Extract additional data
    J_min = [ep.get('J_min', None) for ep in episodes_data]
    J_mean = [ep.get('J_mean', None) for ep in episodes_data]
    num_points = [ep.get('num_surface_points', None) for ep in episodes_data]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Training Losses - {data["config"]}', fontsize=14, fontweight='bold')

    # ============================================================================
    # Plot 1: Physics Loss
    # ============================================================================
    ax = axes[0, 0]
    physics_filtered = filter_none(loss_physics)
    if any(~np.isnan(physics_filtered)):
        ax.plot(episodes, physics_filtered, 'o-', linewidth=2, markersize=4, label='Physics Loss', color='#2E86AB')
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Loss (EndLayerMassLoss)', fontsize=11)
        ax.set_title('Physics Loss (Grid Mass Matching)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add statistics
        valid_physics = [x for x in physics_filtered if not np.isnan(x)]
        if valid_physics:
            ax.text(0.02, 0.98, f'Initial: {valid_physics[0]:.2f}\nFinal: {valid_physics[-1]:.2f}\nReduction: {(valid_physics[0]-valid_physics[-1]):.2f}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No physics loss data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Physics Loss (No Data)', fontsize=12)

    # ============================================================================
    # Plot 2: Render Loss (Total)
    # ============================================================================
    ax = axes[0, 1]
    render_filtered = filter_none(loss_render_total)
    if any(~np.isnan(render_filtered)):
        ax.plot(episodes, render_filtered, 'o-', linewidth=2, markersize=4, label='Total Render Loss', color='#A23B72')
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Loss (Weighted Sum)', fontsize=11)
        ax.set_title('Render Loss (Total)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add statistics
        valid_render = [x for x in render_filtered if not np.isnan(x)]
        if valid_render:
            ax.text(0.02, 0.98, f'Initial: {valid_render[0]:.2f}\nFinal: {valid_render[-1]:.2f}\nReduction: {(valid_render[0]-valid_render[-1]):.2f}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No render loss data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Render Loss (No Data)', fontsize=12)

    # ============================================================================
    # Plot 3: Render Loss Components
    # ============================================================================
    ax = axes[1, 0]

    components = {
        'Alpha': (filter_none(loss_alpha), '#E63946'),
        'Depth': (filter_none(loss_depth), '#F77F00'),
        'Photo': (filter_none(loss_photo), '#06A77D'),
        'Edge': (filter_none(loss_edge), '#118AB2'),
        'Cov Align': (filter_none(loss_cov_align), '#073B4C'),
    }

    has_data = False
    for name, (values, color) in components.items():
        if any(~np.isnan(values)):
            ax.plot(episodes, values, 'o-', linewidth=1.5, markersize=3, label=name, color=color, alpha=0.8)
            has_data = True

    if has_data:
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Loss Value', fontsize=11)
        ax.set_title('Render Loss Components', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    else:
        ax.text(0.5, 0.5, 'No component data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Render Components (No Data)', fontsize=12)

    # ============================================================================
    # Plot 4: Jacobian Statistics (Compression/Expansion)
    # ============================================================================
    ax = axes[1, 1]

    J_min_filtered = filter_none(J_min)
    J_mean_filtered = filter_none(J_mean)

    has_jacobian = any(~np.isnan(J_min_filtered)) or any(~np.isnan(J_mean_filtered))

    if has_jacobian:
        if any(~np.isnan(J_min_filtered)):
            ax.plot(episodes, J_min_filtered, 'o-', linewidth=1.5, markersize=4,
                   label='J_min (worst compression)', color='#E63946', alpha=0.8)
        if any(~np.isnan(J_mean_filtered)):
            ax.plot(episodes, J_mean_filtered, 'o-', linewidth=1.5, markersize=4,
                   label='J_mean (avg volume)', color='#2E86AB', alpha=0.8)

        # Add reference line at J=1.0 (no volume change)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='J=1 (neutral)')

        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('det(F) - Jacobian', fontsize=11)
        ax.set_title('Volume Preservation (det F)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

        # Add interpretation text
        ax.text(0.02, 0.98, 'J < 1: Compression\nJ > 1: Expansion\nJ = 1: Volume preserved',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No Jacobian data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Jacobian (No Data)', fontsize=12)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Plot] Saved loss curves to: {output_path}")


def plot_losses_standalone(json_path: str, output_path: Optional[str] = None):
    """
    Standalone plotting function for command-line use.

    Usage:
        python -c "from utils.plotting_utils import plot_losses_standalone; plot_losses_standalone('output/training_losses.json')"

    Args:
        json_path: Path to training_losses.json
        output_path: Optional output path (defaults to same directory as JSON)
    """
    json_path = Path(json_path)

    if output_path is None:
        output_path = json_path.parent / "loss_curves.png"
    else:
        output_path = Path(output_path)

    if not json_path.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        return

    plot_training_losses(json_path, output_path)
    print(f"✅ Plot saved to: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plotting_utils.py <json_path> [output_path]")
        print("Example: python plotting_utils.py output/spot/training_losses.json")
        sys.exit(1)

    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    plot_losses_standalone(json_path, output_path)
