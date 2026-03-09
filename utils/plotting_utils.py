"""
Plotting Utilities - Loss Visualization

Utilities for plotting training losses and metrics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def plot_training_losses(json_path: Path, output_path: Path, figsize=(18, 14)):
    """
    Plot physics and render losses from training JSON log.

    Layout (3×3 if dFc data present, else 2×3):
    Row 1: Physics Loss | Volume (J) | Render Total
    Row 2: Alpha Loss | Depth Loss | Edge/Cov Align
    Row 3: dFc J before/after | dFc Step Norm | Surface vs Interior Grads

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
    loss_edge = [ep.get('loss_edge', None) for ep in episodes_data]
    loss_cov_align = [ep.get('loss_cov_align', None) for ep in episodes_data]

    # Filter out None values
    def filter_none(lst):
        return [x if x is not None else np.nan for x in lst]

    # Extract additional data
    J_min = [ep.get('J_min', None) for ep in episodes_data]
    J_mean = [ep.get('J_mean', None) for ep in episodes_data]

    # Check if dFc tracking data exists
    has_dfc = any(ep.get('J_min_before_dfc') is not None for ep in episodes_data)
    nrows = 3 if has_dfc else 2
    fig, axes = plt.subplots(nrows, 3, figsize=figsize)
    fig.suptitle(f'Training Losses - {data["config"]}', fontsize=16, fontweight='bold', y=0.995)

    # ============================================================================
    # Row 1, Col 1: Physics Loss
    # ============================================================================
    ax = axes[0, 0]
    physics_filtered = filter_none(loss_physics)
    if any(~np.isnan(physics_filtered)):
        ax.plot(episodes, physics_filtered, 'o-', linewidth=2.5, markersize=5,
                label='Physics Loss', color='#2E86AB')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('Physics Loss', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics box
        valid_physics = [x for x in physics_filtered if not np.isnan(x)]
        if valid_physics:
            reduction_pct = ((valid_physics[0] - valid_physics[-1]) / valid_physics[0] * 100)
            ax.text(0.98, 0.97,
                   f'Start: {valid_physics[0]:.1f}\nEnd: {valid_physics[-1]:.1f}\n↓ {reduction_pct:.1f}%',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='#2E86AB', linewidth=2))
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        ax.set_title('Physics Loss (No Data)', fontsize=13)

    # ============================================================================
    # Row 1, Col 2: Volume (Jacobian)
    # ============================================================================
    ax = axes[0, 1]
    J_min_filtered = filter_none(J_min)
    J_mean_filtered = filter_none(J_mean)

    has_jacobian = any(~np.isnan(J_min_filtered)) or any(~np.isnan(J_mean_filtered))

    if has_jacobian:
        if any(~np.isnan(J_min_filtered)):
            ax.plot(episodes, J_min_filtered, 'o-', linewidth=2.5, markersize=5,
                   label='J_min (max compression)', color='#E63946')
        if any(~np.isnan(J_mean_filtered)):
            ax.plot(episodes, J_mean_filtered, 's-', linewidth=2.5, markersize=5,
                   label='J_mean (avg volume)', color='#06A77D')

        # Reference line at J=1.0
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.6, label='J=1 (no change)')

        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('det(F)', fontsize=12)
        ax.set_title('Volume Preservation', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)

        # Set y-axis limits to focus on relevant range
        all_j = J_min_filtered + J_mean_filtered
        valid_j = [x for x in all_j if not np.isnan(x)]
        if valid_j:
            j_range = max(valid_j) - min(valid_j)
            ax.set_ylim([min(valid_j) - 0.1*j_range, max(valid_j) + 0.1*j_range])
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        ax.set_title('Volume (No Data)', fontsize=13)

    # ============================================================================
    # Row 1, Col 3: Render Loss (Total)
    # ============================================================================
    ax = axes[0, 2]
    render_filtered = filter_none(loss_render_total)
    if any(~np.isnan(render_filtered)):
        ax.plot(episodes, render_filtered, 'o-', linewidth=2.5, markersize=5,
                label='Total Render', color='#A23B72')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('Render Loss (Total)', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics box
        valid_render = [x for x in render_filtered if not np.isnan(x)]
        if valid_render:
            reduction_pct = ((valid_render[0] - valid_render[-1]) / valid_render[0] * 100)
            ax.text(0.98, 0.97,
                   f'Start: {valid_render[0]:.2f}\nEnd: {valid_render[-1]:.2f}\n↓ {reduction_pct:.1f}%',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7, edgecolor='#A23B72', linewidth=2))
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        ax.set_title('Render Loss (No Data)', fontsize=13)

    # ============================================================================
    # Row 2, Col 1: Alpha Loss
    # ============================================================================
    ax = axes[1, 0]
    alpha_filtered = filter_none(loss_alpha)
    if any(~np.isnan(alpha_filtered)):
        ax.plot(episodes, alpha_filtered, 'o-', linewidth=2.5, markersize=5,
                label='Alpha (Silhouette)', color='#E63946')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('Alpha Loss', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics
        valid_alpha = [x for x in alpha_filtered if not np.isnan(x)]
        if valid_alpha:
            reduction_pct = ((valid_alpha[0] - valid_alpha[-1]) / valid_alpha[0] * 100)
            ax.text(0.98, 0.97,
                   f'Start: {valid_alpha[0]:.3f}\nEnd: {valid_alpha[-1]:.3f}\n↓ {reduction_pct:.1f}%',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7, edgecolor='#E63946', linewidth=2))
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        ax.set_title('Alpha Loss (No Data)', fontsize=13)

    # ============================================================================
    # Row 2, Col 2: Depth Loss
    # ============================================================================
    ax = axes[1, 1]
    depth_filtered = filter_none(loss_depth)
    if any(~np.isnan(depth_filtered)):
        ax.plot(episodes, depth_filtered, 'o-', linewidth=2.5, markersize=5,
                label='Depth (3D Structure)', color='#F77F00')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('Depth Loss', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics
        valid_depth = [x for x in depth_filtered if not np.isnan(x)]
        if valid_depth:
            reduction_pct = ((valid_depth[0] - valid_depth[-1]) / valid_depth[0] * 100)
            ax.text(0.98, 0.97,
                   f'Start: {valid_depth[0]:.2f}\nEnd: {valid_depth[-1]:.2f}\n↓ {reduction_pct:.1f}%',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.7, edgecolor='#F77F00', linewidth=2))
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        ax.set_title('Depth Loss (No Data)', fontsize=13)

    # ============================================================================
    # Row 2, Col 3: Edge / Cov Align
    # ============================================================================
    ax = axes[1, 2]
    edge_filtered = filter_none(loss_edge)
    cov_filtered = filter_none(loss_cov_align)

    has_edge_cov = any(~np.isnan(edge_filtered)) or any(~np.isnan(cov_filtered))

    if has_edge_cov:
        if any(~np.isnan(edge_filtered)):
            ax.plot(episodes, edge_filtered, 'o-', linewidth=2.5, markersize=5,
                   label='Edge (Sharpness)', color='#118AB2', alpha=0.85)
        if any(~np.isnan(cov_filtered)):
            ax.plot(episodes, cov_filtered, 's-', linewidth=2.5, markersize=5,
                   label='Cov Align (Curvature)', color='#073B4C', alpha=0.85)

        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('Edge / Covariance Align', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
    else:
        ax.text(0.5, 0.5, 'No Edge/Cov Data\n(Losses disabled)', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Edge / Cov Align (No Data)', fontsize=13)

    # ============================================================================
    # Row 3: dFc Correction Tracking (only if data present)
    # ============================================================================
    if has_dfc:
        # Row 3, Col 1: J before/after dFc correction
        ax = axes[2, 0]
        j_before = filter_none([ep.get('J_min_before_dfc') for ep in episodes_data])
        j_after = filter_none([ep.get('J_min_after_dfc') for ep in episodes_data])
        if any(~np.isnan(j_before)):
            ax.plot(episodes, j_before, 'o-', linewidth=2, markersize=4,
                    label='J_min before dFc', color='#E63946')
        if any(~np.isnan(j_after)):
            ax.plot(episodes, j_after, 's-', linewidth=2, markersize=4,
                    label='J_min after dFc', color='#06A77D')
        ax.axhline(y=0.3, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='J guard threshold')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('det(F)', fontsize=12)
        ax.set_title('dFc: J_min Before/After', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)

        # Row 3, Col 2: dFc step norm
        ax = axes[2, 1]
        step_norm = filter_none([ep.get('dfc_step_norm') for ep in episodes_data])
        j_guard = filter_none([ep.get('dfc_J_guard_count', 0) for ep in episodes_data])
        if any(~np.isnan(step_norm)):
            ax.plot(episodes, step_norm, 'o-', linewidth=2, markersize=4,
                    label='||dFc step||', color='#2E86AB')
            ax.set_ylabel('Step Norm', fontsize=12, color='#2E86AB')

            # Secondary axis for J guard count
            ax2 = ax.twinx()
            if any(~np.isnan(j_guard)):
                ax2.bar(episodes, j_guard, alpha=0.3, color='#E63946', label='J guard clipped')
                ax2.set_ylabel('Guard Count', fontsize=10, color='#E63946')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_title('dFc: Step Norm & Guard', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Row 3, Col 3: Surface vs Interior gradient norms
        ax = axes[2, 2]
        g_surface = filter_none([ep.get('dfc_grad_norm_surface') for ep in episodes_data])
        g_interior = filter_none([ep.get('dfc_grad_norm_interior') for ep in episodes_data])
        if any(~np.isnan(g_surface)):
            ax.plot(episodes, g_surface, 'o-', linewidth=2, markersize=4,
                    label='Surface grad norm', color='#118AB2')
        if any(~np.isnan(g_interior)):
            ax.plot(episodes, g_interior, 's-', linewidth=2, markersize=4,
                    label='Interior grad norm', color='#073B4C', alpha=0.7)
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Grad Norm', fontsize=12)
        ax.set_title('dFc: Surface vs Interior Grads', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.99])
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
