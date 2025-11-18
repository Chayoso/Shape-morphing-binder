"""
Visualize Multi-scale F-field and Upsampling Process

This script visualizes:
1. Multi-scale F-field interpolation (coarse MPM → fine rendering)
2. Upsampling stages (clustering, subdivision, interpolation)
3. Particle density evolution
4. Deformation gradient magnitudes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import torch
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sampling import upsample
from utils.io_utils import load_point_cloud


def visualize_multiscale_F(x_coarse, F_coarse, x_fine, F_fine, output_path):
    """
    Visualize multi-scale F-field interpolation.

    Args:
        x_coarse: (N_coarse, 3) coarse particle positions (from MPM)
        F_coarse: (N_coarse, 3, 3) coarse deformation gradients
        x_fine: (N_fine, 3) fine particle positions (after upsampling)
        F_fine: (N_fine, 3, 3) fine deformation gradients (interpolated)
        output_path: Where to save visualization
    """
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Title
    fig.suptitle('Multi-scale F-field Interpolation & Upsampling',
                 fontsize=16, fontweight='bold')

    # ============================================================================
    # Row 1: Particle Distribution (2D projection)
    # ============================================================================

    # Coarse particles (MPM)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x_coarse[:, 0], x_coarse[:, 1], c='blue', s=30, alpha=0.6, label='Coarse MPM')
    ax1.set_title(f'Coarse Particles (MPM)\nN = {len(x_coarse)}', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Fine particles (after upsampling)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(x_fine[:, 0], x_fine[:, 1], c='red', s=10, alpha=0.4, label='Fine Upsampled')
    ax2.set_title(f'Fine Particles (Upsampled)\nN = {len(x_fine)}', fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Overlay comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(x_fine[:, 0], x_fine[:, 1], c='red', s=5, alpha=0.3, label='Fine')
    ax3.scatter(x_coarse[:, 0], x_coarse[:, 1], c='blue', s=50, alpha=0.8,
                edgecolors='black', linewidths=1, label='Coarse', zorder=10)
    ax3.set_title(f'Overlay: Coarse → Fine\nUpsampling Ratio: {len(x_fine)/len(x_coarse):.1f}x',
                  fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Particle density histogram
    ax4 = fig.add_subplot(gs[0, 3])
    from scipy.spatial import cKDTree
    tree_fine = cKDTree(x_fine[:, :2])
    distances, _ = tree_fine.query(x_fine[:, :2], k=10)
    mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self
    ax4.hist(mean_distances, bins=50, alpha=0.7, color='red', label='Fine', density=True)

    if len(x_coarse) > 10:
        tree_coarse = cKDTree(x_coarse[:, :2])
        distances_c, _ = tree_coarse.query(x_coarse[:, :2], k=min(10, len(x_coarse)))
        mean_distances_c = distances_c[:, 1:].mean(axis=1)
        ax4.hist(mean_distances_c, bins=30, alpha=0.7, color='blue', label='Coarse', density=True)

    ax4.set_title('Particle Spacing Distribution', fontweight='bold')
    ax4.set_xlabel('Mean Distance to 10 Neighbors')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ============================================================================
    # Row 2: F-field Magnitude Visualization
    # ============================================================================

    # Compute F magnitudes (Frobenius norm)
    F_coarse_mag = np.linalg.norm(F_coarse.reshape(len(F_coarse), -1), axis=1)
    F_fine_mag = np.linalg.norm(F_fine.reshape(len(F_fine), -1), axis=1)

    # Coarse F magnitude
    ax5 = fig.add_subplot(gs[1, 0])
    scatter1 = ax5.scatter(x_coarse[:, 0], x_coarse[:, 1], c=F_coarse_mag,
                           s=100, cmap='viridis', alpha=0.8, edgecolors='black', linewidths=0.5)
    ax5.set_title('Coarse F Magnitude (||F||_F)', fontweight='bold')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    plt.colorbar(scatter1, ax=ax5, label='||F||')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')

    # Fine F magnitude
    ax6 = fig.add_subplot(gs[1, 1])
    scatter2 = ax6.scatter(x_fine[:, 0], x_fine[:, 1], c=F_fine_mag,
                           s=20, cmap='viridis', alpha=0.6)
    ax6.set_title('Fine F Magnitude (Interpolated)', fontweight='bold')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    plt.colorbar(scatter2, ax=ax6, label='||F||')
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')

    # F magnitude comparison histogram
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.hist(F_coarse_mag, bins=30, alpha=0.6, color='blue', label='Coarse', density=True)
    ax7.hist(F_fine_mag, bins=50, alpha=0.6, color='red', label='Fine', density=True)
    ax7.axvline(np.mean(F_coarse_mag), color='blue', linestyle='--', linewidth=2, label=f'Coarse Mean: {np.mean(F_coarse_mag):.3f}')
    ax7.axvline(np.mean(F_fine_mag), color='red', linestyle='--', linewidth=2, label=f'Fine Mean: {np.mean(F_fine_mag):.3f}')
    ax7.set_title('F Magnitude Distribution', fontweight='bold')
    ax7.set_xlabel('||F||_F')
    ax7.set_ylabel('Density')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # Determinant (volume change)
    ax8 = fig.add_subplot(gs[1, 3])
    det_coarse = np.linalg.det(F_coarse)
    det_fine = np.linalg.det(F_fine)
    ax8.hist(det_coarse, bins=30, alpha=0.6, color='blue', label='Coarse', density=True)
    ax8.hist(det_fine, bins=50, alpha=0.6, color='red', label='Fine', density=True)
    ax8.axvline(1.0, color='green', linestyle='--', linewidth=2, label='det(F)=1 (no volume change)')
    ax8.axvline(np.mean(det_coarse), color='blue', linestyle=':', linewidth=2, alpha=0.7)
    ax8.axvline(np.mean(det_fine), color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax8.set_title('Determinant Distribution (Volume Change)', fontweight='bold')
    ax8.set_xlabel('det(F)')
    ax8.set_ylabel('Density')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # ============================================================================
    # Row 3: F-field Directional Analysis
    # ============================================================================

    # Compute principal stretches (eigenvalues of F^T F)
    def compute_principal_stretches(F):
        """Compute principal stretches from F."""
        FTF = F.transpose(0, 2, 1) @ F  # (N, 3, 3)
        eigenvalues = np.linalg.eigvalsh(FTF)  # (N, 3)
        stretches = np.sqrt(np.abs(eigenvalues))  # λ = sqrt(eigenvalue)
        return stretches

    stretches_coarse = compute_principal_stretches(F_coarse)
    stretches_fine = compute_principal_stretches(F_fine)

    # Plot principal stretch distributions
    ax9 = fig.add_subplot(gs[2, 0])
    for i, label in enumerate(['λ₁ (max)', 'λ₂ (mid)', 'λ₃ (min)']):
        ax9.hist(stretches_coarse[:, -(i+1)], bins=30, alpha=0.5, label=f'Coarse {label}')
    ax9.axvline(1.0, color='black', linestyle='--', linewidth=2, label='λ=1 (no stretch)')
    ax9.set_title('Principal Stretches (Coarse)', fontweight='bold')
    ax9.set_xlabel('Stretch λ')
    ax9.set_ylabel('Count')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    ax10 = fig.add_subplot(gs[2, 1])
    for i, label in enumerate(['λ₁ (max)', 'λ₂ (mid)', 'λ₃ (min)']):
        ax10.hist(stretches_fine[:, -(i+1)], bins=50, alpha=0.5, label=f'Fine {label}')
    ax10.axvline(1.0, color='black', linestyle='--', linewidth=2, label='λ=1 (no stretch)')
    ax10.set_title('Principal Stretches (Fine)', fontweight='bold')
    ax10.set_xlabel('Stretch λ')
    ax10.set_ylabel('Count')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)

    # Anisotropy measure (ratio of max/min stretch)
    anisotropy_coarse = stretches_coarse[:, 2] / (stretches_coarse[:, 0] + 1e-6)
    anisotropy_fine = stretches_fine[:, 2] / (stretches_fine[:, 0] + 1e-6)

    ax11 = fig.add_subplot(gs[2, 2])
    ax11.hist(anisotropy_coarse, bins=30, alpha=0.6, color='blue', label='Coarse', density=True)
    ax11.hist(anisotropy_fine, bins=50, alpha=0.6, color='red', label='Fine', density=True)
    ax11.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Ratio=1 (isotropic)')
    ax11.set_title('Anisotropy (λ_max / λ_min)', fontweight='bold')
    ax11.set_xlabel('Stretch Ratio')
    ax11.set_ylabel('Density')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    ax11.set_xlim(0, 5)

    # Statistics summary
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    stats_text = f"""
    MULTI-SCALE F-FIELD STATISTICS

    Coarse (MPM):
      • Particles: {len(x_coarse)}
      • ||F|| mean: {np.mean(F_coarse_mag):.4f}
      • det(F) mean: {np.mean(det_coarse):.4f}
      • det(F) std: {np.std(det_coarse):.4f}
      • Anisotropy: {np.mean(anisotropy_coarse):.4f}

    Fine (Upsampled):
      • Particles: {len(x_fine)}
      • ||F|| mean: {np.mean(F_fine_mag):.4f}
      • det(F) mean: {np.mean(det_fine):.4f}
      • det(F) std: {np.std(det_fine):.4f}
      • Anisotropy: {np.mean(anisotropy_fine):.4f}

    Upsampling:
      • Ratio: {len(x_fine)/len(x_coarse):.2f}x
      • F interpolation: Multi-scale RBF
      • Volume preservation: {np.abs(np.mean(det_fine) - np.mean(det_coarse)):.4f}
    """

    ax12.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round',
              facecolor='wheat', alpha=0.5))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved multi-scale F-field visualization to: {output_path}")


def visualize_upsampling_stages(x_stages, F_stages, stage_names, output_path):
    """
    Visualize intermediate stages of upsampling process.

    Args:
        x_stages: List of (N_i, 3) particle positions at each stage
        F_stages: List of (N_i, 3, 3) F-fields at each stage
        stage_names: List of stage names
        output_path: Where to save
    """
    n_stages = len(x_stages)
    fig, axes = plt.subplots(2, n_stages, figsize=(6*n_stages, 12))

    fig.suptitle('Upsampling Pipeline: Stage-by-Stage', fontsize=16, fontweight='bold')

    for i, (x, F, name) in enumerate(zip(x_stages, F_stages, stage_names)):
        # Top row: Particle positions
        ax_pos = axes[0, i] if n_stages > 1 else axes[0]
        ax_pos.scatter(x[:, 0], x[:, 1], s=20, alpha=0.6, c=np.arange(len(x)), cmap='tab20')
        ax_pos.set_title(f'{name}\nN = {len(x)}', fontweight='bold')
        ax_pos.set_xlabel('X')
        ax_pos.set_ylabel('Y')
        ax_pos.set_aspect('equal')
        ax_pos.grid(True, alpha=0.3)

        # Bottom row: F magnitude
        ax_F = axes[1, i] if n_stages > 1 else axes[1]
        F_mag = np.linalg.norm(F.reshape(len(F), -1), axis=1)
        scatter = ax_F.scatter(x[:, 0], x[:, 1], c=F_mag, s=30, alpha=0.7, cmap='viridis')
        ax_F.set_title(f'||F|| (mean: {np.mean(F_mag):.3f})', fontweight='bold')
        ax_F.set_xlabel('X')
        ax_F.set_ylabel('Y')
        ax_F.set_aspect('equal')
        ax_F.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax_F, label='||F||')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved upsampling stages to: {output_path}")


def main():
    """
    Main visualization function.
    Loads data from a training episode and visualizes multi-scale F-field.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Visualize multi-scale F-field and upsampling')
    parser.add_argument('--episode_dir', type=str, required=True,
                        help='Path to episode directory (e.g., output/bunny/ep000)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for visualizations (default: same as episode_dir)')
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    output_dir = Path(args.output_dir) if args.output_dir else episode_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Multi-scale F-field Visualization")
    print(f"{'='*70}")
    print(f"Episode directory: {episode_dir}")
    print(f"Output directory: {output_dir}")

    # Load coarse state (from MPM)
    coarse_npz = episode_dir / "state_coarse.npz"
    if not coarse_npz.exists():
        print(f"[ERROR] Coarse state not found: {coarse_npz}")
        print("Please ensure the episode has been saved with export_stages=True")
        return

    coarse_data = np.load(coarse_npz)
    x_coarse = coarse_data['x']
    F_coarse = coarse_data['F']

    print(f"\n[Coarse State]")
    print(f"  Particles: {len(x_coarse)}")
    print(f"  F shape: {F_coarse.shape}")

    # Load fine state (after upsampling)
    fine_npz = episode_dir / "state_fine.npz"
    if not fine_npz.exists():
        print(f"[ERROR] Fine state not found: {fine_npz}")
        return

    fine_data = np.load(fine_npz)
    x_fine = fine_data['x']
    F_fine = fine_data['F']

    print(f"\n[Fine State]")
    print(f"  Particles: {len(x_fine)}")
    print(f"  F shape: {F_fine.shape}")
    print(f"  Upsampling ratio: {len(x_fine)/len(x_coarse):.2f}x")

    # Generate visualizations
    print(f"\n[Generating Visualizations]")

    # Main multi-scale F visualization
    output_path = output_dir / "multiscale_F_visualization.png"
    visualize_multiscale_F(x_coarse, F_coarse, x_fine, F_fine, output_path)

    # Check for intermediate stages
    stages_found = []
    stage_names_found = []

    for stage_file in sorted(episode_dir.glob("state_stage*.npz")):
        stage_data = np.load(stage_file)
        stages_found.append((stage_data['x'], stage_data['F']))
        stage_name = stage_file.stem.replace('state_', '').replace('_', ' ').title()
        stage_names_found.append(stage_name)

    if stages_found:
        print(f"  Found {len(stages_found)} intermediate stages")
        x_stages = [s[0] for s in stages_found]
        F_stages = [s[1] for s in stages_found]
        output_path = output_dir / "upsampling_stages.png"
        visualize_upsampling_stages(x_stages, F_stages, stage_names_found, output_path)
    else:
        print(f"  No intermediate stages found (set export_stages=True to save them)")

    print(f"\n{'='*70}")
    print(f"✅ Visualization complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
