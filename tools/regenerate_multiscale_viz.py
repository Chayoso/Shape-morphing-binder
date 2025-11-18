"""
Regenerate Multi-scale F-field visualization with adjusted aspect ratio
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.spatial import cKDTree


def visualize_multiscale_F(x_coarse, F_coarse, x_fine, F_fine, output_path):
    """
    Visualize multi-scale F-field interpolation with better aspect ratio.
    """
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Title
    fig.suptitle('Multi-scale F-field Interpolation & Upsampling',
                 fontsize=18, fontweight='bold')

    # ============================================================================
    # Row 1: Particle Distribution (2D projection)
    # ============================================================================

    # Coarse particles (MPM)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x_coarse[:, 0], x_coarse[:, 1], c='blue', s=30, alpha=0.6, label='Coarse MPM')
    ax1.set_title(f'Coarse Particles (MPM)\nN = {len(x_coarse)}', fontweight='bold', fontsize=12)
    ax1.set_xlabel('X', fontsize=11)
    ax1.set_ylabel('Y', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Fine particles (after upsampling)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(x_fine[:, 0], x_fine[:, 1], c='red', s=10, alpha=0.4, label='Fine Upsampled')
    ax2.set_title(f'Fine Particles (Upsampled)\nN = {len(x_fine)}', fontweight='bold', fontsize=12)
    ax2.set_xlabel('X', fontsize=11)
    ax2.set_ylabel('Y', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Overlay comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(x_fine[:, 0], x_fine[:, 1], c='red', s=5, alpha=0.3, label='Fine')
    ax3.scatter(x_coarse[:, 0], x_coarse[:, 1], c='blue', s=50, alpha=0.8,
                edgecolors='black', linewidths=1, label='Coarse', zorder=10)
    ax3.set_title(f'Overlay: Coarse → Fine\nUpsampling Ratio: {len(x_fine)/len(x_coarse):.1f}x',
                  fontweight='bold', fontsize=12)
    ax3.set_xlabel('X', fontsize=11)
    ax3.set_ylabel('Y', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Particle density histogram
    ax4 = fig.add_subplot(gs[0, 3])
    tree_fine = cKDTree(x_fine[:, :2])
    distances, _ = tree_fine.query(x_fine[:, :2], k=10)
    mean_distances = distances[:, 1:].mean(axis=1)
    ax4.hist(mean_distances, bins=50, alpha=0.7, color='red', label='Fine', density=True)

    if len(x_coarse) > 10:
        tree_coarse = cKDTree(x_coarse[:, :2])
        distances_c, _ = tree_coarse.query(x_coarse[:, :2], k=min(10, len(x_coarse)))
        mean_distances_c = distances_c[:, 1:].mean(axis=1)
        ax4.hist(mean_distances_c, bins=30, alpha=0.7, color='blue', label='Coarse', density=True)

    ax4.set_title('Particle Spacing Distribution', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Mean Distance to 10 Neighbors', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
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
    ax5.set_title('Coarse F Magnitude (||F||_F)', fontweight='bold', fontsize=12)
    ax5.set_xlabel('X', fontsize=11)
    ax5.set_ylabel('Y', fontsize=11)
    plt.colorbar(scatter1, ax=ax5, label='||F||')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')

    # Fine F magnitude
    ax6 = fig.add_subplot(gs[1, 1])
    scatter2 = ax6.scatter(x_fine[:, 0], x_fine[:, 1], c=F_fine_mag,
                           s=20, cmap='viridis', alpha=0.6)
    ax6.set_title('Fine F Magnitude (Interpolated)', fontweight='bold', fontsize=12)
    ax6.set_xlabel('X', fontsize=11)
    ax6.set_ylabel('Y', fontsize=11)
    plt.colorbar(scatter2, ax=ax6, label='||F||')
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')

    # F magnitude comparison histogram
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.hist(F_coarse_mag, bins=30, alpha=0.6, color='blue', label='Coarse', density=True)
    ax7.hist(F_fine_mag, bins=50, alpha=0.6, color='red', label='Fine', density=True)
    ax7.axvline(np.mean(F_coarse_mag), color='blue', linestyle='--', linewidth=2, label=f'Coarse Mean: {np.mean(F_coarse_mag):.3f}')
    ax7.axvline(np.mean(F_fine_mag), color='red', linestyle='--', linewidth=2, label=f'Fine Mean: {np.mean(F_fine_mag):.3f}')
    ax7.set_title('F Magnitude Distribution', fontweight='bold', fontsize=12)
    ax7.set_xlabel('||F||_F', fontsize=11)
    ax7.set_ylabel('Density', fontsize=11)
    ax7.legend(fontsize=9)
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
    ax8.set_title('Determinant Distribution (Volume Change)', fontweight='bold', fontsize=12)
    ax8.set_xlabel('det(F)', fontsize=11)
    ax8.set_ylabel('Density', fontsize=11)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # ============================================================================
    # Row 3: F-field Directional Analysis
    # ============================================================================

    # Compute principal stretches
    def compute_principal_stretches(F):
        FTF = F.transpose(0, 2, 1) @ F
        eigenvalues = np.linalg.eigvalsh(FTF)
        stretches = np.sqrt(np.abs(eigenvalues))
        return stretches

    stretches_coarse = compute_principal_stretches(F_coarse)
    stretches_fine = compute_principal_stretches(F_fine)

    # Plot principal stretch distributions
    ax9 = fig.add_subplot(gs[2, 0])
    for i, label in enumerate(['λ₁ (max)', 'λ₂ (mid)', 'λ₃ (min)']):
        ax9.hist(stretches_coarse[:, -(i+1)], bins=30, alpha=0.5, label=f'Coarse {label}')
    ax9.axvline(1.0, color='black', linestyle='--', linewidth=2, label='λ=1 (no stretch)')
    ax9.set_title('Principal Stretches (Coarse)', fontweight='bold', fontsize=12)
    ax9.set_xlabel('Stretch λ', fontsize=11)
    ax9.set_ylabel('Count', fontsize=11)
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)

    ax10 = fig.add_subplot(gs[2, 1])
    for i, label in enumerate(['λ₁ (max)', 'λ₂ (mid)', 'λ₃ (min)']):
        ax10.hist(stretches_fine[:, -(i+1)], bins=50, alpha=0.5, label=f'Fine {label}')
    ax10.axvline(1.0, color='black', linestyle='--', linewidth=2, label='λ=1 (no stretch)')
    ax10.set_title('Principal Stretches (Fine)', fontweight='bold', fontsize=12)
    ax10.set_xlabel('Stretch λ', fontsize=11)
    ax10.set_ylabel('Count', fontsize=11)
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3)

    # Anisotropy measure
    anisotropy_coarse = stretches_coarse[:, 2] / (stretches_coarse[:, 0] + 1e-6)
    anisotropy_fine = stretches_fine[:, 2] / (stretches_fine[:, 0] + 1e-6)

    ax11 = fig.add_subplot(gs[2, 2])
    ax11.hist(anisotropy_coarse, bins=30, alpha=0.6, color='blue', label='Coarse', density=True)
    ax11.hist(anisotropy_fine, bins=50, alpha=0.6, color='red', label='Fine', density=True)
    ax11.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Ratio=1 (isotropic)')
    ax11.set_title('Anisotropy (λ_max / λ_min)', fontweight='bold', fontsize=12)
    ax11.set_xlabel('Stretch Ratio', fontsize=11)
    ax11.set_ylabel('Density', fontsize=11)
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

    ax12.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round',
              facecolor='wheat', alpha=0.5))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization: {output_path}")


def main():
    episode_dir = Path("output/physics/bunny_1/ep001")

    print(f"\n{'='*70}")
    print(f"Regenerating Multi-scale F-field Visualization")
    print(f"Episode: {episode_dir}")
    print(f"{'='*70}")

    # Load coarse state
    coarse_npz = episode_dir / "state_coarse.npz"
    if not coarse_npz.exists():
        print(f"ERROR: Coarse state not found: {coarse_npz}")
        return

    coarse_data = np.load(coarse_npz)
    x_coarse = coarse_data['x']
    F_coarse = coarse_data['F']
    print(f"\n✓ Loaded coarse state: {len(x_coarse)} particles")

    # Load fine state
    fine_npz = episode_dir / "state_fine.npz"
    if not fine_npz.exists():
        print(f"ERROR: Fine state not found: {fine_npz}")
        return

    fine_data = np.load(fine_npz)
    x_fine = fine_data['x']
    F_fine = fine_data['F']
    print(f"✓ Loaded fine state: {len(x_fine)} particles")
    print(f"✓ Upsampling ratio: {len(x_fine)/len(x_coarse):.2f}x")

    # Generate visualization
    output_path = episode_dir / "F_field_multiscale.png"
    print(f"\n→ Generating visualization with improved aspect ratio (24×18)...")
    visualize_multiscale_F(x_coarse, F_coarse, x_fine, F_fine, output_path)

    print(f"\n{'='*70}")
    print(f"✅ Complete! Visualization saved with improved aspect ratio")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
