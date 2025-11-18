#!/usr/bin/env python3
"""
Visualize Multi-scale F-field from Gaussian Covariances

Since Σ ∝ F · F^T, we can extract F-field information from the covariances.
This shows:
1. Covariance eigenvalues (principal deformations)
2. Anisotropy (stretch ratios)
3. Volume change (determinant)
4. Spatial distribution of deformation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys

def analyze_covariances(cov):
    """
    Extract deformation metrics from covariances.

    Args:
        cov: (N, 3, 3) covariance matrices

    Returns:
        Dictionary of metrics
    """
    N = len(cov)

    # Compute eigenvalues (principal axes of ellipsoid)
    eigenvalues = np.linalg.eigvalsh(cov)  # (N, 3), sorted ascending

    # Determinant (volume of ellipsoid)
    det = np.linalg.det(cov)

    # Frobenius norm (total deformation magnitude)
    frob_norm = np.linalg.norm(cov.reshape(N, -1), axis=1)

    # Anisotropy (ratio of max to min eigenvalue)
    anisotropy = eigenvalues[:, 2] / (eigenvalues[:, 0] + 1e-8)

    # Isotropy score (how spherical is the ellipsoid)
    isotropy = 1.0 / anisotropy

    return {
        'eigenvalues': eigenvalues,
        'det': det,
        'frob_norm': frob_norm,
        'anisotropy': anisotropy,
        'isotropy': isotropy,
        'lambda_min': eigenvalues[:, 0],
        'lambda_mid': eigenvalues[:, 1],
        'lambda_max': eigenvalues[:, 2]
    }


def visualize_F_field_metrics(mu, cov, output_path):
    """
    Visualize F-field-related metrics from Gaussian covariances.

    Args:
        mu: (N, 3) positions
        cov: (N, 3, 3) covariances
        output_path: Where to save
    """
    metrics = analyze_covariances(cov)
    N = len(mu)

    fig = plt.figure(figsize=(28, 20))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    fig.suptitle('Multi-scale F-field Visualization (from Gaussian Covariances)',
                 fontsize=18, fontweight='bold')

    # ============================================================================
    # Row 1: 3D Spatial Distribution of Deformation
    # ============================================================================

    # Create darker versions of colormaps
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    def darken_colormap(cmap_name, factor=0.7):
        """Create a darker version of a colormap."""
        original_cmap = plt.colormaps[cmap_name]
        colors = original_cmap(np.linspace(0, 1, 256))
        colors[:, :3] *= factor  # Darken RGB channels
        return mcolors.LinearSegmentedColormap.from_list(f'{cmap_name}_dark', colors)

    cmap_hot_dark = darken_colormap('hot', 0.7)
    cmap_viridis_dark = darken_colormap('viridis', 0.7)
    cmap_plasma_dark = darken_colormap('plasma', 0.7)
    cmap_coolwarm_dark = darken_colormap('coolwarm', 0.7)

    # Panel 1: Anisotropy (stretch ratio)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_box_aspect([1.5, 1, 1])  # Stretch X axis
    scatter1 = ax1.scatter(mu[:, 0], mu[:, 1], mu[:, 2],
                          c=metrics['anisotropy'], s=10, alpha=0.6,
                          cmap=cmap_hot_dark, vmin=1, vmax=np.percentile(metrics['anisotropy'], 95))
    ax1.set_title(f'Anisotropy (λ_max / λ_min)\nMean: {metrics["anisotropy"].mean():.2f}',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=25, azim=315)
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.5, pad=0.1)
    cbar1.set_label('Anisotropy', fontsize=10)

    # Panel 2: Volume (determinant)
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.set_box_aspect([1.5, 1, 1])  # Stretch X axis
    scatter2 = ax2.scatter(mu[:, 0], mu[:, 1], mu[:, 2],
                          c=np.log10(metrics['det'] + 1e-8), s=10, alpha=0.6,
                          cmap=cmap_viridis_dark)
    ax2.set_title(f'log₁₀(det(Σ)) - Volume\nMean: {metrics["det"].mean():.2e}',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=25, azim=315)
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.5, pad=0.1)
    cbar2.set_label('log₁₀(det)', fontsize=10)

    # Panel 3: Maximum principal axis (max deformation)
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax3.set_box_aspect([1.5, 1, 1])  # Stretch X axis
    scatter3 = ax3.scatter(mu[:, 0], mu[:, 1], mu[:, 2],
                          c=metrics['lambda_max'], s=10, alpha=0.6,
                          cmap=cmap_plasma_dark)
    ax3.set_title(f'λ_max (Maximum Stretch)\nMean: {metrics["lambda_max"].mean():.4f}',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.view_init(elev=25, azim=315)
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.5, pad=0.1)
    cbar3.set_label('λ_max', fontsize=10)

    # Panel 4: Frobenius norm (total deformation)
    ax4 = fig.add_subplot(gs[0, 3], projection='3d')
    ax4.set_box_aspect([1.5, 1, 1])  # Stretch X axis
    scatter4 = ax4.scatter(mu[:, 0], mu[:, 1], mu[:, 2],
                          c=metrics['frob_norm'], s=10, alpha=0.6,
                          cmap=cmap_coolwarm_dark)
    ax4.set_title(f'||Σ||_F (Total Deformation)\nMean: {metrics["frob_norm"].mean():.4f}',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.view_init(elev=25, azim=315)
    cbar4 = plt.colorbar(scatter4, ax=ax4, shrink=0.5, pad=0.1)
    cbar4.set_label('||Σ||_F', fontsize=10)

    # ============================================================================
    # Row 2: Distribution Histograms
    # ============================================================================

    # Panel 5: Eigenvalue distribution
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(metrics['lambda_min'], bins=50, alpha=0.6, color='blue', label='λ_min', density=True)
    ax5.hist(metrics['lambda_mid'], bins=50, alpha=0.6, color='green', label='λ_mid', density=True)
    ax5.hist(metrics['lambda_max'], bins=50, alpha=0.6, color='red', label='λ_max', density=True)
    ax5.set_title('Eigenvalue Distribution (Principal Axes)', fontweight='bold')
    ax5.set_xlabel('Eigenvalue λ')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, np.percentile(metrics['lambda_max'], 99))

    # Panel 6: Anisotropy distribution
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(metrics['anisotropy'], bins=50, alpha=0.7, color='orange', density=True)
    ax6.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Isotropic (ratio=1)')
    ax6.axvline(metrics['anisotropy'].mean(), color='red', linestyle='-', linewidth=2,
               label=f'Mean: {metrics["anisotropy"].mean():.2f}')
    ax6.set_title('Anisotropy Distribution', fontweight='bold')
    ax6.set_xlabel('λ_max / λ_min')
    ax6.set_ylabel('Density')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(1, min(20, np.percentile(metrics['anisotropy'], 99)))

    # Panel 7: Volume distribution
    ax7 = fig.add_subplot(gs[1, 2])
    det_filtered = metrics['det'][metrics['det'] > 0]
    ax7.hist(np.log10(det_filtered), bins=50, alpha=0.7, color='teal', density=True)
    ax7.axvline(np.log10(det_filtered.mean()), color='red', linestyle='-', linewidth=2,
               label=f'Mean: {det_filtered.mean():.2e}')
    ax7.set_title('Volume Distribution (det(Σ))', fontweight='bold')
    ax7.set_xlabel('log₁₀(det(Σ))')
    ax7.set_ylabel('Density')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # Panel 8: Frobenius norm distribution
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hist(metrics['frob_norm'], bins=50, alpha=0.7, color='purple', density=True)
    ax8.axvline(metrics['frob_norm'].mean(), color='red', linestyle='-', linewidth=2,
               label=f'Mean: {metrics["frob_norm"].mean():.4f}')
    ax8.set_title('Frobenius Norm Distribution', fontweight='bold')
    ax8.set_xlabel('||Σ||_F')
    ax8.set_ylabel('Density')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # ============================================================================
    # Row 3: 2D Projections and Statistics
    # ============================================================================

    # Panel 9: XY projection colored by anisotropy
    ax9 = fig.add_subplot(gs[2, 0])
    scatter9 = ax9.scatter(mu[:, 0], mu[:, 1], c=metrics['anisotropy'],
                          s=5, alpha=0.5, cmap='hot',
                          vmin=1, vmax=np.percentile(metrics['anisotropy'], 95))
    ax9.set_title('XY Projection: Anisotropy', fontweight='bold')
    ax9.set_xlabel('X')
    ax9.set_ylabel('Y')
    ax9.set_aspect('equal')
    ax9.grid(True, alpha=0.3)
    plt.colorbar(scatter9, ax=ax9, label='Anisotropy')

    # Panel 10: XZ projection colored by max eigenvalue
    ax10 = fig.add_subplot(gs[2, 1])
    scatter10 = ax10.scatter(mu[:, 0], mu[:, 2], c=metrics['lambda_max'],
                            s=5, alpha=0.5, cmap='plasma')
    ax10.set_title('XZ Projection: λ_max', fontweight='bold')
    ax10.set_xlabel('X')
    ax10.set_ylabel('Z')
    ax10.set_aspect('equal')
    ax10.grid(True, alpha=0.3)
    plt.colorbar(scatter10, ax=ax10, label='λ_max')

    # Panel 11: YZ projection colored by volume
    ax11 = fig.add_subplot(gs[2, 2])
    scatter11 = ax11.scatter(mu[:, 1], mu[:, 2],
                            c=np.log10(metrics['det'] + 1e-8),
                            s=5, alpha=0.5, cmap='viridis')
    ax11.set_title('YZ Projection: log₁₀(det)', fontweight='bold')
    ax11.set_xlabel('Y')
    ax11.set_ylabel('Z')
    ax11.set_aspect('equal')
    ax11.grid(True, alpha=0.3)
    plt.colorbar(scatter11, ax=ax11, label='log₁₀(det)')

    # Panel 12: Statistics summary
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    stats_text = f"""
    MULTI-SCALE F-FIELD STATISTICS

    Particle Count:
      • Total Gaussians: {N:,}

    Eigenvalues (Principal Stretches):
      • λ_min (mean): {metrics['lambda_min'].mean():.6f}
      • λ_mid (mean): {metrics['lambda_mid'].mean():.6f}
      • λ_max (mean): {metrics['lambda_max'].mean():.6f}

    Anisotropy:
      • Mean ratio: {metrics['anisotropy'].mean():.4f}
      • Median ratio: {np.median(metrics['anisotropy']):.4f}
      • 95th percentile: {np.percentile(metrics['anisotropy'], 95):.4f}

    Volume (Determinant):
      • Mean det(Σ): {metrics['det'].mean():.6e}
      • Std det(Σ): {metrics['det'].std():.6e}

    Deformation Magnitude:
      • Mean ||Σ||_F: {metrics['frob_norm'].mean():.6f}
      • Std ||Σ||_F: {metrics['frob_norm'].std():.6f}

    Note: Σ = F · F^T (covariance from F-field)
    """

    ax12.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
              verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved F-field visualization to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize F-field from Gaussian covariances')
    parser.add_argument('--episode_dir', type=str, required=True,
                        help='Path to episode directory (e.g., output/bunny_v2/ep000)')
    parser.add_argument('--max_particles', type=int, default=100000,
                        help='Maximum particles to visualize (downsample if larger)')
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)

    # Load Gaussian data
    gaussians_path = episode_dir / f"{episode_dir.name}_gaussians.npz"
    if not gaussians_path.exists():
        print(f"❌ Gaussians file not found: {gaussians_path}")
        return

    print(f"Loading: {gaussians_path}")
    data = np.load(gaussians_path)
    mu = data['mu']
    cov = data['cov']

    print(f"  • Particles: {len(mu):,}")
    print(f"  • mu shape: {mu.shape}")
    print(f"  • cov shape: {cov.shape}")

    # Downsample if too large
    if len(mu) > args.max_particles:
        print(f"  • Downsampling from {len(mu):,} to {args.max_particles:,} particles...")
        indices = np.random.choice(len(mu), args.max_particles, replace=False)
        mu = mu[indices]
        cov = cov[indices]
        print(f"  • Downsampled: {len(mu):,} particles")

    # Generate visualization
    output_path = episode_dir / "F_field_multiscale.png"
    print(f"\nGenerating F-field visualization...")
    visualize_F_field_metrics(mu, cov, output_path)

    print(f"\n✅ Done! Check: {output_path}")


if __name__ == "__main__":
    main()
