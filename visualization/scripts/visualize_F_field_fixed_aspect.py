#!/usr/bin/env python3
"""
Visualize Multi-scale F-field from Gaussian Covariances with fixed aspect ratio.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys


def analyze_covariances(cov):
    """Extract deformation metrics from covariances."""
    N = len(cov)

    eigenvalues = np.linalg.eigvalsh(cov)
    det = np.linalg.det(cov)
    frob_norm = np.linalg.norm(cov.reshape(N, -1), axis=1)
    anisotropy = eigenvalues[:, 2] / (eigenvalues[:, 0] + 1e-8)

    return {
        'eigenvalues': eigenvalues,
        'lambda_min': eigenvalues[:, 0],
        'lambda_mid': eigenvalues[:, 1],
        'lambda_max': eigenvalues[:, 2],
        'det': det,
        'frob_norm': frob_norm,
        'anisotropy': anisotropy,
    }


def set_3d_equal_aspect(ax, X, Y, Z):
    """Set equal aspect ratio for 3D plot."""
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])


def visualize_F_field(mu, cov, save_path):
    """Create comprehensive F-field visualization with fixed aspect ratio."""

    # Subsample if too many particles
    max_particles = 100000
    if len(mu) > max_particles:
        indices = np.random.choice(len(mu), max_particles, replace=False)
        mu = mu[indices]
        cov = cov[indices]

    # Analyze
    metrics = analyze_covariances(cov)

    # Create figure
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle('Multi-scale F-field Visualization (from Gaussian Covariances)',
                 fontsize=18, fontweight='bold')

    # ========================================================================
    # Row 1: 3D Spatial Views with FIXED ASPECT RATIO
    # ========================================================================

    # Panel 1: Anisotropy
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    scatter1 = ax1.scatter(mu[:, 0], mu[:, 1], mu[:, 2],
                          c=metrics['anisotropy'], s=10, alpha=0.6,
                          cmap='hot', vmin=1, vmax=np.percentile(metrics['anisotropy'], 95))
    ax1.set_title(f'Anisotropy (λ_max / λ_min)\nMean: {metrics["anisotropy"].mean():.2f}',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    set_3d_equal_aspect(ax1, mu[:, 0], mu[:, 1], mu[:, 2])
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, pad=0.1)

    # Panel 2: Volume
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    scatter2 = ax2.scatter(mu[:, 0], mu[:, 1], mu[:, 2],
                          c=np.log10(metrics['det'] + 1e-8), s=10, alpha=0.6,
                          cmap='viridis')
    ax2.set_title(f'log₁₀(det(Σ)) - Volume\nMean: {metrics["det"].mean():.2e}',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    set_3d_equal_aspect(ax2, mu[:, 0], mu[:, 1], mu[:, 2])
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, pad=0.1)

    # Panel 3: Max eigenvalue
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    scatter3 = ax3.scatter(mu[:, 0], mu[:, 1], mu[:, 2],
                          c=metrics['lambda_max'], s=10, alpha=0.6,
                          cmap='plasma')
    ax3.set_title(f'λ_max (Maximum Stretch)\nMean: {metrics["lambda_max"].mean():.4f}',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    set_3d_equal_aspect(ax3, mu[:, 0], mu[:, 1], mu[:, 2])
    plt.colorbar(scatter3, ax=ax3, shrink=0.5, pad=0.1)

    # Panel 4: Frobenius norm
    ax4 = fig.add_subplot(gs[0, 3], projection='3d')
    scatter4 = ax4.scatter(mu[:, 0], mu[:, 1], mu[:, 2],
                          c=metrics['frob_norm'], s=10, alpha=0.6,
                          cmap='coolwarm')
    ax4.set_title(f'||Σ||_F (Total Deformation)\nMean: {metrics["frob_norm"].mean():.4f}',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    set_3d_equal_aspect(ax4, mu[:, 0], mu[:, 1], mu[:, 2])
    plt.colorbar(scatter4, ax=ax4, shrink=0.5, pad=0.1)

    # ========================================================================
    # Row 2: Statistical Distributions
    # ========================================================================

    # Panel 5: Eigenvalue distribution
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(metrics['lambda_min'], bins=50, alpha=0.6, label='λ_min', color='blue')
    ax5.hist(metrics['lambda_mid'], bins=50, alpha=0.6, label='λ_mid', color='green')
    ax5.hist(metrics['lambda_max'], bins=50, alpha=0.6, label='λ_max', color='red')
    ax5.set_xlabel('Eigenvalue (Principal Axis)', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Eigenvalue Distribution (Principal Axes)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    # Set tighter x-limits based on data
    ax5.set_xlim(np.percentile(metrics['lambda_min'], 1), np.percentile(metrics['lambda_max'], 99))

    # Panel 6: Anisotropy distribution
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(metrics['anisotropy'], bins=50, alpha=0.7, color='orange')
    ax6.axvline(metrics['anisotropy'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {metrics["anisotropy"].mean():.2f}')
    ax6.axvline(1.0, color='black', linestyle=':', linewidth=2, label='Isotropic (1.0)')
    ax6.set_xlabel('Anisotropy', fontsize=11)
    ax6.set_ylabel('Count', fontsize=11)
    ax6.set_title('Anisotropy Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    # Set tighter x-limits
    ax6.set_xlim(0.9, np.percentile(metrics['anisotropy'], 99))

    # Panel 7: Volume distribution
    ax7 = fig.add_subplot(gs[1, 2])
    log_det = np.log10(metrics['det'] + 1e-8)
    ax7.hist(log_det, bins=50, alpha=0.7, color='cyan')
    ax7.set_xlabel('log₁₀(det(Σ))', fontsize=11)
    ax7.set_ylabel('Count', fontsize=11)
    ax7.set_title('Volume Distribution (det(Σ))', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    # Set tighter x-limits
    ax7.set_xlim(np.percentile(log_det, 1), np.percentile(log_det, 99))

    # Panel 8: Frobenius norm distribution
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hist(metrics['frob_norm'], bins=50, alpha=0.7, color='purple')
    ax8.set_xlabel('||Σ||_F', fontsize=11)
    ax8.set_ylabel('Count', fontsize=11)
    ax8.set_title('Frobenius Norm Distribution', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    # Set tighter x-limits
    ax8.set_xlim(np.percentile(metrics['frob_norm'], 1), np.percentile(metrics['frob_norm'], 99))

    # ========================================================================
    # Row 3: 2D Projections
    # ========================================================================

    # Panel 9: XY projection
    ax9 = fig.add_subplot(gs[2, 0])
    scatter9 = ax9.scatter(mu[:, 0], mu[:, 1], c=metrics['anisotropy'],
                          s=5, alpha=0.5, cmap='hot',
                          vmin=1, vmax=np.percentile(metrics['anisotropy'], 95))
    ax9.set_title('XY Projection: Anisotropy', fontsize=12, fontweight='bold')
    ax9.set_xlabel('X')
    ax9.set_ylabel('Y')
    ax9.set_aspect('equal')
    ax9.grid(True, alpha=0.3)
    plt.colorbar(scatter9, ax=ax9, label='Anisotropy')

    # Panel 10: XZ projection
    ax10 = fig.add_subplot(gs[2, 1])
    scatter10 = ax10.scatter(mu[:, 0], mu[:, 2], c=metrics['lambda_max'],
                            s=5, alpha=0.5, cmap='plasma')
    ax10.set_title('XZ Projection: λ_max', fontsize=12, fontweight='bold')
    ax10.set_xlabel('X')
    ax10.set_ylabel('Z')
    ax10.set_aspect('equal')
    ax10.grid(True, alpha=0.3)
    plt.colorbar(scatter10, ax=ax10, label='λ_max')

    # Panel 11: YZ projection
    ax11 = fig.add_subplot(gs[2, 2])
    scatter11 = ax11.scatter(mu[:, 1], mu[:, 2],
                            c=np.log10(metrics['det'] + 1e-8),
                            s=5, alpha=0.5, cmap='viridis')
    ax11.set_title('YZ Projection: log₁₀(det)', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Y')
    ax11.set_ylabel('Z')
    ax11.set_aspect('equal')
    ax11.grid(True, alpha=0.3)
    plt.colorbar(scatter11, ax=ax11, label='log₁₀(det)')

    # Panel 12: Statistics
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    stats_text = "MEAN / STD / [MIN, MAX]\n"
    stats_text += "="*40 + "\n\n"
    stats_text += f"Eigenvalues (Principal Stretches):\n"
    stats_text += f"  λ_min: {metrics['lambda_min'].mean():.6f} ± {metrics['lambda_min'].std():.6f}\n"
    stats_text += f"         [{metrics['lambda_min'].min():.6f}, {metrics['lambda_min'].max():.6f}]\n"
    stats_text += f"  λ_mid: {metrics['lambda_mid'].mean():.6f} ± {metrics['lambda_mid'].std():.6f}\n"
    stats_text += f"         [{metrics['lambda_mid'].min():.6f}, {metrics['lambda_mid'].max():.6f}]\n"
    stats_text += f"  λ_max: {metrics['lambda_max'].mean():.6f} ± {metrics['lambda_max'].std():.6f}\n"
    stats_text += f"         [{metrics['lambda_max'].min():.6f}, {metrics['lambda_max'].max():.6f}]\n\n"
    stats_text += f"Anisotropy (λ_max/λ_min):\n"
    stats_text += f"  Mean: {metrics['anisotropy'].mean():.4f} ± {metrics['anisotropy'].std():.4f}\n"
    stats_text += f"  Median: {np.median(metrics['anisotropy']):.4f}\n"
    stats_text += f"  95th percentile: {np.percentile(metrics['anisotropy'], 95):.4f}\n\n"
    stats_text += f"Volume (det(Σ)):\n"
    stats_text += f"  Mean: {metrics['det'].mean():.6e}\n"
    stats_text += f"  Median: {np.median(metrics['det']):.6e}\n\n"
    stats_text += f"F eigenvalue (√λ_mid): {np.sqrt(metrics['lambda_mid'].mean()):.6f}\n"
    stats_text += f"Total particles: {len(mu):,}\n"

    ax12.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_dir', type=str, required=True)
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    ep_name = episode_dir.name

    # Load Gaussians
    gaussians_path = episode_dir / f"{ep_name}_gaussians.npz"
    if not gaussians_path.exists():
        print(f"❌ Gaussians not found: {gaussians_path}")
        return

    print(f"Loading: {gaussians_path}")
    data = np.load(gaussians_path)
    mu = data['mu']
    cov = data['cov']

    print(f"Particles: {len(mu):,}")

    # Create visualization
    save_path = episode_dir / "F_field_fixed_aspect.png"
    visualize_F_field(mu, cov, save_path)


if __name__ == "__main__":
    main()
