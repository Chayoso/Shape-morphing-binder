"""
Simple Visualizer for Multi-scale F-field from Existing Episode Data

Works with standard episode output (no special export needed).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys


def load_episode_data(episode_dir):
    """Load data from episode directory."""
    episode_dir = Path(episode_dir)

    # Load Gaussian data (fine particles after upsampling)
    gaussian_file = list(episode_dir.glob("*_gaussians.npz"))[0]
    gaussian_data = np.load(gaussian_file)

    x_fine = gaussian_data['mu']  # (N, 3) positions

    # Try to get F if available
    F_fine = None
    if 'F' in gaussian_data:
        F_fine = gaussian_data['F']
    elif 'F_interp' in gaussian_data:
        F_fine = gaussian_data['F_interp']

    # Get covariances
    cov_fine = None
    if 'cov' in gaussian_data:
        cov_fine = gaussian_data['cov']

    return x_fine, F_fine, cov_fine


def visualize_particle_distribution(x_fine, output_path):
    """Visualize upsampled particle distribution."""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('Multi-scale Upsampling Visualization', fontsize=16, fontweight='bold')

    # ============================================================================
    # 3D Scatter Views
    # ============================================================================

    # XY view
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(x_fine[:, 0], x_fine[:, 1], c=x_fine[:, 2],
                           s=10, alpha=0.5, cmap='viridis')
    ax1.set_title(f'XY View\nN = {len(x_fine):,} particles', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Z')

    # XZ view
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(x_fine[:, 0], x_fine[:, 2], c=x_fine[:, 1],
                           s=10, alpha=0.5, cmap='plasma')
    ax2.set_title('XZ View', fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Y')

    # YZ view
    ax3 = fig.add_subplot(gs[0, 2])
    scatter3 = ax3.scatter(x_fine[:, 1], x_fine[:, 2], c=x_fine[:, 0],
                           s=10, alpha=0.5, cmap='coolwarm')
    ax3.set_title('YZ View', fontweight='bold')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='X')

    # ============================================================================
    # Density Analysis
    # ============================================================================

    # Compute local density
    from scipy.spatial import cKDTree
    tree = cKDTree(x_fine)
    k = min(20, len(x_fine))
    distances, _ = tree.query(x_fine, k=k)
    mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self
    density = 1.0 / (mean_distances ** 3 + 1e-6)  # Inverse distance cubed

    # Density map (XY projection)
    ax4 = fig.add_subplot(gs[1, 0])
    scatter4 = ax4.scatter(x_fine[:, 0], x_fine[:, 1], c=density,
                           s=20, alpha=0.6, cmap='hot', vmin=np.percentile(density, 5),
                           vmax=np.percentile(density, 95))
    ax4.set_title('Local Density Map (XY)', fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax4, label='Density')

    # Density histogram
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(mean_distances, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax5.axvline(np.mean(mean_distances), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(mean_distances):.4f}')
    ax5.axvline(np.median(mean_distances), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(mean_distances):.4f}')
    ax5.set_title('Particle Spacing Distribution', fontweight='bold')
    ax5.set_xlabel(f'Mean Distance to {k-1} Neighbors')
    ax5.set_ylabel('Count')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    bounds = np.array([x_fine.min(axis=0), x_fine.max(axis=0)])
    volume = np.prod(bounds[1] - bounds[0])

    stats_text = f"""
    PARTICLE STATISTICS

    Total Particles: {len(x_fine):,}

    Bounding Box:
      X: [{bounds[0, 0]:.3f}, {bounds[1, 0]:.3f}]
      Y: [{bounds[0, 1]:.3f}, {bounds[1, 1]:.3f}]
      Z: [{bounds[0, 2]:.3f}, {bounds[1, 2]:.3f}]
      Volume: {volume:.6f}

    Particle Spacing:
      Mean: {np.mean(mean_distances):.6f}
      Std: {np.std(mean_distances):.6f}
      Min: {np.min(mean_distances):.6f}
      Max: {np.max(mean_distances):.6f}

    Density:
      Mean: {np.mean(density):.2f} particles/unit³
      Median: {np.median(density):.2f}
    """

    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor='lightblue', alpha=0.5))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved particle distribution to: {output_path}")


def visualize_F_field(x_fine, F_fine, output_path):
    """Visualize F-field magnitudes and properties."""
    if F_fine is None:
        print("⚠️  F-field not available in data")
        return

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('F-field (Deformation Gradient) Visualization', fontsize=16, fontweight='bold')

    # Compute F properties
    F_mag = np.linalg.norm(F_fine.reshape(len(F_fine), -1), axis=1)
    det_F = np.linalg.det(F_fine)

    # Principal stretches
    FTF = F_fine.transpose(0, 2, 1) @ F_fine
    eigenvalues = np.linalg.eigvalsh(FTF)
    stretches = np.sqrt(np.abs(eigenvalues))

    # F magnitude spatial map (XY)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(x_fine[:, 0], x_fine[:, 1], c=F_mag,
                           s=20, alpha=0.6, cmap='viridis')
    ax1.set_title('||F|| Frobenius Norm (XY)', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='||F||_F')

    # Determinant spatial map
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(x_fine[:, 0], x_fine[:, 1], c=det_F,
                           s=20, alpha=0.6, cmap='RdBu_r', vmin=0.5, vmax=1.5)
    ax2.set_title('det(F) - Volume Change (XY)', fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='det(F)')

    # Max stretch spatial map
    ax3 = fig.add_subplot(gs[0, 2])
    scatter3 = ax3.scatter(x_fine[:, 0], x_fine[:, 1], c=stretches[:, 2],
                           s=20, alpha=0.6, cmap='plasma')
    ax3.set_title('Max Principal Stretch λ₁ (XY)', fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='λ₁')

    # F magnitude histogram
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(F_mag, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(np.mean(F_mag), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(F_mag):.3f}')
    ax4.set_title('||F|| Distribution', fontweight='bold')
    ax4.set_xlabel('||F||_F')
    ax4.set_ylabel('Count')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Determinant histogram
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(det_F, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax5.axvline(1.0, color='green', linestyle='--', linewidth=2, label='det(F)=1 (no volume change)')
    ax5.axvline(np.mean(det_F), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(det_F):.3f}')
    ax5.set_title('det(F) Distribution (Volume Change)', fontweight='bold')
    ax5.set_xlabel('det(F)')
    ax5.set_ylabel('Count')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Principal stretches
    ax6 = fig.add_subplot(gs[1, 2])
    for i, label in enumerate(['λ₃ (min)', 'λ₂ (mid)', 'λ₁ (max)']):
        ax6.hist(stretches[:, i], bins=50, alpha=0.5, label=label)
    ax6.axvline(1.0, color='black', linestyle='--', linewidth=2, label='λ=1 (no stretch)')
    ax6.set_title('Principal Stretches', fontweight='bold')
    ax6.set_xlabel('Stretch λ')
    ax6.set_ylabel('Count')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved F-field visualization to: {output_path}")


def visualize_covariances(x_fine, cov_fine, output_path):
    """Visualize covariance properties."""
    if cov_fine is None:
        print("⚠️  Covariances not available in data")
        return

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('Covariance Σ = s²·F·F^T Visualization', fontsize=16, fontweight='bold')

    # Compute covariance properties
    eigenvalues = np.linalg.eigvalsh(cov_fine)  # (N, 3)
    scales = np.sqrt(np.abs(eigenvalues))  # Gaussian scales

    # Anisotropy
    anisotropy = scales[:, 2] / (scales[:, 0] + 1e-6)

    # Scale spatial maps
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(x_fine[:, 0], x_fine[:, 1], c=scales[:, 2],
                           s=20, alpha=0.6, cmap='viridis')
    ax1.set_title('Max Gaussian Scale (XY)', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Scale')

    # Anisotropy map
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(x_fine[:, 0], x_fine[:, 1], c=anisotropy,
                           s=20, alpha=0.6, cmap='plasma', vmin=1, vmax=5)
    ax2.set_title('Anisotropy (max/min scale)', fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Ratio')

    # Scale histograms
    ax3 = fig.add_subplot(gs[0, 2])
    for i, label in enumerate(['scale₃ (min)', 'scale₂ (mid)', 'scale₁ (max)']):
        ax3.hist(scales[:, i], bins=50, alpha=0.5, label=label)
    ax3.set_title('Gaussian Scale Distribution', fontweight='bold')
    ax3.set_xlabel('Scale')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved covariance visualization to: {output_path}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize multi-scale upsampling from episode data')
    parser.add_argument('--episode', '-e', type=str, required=True,
                        help='Episode directory (e.g., output/bunny/ep000)')
    args = parser.parse_args()

    episode_dir = Path(args.episode)

    if not episode_dir.exists():
        print(f"❌ Episode directory not found: {episode_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Multi-scale Upsampling Visualization")
    print(f"{'='*70}")
    print(f"Episode: {episode_dir}")

    # Load data
    print(f"\n[Loading Data]")
    x_fine, F_fine, cov_fine = load_episode_data(episode_dir)
    print(f"  Particles: {len(x_fine):,}")
    print(f"  F available: {F_fine is not None}")
    print(f"  Cov available: {cov_fine is not None}")

    # Generate visualizations
    print(f"\n[Generating Visualizations]")

    output_dir = episode_dir

    # Particle distribution
    visualize_particle_distribution(x_fine, output_dir / "multiscale_particles.png")

    # F-field
    if F_fine is not None:
        visualize_F_field(x_fine, F_fine, output_dir / "multiscale_F_field.png")

    # Covariances
    if cov_fine is not None:
        visualize_covariances(x_fine, cov_fine, output_dir / "multiscale_covariances.png")

    print(f"\n{'='*70}")
    print(f"✅ Visualization Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
