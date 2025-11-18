#!/usr/bin/env python3
"""
Plot F-field eigenvalue metrics from Gaussian data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path


def load_F_field_data(gaussians_path):
    """Load F-field data from Gaussian NPZ file."""
    data = np.load(gaussians_path)
    mu = data['mu']
    cov = data['cov']

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)  # (N, 3), sorted ascending

    # Determinant (volume)
    det = np.linalg.det(cov)

    # Frobenius norm
    frob_norm = np.linalg.norm(cov.reshape(len(cov), -1), axis=1)

    # Anisotropy
    anisotropy = eigenvalues[:, 2] / (eigenvalues[:, 0] + 1e-8)

    return {
        'mu': mu,
        'eigenvalues': eigenvalues,
        'det': det,
        'frob_norm': frob_norm,
        'anisotropy': anisotropy,
        'num_particles': len(mu)
    }


def plot_F_field_progression(output_dir, episodes, output_path):
    """
    Plot F-field metrics across multiple episodes.

    Args:
        output_dir: Output directory path
        episodes: List of episode numbers to plot
        output_path: Where to save the plot
    """
    output_dir = Path(output_dir)

    # Load data for all episodes
    episode_data = []
    for ep_num in episodes:
        ep_dir = output_dir / f"ep{ep_num:03d}"
        gaussians_path = ep_dir / f"ep{ep_num:03d}_gaussians.npz"

        if not gaussians_path.exists():
            print(f"⚠️  Missing: {gaussians_path.name}")
            continue

        data = load_F_field_data(gaussians_path)
        episode_data.append((ep_num, data))
        print(f"✅ Loaded ep{ep_num:03d}: {data['num_particles']:,} particles")

    if len(episode_data) == 0:
        print("❌ No data loaded")
        return

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('F-field Evolution During Training', fontsize=18, fontweight='bold')

    # Extract metrics for plotting
    episodes_loaded = [ep for ep, _ in episode_data]

    lambda_min_mean = [d['eigenvalues'][:, 0].mean() for _, d in episode_data]
    lambda_mid_mean = [d['eigenvalues'][:, 1].mean() for _, d in episode_data]
    lambda_max_mean = [d['eigenvalues'][:, 2].mean() for _, d in episode_data]

    lambda_min_std = [d['eigenvalues'][:, 0].std() for _, d in episode_data]
    lambda_mid_std = [d['eigenvalues'][:, 1].std() for _, d in episode_data]
    lambda_max_std = [d['eigenvalues'][:, 2].std() for _, d in episode_data]

    anisotropy_mean = [d['anisotropy'].mean() for _, d in episode_data]
    anisotropy_std = [d['anisotropy'].std() for _, d in episode_data]

    det_mean = [d['det'].mean() for _, d in episode_data]
    frob_mean = [d['frob_norm'].mean() for _, d in episode_data]

    # ========================================================================
    # Plot 1: Eigenvalue Means
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes_loaded, lambda_min_mean, 'o-', label='λ_min', linewidth=2, markersize=6)
    ax1.plot(episodes_loaded, lambda_mid_mean, 's-', label='λ_mid', linewidth=2, markersize=6)
    ax1.plot(episodes_loaded, lambda_max_mean, '^-', label='λ_max', linewidth=2, markersize=6)
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Mean Eigenvalue', fontsize=11)
    ax1.set_title('F-field Eigenvalues (Mean)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 2: Eigenvalue Std Dev
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes_loaded, lambda_min_std, 'o-', label='λ_min std', linewidth=2, markersize=6)
    ax2.plot(episodes_loaded, lambda_mid_std, 's-', label='λ_mid std', linewidth=2, markersize=6)
    ax2.plot(episodes_loaded, lambda_max_std, '^-', label='λ_max std', linewidth=2, markersize=6)
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Std Dev', fontsize=11)
    ax2.set_title('F-field Eigenvalue Variation', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 3: Anisotropy
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(episodes_loaded, anisotropy_mean, 'o-', linewidth=2, markersize=6, color='purple')
    ax3.fill_between(episodes_loaded,
                     np.array(anisotropy_mean) - np.array(anisotropy_std),
                     np.array(anisotropy_mean) + np.array(anisotropy_std),
                     alpha=0.3, color='purple')
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Isotropic (1.0)')
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Anisotropy (λ_max/λ_min)', fontsize=11)
    ax3.set_title('Gaussian Anisotropy', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 4: Determinant (Volume)
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(episodes_loaded, det_mean, 'o-', linewidth=2, markersize=6, color='orange')
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Mean det(Σ)', fontsize=11)
    ax4.set_title('Gaussian Volume (det)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # ========================================================================
    # Plot 5: Frobenius Norm
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(episodes_loaded, frob_mean, 'o-', linewidth=2, markersize=6, color='green')
    ax5.set_xlabel('Episode', fontsize=11)
    ax5.set_ylabel('Mean ||Σ||_F', fontsize=11)
    ax5.set_title('Frobenius Norm (Total Magnitude)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 6: F Eigenvalue (√λ)
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    F_eigen_min = np.sqrt(lambda_min_mean)
    F_eigen_mid = np.sqrt(lambda_mid_mean)
    F_eigen_max = np.sqrt(lambda_max_mean)

    ax6.plot(episodes_loaded, F_eigen_min, 'o-', label='√λ_min', linewidth=2, markersize=6)
    ax6.plot(episodes_loaded, F_eigen_mid, 's-', label='√λ_mid', linewidth=2, markersize=6)
    ax6.plot(episodes_loaded, F_eigen_max, '^-', label='√λ_max', linewidth=2, markersize=6)
    ax6.set_xlabel('Episode', fontsize=11)
    ax6.set_ylabel('F Eigenvalue (√λ)', fontsize=11)
    ax6.set_title('Deformation Gradient Eigenvalues', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 7: Eigenvalue Distribution (First Episode)
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    first_data = episode_data[0][1]
    ax7.hist(first_data['eigenvalues'][:, 0], bins=50, alpha=0.6, label='λ_min', color='blue')
    ax7.hist(first_data['eigenvalues'][:, 1], bins=50, alpha=0.6, label='λ_mid', color='green')
    ax7.hist(first_data['eigenvalues'][:, 2], bins=50, alpha=0.6, label='λ_max', color='red')
    ax7.set_xlabel('Eigenvalue', fontsize=11)
    ax7.set_ylabel('Count', fontsize=11)
    ax7.set_title(f'Eigenvalue Distribution (ep{episode_data[0][0]:03d})', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 8: Eigenvalue Distribution (Last Episode)
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    last_data = episode_data[-1][1]
    ax8.hist(last_data['eigenvalues'][:, 0], bins=50, alpha=0.6, label='λ_min', color='blue')
    ax8.hist(last_data['eigenvalues'][:, 1], bins=50, alpha=0.6, label='λ_mid', color='green')
    ax8.hist(last_data['eigenvalues'][:, 2], bins=50, alpha=0.6, label='λ_max', color='red')
    ax8.set_xlabel('Eigenvalue', fontsize=11)
    ax8.set_ylabel('Count', fontsize=11)
    ax8.set_title(f'Eigenvalue Distribution (ep{episode_data[-1][0]:03d})', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 9: Statistics Table
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    first_ep, first_d = episode_data[0]
    last_ep, last_d = episode_data[-1]

    stats_text = "F-FIELD STATISTICS\n"
    stats_text += "="*35 + "\n\n"

    stats_text += f"Episode {first_ep} (Initial):\n"
    stats_text += f"  λ_mid: {first_d['eigenvalues'][:, 1].mean():.6f}\n"
    stats_text += f"  Anisotropy: {first_d['anisotropy'].mean():.4f}\n"
    stats_text += f"  √λ_mid: {np.sqrt(first_d['eigenvalues'][:, 1].mean()):.6f}\n"
    stats_text += f"  Particles: {first_d['num_particles']:,}\n\n"

    stats_text += f"Episode {last_ep} (Final):\n"
    stats_text += f"  λ_mid: {last_d['eigenvalues'][:, 1].mean():.6f}\n"
    stats_text += f"  Anisotropy: {last_d['anisotropy'].mean():.4f}\n"
    stats_text += f"  √λ_mid: {np.sqrt(last_d['eigenvalues'][:, 1].mean()):.6f}\n"
    stats_text += f"  Particles: {last_d['num_particles']:,}\n\n"

    stats_text += "Interpretation:\n"
    stats_text += f"  Anisotropy change: {last_d['anisotropy'].mean() - first_d['anisotropy'].mean():+.4f}\n"
    stats_text += f"  Size change: {(last_d['eigenvalues'][:, 1].mean() / first_d['eigenvalues'][:, 1].mean() - 1)*100:.1f}%\n"

    ax9.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ F-field plot saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Plot F-field metrics')
    parser.add_argument('--output_dir', type=str, default='output/bunny',
                        help='Output directory')
    parser.add_argument('--episodes', type=str, default='0,5,10,15,20,25,29',
                        help='Comma-separated episode numbers')
    parser.add_argument('--output_plot', type=str, default='F_field_progression.png',
                        help='Output plot path')
    args = parser.parse_args()

    episodes = [int(x.strip()) for x in args.episodes.split(',')]

    print(f"Output directory: {args.output_dir}")
    print(f"Episodes: {episodes}")
    print("="*70)

    plot_F_field_progression(args.output_dir, episodes, args.output_plot)


if __name__ == "__main__":
    main()
