"""
Visualize Subdivision Upsampling Process

Shows:
1. Parent particles with deformation magnitude
2. Child allocation (num_children per parent)
3. Parent-child relationships with connections
4. Before/after comparison

Usage:
    python visualize_subdivision.py --episode output/bob_v1/ep029
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import argparse
from pathlib import Path
import sys


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def load_ply(ply_path):
    """Load PLY file (simple parser for point clouds)"""
    with open(ply_path, 'r') as f:
        lines = f.readlines()

    # Find header end
    header_end = 0
    vertex_count = 0
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        if line.strip() == 'end_header':
            header_end = i + 1
            break

    # Parse vertices
    points = []
    for i in range(header_end, header_end + vertex_count):
        parts = lines[i].strip().split()
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        points.append([x, y, z])

    return np.array(points)


def compute_deformation_metric(points):
    """
    Simulate deformation magnitude based on distance from centroid.

    In real code, this comes from det(F), but we don't have F saved.
    This is just for visualization purposes.
    """
    centroid = points.mean(axis=0)
    dist_from_center = np.linalg.norm(points - centroid, axis=1)

    # Normalize to [0, 1]
    dist_norm = (dist_from_center - dist_from_center.min()) / (dist_from_center.max() - dist_from_center.min() + 1e-8)

    # Simulate: edges have higher "deformation" (more interesting)
    # Use distance as proxy (particles far from center = surface = high deformation)
    deformation = dist_norm ** 2  # Quadratic to emphasize extremes

    return deformation


def allocate_children(deformation, target_ratio=90, max_children_per_parent=20):
    """
    Simulate subdivision allocation.

    Args:
        deformation: (N,) deformation magnitude per parent
        target_ratio: Upsampling ratio (e.g., 90x)
        max_children_per_parent: Cap to prevent clustering

    Returns:
        num_children: (N,) number of children per parent
    """
    N = len(deformation)
    num_new = int(N * (target_ratio - 1))  # Total new particles

    # Check if deformation is negligible
    mean_dev = deformation.mean()

    if mean_dev < 0.02:
        # Uniform subdivision
        base_children = num_new // N
        remainder = num_new % N
        num_children = np.full(N, base_children, dtype=int)
        num_children[:remainder] += 1
    else:
        # Proportional allocation
        prob = deformation / (deformation.sum() + 1e-8)
        num_children_float = prob * num_new

        # Apply cap
        num_children_float = np.clip(num_children_float, 0, max_children_per_parent)

        # Redistribute excess (simplified)
        total_allocated = num_children_float.sum()
        if total_allocated < num_new:
            deficit = num_new - total_allocated
            under_cap = num_children_float < max_children_per_parent
            if under_cap.sum() > 0:
                redistrib = prob[under_cap] / (prob[under_cap].sum() + 1e-8)
                num_children_float[under_cap] += redistrib * deficit

        # Stochastic rounding
        num_children = np.floor(num_children_float).astype(int)
        frac = num_children_float - num_children
        rand = np.random.rand(N)
        num_children += (frac > rand).astype(int)

        # Final adjustment to hit target
        diff = num_new - num_children.sum()
        if diff > 0:
            # Add to highest fractional parts
            top_idx = np.argsort(frac)[-diff:]
            num_children[top_idx] += 1
        elif diff < 0:
            # Remove from lowest
            bottom_idx = np.argsort(frac)[:abs(diff)]
            num_children[bottom_idx] = np.maximum(0, num_children[bottom_idx] - 1)

    return num_children


def generate_children(parents, num_children, jitter_scale=0.15, k=8):
    """
    Generate child particles with jitter.

    Args:
        parents: (N, 3) parent positions
        num_children: (N,) number of children per parent
        jitter_scale: Jitter magnitude relative to local spacing
        k: Number of neighbors for spacing estimation

    Returns:
        children: (M, 3) child positions
        parent_indices: (M,) parent index for each child
    """
    from scipy.spatial import KDTree

    N = len(parents)
    tree = KDTree(parents)

    children_list = []
    parent_indices_list = []

    for i in range(N):
        n_child = num_children[i]
        if n_child > 0:
            # Estimate local spacing
            dists, _ = tree.query(parents[i], k=min(k, N))
            local_spacing = dists.mean()

            # Generate children with jitter
            jitter = np.random.randn(n_child, 3) * jitter_scale * local_spacing
            child_pos = parents[i] + jitter

            children_list.append(child_pos)
            parent_indices_list.extend([i] * n_child)

    if len(children_list) > 0:
        children = np.vstack(children_list)
        parent_indices = np.array(parent_indices_list)
    else:
        children = np.empty((0, 3))
        parent_indices = np.array([], dtype=int)

    return children, parent_indices


def visualize_subdivision(episode_path, output_path=None, downsample=1.0):
    """
    Visualize subdivision upsampling process.

    Args:
        episode_path: Path to episode directory (e.g., output/bob_v1/ep029)
        output_path: Where to save visualization (default: episode_path/subdivision_visualization.png)
        downsample: Fraction of particles to use (1.0 = all, 0.1 = 10%)
    """
    episode_path = Path(episode_path)

    # Load Gaussian point cloud
    # Try multiple locations
    possible_paths = [
        episode_path / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        list(episode_path.glob("*_surface_*.ply")),
        list(episode_path.glob("*.ply"))
    ]

    ply_path = None
    for path in possible_paths:
        if isinstance(path, list):
            if len(path) > 0:
                ply_path = path[0]
                break
        elif path.exists():
            ply_path = path
            break

    if ply_path is None:
        print(f"❌ No PLY file found in {episode_path}")
        print(f"Available files:")
        for f in episode_path.rglob("*.ply"):
            print(f"  {f}")
        return

    points = load_ply(ply_path)
    print(f"Loaded {len(points):,} particles from {ply_path.name}")

    # Store original points for Panel 1
    original_points = points.copy()

    # Downsample if requested (for faster visualization)
    if downsample < 1.0:
        N_original = len(points)
        N_keep = int(N_original * downsample)
        indices = np.random.choice(N_original, N_keep, replace=False)
        points = points[indices]
        print(f"Downsampled to {len(points):,} particles ({downsample*100:.0f}%)")

        # Also downsample original points for Panel 1
        original_points = points.copy()

    # Simulate subdivision (since we don't have original coarse particles saved)
    # In reality, these would be the MPM particles before upsampling
    # Here we'll simulate by taking a subset
    target_ratio = 90
    N_coarse = len(points) // target_ratio

    # Select "parent" particles (simulate coarse particles)
    parent_indices = np.random.choice(len(points), N_coarse, replace=False)
    parents = points[parent_indices]
    print(f"\nSimulating coarse particles: {N_coarse:,}")

    # Compute deformation metric
    deformation = compute_deformation_metric(parents)
    print(f"Deformation: mean={deformation.mean():.4f}, range=[{deformation.min():.4f}, {deformation.max():.4f}]")

    # Allocate children
    num_children = allocate_children(deformation, target_ratio=target_ratio)
    print(f"\nSubdivision allocation:")
    print(f"  Mean children/parent: {num_children.mean():.2f}")
    print(f"  Max children/parent: {num_children.max()}")
    print(f"  Parents with children: {(num_children > 0).sum()}/{N_coarse}")

    # Generate children
    children, child_parent_map = generate_children(parents, num_children)
    print(f"\nGenerated {len(children):,} children")

    # Combine
    all_points = np.vstack([original_points, children])
    print(f"Total particles: {len(all_points):,}")

    # ========================================================================
    # Compute common axis limits for consistent camera view across all panels
    # ========================================================================
    all_data = original_points  # Use original loaded points for consistent bounds
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
    z_min, z_max = all_data[:, 2].min(), all_data[:, 2].max()

    # Add 10% padding for better visualization
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    padding = 0.1

    x_lim = [x_min - padding * x_range, x_max + padding * x_range]
    y_lim = [y_min - padding * y_range, y_max + padding * y_range]
    z_lim = [z_min - padding * z_range, z_max + padding * z_range]

    # Create figure with 2x3 layout (6 panels) using constrained_layout for equal panel sizes
    fig = plt.figure(figsize=(24, 16), constrained_layout=True)
    fig.suptitle('Subdivision Upsampling Process', fontsize=18, fontweight='bold')

    # ========================================================================
    # Panel 1: Original Shape (Loaded PLY file)
    # ========================================================================
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    # Show ORIGINAL loaded particles (the actual bunny shape!)
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                c='steelblue', s=8, alpha=0.5, edgecolors='none')

    ax1.set_title(f'1. Original Shape\n({len(original_points):,} particles)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=45)

    # Apply consistent axis limits
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_zlim(z_lim)
    ax1.set_box_aspect([x_range, y_range, z_range])

    # ========================================================================
    # Panel 2: Parent Selection (Highlight parents, fade others)
    # ========================================================================
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')

    # Show only selected parents (no background)
    scatter2 = ax2.scatter(parents[:, 0], parents[:, 1], parents[:, 2],
                           c=deformation, cmap='hot', s=100, alpha=1.0,
                           edgecolors='black', linewidths=1.5, vmin=0, vmax=1,
                           label='Selected parents', zorder=10)

    ax2.set_title(f'2. Parent Selection\n({N_coarse:,} parents selected)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=20, azim=45)

    # Apply consistent axis limits
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.set_zlim(z_lim)
    ax2.set_box_aspect([x_range, y_range, z_range])

    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.3, pad=0.05, aspect=10)
    cbar2.set_label('Deformation', fontsize=8)
    ax2.legend(fontsize=7, loc='upper left')

    # ========================================================================
    # Panel 3: Parent-Child Relationships (Top 20)
    # ========================================================================
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')

    # Sample high-deformation parents to show connections
    sample_size = min(20, N_coarse)
    top_deform_idx = np.argsort(deformation)[-sample_size:]

    # Plot sample parents
    ax3.scatter(parents[top_deform_idx, 0], parents[top_deform_idx, 1], parents[top_deform_idx, 2],
                c='red', s=80, alpha=1.0, marker='o', edgecolors='black', linewidths=2,
                label='Parents', zorder=10)

    # Plot their children and connections
    for parent_idx in top_deform_idx:
        # Find children of this parent
        child_mask = (child_parent_map == parent_idx)
        if child_mask.sum() > 0:
            child_pos = children[child_mask]

            # Plot children
            ax3.scatter(child_pos[:, 0], child_pos[:, 1], child_pos[:, 2],
                        c='darkorange', s=20, alpha=0.5, marker='.', label='Children' if parent_idx == top_deform_idx[0] else '')

            # Draw connections (only first few to avoid clutter)
            for i in range(min(5, len(child_pos))):
                arrow = Arrow3D([parents[parent_idx, 0], child_pos[i, 0]],
                                [parents[parent_idx, 1], child_pos[i, 1]],
                                [parents[parent_idx, 2], child_pos[i, 2]],
                                mutation_scale=10, lw=0.5, arrowstyle='->', color='gray', alpha=0.3)
                ax3.add_artist(arrow)

    ax3.set_title(f'3. Parent-Child Relationships\n(Top {sample_size} high-deformation parents)',
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.view_init(elev=20, azim=45)

    # Apply consistent axis limits
    ax3.set_xlim(x_lim)
    ax3.set_ylim(y_lim)
    ax3.set_zlim(z_lim)
    ax3.set_box_aspect([x_range, y_range, z_range])

    ax3.legend(fontsize=9, loc='upper left')

    # ========================================================================
    # Panel 4: Upsampled Children Only (No parents)
    # ========================================================================
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')

    # Show only children particles (upsampled result)
    ax4.scatter(children[:, 0], children[:, 1], children[:, 2],
                c='darkorange', s=20, alpha=0.6, edgecolors='none')

    ax4.set_title(f'4. Upsampled Children\n({len(children):,} particles)',
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.view_init(elev=20, azim=45)

    # Apply consistent axis limits
    ax4.set_xlim(x_lim)
    ax4.set_ylim(y_lim)
    ax4.set_zlim(z_lim)
    ax4.set_box_aspect([x_range, y_range, z_range])

    # ========================================================================
    # Panel 5: After Upsampling (All Original + Children Combined)
    # ========================================================================
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')

    # Show ALL original particles and children combined (the final upsampled result)
    # Render everything in the same color as original shape
    ax5.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
                c='steelblue', s=8, alpha=0.5, edgecolors='none')

    ax5.set_title(f'5. After Upsampling (Combined)\n({len(all_points):,} particles total)',
                  fontsize=14, fontweight='bold')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.view_init(elev=20, azim=45)

    # Apply consistent axis limits
    ax5.set_xlim(x_lim)
    ax5.set_ylim(y_lim)
    ax5.set_zlim(z_lim)
    ax5.set_box_aspect([x_range, y_range, z_range])

    # Save
    if output_path is None:
        output_path = episode_path / "subdivision_visualization.png"
    else:
        output_path = Path(output_path)

    # IMPROVED: Higher DPI for better quality
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved visualization: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize subdivision upsampling')
    parser.add_argument('-e', '--episode', type=str, required=True,
                        help='Episode directory (e.g., output/bob_v1/ep029)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path (default: episode/subdivision_visualization.png)')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='Downsample factor (0-1, default: 1.0 = use all particles)')

    args = parser.parse_args()

    visualize_subdivision(args.episode, args.output, args.downsample)


if __name__ == '__main__':
    main()
