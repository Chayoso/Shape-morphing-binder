"""
Visualize Surface Extraction: Compare Different Upsampling Methods

This script visualizes:
1. Original low-res mesh vertices
2. Subdivision with high jitter (current approach)
3. Subdivision with low jitter (fails to cover surface)
4. Mesh surface sampling (correct approach)

Usage:
    python visualize_surface_extraction.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

# ============================================================================
# Helper Functions
# ============================================================================

def subdivision_upsample(vertices, target_count, jitter_scale=0.15):
    """
    Simulate subdivision upsampling with jitter.

    Args:
        vertices: (N, 3) original vertices
        target_count: number of particles to create
        jitter_scale: jitter strength relative to local spacing

    Returns:
        (M, 3) upsampled particles
    """
    N = len(vertices)
    num_children_per_parent = target_count // N

    particles = []

    for i, vertex in enumerate(vertices):
        # Estimate local spacing (distance to nearest neighbors)
        distances = np.linalg.norm(vertices - vertex, axis=1)
        distances = distances[distances > 0]  # Exclude self
        if len(distances) > 0:
            local_spacing = np.mean(np.sort(distances)[:min(8, len(distances))])
        else:
            local_spacing = 1.0

        # Create children with random jitter
        for _ in range(num_children_per_parent):
            jitter = np.random.randn(3) * jitter_scale * local_spacing
            child = vertex + jitter
            particles.append(child)

    return np.array(particles)


def mesh_surface_sample(mesh, count):
    """
    Sample points uniformly on mesh surface (faces).

    Args:
        mesh: trimesh.Trimesh object
        count: number of points to sample

    Returns:
        (N, 3) sampled points on surface
    """
    points, face_indices = trimesh.sample.sample_surface(mesh, count)
    return points


def compute_coverage(particles, mesh, threshold=0.02):
    """
    Compute what percentage of mesh faces are covered by particles.

    Args:
        particles: (N, 3) particle positions
        mesh: trimesh.Trimesh object
        threshold: max distance to consider "covered"

    Returns:
        coverage_ratio, uncovered_faces
    """
    from scipy.spatial import cKDTree

    # Get face centers
    face_centers = mesh.vertices[mesh.faces].mean(axis=1)

    # Find nearest particle to each face
    tree = cKDTree(particles)
    distances, _ = tree.query(face_centers, k=1)

    # Check coverage
    covered = distances < threshold
    coverage_ratio = covered.sum() / len(face_centers)
    uncovered_faces = np.where(~covered)[0]

    return coverage_ratio, uncovered_faces, distances


def compute_particle_to_mesh_distance(particles, mesh):
    """
    Compute distance from each particle to nearest mesh surface.

    Args:
        particles: (N, 3) particle positions
        mesh: trimesh.Trimesh object

    Returns:
        (N,) distances to mesh
    """
    closest_point, distance, face_idx = trimesh.proximity.closest_point(
        mesh, particles
    )
    return distance


# ============================================================================
# Main Visualization
# ============================================================================

def visualize_surface_extraction(mesh_path="assets/bunny.obj"):
    """
    Create comprehensive visualization comparing upsampling methods.
    """
    # Load mesh
    print(f"Loading mesh: {mesh_path}")
    try:
        mesh = trimesh.load(mesh_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        print("Creating simple sphere mesh for demonstration...")
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)

    print(f"Mesh info:")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Faces: {len(mesh.faces):,}")

    # Subsample vertices for visualization (simulate low-res)
    subsample_ratio = min(1.0, 500 / len(mesh.vertices))  # Use ~500 vertices
    subsample_count = max(100, int(len(mesh.vertices) * subsample_ratio))
    vertex_indices = np.random.choice(len(mesh.vertices), subsample_count, replace=False)
    vertices_lowres = mesh.vertices[vertex_indices]

    print(f"  Low-res vertices: {len(vertices_lowres):,}")

    # Generate upsampled particles
    target_count = 5000  # Enough to see patterns

    print(f"\nGenerating upsampled particles (target: {target_count:,})...")

    # Method 1: High jitter (current approach)
    print("  1. Subdivision with high jitter (0.4)...")
    particles_high_jitter = subdivision_upsample(vertices_lowres, target_count, jitter_scale=0.4)

    # Method 2: Low jitter (gaps)
    print("  2. Subdivision with low jitter (0.05)...")
    particles_low_jitter = subdivision_upsample(vertices_lowres, target_count, jitter_scale=0.05)

    # Method 3: Mesh surface sampling (correct)
    print("  3. Mesh surface sampling...")
    particles_mesh_surface = mesh_surface_sample(mesh, target_count)

    # Compute statistics
    print(f"\nComputing statistics...")

    # Coverage analysis
    coverage_high, uncov_high, dist_high = compute_coverage(particles_high_jitter, mesh, 0.02)
    coverage_low, uncov_low, dist_low = compute_coverage(particles_low_jitter, mesh, 0.02)
    coverage_mesh, uncov_mesh, dist_mesh = compute_coverage(particles_mesh_surface, mesh, 0.02)

    # Distance to mesh analysis
    dist_to_mesh_high = compute_particle_to_mesh_distance(particles_high_jitter, mesh)
    dist_to_mesh_low = compute_particle_to_mesh_distance(particles_low_jitter, mesh)
    dist_to_mesh_mesh = compute_particle_to_mesh_distance(particles_mesh_surface, mesh)

    print(f"\nCoverage (faces with particle < 0.02):")
    print(f"  High jitter: {coverage_high*100:.1f}%")
    print(f"  Low jitter:  {coverage_low*100:.1f}%")
    print(f"  Mesh surface: {coverage_mesh*100:.1f}%")

    print(f"\nParticle-to-mesh distance:")
    print(f"  High jitter:  mean={dist_to_mesh_high.mean():.4f}, max={dist_to_mesh_high.max():.4f}")
    print(f"  Low jitter:   mean={dist_to_mesh_low.mean():.4f}, max={dist_to_mesh_low.max():.4f}")
    print(f"  Mesh surface: mean={dist_to_mesh_mesh.mean():.4f}, max={dist_to_mesh_mesh.max():.4f}")

    # ========================================================================
    # Create Visualization
    # ========================================================================

    fig = plt.figure(figsize=(20, 12))

    # ========================================================================
    # Row 1: 3D Particle Distributions
    # ========================================================================

    # Plot 1: Low-res vertices only
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax1.scatter(vertices_lowres[:, 0], vertices_lowres[:, 1], vertices_lowres[:, 2],
                c='red', s=50, alpha=0.8, label='Low-res vertices')
    ax1.set_title(f'Original Low-Res\n({len(vertices_lowres):,} vertices)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    set_axes_equal(ax1)

    # Plot 2: Subdivision with high jitter
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    # Color by distance to mesh
    colors_high = plt.cm.RdYlGn_r(np.clip(dist_to_mesh_high / 0.1, 0, 1))
    ax2.scatter(particles_high_jitter[:, 0], particles_high_jitter[:, 1], particles_high_jitter[:, 2],
                c=colors_high, s=5, alpha=0.6)
    ax2.set_title(f'Subdivision (jitter=0.4)\n{len(particles_high_jitter):,} particles\n'
                  f'On surface: {(dist_to_mesh_high < 0.01).sum()/len(dist_to_mesh_high)*100:.1f}%',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    set_axes_equal(ax2)

    # Plot 3: Subdivision with low jitter
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    colors_low = plt.cm.RdYlGn_r(np.clip(dist_to_mesh_low / 0.1, 0, 1))
    ax3.scatter(particles_low_jitter[:, 0], particles_low_jitter[:, 1], particles_low_jitter[:, 2],
                c=colors_low, s=5, alpha=0.6)
    ax3.set_title(f'Subdivision (jitter=0.05)\n{len(particles_low_jitter):,} particles\n'
                  f'On surface: {(dist_to_mesh_low < 0.01).sum()/len(dist_to_mesh_low)*100:.1f}%',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    set_axes_equal(ax3)

    # Plot 4: Mesh surface sampling
    ax4 = fig.add_subplot(2, 4, 4, projection='3d')
    colors_mesh = plt.cm.RdYlGn_r(np.clip(dist_to_mesh_mesh / 0.1, 0, 1))
    ax4.scatter(particles_mesh_surface[:, 0], particles_mesh_surface[:, 1], particles_mesh_surface[:, 2],
                c=colors_mesh, s=5, alpha=0.6)
    ax4.set_title(f'Mesh Surface Sampling\n{len(particles_mesh_surface):,} particles\n'
                  f'On surface: {(dist_to_mesh_mesh < 0.01).sum()/len(dist_to_mesh_mesh)*100:.1f}%',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    set_axes_equal(ax4)

    # ========================================================================
    # Row 2: Statistics and Analysis
    # ========================================================================

    # Plot 5: Distance distribution histogram
    ax5 = fig.add_subplot(2, 4, 5)
    bins = np.linspace(0, 0.15, 50)
    ax5.hist(dist_to_mesh_high, bins=bins, alpha=0.5, label='High jitter (0.4)', color='orange', density=True)
    ax5.hist(dist_to_mesh_low, bins=bins, alpha=0.5, label='Low jitter (0.05)', color='blue', density=True)
    ax5.hist(dist_to_mesh_mesh, bins=bins, alpha=0.5, label='Mesh surface', color='green', density=True)
    ax5.axvline(0.01, color='red', linestyle='--', linewidth=2, label='Surface threshold')
    ax5.set_xlabel('Distance to Mesh', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('Particle-to-Mesh Distance Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Coverage comparison
    ax6 = fig.add_subplot(2, 4, 6)
    methods = ['High jitter\n(0.4)', 'Low jitter\n(0.05)', 'Mesh\nsurface']
    coverages = [coverage_high * 100, coverage_low * 100, coverage_mesh * 100]
    colors_bar = ['orange', 'blue', 'green']
    bars = ax6.bar(methods, coverages, color=colors_bar, alpha=0.7)
    ax6.set_ylabel('Coverage (%)', fontsize=11)
    ax6.set_title('Mesh Face Coverage\n(faces with particle < 0.02)', fontsize=12, fontweight='bold')
    ax6.set_ylim([0, 105])
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, coverages):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 7: Cumulative distance distribution
    ax7 = fig.add_subplot(2, 4, 7)
    sorted_high = np.sort(dist_to_mesh_high)
    sorted_low = np.sort(dist_to_mesh_low)
    sorted_mesh = np.sort(dist_to_mesh_mesh)

    cumulative = np.linspace(0, 100, len(sorted_high))

    ax7.plot(sorted_high, cumulative, label='High jitter (0.4)', color='orange', linewidth=2)
    ax7.plot(sorted_low, cumulative, label='Low jitter (0.05)', color='blue', linewidth=2)
    ax7.plot(sorted_mesh, cumulative, label='Mesh surface', color='green', linewidth=2)

    # Add reference lines
    ax7.axvline(0.01, color='red', linestyle='--', alpha=0.5, label='0.01 (surface)')
    ax7.axvline(0.05, color='purple', linestyle='--', alpha=0.5, label='0.05 (near surface)')

    ax7.set_xlabel('Distance to Mesh', fontsize=11)
    ax7.set_ylabel('Cumulative % of Particles', fontsize=11)
    ax7.set_title('Cumulative Distance Distribution', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, 0.15])

    # Plot 8: Summary statistics table
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')

    # Create summary table
    table_data = [
        ['Metric', 'High Jitter', 'Low Jitter', 'Mesh Surface'],
        ['', '(0.4)', '(0.05)', ''],
        ['', '', '', ''],
        ['Particles', f'{len(particles_high_jitter):,}', f'{len(particles_low_jitter):,}', f'{len(particles_mesh_surface):,}'],
        ['', '', '', ''],
        ['On surface', f'{(dist_to_mesh_high < 0.01).sum()/len(dist_to_mesh_high)*100:.1f}%',
         f'{(dist_to_mesh_low < 0.01).sum()/len(dist_to_mesh_low)*100:.1f}%',
         f'{(dist_to_mesh_mesh < 0.01).sum()/len(dist_to_mesh_mesh)*100:.1f}%'],
        ['Near surface', f'{(dist_to_mesh_high < 0.05).sum()/len(dist_to_mesh_high)*100:.1f}%',
         f'{(dist_to_mesh_low < 0.05).sum()/len(dist_to_mesh_low)*100:.1f}%',
         f'{(dist_to_mesh_mesh < 0.05).sum()/len(dist_to_mesh_mesh)*100:.1f}%'],
        ['', '', '', ''],
        ['Mean dist', f'{dist_to_mesh_high.mean():.4f}', f'{dist_to_mesh_low.mean():.4f}', f'{dist_to_mesh_mesh.mean():.4f}'],
        ['Max dist', f'{dist_to_mesh_high.max():.4f}', f'{dist_to_mesh_low.max():.4f}', f'{dist_to_mesh_mesh.max():.4f}'],
        ['', '', '', ''],
        ['Coverage', f'{coverage_high*100:.1f}%', f'{coverage_low*100:.1f}%', f'{coverage_mesh*100:.1f}%'],
    ]

    table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.23, 0.23, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code results
    # High jitter column
    table[(5, 1)].set_facecolor('#FFCCCC')  # Low on-surface %
    # Low jitter column
    table[(11, 2)].set_facecolor('#FFCCCC')  # Low coverage
    # Mesh surface column (all good)
    table[(5, 3)].set_facecolor('#CCFFCC')
    table[(11, 3)].set_facecolor('#CCFFCC')

    ax8.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    # ========================================================================
    # Overall title
    # ========================================================================

    fig.suptitle('Surface Extraction Comparison: Subdivision vs Mesh Surface Sampling',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = 'surface_extraction_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys

    mesh_path = "assets/bunny.obj"
    if len(sys.argv) > 1:
        mesh_path = sys.argv[1]

    print("="*70)
    print("Surface Extraction Visualization")
    print("="*70)
    print()

    visualize_surface_extraction(mesh_path)

    print("\n" + "="*70)
    print("Key Insights:")
    print("="*70)
    print("1. High jitter (0.4): Creates particles everywhere, many INSIDE mesh")
    print("   → Good coverage but wrong curvature for interior particles")
    print()
    print("2. Low jitter (0.05): Particles cluster around vertices")
    print("   → GAPS between vertices, incomplete surface coverage")
    print()
    print("3. Mesh surface sampling: Particles uniformly ON mesh faces")
    print("   → Perfect coverage, all particles on true surface ✓")
    print("="*70)
