#!/usr/bin/env python3
"""
Visualize particle positions vs target mesh to check coverage.

Usage:
    python visualize_particles.py output/sphere/bunny/ep049/
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import trimesh


def load_mesh(mesh_path):
    """Load OBJ mesh."""
    mesh = trimesh.load(mesh_path)
    return np.array(mesh.vertices)


def load_particles_from_stage(stage_png_path):
    """
    Extract particles from STAGE1.png (crude but works).
    Better: load from actual NPZ if available.
    """
    # For now, return None - need actual particle data
    return None


def visualize_coverage(particles, target_mesh_vertices, title="Particle Coverage"):
    """Visualize particles vs target mesh."""
    fig = plt.figure(figsize=(15, 5))

    # View 1: XY plane
    ax1 = fig.add_subplot(131)
    ax1.scatter(target_mesh_vertices[:, 0], target_mesh_vertices[:, 1],
                c='red', s=1, alpha=0.3, label='Target')
    if particles is not None:
        ax1.scatter(particles[:, 0], particles[:, 1],
                   c='blue', s=5, alpha=0.6, label='Particles')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('XY View')
    ax1.legend()
    ax1.axis('equal')

    # View 2: XZ plane (side view - see ears!)
    ax2 = fig.add_subplot(132)
    ax2.scatter(target_mesh_vertices[:, 0], target_mesh_vertices[:, 2],
                c='red', s=1, alpha=0.3, label='Target')
    if particles is not None:
        ax2.scatter(particles[:, 0], particles[:, 2],
                   c='blue', s=5, alpha=0.6, label='Particles')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z (Height)')
    ax2.set_title('XZ View (EARS SHOULD BE VISIBLE HERE!)')
    ax2.legend()
    ax2.axis('equal')

    # View 3: 3D
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(target_mesh_vertices[:, 0], target_mesh_vertices[:, 1],
                target_mesh_vertices[:, 2], c='red', s=1, alpha=0.2, label='Target')
    if particles is not None:
        ax3.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                   c='blue', s=20, alpha=0.8, label='Particles')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D View')
    ax3.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('particle_coverage_check.png', dpi=150)
    print(f"Saved visualization to: particle_coverage_check.png")
    plt.show()


def check_ear_coverage(particles, target_mesh_vertices):
    """Check if particles cover the ear region."""
    # Bunny ears are typically at:
    # - High Z (top of head)
    # - Moderate X (slightly off center)

    # Find ear region in target
    z_max = target_mesh_vertices[:, 2].max()
    ear_threshold = z_max * 0.7  # Top 30% of Z

    target_ear_verts = target_mesh_vertices[target_mesh_vertices[:, 2] > ear_threshold]

    if particles is not None:
        particle_ear_count = (particles[:, 2] > ear_threshold).sum()
        print(f"\n🔍 Ear Region Analysis:")
        print(f"  Target vertices in ear region: {len(target_ear_verts)}")
        print(f"  Particles in ear region: {particle_ear_count}")
        print(f"  Coverage: {particle_ear_count / len(target_ear_verts) * 100:.1f}%")

        if particle_ear_count < 10:
            print(f"  ❌ PROBLEM: Very few particles in ear region!")
            print(f"  → Physics optimizer likely stuck in local minimum")
    else:
        print(f"\n🔍 Ear Region in Target:")
        print(f"  Target vertices in ear region: {len(target_ear_verts)}")
        print(f"  Ear Z range: [{target_ear_verts[:, 2].min():.2f}, {target_ear_verts[:, 2].max():.2f}]")


if __name__ == '__main__':
    # For now, just load and visualize target mesh
    target_mesh = load_mesh('assets/bunny.obj')

    print(f"Target mesh stats:")
    print(f"  Vertices: {len(target_mesh)}")
    print(f"  X range: [{target_mesh[:, 0].min():.2f}, {target_mesh[:, 0].max():.2f}]")
    print(f"  Y range: [{target_mesh[:, 1].min():.2f}, {target_mesh[:, 1].max():.2f}]")
    print(f"  Z range: [{target_mesh[:, 2].min():.2f}, {target_mesh[:, 2].max():.2f}]")

    check_ear_coverage(None, target_mesh)
    visualize_coverage(None, target_mesh, title="Target Bunny Mesh (No particles loaded yet)")
