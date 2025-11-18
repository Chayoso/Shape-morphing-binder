#!/usr/bin/env python3
"""
Debug script to visualize the reconstruction quality and understand IoU issues
"""

import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_and_reconstruct(gaussians_path):
    """Load Gaussians and reconstruct mesh."""
    print(f"\nLoading: {gaussians_path}")
    data = np.load(gaussians_path)
    points = data['mu']
    print(f"  Loaded {len(points):,} points")

    # Convert to Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # Poisson reconstruction
    print("  Running Poisson reconstruction...")
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False
    )

    print(f"  Vertices before filtering: {len(mesh_o3d.vertices):,}")

    # Remove low density vertices
    densities = np.asarray(densities)
    print(f"  Density range: [{densities.min():.4f}, {densities.max():.4f}]")
    print(f"  Density percentiles: 5%={np.percentile(densities, 5):.4f}, 10%={np.percentile(densities, 10):.4f}, 25%={np.percentile(densities, 25):.4f}")

    density_threshold = np.percentile(densities, 5)
    vertices_to_remove = densities < density_threshold
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    print(f"  Vertices after filtering: {len(mesh_o3d.vertices):,}")

    # Convert to trimesh
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    print(f"  Is watertight: {mesh.is_watertight}")
    print(f"  Volume: {mesh.volume:.6f}")
    print(f"  Bounds: {mesh.bounds}")

    return mesh, points


def visualize_comparison(mesh_pred, mesh_target, points, output_path):
    """Visualize predicted vs target mesh."""
    fig = plt.figure(figsize=(20, 8))

    # Plot 1: Point cloud
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, alpha=0.3, c='blue')
    ax1.set_title(f'Original Point Cloud\n{len(points):,} points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot 2: Reconstructed mesh
    ax2 = fig.add_subplot(132, projection='3d')
    vertices = mesh_pred.vertices
    ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.1, alpha=0.3, c='red')
    ax2.set_title(f'Reconstructed Mesh\n{len(mesh_pred.vertices):,} vertices, {len(mesh_pred.faces):,} faces')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Plot 3: Target mesh
    ax3 = fig.add_subplot(133, projection='3d')
    vertices = mesh_target.vertices
    ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.5, alpha=0.3, c='green')
    ax3.set_title(f'Target Mesh\n{len(mesh_target.vertices):,} vertices, {len(mesh_target.faces):,} faces')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization: {output_path}")


def test_iou_with_different_voxel_sizes(mesh1, mesh2):
    """Test IoU with different voxel sizes."""
    print("\n  Testing IoU with different voxel sizes:")

    voxel_sizes = [0.05, 0.02, 0.01, 0.005]

    for voxel_size in voxel_sizes:
        try:
            # Create voxel grids
            voxel1 = mesh1.voxelized(pitch=voxel_size)
            voxel2 = mesh2.voxelized(pitch=voxel_size)

            matrix1 = voxel1.matrix
            matrix2 = voxel2.matrix

            # Align grids
            max_shape = np.maximum(matrix1.shape, matrix2.shape)

            def resize_to_shape(arr, target_shape):
                result = np.zeros(target_shape, dtype=bool)
                slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
                result[slices] = arr[slices]
                return result

            matrix1_resized = resize_to_shape(matrix1, max_shape)
            matrix2_resized = resize_to_shape(matrix2, max_shape)

            # Calculate IoU
            intersection = np.logical_and(matrix1_resized, matrix2_resized).sum()
            union = np.logical_or(matrix1_resized, matrix2_resized).sum()

            iou = intersection / union if union > 0 else 0.0

            print(f"    Voxel size {voxel_size:.4f}: IoU = {iou:.6f} (intersection={intersection:,}, union={union:,})")

        except Exception as e:
            print(f"    Voxel size {voxel_size:.4f}: Failed - {e}")


def main():
    """Debug IoU calculation issues."""

    print("="*70)
    print("DEBUG: Reconstruction and IoU Analysis")
    print("="*70)

    # Test bunny_1
    gaussians_path = Path("output/physics/bunny_1/ep019/ep019_gaussians.npz")
    target_path = Path("assets/bunny.obj")

    mesh_pred, points = load_and_reconstruct(gaussians_path)

    print(f"\nLoading target mesh: {target_path}")
    mesh_target = trimesh.load(target_path)
    print(f"  Target vertices: {len(mesh_target.vertices):,}")
    print(f"  Target faces: {len(mesh_target.faces):,}")
    print(f"  Target is watertight: {mesh_target.is_watertight}")
    print(f"  Target volume: {mesh_target.volume:.6f}")
    print(f"  Target bounds: {mesh_target.bounds}")

    # Normalize both to unit sphere
    print("\nNormalizing meshes...")
    mesh_pred.vertices -= np.mean(mesh_pred.vertices, axis=0)
    pred_scale = np.max(np.linalg.norm(mesh_pred.vertices, axis=1))
    mesh_pred.vertices /= pred_scale

    mesh_target.vertices -= mesh_target.centroid
    target_scale = np.max(np.linalg.norm(mesh_target.vertices, axis=1))
    mesh_target.vertices /= target_scale

    print(f"  Predicted scale factor: {pred_scale:.4f}")
    print(f"  Target scale factor: {target_scale:.4f}")

    # Test IoU with different voxel sizes
    test_iou_with_different_voxel_sizes(mesh_pred, mesh_target)

    # Visualize
    output_path = Path("output/physics/bunny_1/ep019/debug_reconstruction.png")
    visualize_comparison(mesh_pred, mesh_target, points, output_path)

    # Export meshes for manual inspection
    mesh_pred.export("output/physics/bunny_1/ep019/reconstructed_mesh.obj")
    print(f"\n  Exported reconstructed mesh: output/physics/bunny_1/ep019/reconstructed_mesh.obj")

    print("\n" + "="*70)
    print("✅ Debug complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
