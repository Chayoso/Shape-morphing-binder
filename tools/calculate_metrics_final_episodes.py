#!/usr/bin/env python3
"""
Calculate IoU and Chamfer Distance for Final Episodes

For each experiment, loads the final episode's reconstructed mesh and compares it
against the target mesh using:
1. Intersection over Union (IoU) via voxelization
2. Chamfer Distance (both directions)
"""

import numpy as np
import trimesh
from pathlib import Path
import yaml
import json
from scipy.spatial import cKDTree


def load_mesh(mesh_path):
    """Load mesh, point cloud, or NPZ Gaussians file."""
    mesh_path = Path(mesh_path)

    # Handle NPZ Gaussians file (contains 'mu' positions)
    if mesh_path.suffix == '.npz':
        print(f"    Note: {mesh_path.name} is Gaussians NPZ, loading particle positions...")
        data = np.load(mesh_path)
        if 'mu' in data:
            points = data['mu']
            print(f"      → Loaded {len(points):,} Gaussian positions")

            # Reconstruct mesh using Poisson surface reconstruction or ball pivoting
            try:
                import open3d as o3d

                # Convert to Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                # Estimate normals with better parameters
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))

                # Orient normals towards consistent direction
                pcd.orient_normals_consistent_tangent_plane(k=20)

                # Optionally orient towards camera/viewpoint for better results
                camera_location = np.array([0.0, 0.0, 100.0])
                pcd.orient_normals_towards_camera_location(camera_location)

                # Use Poisson surface reconstruction with better depth
                print(f"      → Running Poisson surface reconstruction...")
                mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8, width=0, scale=1.05, linear_fit=False
                )

                # Remove low density vertices (artifacts) more aggressively
                densities = np.asarray(densities)
                density_threshold = np.percentile(densities, 10)  # More aggressive filtering
                vertices_to_remove = densities < density_threshold
                mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

                # Clean up the mesh
                mesh_o3d.remove_duplicated_vertices()
                mesh_o3d.remove_duplicated_triangles()
                mesh_o3d.remove_degenerate_triangles()
                mesh_o3d.remove_unreferenced_vertices()

                # Convert back to trimesh
                vertices = np.asarray(mesh_o3d.vertices)
                faces = np.asarray(mesh_o3d.triangles)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                # Fix mesh orientation if volume is negative
                if mesh.volume < 0:
                    mesh.invert()
                    print(f"      → Fixed mesh orientation (was inverted)")

                # Try to make watertight
                if not mesh.is_watertight:
                    mesh.fill_holes()

                print(f"      → Reconstructed mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
                print(f"      → Volume: {mesh.volume:.4f}, Watertight: {mesh.is_watertight}")
                return mesh

            except ImportError:
                print(f"      → Open3D not available, falling back to convex hull...")
                try:
                    mesh = trimesh.convex.convex_hull(points)
                    print(f"      → Reconstructed mesh (convex hull): {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
                    return mesh
                except Exception as e:
                    print(f"      → Failed to reconstruct mesh: {e}, using point cloud")
                    return trimesh.PointCloud(points)

            except Exception as e:
                print(f"      → Poisson reconstruction failed: {e}, using point cloud")
                return trimesh.PointCloud(points)
        else:
            raise ValueError(f"NPZ file {mesh_path} does not contain 'mu' key")

    # Handle regular mesh/PLY files
    data = trimesh.load(mesh_path)

    # If it's a point cloud (no faces), create a convex hull or use points directly
    if isinstance(data, trimesh.PointCloud) or (hasattr(data, 'faces') and len(data.faces) == 0):
        print(f"    Note: {Path(mesh_path).name} is a point cloud, reconstructing mesh...")
        # Get points
        if isinstance(data, trimesh.PointCloud):
            points = data.vertices
        else:
            points = data.vertices

        # Reconstruct mesh using Alpha Shape or Ball Pivoting
        try:
            # Try alpha shape (better for smooth surfaces)
            import scipy.spatial
            from scipy.spatial import Delaunay

            # Simple approach: use convex hull for now
            mesh = trimesh.convex.convex_hull(points)
            print(f"      → Reconstructed using convex hull: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh

        except Exception as e:
            print(f"      → Failed to reconstruct mesh: {e}")
            # Return as point cloud for Chamfer distance only
            return trimesh.PointCloud(points)

    return data


def calculate_chamfer_distance(mesh1, mesh2, num_samples=10000):
    """
    Calculate bidirectional Chamfer Distance between two meshes or point clouds.

    Args:
        mesh1: First mesh/point cloud (predicted)
        mesh2: Second mesh/point cloud (target)
        num_samples: Number of points to sample from each mesh

    Returns:
        dict with chamfer_dist, chamfer_1to2, chamfer_2to1
    """
    # Get points from meshes or point clouds
    if isinstance(mesh1, trimesh.PointCloud):
        points1 = mesh1.vertices
        # Subsample if too many points
        if len(points1) > num_samples:
            indices = np.random.choice(len(points1), num_samples, replace=False)
            points1 = points1[indices]
    else:
        points1 = mesh1.sample(num_samples)

    if isinstance(mesh2, trimesh.PointCloud):
        points2 = mesh2.vertices
        if len(points2) > num_samples:
            indices = np.random.choice(len(points2), num_samples, replace=False)
            points2 = points2[indices]
    else:
        points2 = mesh2.sample(num_samples)

    # Build KD-trees
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    # Compute distances
    dist_1to2, _ = tree2.query(points1)
    dist_2to1, _ = tree1.query(points2)

    # Chamfer distance (average of both directions)
    chamfer_1to2 = np.mean(dist_1to2)
    chamfer_2to1 = np.mean(dist_2to1)
    chamfer_dist = (chamfer_1to2 + chamfer_2to1) / 2.0

    return {
        'chamfer_distance': float(chamfer_dist),
        'chamfer_predicted_to_target': float(chamfer_1to2),
        'chamfer_target_to_predicted': float(chamfer_2to1),
        'chamfer_max': float(max(chamfer_1to2, chamfer_2to1))
    }


def calculate_iou_voxel(mesh1, mesh2, voxel_size=0.1):
    """
    Calculate IoU using voxelization.

    Args:
        mesh1: First mesh (predicted)
        mesh2: Second mesh (target)
        voxel_size: Voxel grid resolution

    Returns:
        float: IoU score
    """
    # Determine bounding box that encompasses both meshes
    bounds1 = mesh1.bounds
    bounds2 = mesh2.bounds

    min_bound = np.minimum(bounds1[0], bounds2[0]) - voxel_size
    max_bound = np.maximum(bounds1[1], bounds2[1]) + voxel_size

    # Create voxel grids
    voxel1 = mesh1.voxelized(pitch=voxel_size)
    voxel2 = mesh2.voxelized(pitch=voxel_size)

    # Get filled voxel matrices
    matrix1 = voxel1.matrix
    matrix2 = voxel2.matrix

    # Align the grids (they may have different origins/sizes)
    # Get their transforms and create aligned boolean arrays

    # Simple approach: create unified grid and fill both
    grid_shape = np.maximum(matrix1.shape, matrix2.shape)

    # Resize arrays if needed
    def resize_to_shape(arr, target_shape):
        result = np.zeros(target_shape, dtype=bool)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
        result[slices] = arr[slices]
        return result

    matrix1_resized = resize_to_shape(matrix1, grid_shape)
    matrix2_resized = resize_to_shape(matrix2, grid_shape)

    # Calculate IoU
    intersection = np.logical_and(matrix1_resized, matrix2_resized).sum()
    union = np.logical_or(matrix1_resized, matrix2_resized).sum()

    if union == 0:
        return 0.0

    iou = intersection / union
    return float(iou)


def calculate_volumetric_iou(mesh1, mesh2, voxel_size=0.1):
    """
    More robust volumetric IoU calculation using watertight meshes.
    """
    try:
        # Ensure meshes are watertight
        if not mesh1.is_watertight:
            print(f"    Warning: Predicted mesh is not watertight, attempting to fix...")
            mesh1.fill_holes()

        if not mesh2.is_watertight:
            print(f"    Warning: Target mesh is not watertight, attempting to fix...")
            mesh2.fill_holes()

        # Use voxelization
        iou = calculate_iou_voxel(mesh1, mesh2, voxel_size)
        return iou

    except Exception as e:
        print(f"    Error calculating IoU: {e}")
        return None


def process_experiment(exp_dir):
    """
    Process a single experiment directory.

    Args:
        exp_dir: Path to experiment directory (e.g., output/physics/bunny_1/)

    Returns:
        dict with metrics or None if failed
    """
    exp_dir = Path(exp_dir)
    exp_name = exp_dir.name

    print(f"\n{'='*70}")
    print(f"Processing: {exp_name}")
    print(f"{'='*70}")

    # Find config file
    config_path = Path("configs/physics") / f"{exp_name}.yaml"
    if not config_path.exists():
        print(f"  ⚠ Config not found: {config_path}")
        return None

    # Load config to get target mesh
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    target_mesh_path = config.get('target_mesh_path')
    if not target_mesh_path:
        print(f"  ⚠ No target_mesh_path in config")
        return None

    target_mesh_path = Path(target_mesh_path)
    if not target_mesh_path.exists():
        print(f"  ⚠ Target mesh not found: {target_mesh_path}")
        return None

    # Find final episode
    episode_dirs = sorted(exp_dir.glob("ep*"))
    if not episode_dirs:
        print(f"  ⚠ No episodes found")
        return None

    final_episode = episode_dirs[-1]
    final_episode_num = int(final_episode.name.replace('ep', ''))

    print(f"  Final episode: {final_episode.name}")

    # Find predicted data - prefer Gaussians NPZ (100K particles) over surface PLY (coarse)
    gaussians_file = final_episode / f"{final_episode.name}_gaussians.npz"

    if gaussians_file.exists():
        predicted_mesh_path = gaussians_file
        print(f"  Predicted data: {predicted_mesh_path.name} (full Gaussians)")
    else:
        # Fall back to PLY if no gaussians
        predicted_mesh_files = list(final_episode.glob("*_surface_*.ply"))
        if not predicted_mesh_files:
            print(f"  ⚠ No surface data found in {final_episode}")
            return None
        predicted_mesh_path = predicted_mesh_files[0]
        print(f"  Predicted data: {predicted_mesh_path.name} (PLY fallback)")

    print(f"  Target mesh: {target_mesh_path.name}")

    # Load meshes
    print(f"\n  Loading meshes...")
    try:
        predicted_mesh = load_mesh(predicted_mesh_path)
        target_mesh = load_mesh(target_mesh_path)

        print(f"    Predicted: {len(predicted_mesh.vertices):,} vertices, {len(predicted_mesh.faces):,} faces")
        print(f"    Target: {len(target_mesh.vertices):,} vertices, {len(target_mesh.faces):,} faces")
    except Exception as e:
        print(f"  ⚠ Failed to load meshes: {e}")
        return None

    # Normalize meshes to same scale and center
    print(f"\n  Normalizing meshes...")

    # Get centroids
    if isinstance(predicted_mesh, trimesh.PointCloud):
        pred_centroid = np.mean(predicted_mesh.vertices, axis=0)
    else:
        pred_centroid = predicted_mesh.centroid

    if isinstance(target_mesh, trimesh.PointCloud):
        target_centroid = np.mean(target_mesh.vertices, axis=0)
    else:
        target_centroid = target_mesh.centroid

    # Center both meshes at origin
    predicted_mesh.vertices -= pred_centroid
    target_mesh.vertices -= target_centroid

    # Scale to unit sphere (max distance from origin = 1)
    if len(predicted_mesh.vertices) > 0:
        predicted_scale = np.max(np.linalg.norm(predicted_mesh.vertices, axis=1))
        if predicted_scale > 0:
            predicted_mesh.vertices /= predicted_scale
    else:
        print(f"    Warning: Predicted mesh has no vertices!")
        return None

    if len(target_mesh.vertices) > 0:
        target_scale = np.max(np.linalg.norm(target_mesh.vertices, axis=1))
        if target_scale > 0:
            target_mesh.vertices /= target_scale
    else:
        print(f"    Warning: Target mesh has no vertices!")
        return None

    print(f"    Normalized to unit sphere")

    # Calculate Chamfer Distance
    print(f"\n  Calculating Chamfer Distance...")
    chamfer_metrics = calculate_chamfer_distance(predicted_mesh, target_mesh, num_samples=50000)

    print(f"    Chamfer Distance: {chamfer_metrics['chamfer_distance']:.6f}")
    print(f"    Predicted → Target: {chamfer_metrics['chamfer_predicted_to_target']:.6f}")
    print(f"    Target → Predicted: {chamfer_metrics['chamfer_target_to_predicted']:.6f}")

    # Calculate IoU (only if both are meshes, not point clouds)
    iou = None
    if not isinstance(predicted_mesh, trimesh.PointCloud) and not isinstance(target_mesh, trimesh.PointCloud):
        print(f"\n  Calculating IoU (voxel-based)...")
        # Use larger voxel size for more stable IoU with complex meshes
        iou = calculate_volumetric_iou(predicted_mesh, target_mesh, voxel_size=0.05)

        if iou is not None:
            print(f"    IoU: {iou:.6f}")
        else:
            print(f"    IoU: Failed to calculate")
    else:
        print(f"\n  ⚠ Skipping IoU (point cloud detected, IoU requires watertight mesh)")
        iou = None

    # Compile results
    try:
        pred_mesh_rel = str(predicted_mesh_path.relative_to(Path.cwd()))
    except ValueError:
        pred_mesh_rel = str(predicted_mesh_path)

    results = {
        'experiment': exp_name,
        'final_episode': final_episode_num,
        'predicted_mesh': pred_mesh_rel,
        'target_mesh': str(target_mesh_path),
        'metrics': {
            **chamfer_metrics,
            'iou': iou
        },
        'mesh_stats': {
            'predicted_vertices': len(predicted_mesh.vertices),
            'predicted_faces': len(predicted_mesh.faces) if not isinstance(predicted_mesh, trimesh.PointCloud) else 0,
            'target_vertices': len(target_mesh.vertices),
            'target_faces': len(target_mesh.faces) if not isinstance(target_mesh, trimesh.PointCloud) else 0,
            'predicted_is_point_cloud': isinstance(predicted_mesh, trimesh.PointCloud),
            'target_is_point_cloud': isinstance(target_mesh, trimesh.PointCloud)
        }
    }

    # Save individual result
    results_file = final_episode / "metrics_final.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ Saved metrics to: {results_file}")

    return results


def main():
    """Process all experiments in physics directory."""

    print(f"\n{'#'*70}")
    print(f"# Calculate IoU and Chamfer Distance for Final Episodes")
    print(f"{'#'*70}")

    physics_dir = Path("output/physics")

    if not physics_dir.exists():
        print(f"❌ Physics output directory not found: {physics_dir}")
        return

    # Find all experiment directories
    exp_dirs = sorted([d for d in physics_dir.iterdir() if d.is_dir()])

    print(f"\nFound {len(exp_dirs)} experiments")

    all_results = []

    for exp_dir in exp_dirs:
        result = process_experiment(exp_dir)
        if result:
            all_results.append(result)

    # Save combined results
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")

    if all_results:
        summary_file = physics_dir / "metrics_summary_final_episodes.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✅ Processed {len(all_results)} experiments")
        print(f"✅ Saved summary to: {summary_file}")

        # Print summary table
        print(f"\n{'Experiment':<20} {'Episode':<10} {'Chamfer↓':<12} {'IoU↑':<10}")
        print(f"{'-'*60}")
        for result in all_results:
            exp = result['experiment']
            ep = f"ep{result['final_episode']:03d}"
            chamfer = result['metrics']['chamfer_distance']
            iou = result['metrics']['iou']
            iou_str = f"{iou:.4f}" if iou is not None else "N/A"
            print(f"{exp:<20} {ep:<10} {chamfer:<12.6f} {iou_str:<10}")
    else:
        print(f"\n⚠ No experiments processed successfully")

    print(f"\n{'#'*70}\n")


if __name__ == "__main__":
    main()
