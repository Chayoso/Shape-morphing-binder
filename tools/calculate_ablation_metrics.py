#!/usr/bin/env python3
"""
Calculate Chamfer Distance and IoU for All Ablation Experiments

Processes all experiments in output/ablation/ directory
"""

import numpy as np
import trimesh
from pathlib import Path
import json
from scipy.spatial import cKDTree
import open3d as o3d


def load_and_reconstruct_mesh(gaussians_path):
    """Load Gaussians and reconstruct mesh using Poisson."""
    data = np.load(gaussians_path)
    if 'mu' not in data:
        return None

    points = data['mu']
    print(f"      → Loaded {len(points):,} Gaussian positions")

    # Convert to Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
    pcd.orient_normals_consistent_tangent_plane(k=20)

    # Orient towards camera
    camera_location = np.array([0.0, 0.0, 100.0])
    pcd.orient_normals_towards_camera_location(camera_location)

    # Poisson reconstruction
    print(f"      → Running Poisson surface reconstruction...")
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, width=0, scale=1.05, linear_fit=False
    )

    # Remove low density vertices
    densities = np.asarray(densities)
    density_threshold = np.percentile(densities, 10)
    vertices_to_remove = densities < density_threshold
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    # Clean up
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_unreferenced_vertices()

    # Convert to trimesh
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Fix orientation if needed
    if mesh.volume < 0:
        mesh.invert()

    # Try to make watertight
    if not mesh.is_watertight:
        mesh.fill_holes()

    print(f"      → Reconstructed mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    return mesh


def calculate_chamfer_distance(mesh1, mesh2, num_samples=50000):
    """Calculate bidirectional Chamfer Distance."""
    # Sample points
    if isinstance(mesh1, trimesh.PointCloud):
        points1 = mesh1.vertices
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

    # Chamfer distance
    chamfer_1to2 = np.mean(dist_1to2)
    chamfer_2to1 = np.mean(dist_2to1)
    chamfer_dist = (chamfer_1to2 + chamfer_2to1) / 2.0

    return {
        'chamfer_distance': float(chamfer_dist),
        'chamfer_predicted_to_target': float(chamfer_1to2),
        'chamfer_target_to_predicted': float(chamfer_2to1),
    }


def calculate_iou(mesh1, mesh2, voxel_size=0.05):
    """Calculate IoU using voxelization."""
    try:
        # Ensure watertight
        if not mesh1.is_watertight:
            mesh1.fill_holes()
        if not mesh2.is_watertight:
            mesh2.fill_holes()

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

        if union == 0:
            return 0.0

        iou = intersection / union
        return float(iou)

    except Exception as e:
        print(f"      Warning: IoU calculation failed: {e}")
        return None


def process_ablation_experiment(exp_dir, target_mesh):
    """Process a single ablation experiment."""
    exp_dir = Path(exp_dir)
    exp_name = exp_dir.name

    print(f"\n{'='*70}")
    print(f"Processing: {exp_name}")
    print(f"{'='*70}")

    # Find final episode
    episode_dirs = sorted(exp_dir.glob("ep*"))
    if not episode_dirs:
        print(f"  ⚠ No episodes found")
        return None

    final_episode = episode_dirs[-1]
    final_episode_num = int(final_episode.name.replace('ep', ''))

    print(f"  Final episode: {final_episode.name}")

    # Find Gaussians file
    gaussians_file = final_episode / f"{final_episode.name}_gaussians.npz"

    if not gaussians_file.exists():
        print(f"  ⚠ Gaussians file not found: {gaussians_file}")
        return None

    print(f"  Gaussians file: {gaussians_file.name}")

    # Load and reconstruct mesh
    print(f"\n  Loading and reconstructing mesh...")
    try:
        predicted_mesh = load_and_reconstruct_mesh(gaussians_file)
        if predicted_mesh is None:
            print(f"  ⚠ Failed to load mesh")
            return None
    except Exception as e:
        print(f"  ⚠ Error loading mesh: {e}")
        return None

    print(f"    Predicted: {len(predicted_mesh.vertices):,} vertices, {len(predicted_mesh.faces):,} faces")
    print(f"    Target: {len(target_mesh.vertices):,} vertices, {len(target_mesh.faces):,} faces")

    # Normalize meshes
    print(f"\n  Normalizing meshes...")

    # Center at origin
    pred_centroid = np.mean(predicted_mesh.vertices, axis=0) if isinstance(predicted_mesh, trimesh.PointCloud) else predicted_mesh.centroid
    target_centroid = target_mesh.centroid

    predicted_mesh.vertices -= pred_centroid
    target_mesh.vertices -= target_centroid

    # Scale to unit sphere
    pred_scale = np.max(np.linalg.norm(predicted_mesh.vertices, axis=1))
    target_scale = np.max(np.linalg.norm(target_mesh.vertices, axis=1))

    if pred_scale > 0:
        predicted_mesh.vertices /= pred_scale
    if target_scale > 0:
        target_mesh.vertices /= target_scale

    print(f"    Normalized to unit sphere")

    # Calculate Chamfer Distance
    print(f"\n  Calculating Chamfer Distance...")
    chamfer_metrics = calculate_chamfer_distance(predicted_mesh, target_mesh, num_samples=50000)

    print(f"    Chamfer Distance: {chamfer_metrics['chamfer_distance']:.6f}")
    print(f"    Predicted → Target: {chamfer_metrics['chamfer_predicted_to_target']:.6f}")
    print(f"    Target → Predicted: {chamfer_metrics['chamfer_target_to_predicted']:.6f}")

    # Calculate IoU
    print(f"\n  Calculating IoU (voxel-based)...")
    iou = calculate_iou(predicted_mesh, target_mesh, voxel_size=0.05)

    if iou is not None:
        print(f"    IoU: {iou:.6f}")
    else:
        print(f"    IoU: Failed to calculate")

    # Compile results
    results = {
        'experiment': exp_name,
        'final_episode': final_episode_num,
        'predicted_mesh': str(gaussians_file),
        'target_mesh': 'assets/bunny.obj',
        'metrics': {
            **chamfer_metrics,
            'iou': iou
        },
        'mesh_stats': {
            'predicted_vertices': len(predicted_mesh.vertices),
            'predicted_faces': len(predicted_mesh.faces) if not isinstance(predicted_mesh, trimesh.PointCloud) else 0,
            'target_vertices': len(target_mesh.vertices),
            'target_faces': len(target_mesh.faces),
        }
    }

    # Save individual result
    results_file = final_episode / "metrics_final.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ Saved metrics to: {results_file}")

    return results


def main():
    """Calculate metrics for all ablation experiments."""

    print(f"\n{'#'*70}")
    print(f"# Calculate Metrics for Ablation Study Experiments")
    print(f"{'#'*70}")

    ablation_dir = Path("output/ablation")

    if not ablation_dir.exists():
        print(f"❌ Ablation directory not found: {ablation_dir}")
        return

    # Load target mesh once
    target_mesh_path = Path("assets/bunny.obj")
    print(f"\nLoading target mesh: {target_mesh_path}")
    target_mesh = trimesh.load(target_mesh_path)
    print(f"  Target: {len(target_mesh.vertices):,} vertices, {len(target_mesh.faces):,} faces")

    # Find all experiment directories
    exp_dirs = sorted([d for d in ablation_dir.iterdir() if d.is_dir()])

    print(f"\nFound {len(exp_dirs)} ablation experiments")

    all_results = []

    for exp_dir in exp_dirs:
        # Reload target mesh for each experiment (gets modified during normalization)
        target_mesh = trimesh.load(target_mesh_path)

        result = process_ablation_experiment(exp_dir, target_mesh)
        if result:
            all_results.append(result)

    # Save combined results
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")

    if all_results:
        summary_file = ablation_dir / "metrics_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✅ Processed {len(all_results)} experiments")
        print(f"✅ Saved summary to: {summary_file}")

        # Print summary table
        print(f"\n{'Experiment':<30} {'Episode':<10} {'Chamfer↓':<14} {'IoU↑':<10}")
        print(f"{'-'*70}")
        for result in sorted(all_results, key=lambda x: x['experiment']):
            exp = result['experiment']
            ep = f"ep{result['final_episode']:03d}"
            chamfer = result['metrics']['chamfer_distance']
            iou = result['metrics']['iou']
            iou_str = f"{iou:.4f}" if iou is not None else "N/A"
            print(f"{exp:<30} {ep:<10} {chamfer:<14.6f} {iou_str:<10}")

        # Generate LaTeX table
        generate_latex_table(all_results)
    else:
        print(f"\n⚠ No experiments processed successfully")

    print(f"\n{'#'*70}\n")


def generate_latex_table(results):
    """Generate LaTeX table for paper."""

    print(f"\n{'='*70}")
    print(f"LaTeX Table")
    print(f"{'='*70}\n")

    # Sort experiments in logical order
    order = {
        '01_physics_only': 1,
        '02_alpha_only': 2,
        '03_depth_only': 3,
        '04_full_model': 4,
        'no_depth_loss': 5,
        'no_alpha_loss': 6,
        'no_render_loss': 7
    }

    sorted_results = sorted(results, key=lambda x: order.get(x['experiment'], 99))

    latex = r"""\begin{table}[t]
\centering
    \caption{\textbf{Ablation Study Results (Sphere to Bunny).} Chamfer Distance (CD, $\downarrow$, $\times 10^{-2}$) and IoU ($\uparrow$, \%).}
    \label{tab:ablation}
    \begin{tabular}{l|c|c}
    \toprule
    \textbf{Configuration} & \textbf{Chamfer Dist. $\downarrow$} & \textbf{IoU $\uparrow$} \\
    \midrule
"""

    for result in sorted_results:
        name = result['experiment'].replace('_', ' ').title()
        cd = result['metrics']['chamfer_distance'] * 100
        iou = result['metrics']['iou'] * 100 if result['metrics']['iou'] else 0.0

        # Bold the full model
        if '04' in result['experiment'] or 'full' in result['experiment'].lower():
            latex += f"    \\textbf{{{name}}} & \\textbf{{{cd:.2f}}} & \\textbf{{{iou:.2f}}} \\\\\n"
        else:
            latex += f"    {name} & {cd:.2f} & {iou:.2f} \\\\\n"

    latex += r"""    \bottomrule
    \end{tabular}
\end{table}
"""

    print(latex)

    # Save to file
    output_file = Path("output/ablation/ablation_table.tex")
    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"\n✅ Saved LaTeX table to: {output_file}\n")


if __name__ == "__main__":
    main()
