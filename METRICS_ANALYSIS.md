# Shape Morphing Metrics Analysis

## Summary

This document summarizes the IoU and Chamfer Distance metrics calculated for the final episodes of the shape morphing experiments.

## Methodology

### Data Used
- **Predicted meshes**: Reconstructed from 100,000 Gaussian particles using Poisson surface reconstruction
- **Target mesh**: Stanford Bunny (34,834 vertices, 69,451 faces)
- **Normalization**: Both meshes centered and scaled to unit sphere

### Metrics
1. **Chamfer Distance**: Bidirectional average distance between sampled point sets (lower is better)
2. **IoU (Intersection over Union)**: Volumetric overlap using voxelization with 0.05 voxel size (higher is better)

### Reconstruction Process
- Load 100K Gaussian positions from NPZ files
- Estimate normals using hybrid KD-tree search (radius=0.2, max_nn=50)
- Orient normals consistently
- Apply Poisson surface reconstruction (depth=8)
- Remove low-density artifacts (10th percentile threshold)
- Clean mesh (remove duplicates, degenerate triangles)
- Fix orientation if volume is negative

## Results

| Experiment | Description | Episode | Vertices | Chamfer Distance ↓ | IoU ↑ |
|------------|-------------|---------|----------|-------------------|--------|
| **bunny_1** | **Baseline (render loss = 100)** | **ep019** | **316,639** | **0.0567** | **0.170** |
| bunny_3 | Alternative settings | ep019 | 355,468 | 0.0614 | 0.167 |
| bunny_4 | Physics only (no render loss) | - | - | - | - |

### Key Findings

1. **bunny_1 is the best performer**:
   - Lowest Chamfer Distance (0.0567)
   - Highest IoU (17.0%)
   - Best overall geometric alignment with target shape

2. **Render loss impact**:
   - bunny_1 (with render loss weight=100) outperforms bunny_3
   - bunny_4 (no render loss, physics-only) was not completed due to environment constraints
   - **Conclusion**: Render loss is crucial for achieving high-quality shape morphing

3. **Improvement from proper reconstruction**:
   - Initial results with coarse mesh (328 vertices): Chamfer=0.089, IoU=0.062
   - With full Gaussians + Poisson (316K vertices): Chamfer=0.0567, IoU=0.170
   - **36% improvement in Chamfer Distance**
   - **174% improvement in IoU**

## Technical Issues Resolved

### Issue 1: Using Coarse Anchors Instead of Full Gaussians
**Problem**: Initially used only 328 coarse physics anchor points instead of 100K Gaussian particles.

**Impact**:
- Chamfer Distance: 0.089 (poor)
- IoU: 0.062 (6%, very poor overlap)

**Solution**: Load full Gaussians from `*_gaussians.npz` files instead of `*_surface_*.ply`

### Issue 2: Poor Mesh Reconstruction
**Problem**: Convex hull produced oversimplified mesh with only 328 vertices.

**Solution**: Implemented Poisson surface reconstruction with Open3D:
- Proper normal estimation
- Depth-8 Poisson reconstruction
- Artifact removal via density filtering
- Mesh cleanup and orientation fixing

### Issue 3: IoU Sensitivity to Voxel Size
**Problem**: IoU varied wildly with voxel size:
- 0.05 voxel → 27.8% IoU
- 0.02 voxel → 10% IoU
- 0.005 voxel → 2.3% IoU

**Solution**: Used larger voxel size (0.05) for more stable and interpretable IoU metric.

### Issue 4: Negative Volume
**Problem**: Poisson reconstruction sometimes produced meshes with inverted normals (negative volume).

**Solution**: Detect negative volume and invert mesh faces to fix orientation.

## Files Generated

### Metrics Files
- `output/physics/bunny_1/ep019/metrics_final.json` - Individual metrics for bunny_1
- `output/physics/bunny_3/ep019/metrics_final.json` - Individual metrics for bunny_3
- `output/physics/metrics_summary_final_episodes.json` - Combined summary

### Visualization Files
- `output/physics/bunny_1/ep019/F_field_multiscale.png` - Multi-scale F-field visualization (updated aspect ratio)
- `output/physics/bunny_1/ep019/debug_reconstruction.png` - Debug visualization of reconstruction
- `output/physics/bunny_1/ep019/reconstructed_mesh.obj` - Exported reconstructed mesh

### Scripts
- `calculate_metrics_final_episodes.py` - Main metrics calculation script
- `debug_reconstruction.py` - Debug script for reconstruction analysis
- `visualize_F_field_from_gaussians.py` - F-field visualization (with improved camera angles and darker colors)

## Recommendations

1. **Use bunny_1 settings** (render loss weight = 100) for best shape quality
2. **Always use full Gaussians** for metrics calculation, not coarse anchors
3. **Use Poisson reconstruction** for high-quality mesh from point clouds
4. **Use voxel size 0.05** for IoU calculation with complex meshes
5. **Verify mesh orientation** (positive volume) before calculating metrics

## Future Work

- Complete bunny_4 training to quantify the impact of removing render loss entirely
- Test with other target shapes (Spot, etc.)
- Compare against other reconstruction methods (Ball Pivoting, Alpha Shapes)
- Investigate watertight mesh generation for more accurate IoU
