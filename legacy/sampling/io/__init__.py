"""
Input/Output utilities for point clouds and meshes.

Includes:
- Point cloud export (PLY)
- Gaussian splatting data export (NPZ)
- Visualization export (PNG)
- Matplotlib setup utilities
"""

from .export import (
    # Visualization
    save_comparison_png,
    save_axis_hist_png,
    
    # Point cloud export
    save_ply_xyz,
    
    # Gaussian export
    save_gaussians_npz,
    
    # Utilities
    setup_matplotlib,
    compute_plot_bounds,
    set_axis_limits,
)

__all__ = [
    # Visualization
    "save_comparison_png",
    "save_axis_hist_png",
    
    # Point cloud export
    "save_ply_xyz",
    
    # Gaussian export
    "save_gaussians_npz",
    
    # Utilities
    "setup_matplotlib",
    "compute_plot_bounds",
    "set_axis_limits",
]