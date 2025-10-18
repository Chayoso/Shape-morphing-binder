"""
sampling - Point Sampling and Surface Processing

A modular system for point cloud sampling, surface tracking, and analysis
for shape morphing and deformable object simulation.

Components:
    - Core: Point sampling, Gumbel sampling, runtime surface, synthesis
    - Analysis: Density, PCA, KNN, feature detection
    - Processing: Deformation, smoothing
    - Geometry: Moving least squares
    - IO: Export utilities
    - Utils: Configuration and helper functions

Example:
    >>> from sampling import default_cfg, synthesize_runtime_surface
    >>> 
    >>> # Get default configuration
    >>> cfg = default_cfg()
    >>> cfg["M"] = 50_000
    >>> 
    >>> # Synthesize runtime surface
    >>> result = synthesize_runtime_surface(x_low, F_low, cfg)
    >>> points = result["points"]
    >>> cov = result["cov"]
    >>> 
    >>> # Save results
    >>> save_ply_xyz("output.ply", points)
"""

__version__ = "2.0.0"

# ============================================================================
# Core Sampling
# ============================================================================
from .core import (
    # Point sampling
    sample_surface_points_diff,
    compute_importance_weights,
    build_tangent_frame,
    
    # Gumbel sampling
    generate_gumbel_noise,
    gumbel_softmax_onehot,
    
    # Runtime surface - Main
    default_cfg,
    synthesize_runtime_surface,
    
    # Runtime surface - Helpers
    extract_config_params,
    run_fast_cov_only_path,
    prepare_inputs,
    
    # Synthesis
    synthesize_samples,
    combine_samples,
)

# ============================================================================
# Analysis
# ============================================================================
from .analysis import (
    # KNN
    HybridFAISSKNN,
    
    # PCA
    batched_pca_surface_optimized,
    compute_weighted_centroid,
    compute_weighted_covariance,
    extract_normal_from_pca,
    compute_local_spacing,
    
    # Detection
    compute_surface_mask_diff,
    soft_quantile,
    compute_surface_threshold,
    compute_surface_probability,
    
    # Density
    density_equalize_diff,
    compute_local_density,
    mask_self_neighbors,
    compute_density_displacement,
)

# ============================================================================
# Processing
# ============================================================================
from .processing import (
    # Deformation
    smooth_F_diff_optimized,
    select_graph_nodes,
    build_graph_laplacian,
    build_interpolation_weights,
    
    # Smoothing
    surface_smoother_diff,
    smooth_normals_diff,
    interpolate_normals_at_points,
    compute_laplacian_displacement,
    split_tangent_normal_components,
)

# ============================================================================
# Geometry
# ============================================================================
from .geometry import (
    compute_mls_signed_distance,
    compute_mls_normal,
    project_to_mls_surface_diff,
)

# ============================================================================
# IO
# ============================================================================
from .io import (
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

# ============================================================================
# Utils
# ============================================================================
from .utils import (
    # Configuration
    default_cfg as _default_cfg_utils,  # Avoid duplicate
    
    # Constants
    EPS_NORMALIZE,
    EPS_SAFE,
    EPS_PCA,
    MIN_PROB,
    TANH_SCALE,
    CLAMP_GUMBEL,
    CLAMP_RANDN,
    CLAMP_SPACING,
    CLAMP_KERNEL_EXP,
    
    # Config dictionaries
    DEFAULT_CONFIG,
    ED_CONFIG,
    POST_EQUALIZE_CONFIG,
    SMOOTHER_CONFIG,
    
    # Utilities
    ensure_torch,
    as_numpy,
    normalize,
    validate_positive_definite,
)


__all__ = [
    "__version__",
    
    # ========================================================================
    # Core - Point sampling
    # ========================================================================
    "sample_surface_points_diff",
    "compute_importance_weights",
    "build_tangent_frame",
    
    # ========================================================================
    # Core - Gumbel sampling
    # ========================================================================
    "generate_gumbel_noise",
    "gumbel_softmax_onehot",
    
    # ========================================================================
    # Core - Runtime surface
    # ========================================================================
    "default_cfg",
    "synthesize_runtime_surface",
    "extract_config_params",
    "run_fast_cov_only_path",
    "prepare_inputs",
    
    # ========================================================================
    # Core - Synthesis
    # ========================================================================
    "synthesize_samples",
    "combine_samples",
    
    # ========================================================================
    # Analysis - KNN
    # ========================================================================
    "HybridFAISSKNN",
    
    # ========================================================================
    # Analysis - PCA
    # ========================================================================
    "batched_pca_surface_optimized",
    "compute_weighted_centroid",
    "compute_weighted_covariance",
    "extract_normal_from_pca",
    "compute_local_spacing",
    
    # ========================================================================
    # Analysis - Detection
    # ========================================================================
    "compute_surface_mask_diff",
    "soft_quantile",
    "compute_surface_threshold",
    "compute_surface_probability",
    
    # ========================================================================
    # Analysis - Density
    # ========================================================================
    "density_equalize_diff",
    "compute_local_density",
    "mask_self_neighbors",
    "compute_density_displacement",
    
    # ========================================================================
    # Processing - Deformation
    # ========================================================================
    "smooth_F_diff_optimized",
    "select_graph_nodes",
    "build_graph_laplacian",
    "build_interpolation_weights",
    
    # ========================================================================
    # Processing - Smoothing
    # ========================================================================
    "surface_smoother_diff",
    "smooth_normals_diff",
    "interpolate_normals_at_points",
    "compute_laplacian_displacement",
    "split_tangent_normal_components",
    
    # ========================================================================
    # Geometry - MLS
    # ========================================================================
    "compute_mls_signed_distance",
    "compute_mls_normal",
    "project_to_mls_surface_diff",
    
    # ========================================================================
    # IO - Visualization
    # ========================================================================
    "save_comparison_png",
    "save_axis_hist_png",
    
    # ========================================================================
    # IO - Export
    # ========================================================================
    "save_ply_xyz",
    "save_gaussians_npz",
    
    # ========================================================================
    # IO - Utilities
    # ========================================================================
    "setup_matplotlib",
    "compute_plot_bounds",
    "set_axis_limits",
    
    # ========================================================================
    # Utils - Configuration
    # ========================================================================
    # "default_cfg" already exported from core
    
    # ========================================================================
    # Utils - Constants
    # ========================================================================
    "EPS_NORMALIZE",
    "EPS_SAFE",
    "EPS_PCA",
    "MIN_PROB",
    "TANH_SCALE",
    "CLAMP_GUMBEL",
    "CLAMP_RANDN",
    "CLAMP_SPACING",
    "CLAMP_KERNEL_EXP",
    "DEFAULT_CONFIG",
    "ED_CONFIG",
    "POST_EQUALIZE_CONFIG",
    "SMOOTHER_CONFIG",
    
    # ========================================================================
    # Utils - Utilities
    # ========================================================================
    "ensure_torch",
    "as_numpy",
    "normalize",
    "validate_positive_definite",
]