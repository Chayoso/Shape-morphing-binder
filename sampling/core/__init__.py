"""
Core sampling operations.

Includes:
- Point sampling methods with adaptive jittering
- Gumbel-Softmax sampling for differentiable sampling
- Runtime surface tracking and synthesis
- Sample synthesis and combination
"""

# ============================================================================
# Gumbel Sampling
# ============================================================================
from .gumbel_sampling import (
    generate_gumbel_noise,
    gumbel_softmax_onehot,
)

# ============================================================================
# Point Sampling
# ============================================================================
from .point_sampling import (
    sample_surface_points_diff,
    compute_importance_weights,
    build_tangent_frame,
    generate_tangent_jitter,
    compute_adaptive_jitter_scale,
)

# ============================================================================
# Runtime Surface (Re-exports from modularized structure)
# ============================================================================
from .runtime_surface import (
    # Main entry points
    default_cfg,
    synthesize_runtime_surface,
    
    # Synthesis helpers
    extract_config_params,
    run_fast_cov_only_path,
    prepare_inputs,
    
    # Re-exported constants (for compatibility)
    EPS_NORMALIZE,
    EPS_SAFE,
    EPS_PCA,
    MIN_PROB,
    TANH_SCALE,
    CLAMP_GUMBEL,
    CLAMP_RANDN,
    CLAMP_SPACING,
    CLAMP_KERNEL_EXP,
    DEFAULT_CONFIG,
    ED_CONFIG,
    POST_EQUALIZE_CONFIG,
    SMOOTHER_CONFIG,
    
    # Re-exported utilities
    HybridFAISSKNN,
    FAISS_AVAILABLE,
    batched_pca_surface_optimized,
    compute_surface_mask_diff,
    density_equalize_diff,
    surface_smoother_diff,
    smooth_F_diff_optimized,
    project_to_mls_surface_diff,
    
    # Export utilities
    save_comparison_png,
    save_axis_hist_png,
    save_ply_xyz,
    save_gaussians_npz,
    
    # Version info
    __version__ as runtime_version,
    __author__ as runtime_author,
)

# ============================================================================
# Synthesis (Placeholder - if synthesis.py has actual functions)
# ============================================================================
try:
    from .synthesis import (
        synthesize_samples,
        combine_samples,
        merge_point_clouds,
        blend_samples,
    )
    _HAS_SYNTHESIS = True
except (ImportError, AttributeError):
    # Fallback if synthesis.py doesn't have these functions
    _HAS_SYNTHESIS = False
    
    def synthesize_samples(*args, **kwargs):
        raise NotImplementedError("synthesize_samples not implemented")
    
    def combine_samples(*args, **kwargs):
        raise NotImplementedError("combine_samples not implemented")
    
    def merge_point_clouds(*args, **kwargs):
        raise NotImplementedError("merge_point_clouds not implemented")
    
    def blend_samples(*args, **kwargs):
        raise NotImplementedError("blend_samples not implemented")


__all__ = [
    # Gumbel sampling
    "generate_gumbel_noise",
    "gumbel_softmax_onehot",
    
    # Point sampling
    "sample_surface_points_diff",
    "compute_importance_weights",
    "build_tangent_frame",
    "generate_tangent_jitter",
    "compute_adaptive_jitter_scale",
    
    # Runtime surface - Main
    "default_cfg",
    "synthesize_runtime_surface",
    
    # Runtime surface - Helpers
    "extract_config_params",
    "run_fast_cov_only_path",
    "prepare_inputs",
    
    # Runtime surface - Constants
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
    
    # Runtime surface - Re-exported modules
    "HybridFAISSKNN",
    "FAISS_AVAILABLE",
    "batched_pca_surface_optimized",
    "compute_surface_mask_diff",
    "density_equalize_diff",
    "surface_smoother_diff",
    "smooth_F_diff_optimized",
    "project_to_mls_surface_diff",
    
    # Export utilities
    "save_comparison_png",
    "save_axis_hist_png",
    "save_ply_xyz",
    "save_gaussians_npz",
    
    # Synthesis
    "synthesize_samples",
    "combine_samples",
    "merge_point_clouds",
    "blend_samples",
    
    # Version
    "runtime_version",
    "runtime_author",
]