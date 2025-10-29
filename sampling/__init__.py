"""
Point Cloud Upsampling Pipeline (Refactored v3.1)

Main Pipeline:
1. Surface Detection (PCA-based planarity)
2. Anchor-Density Map (differentiable density estimation)
3. Importance Sampling (Gumbel-Softmax + tangent jitter; **memory-safe streaming ST**)
4. Taubin Smoothing (differentiable, shrinkage-free)
5. Normal Smoothing (Laplacian)
6. Covariance Construction (F-field interpolation)

Notes:
- v3.x uses a **memory-safe sampling path** (no dense Y ∈ ℝ^{M×N}).
  Configure via sampling.gs_batch / ensure_anchor_coverage / micro_jitter_scale.
- Public API and function signatures remain stable (no breaking change).
"""

# ============================================================================
# Main API
# ============================================================================
from .pipeline import upsample

# ============================================================================
# Configuration
# ============================================================================
from .utils.config import (
    default_cfg,
    bunny_cfg,
    sphere_cfg,
    fast_cfg,
    quality_cfg,
    validate_cfg,
    DEFAULT_CONFIG,  # Added

    # Constants (re-export)
    EPS_NORMALIZE,
    EPS_SAFE,
    EPS_PCA,
    MIN_PROB,
    TANH_SCALE,
    CLAMP_GUMBEL,
    CLAMP_RANDN,
    CLAMP_SPACING,
)

# ============================================================================
# Utilities
# ============================================================================
from .utils.utils import (
    ensure_torch,
    normalize,
    as_numpy,
    validate_positive_definite,
)

from .utils.quantile import (
    safe_quantile,
    robust_quantile_range,
)

from .utils.validation import (
    validate_sampling_config,
    check_config_and_warn,
    get_config_summary,
    safe_get_nested,
)

# ============================================================================
# KNN
# ============================================================================
from .analysis.knn import (
    HybridFAISSKNN,
    FAISS_AVAILABLE,
)

# ============================================================================
# I/O
# ============================================================================
from .io.export import (
    save_comparison_png,
    save_axis_hist_png,
    save_ply_xyz,
    save_gaussians_npz,
    save_stage_progression,
)

# Optional: tiny helpers (non-breaking, only if you want them public)
# from .debug.mem import cuda_mem_snapshot   # e.g., prints alloc/reserved/peak

__version__ = "3.1.0"   # ← bump to reflect memory-safe sampling behavior
__author__ = "CHAYO"

__all__ = [
    # Main API
    "upsample",

    # Configuration
    "default_cfg", 
    "bunny_cfg", 
    "sphere_cfg", 
    "fast_cfg", 
    "quality_cfg", 
    "validate_cfg",
    "EPS_NORMALIZE", 
    "EPS_SAFE", 
    "EPS_PCA", 
    "MIN_PROB", 
    "TANH_SCALE",
    "CLAMP_GUMBEL", 
    "CLAMP_RANDN", 
    "CLAMP_SPACING",
    "DEFAULT_CONFIG",

    # Utilities
    "ensure_torch", 
    "normalize", 
    "as_numpy", 
    "validate_positive_definite",

    # KNN
    "HybridFAISSKNN", 
    "FAISS_AVAILABLE",

    # I/O
    "save_comparison_png", 
    "save_axis_hist_png", 
    "save_ply_xyz", 
    "save_gaussians_npz",
    "save_stage_progression",

    # Version
    "__version__", "__author__",
]
