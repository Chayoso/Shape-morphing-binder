"""
Point Cloud Upsampling Pipeline (Refactored v2.1)

Main Pipeline:
1. Surface Detection (PCA-based planarity)
2. Volume Filtering (soft, differentiable)
3. Importance Sampling (Gumbel-Softmax + tangent jitter)
4. Taubin Smoothing (differentiable, shrinkage-free)
5. Normal Smoothing (Laplacian)
6. Covariance Construction (F-field interpolation)

Author: CHAYO
Version: 2.1.0
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
    
    # Constants
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
)


__version__ = "3.0.0"
__author__ = "CHAYO"


__all__ = [
    # ========================================================================
    # Main API
    # ========================================================================
    "upsample",
    
    # ========================================================================
    # Configuration
    # ========================================================================
    "default_cfg",
    "bunny_cfg",
    "sphere_cfg",
    "fast_cfg",
    "quality_cfg",
    "validate_cfg",
    
    # Constants
    "EPS_NORMALIZE",
    "EPS_SAFE",
    "EPS_PCA",
    "MIN_PROB",
    "TANH_SCALE",
    "CLAMP_GUMBEL",
    "CLAMP_RANDN",
    "CLAMP_SPACING",
    
    # ========================================================================
    # Utilities
    # ========================================================================
    "ensure_torch",
    "normalize",
    "as_numpy",
    "validate_positive_definite",
    
    # ========================================================================
    # KNN
    # ========================================================================
    "HybridFAISSKNN",
    "FAISS_AVAILABLE",
    
    # ========================================================================
    # I/O
    # ========================================================================
    "save_comparison_png",
    "save_axis_hist_png",
    "save_ply_xyz",
    "save_gaussians_npz",
    
    # ========================================================================
    # Version
    # ========================================================================
    "__version__",
    "__author__",
]