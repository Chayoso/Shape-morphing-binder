"""
Common utilities and configuration.
"""

# ============================================================================
# Configuration
# ============================================================================
from .config import (
    # Main config functions
    default_cfg,
    bunny_cfg,
    sphere_cfg,
    fast_cfg,
    quality_cfg,
    validate_cfg,
    
    # Numerical constants
    EPS_NORMALIZE,
    EPS_SAFE,
    EPS_PCA,
    MIN_PROB,
    TANH_SCALE,
    CLAMP_GUMBEL,
    CLAMP_RANDN,
    CLAMP_SPACING,
    
    # Config dictionary
    DEFAULT_CONFIG,
)

# ============================================================================
# Utilities
# ============================================================================
from .utils import (
    ensure_torch,
    as_numpy,
    normalize,
    validate_positive_definite,
)


__all__ = [
    # Configuration
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
    
    # Config dictionary
    "DEFAULT_CONFIG",
    
    # Utilities
    "ensure_torch",
    "as_numpy",
    "normalize",
    "validate_positive_definite",
]