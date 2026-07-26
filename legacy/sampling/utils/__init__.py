"""
Common utilities and configuration (v3.1)

This namespace re-exports:
- Configuration factory/presets/validator
- Numerical constants used across the pipeline
- The DEFAULT_CONFIG dictionary (now includes memory-safe sampling keys):
    sampling.gs_batch
    sampling.ensure_anchor_coverage
    sampling.micro_jitter_scale
- Lightweight utilities (torch checks, normalization, PD validation)

No behavioral changes here; this module is a convenience barrel import.
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

    # Config dictionary (includes memory-safe sampling keys)
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

from .quantile import (
    safe_quantile,
    robust_quantile_range,
)

from .validation import (
    validate_sampling_config,
    check_config_and_warn,
    get_config_summary,
    safe_get_nested,
)

from .config_adapter import (
    adapt_config,
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
    
    # Quantile functions
    "safe_quantile",
    "robust_quantile_range",
    
    # Validation
    "validate_sampling_config",
    "check_config_and_warn",
    "get_config_summary",
    "safe_get_nested",
    
    # Config adapter
    "adapt_config",
]
