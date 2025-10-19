"""
Configuration management for upsampling pipeline.

Author: CHAYO
Date: 2025-10-19
Version: 2.1.0 
"""

from typing import Dict

# ============================================================================
# Numerical Constants
# ============================================================================

EPS_NORMALIZE = 1e-8
EPS_SAFE = 1e-9
EPS_PCA = 1e-12
MIN_PROB = 1e-4
TANH_SCALE = 10.0
CLAMP_GUMBEL = (1e-10, 1.0 - 1e-10)
CLAMP_RANDN = (-3.0, 3.0)
CLAMP_SPACING = (0.3, 2.5)

# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # ========================================================================
    # STEP 1: SURFACE DETECTION (PCA-based, Z-score method)
    # ========================================================================
    "surface_detection": {
        "enabled": True,
        "k": 48,                        # KNN neighbors for PCA
        "soft_tau": 0.5,                # 🔥 Z-score sigmoid temperature (0.08 → 0.5)
        "surface_power": 4.0,           # Probability concentration
    },
    
    # ========================================================================
    # STEP 2: VOLUME FILTERING (Soft)
    # ========================================================================
    "volume_filter": {
        "enabled": True,
        "k": 24,                        # 🔥 48 → 24 (more local for thin structures)
        "consistency_threshold": 0.3,   # 🔥 0.7 → 0.3 (more lenient)
        "temperature": 15.0,            # 🔥 10 → 15 (smoother transition)
        "positive_only": True,          # 🔥 NEW: Ignore opposite-facing normals
        "use_distance_weight": False,   # 🔥 NEW: Distance-based weighting (optional)
        "distance_bandwidth": 0.08,     # 🔥 NEW: Distance decay rate
    },
    
    # ========================================================================
    # STEP 3: IMPORTANCE SAMPLING
    # ========================================================================
    "sampling": {
        "M": 70000,                     # Target number of samples
        "tau": 0.2,                     # Gumbel-Softmax temperature
        "alpha": 0.35,                  # Tangent jitter scale
        "thickness": 0.0,               # Normal jitter scale
        "density_gamma": 2.5,           # Density importance
    },
    
    # ========================================================================
    # STEP 4: TAUBIN SMOOTHING
    # ========================================================================
    "taubin": {
        "enabled": True,
        "iters": 3,                     # Number of iterations
        "k": 24,                        # KNN neighbors
        "lambda_smooth": 0.6,           # Smoothing strength
        "lambda_inflate": -0.53,        # Inflation strength (negative!)
        "tangent_only": True,           # Only smooth tangent component
    },
    
    # ========================================================================
    # STEP 5: NORMAL SMOOTHING
    # ========================================================================
    "normal_smooth": {
        "enabled": True,
        "iters": 2,                     # Number of iterations
        "k": 16,                        # KNN neighbors
        "lambda_smooth": 0.8,           # Smoothing strength
    },
    
    # ========================================================================
    # STEP 6: COVARIANCE CONSTRUCTION
    # ========================================================================
    "covariance": {
        "sigma0": 0.08,                 # Base Gaussian scale
        "k_F": 32,                      # KNN for F interpolation
        "use_F_smoothing": True,        # Smooth F field
        "F_smooth": {
            "num_nodes": 180,
            "node_knn": 8,
            "point_knn": 8,
            "lambda_lap": 1e-2,
        },
    },
    
    # ========================================================================
    # KNN SETTINGS
    # ========================================================================
    "knn": {
        "use_faiss": True,
        "use_ivf": True,
        "tau": 0.15,                    # Soft KNN temperature
        "nlist": 100,
        "nprobe": 10,
    },
    
    # ========================================================================
    # DEBUG & EXPORT
    # ========================================================================
    "debug": {
        "verbose": True,           
        "export_volume_filter": True,   # Export PNG after volume filter
        "export_dir": "debug/",
        "png_dpi": 160,
        "png_ptsize": 0.5,
    },
    
    # ========================================================================
    # PERFORMANCE
    # ========================================================================
    "performance": {
        "use_amp": False,               # Mixed precision (stability > speed)
        "cache_knn": True,
        "clear_cache": True,
    },
}


# ============================================================================
# Preset Configurations
# ============================================================================

def default_cfg() -> Dict:
    """Default configuration."""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


def bunny_cfg() -> Dict:
    """
    Optimized for meshes with thin features (bunny ears, fingers).
    
    Changes:
    - More samples (70K → 80K)
    - Even more lenient volume filter
    - Gentler smoothing to preserve details
    """
    cfg = default_cfg()
    cfg["sampling"]["M"] = 80000
    cfg["volume_filter"]["k"] = 20             
    cfg["volume_filter"]["consistency_threshold"] = 0.25  
    cfg["volume_filter"]["temperature"] = 12.0
    cfg["taubin"]["lambda_smooth"] = 0.5
    cfg["taubin"]["lambda_inflate"] = -0.48
    cfg["normal_smooth"]["lambda_smooth"] = 0.7
    return cfg


def sphere_cfg() -> Dict:
    """
    Optimized for smooth surfaces (sphere, torus).
    
    Changes:
    - Fewer samples (70K → 50K)
    - Stronger smoothing
    - More strict volume filter (no thin structures)
    """
    cfg = default_cfg()
    cfg["sampling"]["M"] = 50000
    cfg["volume_filter"]["k"] = 32              # Larger neighborhood
    cfg["volume_filter"]["consistency_threshold"] = 0.5  # More strict
    cfg["taubin"]["lambda_smooth"] = 0.7
    cfg["taubin"]["lambda_inflate"] = -0.65
    cfg["normal_smooth"]["iters"] = 3
    return cfg


def fast_cfg() -> Dict:
    """
    Fast configuration for real-time (sacrifice quality).
    
    Changes:
    - Fewer samples (70K → 30K)
    - Disable expensive steps
    """
    cfg = default_cfg()
    cfg["sampling"]["M"] = 30000
    cfg["volume_filter"]["enabled"] = False
    cfg["taubin"]["enabled"] = False
    cfg["normal_smooth"]["enabled"] = False
    cfg["covariance"]["use_F_smoothing"] = False
    cfg["performance"]["use_amp"] = True
    cfg["debug"]["verbose"] = False             # Disable debug in fast mode
    return cfg


def quality_cfg() -> Dict:
    """
    High-quality configuration for offline rendering.
    
    Changes:
    - More samples (70K → 100K)
    - Higher k for better estimates
    - More iterations
    """
    cfg = default_cfg()
    cfg["sampling"]["M"] = 100000
    cfg["surface_detection"]["k"] = 64
    cfg["surface_detection"]["soft_tau"] = 0.6  # 🔥 Slightly higher for quality
    cfg["volume_filter"]["k"] = 32              # Higher k for quality
    cfg["volume_filter"]["consistency_threshold"] = 0.35  # Balanced
    cfg["taubin"]["iters"] = 5
    cfg["taubin"]["k"] = 32
    cfg["normal_smooth"]["iters"] = 3
    cfg["normal_smooth"]["k"] = 24
    return cfg


# ============================================================================
# Config Validation
# ============================================================================

def validate_cfg(cfg: Dict) -> None:
    """
    Validate configuration.
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Check M
    M = cfg.get("sampling", {}).get("M", 0)
    if M <= 0:
        raise ValueError(f"sampling.M must be positive, got {M}")
    
    # Check k values
    k_surface = cfg.get("surface_detection", {}).get("k", 0)
    if k_surface <= 0:
        raise ValueError(f"surface_detection.k must be positive, got {k_surface}")
    
    # Check soft_tau (z-score temperature)
    soft_tau = cfg.get("surface_detection", {}).get("soft_tau", 0)
    if soft_tau <= 0:
        raise ValueError(f"surface_detection.soft_tau must be positive, got {soft_tau}")
    
    # Check Taubin lambda
    lambda_smooth = cfg.get("taubin", {}).get("lambda_smooth", 0)
    lambda_inflate = cfg.get("taubin", {}).get("lambda_inflate", 0)
    
    if lambda_smooth < 0 or lambda_smooth > 1:
        raise ValueError(f"taubin.lambda_smooth must be in [0, 1], got {lambda_smooth}")
    
    if lambda_inflate >= 0:
        import warnings
        warnings.warn(
            f"taubin.lambda_inflate should be negative for inflation, got {lambda_inflate}"
        )
    
    # Check volume filter threshold
    consistency_threshold = cfg.get("volume_filter", {}).get("consistency_threshold", 0)
    if consistency_threshold < 0 or consistency_threshold > 1:
        raise ValueError(
            f"volume_filter.consistency_threshold must be in [0, 1], got {consistency_threshold}"
        )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Constants
    "EPS_NORMALIZE",
    "EPS_SAFE",
    "EPS_PCA",
    "MIN_PROB",
    "TANH_SCALE",
    "CLAMP_GUMBEL",
    "CLAMP_RANDN",
    "CLAMP_SPACING",
    
    # Config
    "DEFAULT_CONFIG",
    "default_cfg",
    "bunny_cfg",
    "sphere_cfg",
    "fast_cfg",
    "quality_cfg",
    "validate_cfg",
]