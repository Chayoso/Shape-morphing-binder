"""
Configuration management for upsampling pipeline.

Author: CHAYO
Date: 2025-10-19
Version: 2.2.0
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
        "soft_tau": 0.5,                # Z-score sigmoid temperature
        "surface_power": 4.0,           # Probability concentration
        # internal EMA / hysteresis keys are kept in detect_surface()
    },

    # ========================================================================
    # STEP 2: VOLUME FILTERING (Soft)
    # ========================================================================
    "volume_filter": {
        "enabled": True,
        "k": 24,
        "consistency_threshold": 0.3,
        "temperature": 15.0,
        "positive_only": True,          # Ignore opposite-facing normals
        "use_distance_weight": False,   # Optional: exp(-d/h) weighting
        "distance_bandwidth": 0.08,
    },

    # ========================================================================
    # STEP 3: IMPORTANCE SAMPLING (Memory-safe streaming ST)
    # ========================================================================
    "sampling": {
        "M": 70000,                     # Target number of samples (rendered points)
        "tau": 0.2,                     # Gumbel-Softmax temperature
        "alpha": 0.35,                  # Tangent jitter base scale
        "thickness": 0.0,               # Normal jitter (shell thickness)

        # Memory-safe sampler settings
        "gs_batch": 2048,               # Streaming softmax batch size (controls peak VRAM)
        "ensure_anchor_coverage": True, # If M >= N, include each anchor at least once
        "micro_jitter_scale": 0.2,      # HF jitter multiplier (0.0~0.5 typical)
        "tangent_micro_only": True,     # Tangent micro only
        "plane_snap": True,             # Plane snap
        
        # Hole-fix patches (NEW)
        "prob_floor": 1e-8,             # Minimum probability floor (no dead anchors)
        "uniform_mix": 0.02,            # 2% uniform mass (all anchors reachable)
        "plane_snap_beta": 0.5,         # Keep 50% of normal delta (0=hard, 1=no snap)
        "topk_pool": 8,                 # Top-k candidates per sample (prevents collapse)
        "thickness_gamma": 0.15,        # Adaptive thickness = 15% of local spacing
        
        # Surface-constrained sampling
        "surface_support_q": 0.80,      # 상위 20%만 표면
        "prob_floor_mode": "density",   # 밀도 기반 floor
        "uniform_mix_surface_only": True,
        "coverage_only_surface": True,
        "mask_topk_with_surface": True,
        
        # Density-based floor settings
        "density_floor_tau": 1.0,
        "density_floor_gamma": 2.0,
        
        # One-sided thickness with inside barrier
        "thickness_one_sided": True,    # 바깥쪽만 두께
        "inside_barrier_lambda": 1.0,   # 내부 점 보정
    },

    # ========================================================================
    # STEP 4: TAUBIN SMOOTHING
    # ========================================================================
    "taubin": {
        "enabled": True,
        "iters": 3,
        "k": 24,
        "lambda_smooth": 0.6,
        "lambda_inflate": -0.53,        # Negative!
        "tangent_only": True,
    },

    # ========================================================================
    # STEP 5: NORMAL SMOOTHING
    # ========================================================================
    "normal_smooth": {
        "enabled": True,
        "iters": 2,
        "k": 16,
        "lambda_smooth": 0.8,
    },

    # ========================================================================
    # STEP 6: COVARIANCE CONSTRUCTION
    # ========================================================================
    "covariance": {
        "sigma0": 0.08,
        "k_F": 32,
        "use_F_smoothing": True,
        "use_polar_decomposition": True,  # default ON (recommended)
        "use_adaptive_scale": False,      # optional: σ adapt by local spacing
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
        "export_volume_filter": True,
        "export_anchors": False,        # Export anchor sampling visualization
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
    """Default configuration (deep copy)."""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


def bunny_cfg() -> Dict:
    """
    Optimized for meshes with thin features (bunny ears, fingers).

    Changes:
    - More samples (70K → 80K)
    - More lenient volume filter
    - Gentler smoothing
    - Warmer tau for better spread
    - Slightly smaller streaming batch for safer VRAM on 24GB GPUs
    """
    cfg = default_cfg()
    cfg["sampling"]["M"] = 80000
    cfg["sampling"]["tau"] = 0.3
    cfg["sampling"]["gs_batch"] = 1536
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
    - More strict volume filter
    - Larger streaming batch (memory friendly here)
    """
    cfg = default_cfg()
    cfg["sampling"]["M"] = 50000
    cfg["sampling"]["gs_batch"] = 4096
    cfg["volume_filter"]["k"] = 32
    cfg["volume_filter"]["consistency_threshold"] = 0.5
    cfg["taubin"]["lambda_smooth"] = 0.7
    cfg["taubin"]["lambda_inflate"] = -0.65
    cfg["normal_smooth"]["iters"] = 3
    return cfg


def fast_cfg() -> Dict:
    """
    Fast configuration for real-time (sacrifice quality).

    Changes:
    - Fewer samples (70K → 30K)
    - Disable expensive post-process
    - Enable AMP
    - Smaller batch to reduce spikes on mid GPUs
    """
    cfg = default_cfg()
    cfg["sampling"]["M"] = 30000
    cfg["sampling"]["gs_batch"] = 1024
    cfg["volume_filter"]["enabled"] = False
    cfg["taubin"]["enabled"] = False
    cfg["normal_smooth"]["enabled"] = False
    cfg["covariance"]["use_F_smoothing"] = False
    cfg["performance"]["use_amp"] = True
    cfg["debug"]["verbose"] = False
    return cfg


def quality_cfg() -> Dict:
    """
    High-quality configuration for offline rendering.

    Changes:
    - More samples (70K → 100K)
    - Higher k for better estimates
    - More iterations
    - Larger top-k pool for better coverage
    - Moderate streaming batch to keep peak VRAM stable
    """
    cfg = default_cfg()
    cfg["sampling"]["M"] = 100000
    cfg["sampling"]["gs_batch"] = 2048
    cfg["sampling"]["topk_pool"] = 12
    cfg["surface_detection"]["k"] = 64
    cfg["surface_detection"]["soft_tau"] = 0.6
    cfg["volume_filter"]["k"] = 32
    cfg["volume_filter"]["consistency_threshold"] = 0.35
    cfg["taubin"]["iters"] = 5
    cfg["taubin"]["k"] = 32
    cfg["normal_smooth"]["iters"] = 3
    cfg["normal_smooth"]["k"] = 24
    return cfg

# ============================================================================
# Config Validation
# ============================================================================

def _in_range(name: str, val: float, lo: float, hi: float):
    if not (lo <= val <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {val}")

def validate_cfg(cfg: Dict) -> None:
    """
    Validate configuration.

    Raises:
        ValueError: If configuration is invalid
    """
    # Sampling
    M = int(cfg.get("sampling", {}).get("M", 0))
    if M <= 0:
        raise ValueError(f"sampling.M must be positive, got {M}")

    tau = float(cfg.get("sampling", {}).get("tau", 0.2))
    _in_range("sampling.tau", tau, 1e-4, 5.0)

    alpha = float(cfg.get("sampling", {}).get("alpha", 0.35))
    _in_range("sampling.alpha", alpha, 0.0, 5.0)

    thickness = float(cfg.get("sampling", {}).get("thickness", 0.0))
    if thickness < 0.0:
        raise ValueError(f"sampling.thickness must be >= 0, got {thickness}")

    gs_batch = int(cfg.get("sampling", {}).get("gs_batch", 2048))
    if gs_batch <= 0:
        raise ValueError(f"sampling.gs_batch must be positive, got {gs_batch}")

    # Hole-fix patches
    prob_floor = float(cfg.get("sampling", {}).get("prob_floor", 1e-8))
    if prob_floor < 0.0:
        raise ValueError(f"sampling.prob_floor must be >= 0, got {prob_floor}")

    uniform_mix = float(cfg.get("sampling", {}).get("uniform_mix", 0.02))
    _in_range("sampling.uniform_mix", uniform_mix, 0.0, 1.0)

    plane_snap_beta = float(cfg.get("sampling", {}).get("plane_snap_beta", 0.5))
    _in_range("sampling.plane_snap_beta", plane_snap_beta, 0.0, 1.0)

    topk_pool = int(cfg.get("sampling", {}).get("topk_pool", 8))
    if topk_pool < 0:
        raise ValueError(f"sampling.topk_pool must be >= 0, got {topk_pool}")

    thickness_gamma = float(cfg.get("sampling", {}).get("thickness_gamma", 0.15))
    if thickness_gamma < 0.0:
        raise ValueError(f"sampling.thickness_gamma must be >= 0, got {thickness_gamma}")

    # Surface detection
    k_surface = int(cfg.get("surface_detection", {}).get("k", 0))
    if k_surface <= 0:
        raise ValueError(f"surface_detection.k must be positive, got {k_surface}")

    soft_tau = float(cfg.get("surface_detection", {}).get("soft_tau", 0.5))
    _in_range("surface_detection.soft_tau", soft_tau, 1e-4, 10.0)

    # Taubin
    lambda_smooth = float(cfg.get("taubin", {}).get("lambda_smooth", 0.6))
    _in_range("taubin.lambda_smooth", lambda_smooth, 0.0, 1.0)

    lambda_inflate = float(cfg.get("taubin", {}).get("lambda_inflate", -0.53))
    if lambda_inflate >= 0.0:
        import warnings
        warnings.warn("taubin.lambda_inflate should be negative for inflation.")

    # Volume filter
    consistency_threshold = float(cfg.get("volume_filter", {}).get("consistency_threshold", 0.3))
    _in_range("volume_filter.consistency_threshold", consistency_threshold, 0.0, 1.0)

    # Covariance
    sigma0 = float(cfg.get("covariance", {}).get("sigma0", 0.08))
    _in_range("covariance.sigma0", sigma0, 1e-4, 10.0)

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
