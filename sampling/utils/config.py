"""Configuration management for runtime surface synthesis."""

from typing import Dict

# Numerical constants
EPS_NORMALIZE = 1e-8
EPS_SAFE = 1e-9
EPS_PCA = 1e-12
MIN_PROB = 1e-4
TANH_SCALE = 10.0
CLAMP_GUMBEL = (1e-10, 1.0 - 1e-10)
CLAMP_RANDN = (-3.0, 3.0)
CLAMP_SPACING = (0.3, 2.5)
CLAMP_KERNEL_EXP = -80.0

DEFAULT_CONFIG = {
    "differentiable": True,
    "use_hybrid_faiss": True,
    "use_surface_detection": True,
    "k_surface": 36,
    "thr_percentile": 8.0,
    "ema_beta": 0.95,
    "hysteresis": 0.03,
    "soft_tau": 0.08,
    "M": 50_000,
    "surf_jitter_alpha": 0.6,
    "thickness": 0.00,
    "density_gamma": 2.5,
    "use_F_kernel": True,
    "k_F": 32,
    "h_mul": 1.5,
    "sigma0": 0.08,
}

ED_CONFIG = {
    "enabled": True,
    "num_nodes": 180,
    "node_knn": 8,
    "point_knn_nodes": 8,
    "lambda_lap": 1.0e-2,
}

POST_EQUALIZE_CONFIG = {
    "enabled": True,
    "iters": 8,
    "k": 32,
    "step": 0.45,
    "annealing": 0.9,
    "radius_mul": 1.2,
    "use_mls_projection": True,
    "mls_iters": 2,
    "mls_step": 1.0,
}

SMOOTHER_CONFIG = {
    "enabled": False,
    "iters": 3,
    "k": 24,
    "step": 0.12,
    "lambda_normal": 0.15,
    "mls_every": 2
}


def default_cfg() -> Dict:
    """Default configuration with hybrid FAISS + differentiability."""
    config = DEFAULT_CONFIG.copy()
    config["ed"] = ED_CONFIG.copy()
    config["post_equalize"] = POST_EQUALIZE_CONFIG.copy()
    config["smoother"] = SMOOTHER_CONFIG.copy()
    
    config.update({
        "use_faiss_ivf": True,
        "use_amp": True,
        "sampling_tau": 0.2,
        "surface_power": 4.0,
        "knn_tau": 0.15,
        "ivf_nlist": 100,
        "ivf_nprobe": 10,
        "use_soft_radius": False,
        "soft_radius_candidates": 128,
    })
    
    return config