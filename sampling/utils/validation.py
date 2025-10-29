"""
Configuration Validation and Error Checking

Provides utilities to validate configuration dictionaries and catch common
errors before pipeline execution.
"""

import warnings
from typing import Dict, Any, List, Tuple, Optional


def validate_sampling_config(cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate sampling configuration and return errors/warnings.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        (is_valid, messages) where is_valid=True if no critical errors,
        and messages contains list of warning/error strings
    """
    messages = []
    is_valid = True
    
    # Check critical parameters
    if 'sampling' in cfg:
        s_cfg = cfg['sampling']
        
        # Target number of points
        M = s_cfg.get('M')
        if M is None:
            messages.append("ERROR: 'sampling.M' (target points) is required")
            is_valid = False
        elif M <= 0:
            messages.append(f"ERROR: 'sampling.M' must be positive, got {M}")
            is_valid = False
        elif M > 10_000_000:
            messages.append(f"WARNING: 'sampling.M' is very large ({M:,}), may cause OOM")
            
        # Gumbel temperature
        tau = s_cfg.get('tau', 0.2)
        if tau <= 0:
            messages.append(f"ERROR: 'sampling.tau' must be positive, got {tau}")
            is_valid = False
        elif tau > 5.0:
            messages.append(f"WARNING: 'sampling.tau' is very large ({tau}), sampling will be nearly uniform")
            
        # Jitter scale
        alpha = s_cfg.get('alpha', 0.35)
        if alpha < 0:
            messages.append(f"ERROR: 'sampling.alpha' cannot be negative, got {alpha}")
            is_valid = False
        elif alpha > 2.0:
            messages.append(f"WARNING: 'sampling.alpha' is large ({alpha}), points may stray far from surface")
    
    # Check covariance parameters
    if 'covariance' in cfg:
        c_cfg = cfg['covariance']
        
        sigma0 = c_cfg.get('sigma0')
        if sigma0 is None:
            messages.append("ERROR: 'covariance.sigma0' is required")
            is_valid = False
        elif sigma0 <= 0:
            messages.append(f"ERROR: 'covariance.sigma0' must be positive, got {sigma0}")
            is_valid = False
        elif sigma0 > 1.0:
            messages.append(f"WARNING: 'covariance.sigma0' is very large ({sigma0}), Gaussians may be too big")
    
    # Check KNN parameters
    if 'knn' in cfg:
        k_cfg = cfg['knn']
        
        if 'use_faiss' in k_cfg and not k_cfg['use_faiss']:
            messages.append("INFO: FAISS disabled, KNN may be slow for large point clouds")
            
        tau_knn = k_cfg.get('tau', 0.15)
        if tau_knn <= 0:
            messages.append(f"ERROR: 'knn.tau' must be positive, got {tau_knn}")
            is_valid = False
    
    # Check surface detection
    if 'surface_detection' in cfg:
        surf_cfg = cfg['surface_detection']
        
        k = surf_cfg.get('k', 48)
        if k < 3:
            messages.append(f"ERROR: 'surface_detection.k' must be >= 3 for PCA, got {k}")
            is_valid = False
        elif k < 16:
            messages.append(f"WARNING: 'surface_detection.k' is small ({k}), surface detection may be noisy")
            
        percentile = surf_cfg.get('planarity_percentile', 10.0)
        if percentile <= 0 or percentile > 100:
            messages.append(f"ERROR: 'surface_detection.planarity_percentile' must be in (0, 100], got {percentile}")
            is_valid = False
    
    # Check smoothing parameters
    if 'taubin' in cfg:
        t_cfg = cfg['taubin']
        
        if t_cfg.get('enabled', True):
            iters = t_cfg.get('iters', 3)
            if iters < 0:
                messages.append(f"ERROR: 'taubin.iters' cannot be negative, got {iters}")
                is_valid = False
            elif iters > 20:
                messages.append(f"WARNING: 'taubin.iters' is large ({iters}), may over-smooth")
                
            lambda_smooth = t_cfg.get('lambda_smooth', 0.33)
            lambda_inflate = t_cfg.get('lambda_inflate', -0.53)
            
            if lambda_smooth <= 0:
                messages.append(f"WARNING: 'taubin.lambda_smooth' should be positive, got {lambda_smooth}")
            if lambda_inflate >= 0:
                messages.append(f"WARNING: 'taubin.lambda_inflate' should be negative, got {lambda_inflate}")
    
    return is_valid, messages


def check_config_and_warn(cfg: Dict[str, Any], verbose: bool = True) -> bool:
    """
    Check configuration and print warnings/errors.
    
    Args:
        cfg: Configuration dictionary
        verbose: If True, print all messages. If False, only critical errors.
        
    Returns:
        True if config is valid, False otherwise
    """
    is_valid, messages = validate_sampling_config(cfg)
    
    if messages:
        if verbose:
            print("\n" + "="*70)
            print("Configuration Validation")
            print("="*70)
            
            for msg in messages:
                if msg.startswith("ERROR"):
                    print(f"  ❌ {msg}")
                elif msg.startswith("WARNING"):
                    print(f"  ⚠️  {msg}")
                else:
                    print(f"  ℹ️  {msg}")
            
            if is_valid:
                print("  ✅ Configuration is valid")
            else:
                print("  ❌ Configuration has critical errors!")
            
            print("="*70 + "\n")
        else:
            # Only print errors
            errors = [m for m in messages if m.startswith("ERROR")]
            if errors:
                print("\n⚠️  Configuration Errors:")
                for err in errors:
                    print(f"  {err}")
                print()
    
    return is_valid


def get_config_summary(cfg: Dict[str, Any]) -> str:
    """
    Generate human-readable configuration summary.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Multi-line string summarizing key parameters
    """
    lines = []
    lines.append("Configuration Summary")
    lines.append("="*50)
    
    # Sampling
    if 'sampling' in cfg:
        s = cfg['sampling']
        lines.append("Sampling:")
        lines.append(f"  Target points (M): {s.get('M', 'N/A'):,}")
        lines.append(f"  Gumbel τ: {s.get('tau', 0.2):.3f}")
        lines.append(f"  Jitter α: {s.get('alpha', 0.35):.3f}")
    
    # Covariance
    if 'covariance' in cfg:
        c = cfg['covariance']
        lines.append("Covariance:")
        lines.append(f"  Base scale σ₀: {c.get('sigma0', 0.08):.4f}")
        lines.append(f"  Polar decomposition: {c.get('use_polar_decomposition', True)}")
    
    # Surface detection
    if 'surface_detection' in cfg:
        s = cfg['surface_detection']
        if s.get('enabled', True):
            lines.append("Surface Detection:")
            lines.append(f"  PCA neighbors: {s.get('k', 48)}")
            lines.append(f"  Target percentile: {s.get('planarity_percentile', 10.0):.1f}%")
        else:
            lines.append("Surface Detection: DISABLED")
    
    # Smoothing
    if 'taubin' in cfg:
        t = cfg['taubin']
        if t.get('enabled', True):
            lines.append("Taubin Smoothing:")
            lines.append(f"  Iterations: {t.get('iters', 3)}")
            lines.append(f"  λ: {t.get('lambda_smooth', 0.33):.3f}")
        else:
            lines.append("Taubin Smoothing: DISABLED")
    
    lines.append("="*50)
    
    return "\n".join(lines)


def safe_get_nested(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Safely get nested dictionary value with fallback.
    
    Example:
        >>> cfg = {'a': {'b': {'c': 10}}}
        >>> safe_get_nested(cfg, ['a', 'b', 'c'])  # 10
        >>> safe_get_nested(cfg, ['a', 'x', 'y'], default=0)  # 0
    """
    current = cfg
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


__all__ = [
    'validate_sampling_config',
    'check_config_and_warn',
    'get_config_summary',
    'safe_get_nested',
]

