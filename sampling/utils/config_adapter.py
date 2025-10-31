"""
Configuration Adapter for Backward Compatibility

Maps new simplified config structure to internal pipeline structure.
New YAML: upsample.points.jitter → Internal: sampling.alpha
"""

from typing import Dict, Any
import copy
import torch


def _safe_deepcopy(obj):
    """
    Deep copy that safely handles torch tensors by skipping them.
    """
    if isinstance(obj, torch.Tensor):
        # Skip tensors (they shouldn't be in config anyway)
        return obj
    elif isinstance(obj, dict):
        return {k: _safe_deepcopy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_deepcopy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_safe_deepcopy(item) for item in obj)
    else:
        try:
            return copy.deepcopy(obj)
        except:
            # If deepcopy fails, return as-is
            return obj


def adapt_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt new config structure to internal pipeline structure.
    
    New structure (YAML):
        upsample:
          num_points: 400000
          k_neighbors: 32
          surface: {...}
          points: {tau, jitter}
          smooth: {...}
          covariance: {...}
    
    Internal structure (code):
        surface_detection: {...}
        sampling: {M, tau, alpha, ...}
        taubin: {...}
        normal_smooth: {...}
        covariance: {...}
    
    Args:
        cfg: Configuration from YAML
        
    Returns:
        Adapted configuration for pipeline
    """
    adapted = _safe_deepcopy(cfg)
    
    # Check if using new structure
    if 'upsample' in cfg:
        upsample = cfg['upsample']
        
        # Map simplified names to internal names
        _map_surface_detection(adapted, upsample)
        _map_point_generation(adapted, upsample)
        _map_smoothing(adapted, upsample)
        _map_covariance(adapted, upsample)
        _map_performance(adapted, upsample)
        
        # Keep upsample section for episode scheduling compatibility
        # But also ensure backward-compatible access
        
    return adapted


def _map_surface_detection(adapted: Dict, upsample: Dict) -> None:
    """Map surface detection parameters."""
    if 'surface_detection' in upsample:
        # Already in correct format
        adapted.setdefault('surface_detection', {}).update(upsample['surface_detection'])
    elif 'surface' in upsample:
        # Simplified format
        surf = upsample['surface']
        adapted.setdefault('surface_detection', {}).update({
            'enabled': surf.get('enabled', True),
            'k': surf.get('k', upsample.get('k_neighbors', 32)),
            'planarity_percentile': surf.get('planarity_percentile', 25.0),
            'soft_tau': surf.get('soft_tau', 0.5),
            'surface_power': surf.get('surface_power', 6.0),
            'surface_n_sigma': surf.get('surface_n_sigma', 0.25),
            'ema_beta': surf.get('ema_beta', 0.5),
        })


def _map_point_generation(adapted: Dict, upsample: Dict) -> None:
    """Map point generation parameters."""
    if 'sampling' in upsample:
        # Already in internal format
        adapted.setdefault('sampling', {}).update(upsample['sampling'])
    elif 'points' in upsample:
        # Simplified format
        pts = upsample['points']
        adapted.setdefault('sampling', {}).update({
            'M': upsample.get('num_points', 400000),
            'tau': pts.get('tau', 0.2),
            'alpha': pts.get('jitter', 0.18),
            'prob_floor': pts.get('prob_floor', 1e-4),
            'uniform_mix': pts.get('uniform_mix', 0.15),
        })
    
    # Ensure M is set
    if 'sampling' in adapted and 'num_points' in upsample:
        adapted['sampling'].setdefault('M', upsample['num_points'])


def _map_smoothing(adapted: Dict, upsample: Dict) -> None:
    """Map smoothing parameters."""
    
    # Taubin smoothing
    if 'taubin' in upsample:
        adapted.setdefault('taubin', {}).update(upsample['taubin'])
    elif 'smooth' in upsample:
        smooth = upsample['smooth']
        k = upsample.get('k_neighbors', 32)
        
        adapted.setdefault('taubin', {}).update({
            'enabled': True,
            'iters': smooth.get('taubin_iters', 8),
            'k': smooth.get('k', k),
            'lambda_smooth': smooth.get('taubin_lambda', 0.75),
            'lambda_inflate': smooth.get('taubin_inflate', -0.42),
            'tangent_only': True,
        })
        
        adapted.setdefault('normal_smooth', {}).update({
            'enabled': True,
            'iters': smooth.get('normal_iters', 5),
            'k': smooth.get('k', k),
            'lambda_smooth': smooth.get('normal_lambda', 0.88),
        })
    
    # Normal smoothing (if separate)
    if 'normal_smooth' in upsample:
        adapted.setdefault('normal_smooth', {}).update(upsample['normal_smooth'])


def _map_covariance(adapted: Dict, upsample: Dict) -> None:
    """Map covariance parameters."""
    if 'covariance' not in upsample:
        return
    
    cov = upsample['covariance']
    adapted_cov = adapted.setdefault('covariance', {})
    
    # Direct mapping
    for key in ['sigma0', 'k_F', 'use_F_smoothing', 'use_polar_decomposition', 
                'use_adaptive_scale']:
        if key in cov:
            adapted_cov[key] = cov[key]
    
    # Simplified learnable params
    if 'learnable_enabled' in cov:
        adapted_cov.setdefault('learnable', {}).update({
            'enabled': cov.get('learnable_enabled', True),
            'alpha': cov.get('learnable_alpha', 0.4),
            'lr': cov.get('learnable_lr', 0.002),
            'freeze_after_ep': cov.get('learnable_freeze', -1),
        })
    elif 'learnable' in cov:
        adapted_cov['learnable'] = cov['learnable']
    
    # Simplified density params
    if 'scale_kappa' in cov:
        adapted_cov.setdefault('density', {}).update({
            'use_scale_prior': True,
            'scale_kappa': cov.get('scale_kappa', 0.35),
            'scale_max_up': cov.get('scale_max_up', 0.50),
            'scale_smooth_alpha': cov.get('scale_smooth_alpha', 6.0),
            'allow_shrink': cov.get('allow_shrink', False),
        })
    elif 'density' in cov:
        adapted_cov['density'] = cov['density']
    
    # F-field smoothing
    if 'F_num_nodes' in cov:
        adapted_cov.setdefault('F_smooth', {}).update({
            'num_nodes': cov.get('F_num_nodes', 180),
            'node_knn': cov.get('F_knn', 12),
            'point_knn': cov.get('F_knn', 12),
            'lambda_lap': cov.get('F_lambda', 0.05),
        })
    elif 'F_smooth' in cov:
        adapted_cov['F_smooth'] = cov['F_smooth']
    
    # Curvature-based covariance
    if 'curvature_cov' in cov:
        adapted_cov['curvature_cov'] = cov['curvature_cov']


def _map_performance(adapted: Dict, upsample: Dict) -> None:
    """Map performance and debug settings."""
    if 'performance' in upsample:
        perf = upsample['performance']
        
        # KNN performance
        adapted.setdefault('knn', {}).update({
            'use_faiss': perf.get('use_faiss', True),
            'use_ivf': perf.get('use_ivf', True),
        })
        
        # General performance
        adapted.setdefault('performance', {}).update({
            'cache_knn': perf.get('cache_knn', True),
            'clear_cache': perf.get('clear_cache', True),
        })
    
    # Debug settings
    if 'debug' in upsample:
        adapted.setdefault('debug', {}).update(upsample['debug'])


__all__ = ['adapt_config']

