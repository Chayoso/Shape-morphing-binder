"""
Common utilities and configuration.
"""

# ============================================================================
# Configuration
# ============================================================================
from .config import (
    # Main config function
    default_cfg,
    
    # Numerical constants
    EPS_NORMALIZE,
    EPS_SAFE,
    EPS_PCA,
    MIN_PROB,
    TANH_SCALE,
    CLAMP_GUMBEL,
    CLAMP_RANDN,
    CLAMP_SPACING,
    CLAMP_KERNEL_EXP,
    
    # Config dictionaries
    DEFAULT_CONFIG,
    ED_CONFIG,
    POST_EQUALIZE_CONFIG,
    SMOOTHER_CONFIG,
)

# ============================================================================
# Utilities (with fallback)
# ============================================================================
try:
    from .utils import (
        ensure_torch,
        as_numpy,
        normalize,
        validate_positive_definite,
    )
    _HAS_UTILS = True
except ImportError as e:
    print(f"⚠️ Warning: Some utils not available: {e}")
    _HAS_UTILS = False
    
    # Fallback implementations
    import numpy as np
    import torch
    
    def ensure_torch(x, device='cuda', dtype=torch.float32):
        """Convert to torch tensor."""
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)
    
    def as_numpy(x):
        """Convert to numpy."""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    def normalize(v, eps=1e-8):
        """Normalize vectors."""
        norm = torch.norm(v, dim=-1, keepdim=True)
        return v / torch.clamp(norm, min=eps)
    
    def validate_positive_definite(cov, eps=1e-6):
        """Regularize covariance."""
        eye = torch.eye(cov.shape[-1], device=cov.device).unsqueeze(0)
        return cov + eps * eye


__all__ = [
    # Configuration
    "default_cfg",
    
    # Constants - Epsilon
    "EPS_NORMALIZE",
    "EPS_SAFE",
    "EPS_PCA",
    "MIN_PROB",
    
    # Constants - Scaling
    "TANH_SCALE",
    
    # Constants - Clamp
    "CLAMP_GUMBEL",
    "CLAMP_RANDN",
    "CLAMP_SPACING",
    "CLAMP_KERNEL_EXP",
    
    # Config dictionaries
    "DEFAULT_CONFIG",
    "ED_CONFIG",
    "POST_EQUALIZE_CONFIG",
    "SMOOTHER_CONFIG",
    
    # Utilities - Conversion
    "ensure_torch",
    "as_numpy",
    
    # Utilities - Math
    "normalize",
    "validate_positive_definite",
]