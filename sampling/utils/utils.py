"""Common utility functions."""

import numpy as np
import torch
from .config import EPS_NORMALIZE


def ensure_torch(x, device='cuda', dtype=torch.float32):
    """Convert array-like to torch tensor."""
    if torch.is_tensor(x):
        if x.device == torch.device(device) and x.dtype == dtype:
            return x
        return x.to(device=device, dtype=dtype)
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)


def normalize(v: torch.Tensor, eps: float = EPS_NORMALIZE) -> torch.Tensor:
    """L2-normalize last dimension."""
    norm = torch.norm(v, dim=-1, keepdim=True)
    return v / torch.clamp(norm, min=eps)


def as_numpy(x):
    """Convert to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def validate_positive_definite(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Add regularization to ensure positive definiteness."""
    device = cov.device
    eye = torch.eye(cov.shape[-1], device=device).unsqueeze(0)
    return cov + eps * eye


__all__ = [
    "ensure_torch",
    "normalize",
    "as_numpy",
    "validate_positive_definite",
]