"""Common utility functions."""

import numpy as np
import torch
from .config import EPS_NORMALIZE


def ensure_torch(x, device='cuda', dtype=torch.float32):
    """Convert array-like to torch tensor on the given device/dtype."""
    if torch.is_tensor(x):
        if x.device == torch.device(device) and x.dtype == dtype:
            return x
        return x.to(device=device, dtype=dtype)
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)


def normalize(v: torch.Tensor, eps: float = EPS_NORMALIZE) -> torch.Tensor:
    """L2-normalize last dimension, safe for zeros."""
    norm = torch.norm(v, dim=-1, keepdim=True)
    return v / torch.clamp(norm, min=eps)


def validate_positive_definite(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Add small regularization to ensure positive definiteness."""
    device = cov.device
    eye = torch.eye(cov.shape[-1], device=device).unsqueeze(0)
    return cov + eps * eye


def as_numpy(a):
    """Convert to numpy array."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)