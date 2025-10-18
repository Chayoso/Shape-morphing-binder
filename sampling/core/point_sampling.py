"""Surface point sampling with tangent frame jittering."""

import numpy as np
import torch
from typing import Tuple, Optional
from ..utils.config import EPS_SAFE, EPS_PCA, CLAMP_RANDN, CLAMP_SPACING  
from ..utils.utils import normalize                                            
from .gumbel_sampling import gumbel_softmax_onehot    


def compute_importance_weights(probs: torch.Tensor, spacing: torch.Tensor, density_gamma: float) -> torch.Tensor:
    """Compute importance sampling weights."""
    w_import = (probs ** 1.5) * (1.0 / (spacing ** density_gamma + EPS_SAFE))
    w_import = w_import / (w_import.sum() + EPS_PCA)
    return w_import


def build_tangent_frame(normals: torch.Tensor, M: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build orthonormal tangent frame [t1, t2] perpendicular to normals."""
    a = torch.tensor([1., 0., 0.], device=device, dtype=dtype).expand(M, 3).clone()
    
    col = torch.abs(torch.einsum('md,md->m', normals, a)) > 0.9
    a[col] = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
    
    t1 = normalize(a - torch.einsum('md,md->m', a, normals).unsqueeze(-1) * normals)
    t2 = normalize(torch.cross(normals, t1, dim=1))
    
    return t1, t2


def generate_tangent_jitter(M: int, generator: torch.Generator, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random tangent space jitter."""
    U = torch.randn(M, 1, generator=generator, device=device, dtype=dtype).clamp(*CLAMP_RANDN)
    V = torch.randn(M, 1, generator=generator, device=device, dtype=dtype).clamp(*CLAMP_RANDN)
    
    theta = torch.rand(M, 1, generator=generator, device=device, dtype=dtype) * 2 * np.pi
    c, s = torch.cos(theta), torch.sin(theta)
    U_rot = U * c - V * s
    V_rot = U * s + V * c
    
    return U_rot, V_rot


def compute_adaptive_jitter_scale(h: torch.Tensor, alpha: float, generator: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Compute adaptive jitter scale based on local spacing."""
    M = h.shape[0]
    
    alpha_noise = 0.4 + torch.rand(M, 1, generator=generator, device=device, dtype=dtype) * 1.2
    h_scale = torch.clamp(h / (h.mean() + EPS_SAFE), *CLAMP_SPACING).unsqueeze(-1)
    alpha_adapt = float(alpha) * alpha_noise * h_scale
    
    return alpha_adapt


def sample_surface_points_diff(
    x: torch.Tensor,
    normals: torch.Tensor,
    spacing: torch.Tensor,
    probs: torch.Tensor,
    M: int,
    alpha: float,
    thickness: float,
    density_gamma: float,
    seed: int = 0,
    tau: float = 0.2,
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fully differentiable sampling of M points on the surface."""
    device, dtype = x.device, x.dtype
    
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    w_import = compute_importance_weights(probs, spacing, density_gamma)
    Y = gumbel_softmax_onehot(w_import, M=M, tau=tau, hard=True, seed=seed, generator=generator)
    
    mu_anchors = Y @ x
    n = normalize(Y @ normals)
    h = (Y @ spacing.unsqueeze(1)).squeeze(1)
    
    t1, t2 = build_tangent_frame(n, M, device, dtype)
    U_rot, V_rot = generate_tangent_jitter(M, generator, device, dtype)
    Z = (torch.rand(M, 1, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)
    
    alpha_adapt = compute_adaptive_jitter_scale(h, alpha, generator, device, dtype)
    
    tangent_offset = alpha_adapt * h.unsqueeze(-1) * (U_rot * t1 + V_rot * t2)
    normal_offset = (float(thickness) * Z) * n
    micro = 0.2 * float(alpha) * h.unsqueeze(-1) * torch.randn(M, 3, generator=generator, device=device, dtype=dtype)
    
    mu = mu_anchors + tangent_offset + normal_offset + micro
    
    return mu, n, mu_anchors