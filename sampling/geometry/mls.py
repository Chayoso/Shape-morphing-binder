"""Moving Least Squares (MLS) projection."""

import torch
import torch.nn.functional as F
from typing import Optional
from ..utils.config import EPS_SAFE
from ..utils.utils import normalize


def compute_mls_signed_distance(P: torch.Tensor, Q: torch.Tensor, n: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute MLS signed distance field value."""
    V = P.unsqueeze(1) - Q
    s = (w * (V * n).sum(-1)).sum(1, keepdim=True)
    return s


def compute_mls_normal(Q: torch.Tensor, n: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute MLS average normal."""
    nbar = normalize((w.unsqueeze(-1) * n).sum(1))
    return nbar


def project_to_mls_surface_diff(
    P: torch.Tensor,
    X: torch.Tensor,
    N: torch.Tensor,
    iters: int = 2,
    k: int = 32,
    step: float = 1.0,
    knn=None,
    knn_tau: float = 0.15
) -> torch.Tensor:
    """Project points onto an oriented-point MLS surface s(P)=0."""
    for _ in range(int(iters)):
        if knn is not None:
            idx, w = knn(P, X, k)
            Q = X[idx]
            n = N[idx]
        else:
            D = torch.cdist(P, X)
            logits = -D / float(knn_tau)
            attn = F.softmax(logits, dim=1)
            topw, topi = torch.topk(attn, k=min(k, X.shape[0]), dim=1)
            w = topw / (topw.sum(dim=1, keepdim=True) + EPS_SAFE)
            Q = X[topi]
            n = N[topi]
        
        s = compute_mls_signed_distance(P, Q, n, w)
        nbar = compute_mls_normal(Q, n, w)
        
        P = P - float(step) * s * nbar
    
    return P