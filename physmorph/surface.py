"""Particle-scale roughness of the body's surface, measured against the target surface (measurement only, not in the
objective).

Each outer body particle has a signed offset from the target surface: along the normal of the nearest outer target
point, where the normal is the local plane of that point's outer neighbours within two target spacings, oriented away
from the centroid of the target sample around it. The roughness is the particle-scale part of that offset: each offset
minus the mean offset of the particle's outer body neighbours within two body spacings, in body spacings. A body that
is a smooth offset of the target scores zero however curved or sharp the target is. So a smoother that flattens the body
toward its own neighbours' plane is not rewarded where the target itself is curved, and thin features that match the
target are not counted as rough. Outer sets: the census definition of physmorph.thin.
"""
from __future__ import annotations

import torch

from . import gpu
from .thin import outer_mask


def _normals(P: torch.Tensor, full: torch.Tensor, sp: float, k: int = 16) -> torch.Tensor:
    """Unit normals of the outer points P: the smallest principal axis of their neighbours in P within two spacings,
    oriented away from the centroid of the full sample's 32 nearest points (the interior lies behind the surface)."""
    d, idx = gpu.knn(P, k)
    w = (d <= 2.0 * sp).double()
    nb = P[idx].double()
    c = (w[..., None] * nb).sum(1) / w.sum(1, keepdim=True)
    r = (nb - c[:, None]) * w[..., None]
    cov = r.transpose(1, 2) @ r
    n = torch.linalg.eigh(cov)[1][:, :, 0]
    inner = full[gpu.KNN(full).query(P, 32)[1]].double().mean(1)
    s = torch.sign(((P.double() - inner) * n).sum(1))
    return (n * torch.where(s == 0, torch.ones_like(s), s)[:, None]).to(P.dtype)


def surface_roughness(x, target) -> dict:
    """rms and p90 of the particle-scale offset (body spacings), with the body's outer share it is taken over."""
    X, T = gpu.tensor(x), gpu.tensor(target)
    sp_b = gpu.median(gpu.knn(X, 2)[0][:, 1])
    sp_t = gpu.median(gpu.knn(T, 2)[0][:, 1])
    To = T[outer_mask(T, sp_t)]
    Xo = X[outer_mask(X, sp_b)]
    if len(To) < 4 or len(Xo) < 4:
        return {"surf_rough": None, "surf_rough_p90": None}
    n = _normals(To, T, sp_t)
    j = gpu.KNN(To).query(Xo, 1)[1][:, 0]
    s = ((Xo - To[j]) * n[j]).sum(1).double()
    d, idx = gpu.knn(Xo, 17)
    w = ((d[:, 1:] <= 2.0 * sp_b) & (idx[:, 1:] != torch.arange(len(Xo), device=Xo.device)[:, None])).double()
    has = w.sum(1) > 0
    hp = (s - (w * s[idx[:, 1:]]).sum(1) / w.sum(1).clamp_min(1.0))[has] / sp_b
    if not len(hp):
        return {"surf_rough": None, "surf_rough_p90": None}
    return {"surf_rough": float(hp.square().mean().sqrt()),
            "surf_rough_p90": float(torch.quantile(hp.abs(), 0.9)),
            "surf_outer_share": len(Xo) / len(X)}
