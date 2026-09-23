"""Massless render-child construction in frozen material coordinates.

The simulation owns one state per parent particle.  This module only expands that
state into smaller Gaussian primitives; it never creates MPM mass or mutable state.
"""
from __future__ import annotations

import numpy as np
import torch


def tangent_child_offsets(rest_x: np.ndarray, mask: np.ndarray | None, sigma0: float,
                          count: int = 4, offset_scale: float = 0.35,
                          k: int = 16) -> np.ndarray:
    """Frozen local-PCA offsets ``(N,count,3)`` with zero centroid per parent.

    PCA uses only active render parents, preventing volume-interior neighbours from
    defining a spurious tangent plane.  The symmetric patterns make eigenvector signs
    irrelevant and preserve each parent's center of mass exactly.
    """
    from scipy.spatial import cKDTree

    rest = np.ascontiguousarray(rest_x, np.float32)
    if rest.ndim != 2 or rest.shape[1] != 3:
        raise ValueError("rest_x must have shape (N,3)")
    count = int(count)
    if count < 1 or count > 4:
        raise ValueError("render child count must be in [1,4]")
    if not np.isfinite(sigma0) or float(sigma0) <= 0:
        raise ValueError("sigma0 must be finite and positive")
    if not np.isfinite(offset_scale) or float(offset_scale) < 0:
        raise ValueError("offset_scale must be finite and non-negative")
    active = np.ones(len(rest), dtype=bool) if mask is None else np.asarray(mask, bool)
    if active.shape != (len(rest),):
        raise ValueError("mask must have shape (N,)")
    ids = np.flatnonzero(active)
    if len(ids) < 3 and count > 1:
        raise ValueError("at least three active parents are required for tangent children")
    out = np.zeros((len(rest), count, 3), np.float32)
    if count == 1 or not len(ids) or float(offset_scale) == 0:
        return out

    points = rest[ids]
    degree = min(max(int(k), 2), len(points) - 1)
    _, nn = cKDTree(points).query(points, k=degree + 1, workers=-1)
    local = points[nn[:, 1:]] - points[:, None, :]
    cov = np.einsum("nki,nkj->nij", local, local) / float(degree)
    _, basis = np.linalg.eigh(cov)
    t1, t2 = basis[:, :, 2], basis[:, :, 1]
    if count == 2:
        pattern = np.array([[1.0, 0.0], [-1.0, 0.0]], np.float32)
    elif count == 3:
        pattern = np.array([[1.0, 0.0], [-0.5, 0.8660254],
                            [-0.5, -0.8660254]], np.float32)
    else:
        # Four quarter-cell corners.  0.35*sigma0 offsets with a 0.55*sigma0
        # children overlap enough to avoid turning one point artifact into four.
        pattern = np.array([[1.0, 1.0], [1.0, -1.0],
                            [-1.0, 1.0], [-1.0, -1.0]], np.float32)
    delta = float(sigma0) * float(offset_scale)
    out[ids] = delta * (pattern[None, :, :1] * t1[:, None, :]
                        + pattern[None, :, 1:] * t2[:, None, :])
    # Numerical eigensolvers can leave tiny nonzero sums; enforce the contract.
    out[ids] -= out[ids].mean(1, keepdims=True)
    return np.ascontiguousarray(out)


def expand_children_torch(x: torch.Tensor, F: torch.Tensor | None,
                          offsets: torch.Tensor,
                          mask: torch.Tensor | None = None
                          ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the parent mask once, then return differentiable child centers/F."""
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (N,3)")
    Fm = None if F is None else F.reshape(-1, 3, 3)
    off = offsets.to(dtype=x.dtype, device=x.device)
    if off.ndim != 3 or off.shape[0] != len(x) or off.shape[2] != 3:
        raise ValueError("offsets must have shape (N,C,3)")
    if Fm is not None and len(Fm) != len(x):
        raise ValueError("F must have one matrix per parent")
    if mask is not None:
        keep = mask.to(device=x.device, dtype=torch.bool)
        if keep.shape != (len(x),):
            raise ValueError("mask must have shape (N,)")
        x, off = x[keep], off[keep]
        if Fm is not None:
            Fm = Fm[keep]
    if Fm is None:
        center = x[:, None, :] + off
        child_F = None
    else:
        center = x[:, None, :] + torch.einsum("nij,ncj->nci", Fm, off)
        child_F = Fm[:, None, :, :].expand(-1, off.shape[1], -1, -1)
    return center.reshape(-1, 3), (None if child_F is None
                                   else child_F.reshape(-1, 3, 3))


def expand_children_numpy(x: np.ndarray, F: np.ndarray, offsets: np.ndarray,
                          mask: np.ndarray | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """NumPy equivalent used by offline deliverable rendering."""
    xt = torch.as_tensor(np.ascontiguousarray(x, np.float32))
    Ft = torch.as_tensor(np.ascontiguousarray(F, np.float32))
    ot = torch.as_tensor(np.ascontiguousarray(offsets, np.float32))
    mt = None if mask is None else torch.as_tensor(np.asarray(mask, bool))
    xc, Fc = expand_children_torch(xt, Ft, ot, mt)
    return (np.ascontiguousarray(xc.numpy(), np.float32),
            np.ascontiguousarray(Fc.numpy(), np.float32))


def tangent_child_basis(rest_x: np.ndarray, mask: np.ndarray | None,
                        k: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """The frozen per-parent tangent frame ``(t1, t2)`` (each (N,3)) used by
    tangent_child_offsets — recomputed with the same active-parent PCA so the
    dressing DOFs (docs/local_global_design.md §4.1) live in exactly the plane
    the baseline pattern lives in. Inactive parents get zero rows."""
    from scipy.spatial import cKDTree

    rest = np.ascontiguousarray(rest_x, np.float32)
    active = np.ones(len(rest), dtype=bool) if mask is None else np.asarray(mask, bool)
    ids = np.flatnonzero(active)
    t1 = np.zeros((len(rest), 3), np.float32)
    t2 = np.zeros((len(rest), 3), np.float32)
    if len(ids) < 3:
        return t1, t2
    points = rest[ids]
    degree = min(max(int(k), 2), len(points) - 1)
    _, nn = cKDTree(points).query(points, k=degree + 1, workers=-1)
    local = points[nn[:, 1:]] - points[:, None, :]
    cov = np.einsum("nki,nkj->nij", local, local) / float(degree)
    _, basis = np.linalg.eigh(cov)
    t1[ids] = basis[:, :, 2]
    t2[ids] = basis[:, :, 1]
    return t1, t2


def offsets_to_coeffs(offsets: np.ndarray, t1: np.ndarray,
                      t2: np.ndarray) -> np.ndarray:
    """(N,C,3) tangent-plane offsets -> (N,C,2) coefficients in the frozen basis."""
    a = np.einsum("ncj,nj->nc", offsets, t1)
    b = np.einsum("ncj,nj->nc", offsets, t2)
    return np.ascontiguousarray(np.stack([a, b], axis=2), np.float32)


def coeffs_to_offsets_torch(coeff: torch.Tensor, t1: torch.Tensor,
                            t2: torch.Tensor) -> torch.Tensor:
    """(N,C,2) coefficients -> (N,C,3) material offsets, differentiable."""
    return (coeff[..., 0:1] * t1[:, None, :] + coeff[..., 1:2] * t2[:, None, :])


def dressing_feasible_map(coeff: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor,
                          F: torch.Tensor, cap: float) -> torch.Tensor:
    """Exact joint feasibility (design §4.2): (1) project onto the per-parent
    zero-centroid plane (subtract the child mean of the coefficients — the basis is
    shared per parent, so this IS the centroid projection), then (2) uniformly
    rescale each parent's child set by min(1, cap / max_c ||F_p off_pc||). Uniform
    rescaling commutes with the zero-centroid constraint, so BOTH constraints hold
    exactly on return — unlike clamp-then-center (Codex 15: [r,r,-r] violated the
    cap). A feasible map, not the Euclidean projection; sufficient for a
    box-constrained descent, deterministic, order-independent."""
    c = coeff - coeff.mean(dim=1, keepdim=True)
    off = coeffs_to_offsets_torch(c, t1, t2)
    world = torch.einsum("nij,ncj->nci", F.reshape(-1, 3, 3), off)
    m = world.norm(dim=2).amax(dim=1)                       # (N,) max child reach
    s = torch.clamp(cap / m.clamp_min(1e-12), max=1.0)
    return c * s[:, None, None]
