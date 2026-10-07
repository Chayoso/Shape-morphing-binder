"""The outer particle layer of a window: membership, normals and same-side neighbourhoods.

Frozen at the window start and read by the relaxation projection and the u control in the
MPM kernels (k_layer_resid / k_layer_project). All on the device.

`local` (D122, --surface_density): a sample whose density varies by design carries each particle's spacing relative
to the base spacing (w^(1/3), w its rest volume over the mean); every length in spacings (the layer's depth, the
neighbour weight's width, the relief reference's reach) is then the particle's own, base spacing x local. Without it
(the uniform sample) every expression below is the one it was.
"""
from __future__ import annotations

import torch

from ... import gpu
from ...render.knn_gpu import knn_self_torch


def _own(spacing: float, local: torch.Tensor | None, k: float = 1.0):
    """k x spacing, per particle when `local` is given."""
    return k * spacing if local is None else k * spacing * local


def layer_by_asymmetry(x: torch.Tensor, spacing: float, k: int = 32, thr_sp: float = 0.5,
                       local: torch.Tensor | None = None):
    """(mask (N,) bool, outward normal (N,3)): the offset of a particle from the centroid of
    its k nearest neighbours is ~0 inside and about half a spacing at the surface (SPH
    surface detection); thr_sp spacings is the depth of the first layer."""
    _, nb = knn_self_torch(x, k + 1)
    off = x - x[nb[:, 1:]].mean(1)
    n = off.norm(dim=1)
    return n >= _own(spacing, local, thr_sp), off / (n[:, None] + 1e-12)


def layer_relax_data(x0: torch.Tensor, spacing: float, k: int = 24, h_sp: float = 2.0,
                     thr_sp: float = 0.5, local: torch.Tensor | None = None):
    """(mask (N,) float, nrm (N,3), nbr (N,k) int, w (N,k)): the layer by neighbourhood
    asymmetry; each layer particle's k nearest layer particles weighted by a Gaussian of
    h_sp spacings times the normal agreement (same side only), rows normalised to 1. Rows of
    particles off the layer hold zeros. A layer particle with no same-side neighbour within the
    weight's reach has no plane to be relaxed onto: its row is itself (residual zero)."""
    N = x0.shape[0]
    mask, nrm = layer_by_asymmetry(x0, spacing, thr_sp=thr_sp, local=local)
    idx = torch.nonzero(mask).squeeze(1)
    nbr = torch.zeros(N, k, dtype=torch.long, device=x0.device)
    w = torch.zeros(N, k, device=x0.device)
    # a layer particle's row is itself until it has neighbours to be measured against: the kernels read
    # the row's weighted centroid, and a row without weight reads the world's origin (D39: 100 such rows in
    # a 300k run, each carried 31 target spacings toward the plane through the origin)
    nbr[idx] = idx[:, None]
    w[idx, 0] = 1.0
    if len(idx) > k:
        P, R = x0[idx].contiguous(), nrm[idx]
        d, nb = knn_self_torch(P, k + 1)
        d, nb = d[:, 1:], nb[:, 1:]
        width = h_sp * spacing if local is None else (h_sp * spacing * local[idx]).to(d.dtype)[:, None]
        ww = torch.exp(-(d / width) ** 2) * torch.clamp((R[nb] * R[:, None, :]).sum(-1), min=0.0)
        total = ww.sum(1, keepdim=True)
        relaxed = torch.nonzero(total[:, 0] > 1e-12).squeeze(1)
        nbr[idx[relaxed]] = idx[nb[relaxed]]
        w[idx[relaxed]] = (ww[relaxed] / total[relaxed]).float()
    return mask.float(), nrm.float(), nbr, w


class TargetRelief:
    """The relaxation's reference: what its own operator, the rough residual d - dbar over the window's layer graph,
    reads on the layer's feet on the target's surface (each layer particle projected onto the tangent plane of its
    nearest point of the target mesh's surface). Relaxed towards it instead of towards zero, the layer keeps the
    relief the target has below a cell and still loses the sampling noise, which the mesh's surface does not have
    (D82: without the relaxation the surface carries the relief and is as rough as a sample). Read by the same
    operator over the same particles, the reference has the layer's own mean. D88 precomputed the operator on the
    mesh and read it at each particle's nearest surface point: nearest points gather on convex creases, the
    reference's mean was +0.01 to +0.03 spacings where d - dbar has none, and that remainder pushed the whole layer
    outward at every step (D105)."""

    def __init__(self, points: torch.Tensor, normals: torch.Tensor, spacing: float):
        self.points, self.normals = points, normals
        self.tree, self.reach = gpu.KNN(points), spacing

    def at(self, x: torch.Tensor, mask: torch.Tensor, nrm: torch.Tensor, nbr: torch.Tensor,
           w: torch.Tensor, local: torch.Tensor | None = None) -> torch.Tensor:
        """(N,) the reference of each layer particle, zero for a particle farther than one spacing from the target's
        surface (it has not arrived) or off the layer; nrm, nbr, w: the window's layer graph (layer_relax_data);
        local: the particles' own spacing over the base (the reach is then one spacing of its own)."""
        d, i = self.tree.query(x, 1)
        i = i.reshape(-1)
        q, m = self.points[i], self.normals[i]
        foot = x - ((x - q) * m).sum(1, keepdim=True) * m
        res = (nrm * (foot - (w[..., None] * foot[nbr]).sum(1))).sum(1)
        near = (mask > 0.5) & (d.float().reshape(-1) < _own(self.reach, None if local is None else local.float()))
        return torch.where(near, res - (w * res[nbr]).sum(1), torch.zeros((), device=x.device))


def layer_spacing(x0: torch.Tensor, local: torch.Tensor | None = None) -> float:
    """The window's particle spacing: the median 8th-neighbour distance of a 20000-particle
    subsample, rescaled to the full density (with `local`, the base spacing: gpu.median_kth_spacing)."""
    return gpu.median_kth_spacing(x0, 8, subsample=20000, local=local)
