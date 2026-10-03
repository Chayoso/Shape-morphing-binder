"""The outer particle layer of a window: membership, normals and same-side neighbourhoods.

Frozen at the window start and read by the relaxation projection and the u control in the
MPM kernels (k_layer_resid / k_layer_project). All on the device.
"""
from __future__ import annotations

import torch

from ... import gpu
from ...render.knn_gpu import knn_self_torch


def layer_by_asymmetry(x: torch.Tensor, spacing: float, k: int = 32, thr_sp: float = 0.5):
    """(mask (N,) bool, outward normal (N,3)): the offset of a particle from the centroid of
    its k nearest neighbours is ~0 inside and about half a spacing at the surface (SPH
    surface detection); thr_sp spacings is the depth of the first layer."""
    _, nb = knn_self_torch(x, k + 1)
    off = x - x[nb[:, 1:]].mean(1)
    n = off.norm(dim=1)
    return n >= thr_sp * spacing, off / (n[:, None] + 1e-12)


def layer_relax_data(x0: torch.Tensor, spacing: float, k: int = 24, h_sp: float = 2.0,
                     thr_sp: float = 0.5):
    """(mask (N,) float, nrm (N,3), nbr (N,k) int, w (N,k)): the layer by neighbourhood
    asymmetry; each layer particle's k nearest layer particles weighted by a Gaussian of
    h_sp spacings times the normal agreement (same side only), rows normalised to 1. Rows of
    particles off the layer hold zeros."""
    N = x0.shape[0]
    mask, nrm = layer_by_asymmetry(x0, spacing, thr_sp=thr_sp)
    idx = torch.nonzero(mask).squeeze(1)
    nbr = torch.zeros(N, k, dtype=torch.long, device=x0.device)
    w = torch.zeros(N, k, device=x0.device)
    if len(idx) > k:
        P, R = x0[idx].contiguous(), nrm[idx]
        d, nb = knn_self_torch(P, k + 1)
        d, nb = d[:, 1:], nb[:, 1:]
        ww = torch.exp(-(d / (h_sp * spacing)) ** 2) * torch.clamp((R[nb] * R[:, None, :]).sum(-1), min=0.0)
        ww = ww / torch.clamp(ww.sum(1, keepdim=True), min=1e-12)
        nbr[idx] = idx[nb]
        w[idx] = ww.float()
    return mask.float(), nrm.float(), nbr, w


def layer_spacing(x0: torch.Tensor) -> float:
    """The window's particle spacing: the median 8th-neighbour distance of a 20000-particle
    subsample, rescaled to the full density."""
    return gpu.median_kth_spacing(x0, 8, subsample=20000)
