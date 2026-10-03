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


def detached_groups(d: torch.Tensor, nb: torch.Tensor, spacing: float):
    """(group (N,) long, size (N,) long) from a self kNN (d, nb): particles are connected when
    nearer than one spacing (single linkage over each particle's listed neighbours); the body is
    the largest connected set (group 0), every other set a detached group with its own positive
    label; size is the number of members of the particle's set."""
    N = d.shape[0]
    ok = d[:, 1:] < spacing
    i = torch.arange(N, device=d.device)[:, None].expand(-1, d.shape[1] - 1)[ok]
    j = nb[:, 1:][ok]
    label = torch.arange(N, device=d.device)
    while True:                                     # min-label propagation with pointer jumping
        new = label.clone()
        new.scatter_reduce_(0, i, label[j], "amin")
        new.scatter_reduce_(0, j, label[i], "amin")
        new = new[new]
        if bool((new == label).all()):
            break
        label = new
    _, inv, cnt = label.unique(return_inverse=True, return_counts=True)
    group = inv + 1
    group[inv == cnt.argmax()] = 0
    return group, cnt[inv]


def outside_neighbours(x: torch.Tensor, pool: torch.Tensor, q: torch.Tensor, group: torch.Tensor,
                       size: torch.Tensor, k: int):
    """For the particles q, their k nearest particles of `pool` that are not members of their own
    group: (indices into x (M,k), distances (M,k), valid (M,k)), nearest first; valid is False
    where the pool holds fewer than k such particles within the query."""
    K = int(min(len(pool), k + int(size[q].max())))
    d, nb = gpu.knn(x[pool], K, queries=x[q])
    nb = pool[nb]
    fellow = group[nb] == group[q][:, None]
    order = torch.sort(fellow.to(torch.int8), dim=1, stable=True).indices[:, :k]   # others first, by distance
    return nb.gather(1, order), d.gather(1, order).float(), ~fellow.gather(1, order)


def layer_relax_data(x0: torch.Tensor, spacing: float, k: int = 24, h_sp: float = 2.0,
                     thr_sp: float = 0.5, k_asym: int = 32, group_query: int = 512):
    """(mask (N,) float, nrm (N,3), nbr (N,k) int, w (N,k)): the layer by neighbourhood
    asymmetry; each layer particle's k nearest layer particles weighted by a Gaussian of
    h_sp spacings times the normal agreement (same side only), rows normalised to 1. Rows of
    particles off the layer hold zeros.

    The layer is the body's. A detached group (detached_groups) has no surface of its own to be
    made regular: its members are not relaxed (their row is themselves, residual zero), and their
    normal, the direction u moves them along, is the asymmetry against the nearest particles that
    are not members of the group, as a single detached particle's already is. Measured against
    itself a group is its own plane and its normals radiate from its own centre: the relaxation
    then holds it where it is and u contracts it into a clump (D20, D21); relaxed against the
    material around it, it is dragged whatever the objective holds it for (D24). Groups of more
    than group_query members keep their own neighbourhoods (the neighbour query's reach)."""
    N = x0.shape[0]
    d_a, nb_a = knn_self_torch(x0, k_asym + 1)
    off = x0 - x0[nb_a[:, 1:]].mean(1)
    group, size = detached_groups(d_a, nb_a, spacing)
    member = (group > 0) & (size > 1) & (size <= group_query)
    q = torch.nonzero(member).squeeze(1)
    if len(q):
        nb_o, _, ok = outside_neighbours(x0, torch.arange(N, device=x0.device), q, group, size, k_asym)
        cen = (x0[nb_o] * ok[..., None]).sum(1) / ok.sum(1, keepdim=True).clamp_min(1)
        has = ok.any(1)
        off[q[has]] = (x0[q] - cen)[has]
    n = off.norm(dim=1)
    mask, nrm = n >= thr_sp * spacing, off / (n[:, None] + 1e-12)
    idx = torch.nonzero(mask).squeeze(1)
    nbr = torch.zeros(N, k, dtype=torch.long, device=x0.device)
    w = torch.zeros(N, k, device=x0.device)
    if len(idx) > k:
        P, R = x0[idx].contiguous(), nrm[idx]
        d, nb = knn_self_torch(P, k + 1)
        d, nb = d[:, 1:], idx[nb[:, 1:]]
        ww = torch.exp(-(d / (h_sp * spacing)) ** 2) * torch.clamp((nrm[nb] * R[:, None, :]).sum(-1), min=0.0)
        ww = ww / torch.clamp(ww.sum(1, keepdim=True), min=1e-12)
        rows = torch.nonzero(member[idx]).squeeze(1)
        if len(rows):                               # a group member's row is itself: no relaxation
            nb[rows] = idx[rows][:, None]
            ww[rows] = 0.0
            ww[rows, 0] = 1.0
        nbr[idx] = nb
        w[idx] = ww.float()
    return mask.float(), nrm.float(), nbr, w


def layer_spacing(x0: torch.Tensor) -> float:
    """The window's particle spacing: the median 8th-neighbour distance of a 20000-particle
    subsample, rescaled to the full density."""
    return gpu.median_kth_spacing(x0, 8, subsample=20000)
