"""The runtime phase after the freeze (tag freeze-2026-10-09): every speed-up in this file leaves the result as it was.

Each test runs the path before the change and the path after it on the same case and asks for the same numbers: bit
for bit on the CPU (Warp's CPU kernels are deterministic), and within the transfers' atomics on CUDA, where two
rollouts of one control already differ (the replay noise every window measures).

(1) The tape without the geometric deformation (RolloutSpec.track_geom False, the window's spec): no term of the
    settled path reads Fg, and leaving its kernels and buffers out changes no other output and no gradient.
(2) The relaxation's reference (window/layer.TargetRelief.at) looks up only the layer's particles on the target's
    surface: the same reference, bit for bit, as the lookup of the whole body.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from physmorph.mpm.function import PersistentAdjoint
from test_volume_exact import _expand, _window_case


def _dev_or_skip(dev):
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")


def _tape_outputs(spec, dc, u):
    """The persistent tape's outputs at (dc, u) and the gradients of a loss on x, F, v and every step's velocity
    (the settled objective's arguments), on both leaves."""
    adj = PersistentAdjoint(spec)
    dc, u = dc.clone().requires_grad_(True), u.clone().requires_grad_(True)
    xT, FT, vT, _, V = adj.apply(_expand(dc), u)
    w = torch.linspace(0.5, 1.5, 3, device=xT.device)
    L = (xT * w).pow(2).sum() * 1e2 + FT.pow(2).sum() + V.pow(2).sum() * 1e2 + (vT * w).sum()
    g_dc, g_u = torch.autograd.grad(L, (dc, u))
    return [t.detach() for t in (xT, FT, vT, V, g_dc, g_u)]


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
@pytest.mark.parametrize("volume", ["off", "carried"])
def test_tape_without_the_geometric_deformation_is_the_same(dev, volume):
    _dev_or_skip(dev)
    spec, dc, u = _window_case(dev)
    if volume != "off":
        N = spec.x0.shape[0]
        J0 = (np.linalg.det(spec.F0) * np.random.default_rng(5).uniform(0.97, 1.03, N)).astype(np.float32)
        spec = dataclasses.replace(spec, volume_exact=volume, J0=J0)
    with_geom = _tape_outputs(dataclasses.replace(spec, track_geom=True), dc, u)
    without = _tape_outputs(dataclasses.replace(spec, track_geom=False), dc, u)
    for a, b in zip(with_geom, without):
        if dev == "cpu":
            assert torch.equal(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6 * max(1.0, float(a.abs().max())))
    assert float(with_geom[4].abs().max()) > 0 and float(with_geom[5].abs().max()) > 0


def _relief_whole_body(relief, x, mask, nrm, nbr, w, local=None):
    """TargetRelief.at as frozen (tag freeze-2026-10-09): every particle looked up on the surface."""
    from physmorph.pipeline.window.layer import _own
    d, i = relief.tree.query(x, 1)
    i = i.reshape(-1)
    q, m = relief.points[i], relief.normals[i]
    foot = x - ((x - q) * m).sum(1, keepdim=True) * m
    res = (nrm * (foot - (w[..., None] * foot[nbr]).sum(1))).sum(1)
    near = (mask > 0.5) & (d.float().reshape(-1) < _own(relief.reach, None if local is None else local.float()))
    return torch.where(near, res - (w * res[nbr]).sum(1), torch.zeros((), device=x.device))


@pytest.mark.parametrize("with_local", [False, True])
def test_relief_reference_from_the_layer_alone_is_the_same(with_local):
    _dev_or_skip("cuda")
    from physmorph.pipeline.window.layer import TargetRelief, layer_relax_data, layer_spacing
    rng = np.random.default_rng(11)
    g = np.arange(24, dtype=np.float32) * 0.1
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    x = torch.tensor(X - X.mean(0) + rng.uniform(-0.03, 0.03, X.shape).astype(np.float32), device="cuda")
    local = torch.tensor(rng.uniform(0.9, 1.1, len(x)).astype(np.float32), device="cuda") if with_local else None
    # the target's surface: a sphere a little inside the block's corners, so that some layer particles are within
    # one spacing of it and some are not
    u = rng.normal(size=(20000, 3)).astype(np.float32)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    pts = torch.tensor(1.25 * u, device="cuda")
    nrm_s = torch.tensor(u, device="cuda")
    sp = layer_spacing(x, local)
    relief = TargetRelief(pts, nrm_s, sp)
    mask, nrm, nbr, w = layer_relax_data(x, sp, k=24, h_sp=2.0, local=local)
    got = relief.at(x, mask, nrm, nbr, w, local)
    ref = _relief_whole_body(relief, x, mask, nrm, nbr, w, local)
    assert torch.equal(got, ref)
    assert int((got != 0).sum()) > 50 and int(((mask > 0.5) & (got == 0)).sum()) > 50
