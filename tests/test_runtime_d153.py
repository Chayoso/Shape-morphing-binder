"""D153, exact runtime levers on the tag freeze-2026-10-15 (Warp CPU, bit for bit).

(1) SKIP_IDLE_LAYER: a step in which neither the layer relaxation (relax_driven, D146: the driven steps) nor u (the
    driven steps) acts leaves the layer's three kernels and their adjoints out. The re-freeze's window case
    (tests/data/refreeze_ref.npz: bonds with fragment particles, the minimum spacing on the symmetric closure, the
    layer with its reference and u, driven and released steps, the polar adjoint), volume off and carried: the
    outputs and the gradients of dFc and u with the kernels left out equal those with every step's kernels, bit for
    bit; and the released steps' scratch positions xu are not written (the kernels did not run there).
(2) The old path (the relaxation in every step) is untouched: there every step is a layer step.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

import physmorph.mpm.traj as TR
from physmorph.mpm.function import PersistentAdjoint

from test_refreeze_reference import _spec

REF = Path(__file__).parent / "data" / "refreeze_ref.npz"


def _window(r, volume, steps="driven"):
    spec, dc, u = _spec(r)
    if volume == "carried":
        spec = dataclasses.replace(spec, volume_exact="carried", J0=r["J0"])
    if steps == "all":                                   # the frozen path before D146: the relaxation in every step
        spec = dataclasses.replace(spec, layer=spec.layer[:7] + (None,))
    adj = PersistentAdjoint(spec)
    d, uu = dc.clone().requires_grad_(True), u.clone().requires_grad_(True)
    xT, FT, vT, _, V = adj.apply(torch.cat((d, torch.zeros_like(d))), uu)
    w = torch.linspace(0.5, 1.5, 3)
    L = (xT * w).pow(2).sum() * 1e2 + FT.pow(2).sum() + V.pow(2).sum() * 1e2 + (vT * w).sum()
    g_dc, g_u = torch.autograd.grad(L, (d, uu))
    out = {k: t.detach().numpy().copy() for k, t in zip(("xT", "FT", "vT", "V", "g_dc", "g_u"),
                                                      (xT, FT, vT, V, g_dc, g_u))}
    tr = adj.traj
    out["x"] = [tr.x[t].numpy().copy() for t in range(tr.T + 1)]
    out["xu"] = [tr.xu[t].numpy().copy() for t in range(tr.T + 1)]
    return out, tr


@pytest.mark.parametrize("volume", ["off", "carried"])
def test_the_idle_layer_steps_are_left_out_bit_for_bit(volume, monkeypatch):
    monkeypatch.setattr(TR, "HAND_ADJOINTS", True)
    r = dict(np.load(REF))
    monkeypatch.setattr(TR, "SKIP_IDLE_LAYER", False)
    full, _ = _window(r, volume)
    monkeypatch.setattr(TR, "SKIP_IDLE_LAYER", True)
    skip, tr = _window(r, volume)
    for k in ("xT", "FT", "vT", "V", "g_dc", "g_u"):
        assert np.array_equal(full[k], skip[k]), k
    for t in range(tr.T + 1):
        assert np.array_equal(full["x"][t], skip["x"][t]), t
    Tc = int(tr.control_steps)
    assert tr.relax_steps == Tc < tr.T                   # the case is the re-freeze's: relaxation in the driven steps
    for t in range(Tc + 1, tr.T + 1):                    # released steps: the kernels did not run, xu stays unwritten
        assert not np.any(skip["xu"][t]) and np.any(full["xu"][t]), t
    for t in range(1, Tc + 1):                           # driven steps: the same kernels as before
        assert np.array_equal(full["xu"][t], skip["xu"][t]), t
    assert float(np.abs(skip["g_u"]).max()) > 0 and float(np.abs(skip["g_dc"]).max()) > 0


def test_the_relaxation_in_every_step_keeps_every_layer_step(monkeypatch):
    monkeypatch.setattr(TR, "HAND_ADJOINTS", True)
    r = dict(np.load(REF))
    monkeypatch.setattr(TR, "SKIP_IDLE_LAYER", False)
    full, _ = _window(r, "off", steps="all")
    monkeypatch.setattr(TR, "SKIP_IDLE_LAYER", True)
    skip, tr = _window(r, "off", steps="all")
    assert tr.relax_steps == tr.T
    for k in ("xT", "FT", "vT", "V", "g_dc", "g_u"):
        assert np.array_equal(full[k], skip[k]), k
    for t in range(1, tr.T + 1):
        assert np.array_equal(full["xu"][t], skip["xu"][t]) and np.any(skip["xu"][t]), t


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the proximity's KD-tree is CuPy's (CUDA)")
def test_the_proximity_reads_the_repeated_points_lists_bit_for_bit(monkeypatch):
    """(3) D47's repeat (the point of the latest solved call evaluated once more, with the tape) also takes the surface
    proximity's candidate lists that call searched at the same positions: no new KD-tree, the value and the gradient
    bit for bit those of a fresh search; any call without the flag searches, a call without the proximity leaves
    nothing to reuse."""
    from physmorph import gpu
    from physmorph.losses.grid_ot import GridSinkhornLoss
    from physmorph.losses.support import SurfaceProximity
    from physmorph.losses.volumetric import rasterize_mass
    from test_transport_support import sheet_on_block
    target = torch.tensor(sheet_on_block(z=.9), dtype=torch.float32)
    mass = torch.ones(len(target), dtype=target.dtype)
    origin, dims = torch.full((3,), -.5, dtype=target.dtype), (6, 6, 6)
    grid = rasterize_mass(target, mass, origin, .4, dims)
    builds = []
    KNN = gpu.KNN

    class Counting(KNN):
        def __init__(self, points):
            builds.append(int(points.shape[0]))
            super().__init__(points)

    support = SurfaceProximity(target.numpy())
    monkeypatch.setattr(gpu, "KNN", Counting)
    loss = GridSinkhornLoss(grid, origin, .4, dims, eps=.2, iters=2000, tol=1e-6, support=support)
    jitter = torch.tensor(np.random.default_rng(5).uniform(-.02, .02, target.shape), dtype=target.dtype)
    x0 = (target + jitter).clone()
    x0[216:, 2] -= 2.5 * support.spacing                     # the sheet held off: the proximity charges its points
    x0 = x0.contiguous()
    assert float(support.penalty(x0)) > 0
    builds.clear()

    def evaluate(x, repeat=False, forget=False):
        leaf = x.detach().clone().requires_grad_(True)
        loss.repeat = repeat
        if forget:
            loss._sup_idx = None                         # the path before D153: a repeat searches its own lists
        e = loss.state_energy(leaf, mass)
        return e.detach(), torch.autograd.grad(e, leaf)[0]

    first = evaluate(x0)
    assert len(builds) == 1
    again = evaluate(x0, repeat=True)
    assert len(builds) == 1 and loss.repeat is False             # no new search at the repeated point
    fresh = evaluate(x0, repeat=True, forget=True)
    assert len(builds) == 2
    for a, b, c in zip(first, again, fresh):
        assert torch.equal(a, b) and torch.equal(b, c)
    assert float(first[1].abs().sum()) > 0
    evaluate(x0 * 1.01)                                          # no flag: searched
    assert len(builds) == 3
    saved, loss.support = loss.support, None
    evaluate(x0)                                                 # without the proximity: nothing kept to reuse
    loss.support = saved
    evaluate(x0, repeat=True)
    assert len(builds) == 4
