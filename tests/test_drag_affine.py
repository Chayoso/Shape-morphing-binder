"""D150 (`MPMParams.drag_affine`, `--drag_affine`): the drag damps APIC's affine momentum m C in P2G as it damps m v, so
the transferred momentum is scaled as a whole.

APIC's angular momentum (Jiang et al. 2015) is L = sum m (x x v) + sum m D axial(C), D = dx^2 / 3 for the cubic
B-spline, and P2G / G2P keep it. The frozen drag scales the first part by (1 - dt drag) and not the second, so a body
whose two parts are opposite (a deforming body with no net spin) gains L = -dt drag L_v every step.

(1) a deforming, elastic body with L = 0 split into opposite parts: with the switch its L stays at float zero over six
    steps; without it, it grows by the drag's share of L_v; the linear momentum is the same either way;
(2) the switch off is the frozen path: drag_c 0 (the kernels' expression unchanged; tests/test_volume_exact.py (1)
    compares the whole window bit for bit against the code before the change);
(3) the adjoint with the switch: the hand-written adjoints against Warp's generated ones, and a window against finite
    differences on dFc and u.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes


def _axial(C):
    return np.stack((C[:, 2, 1] - C[:, 1, 2], C[:, 0, 2] - C[:, 2, 0], C[:, 1, 0] - C[:, 0, 1]), 1)


def _L(x, v, C, m, dx):
    """APIC's angular momentum about the origin, float64: the particles' and the affine part's."""
    x, v, C = (np.asarray(a, np.float64) for a in (x, v, C))
    return m * np.cross(x, v).sum(0), m * dx * dx / 3.0 * _axial(C).sum(0)


def _body(seed=150, n_side=6, h=0.25, dx=0.5):
    """A jittered block about the origin, deformed (F), with a random velocity field of no net momentum and a random
    affine field with a net spin; a rigid rotation added to v cancels the total L: L_v = -L_C, both of order one."""
    rng = np.random.default_rng(seed)
    g = np.arange(n_side) * h
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    x = X - X.mean(0) + rng.uniform(-0.03, 0.03, X.shape)
    x = x - x.mean(0)
    N = len(x)
    v = rng.normal(0, 0.2, (N, 3))
    v -= v.mean(0)
    w = np.array([0.3, -0.2, 0.5])
    S = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    C = rng.normal(0, 0.3, (N, 3, 3)) + 4.0 * S
    Lv, LC = _L(x, v, C, 1.0, dx)
    inertia = (np.sum(x * x) * np.eye(3) - x.T @ x)
    om = -np.linalg.solve(inertia, Lv + LC)
    v = v + np.cross(om, x)
    sym = rng.normal(0, 0.05, (N, 3, 3))
    F0 = np.eye(3) + 0.5 * (sym + sym.transpose(0, 2, 1))
    return (x.astype(np.float32), v.astype(np.float32), C.astype(np.float32), F0.astype(np.float32))


def _run(x, v, C, F0, affine, T=6, dx=0.5):
    prm = MPMParams(dx=dx, dt=1.0 / 240.0, drag=0.9, drag_affine=affine, smoothing=0.955, grid_min=(-4.0,) * 3,
                    nx=16, ny=16, nz=16)
    tr = Trajectory(x, 1.0, 800.0, 400.0, prm, T, v0=v, C0=C, F0=F0, device="cpu", requires_grad=False,
                    vol0=compute_rest_volumes(x, 1.0, prm, "cpu"))
    tr.rollout()
    return tr.x[T].numpy(), tr.v[T].numpy(), tr.C[T].numpy()


def test_a_deforming_body_with_no_spin_keeps_none_under_the_drag():
    x, v, C, F0 = _body()
    Lv, LC = _L(x, v, C, 1.0, 0.5)
    L0 = np.linalg.norm(Lv + LC)
    scale = np.linalg.norm(LC)
    assert scale > 1.0 and L0 < 1e-6 * scale, (L0, scale)          # opposite parts of order one, total at float zero
    on = _run(x, v, C, F0, True)
    off = _run(x, v, C, F0, False)
    L_on = np.linalg.norm(sum(_L(*on, 1.0, 0.5)))
    L_off = np.linalg.norm(sum(_L(*off, 1.0, 0.5)))
    assert L_on < 2e-6 * scale, (L_on, scale)                       # D150: the drag scales L as a whole: 0 stays 0
    assert L_off > 5e-3 * scale, (L_off, scale)                     # frozen: about 6 steps x dt drag x |L_v| (0.0225)
    p_on, p_off = on[1].astype(np.float64).sum(0), off[1].astype(np.float64).sum(0)
    assert np.linalg.norm(p_on - p_off) < 1e-5 * np.abs(v).sum()    # the linear momentum is the same either way


def test_off_is_the_frozen_path():
    assert MPMParams().drag_c == 0.0 and MPMParams(drag=0.9).drag_c == 0.0
    assert MPMParams(drag=0.9, drag_affine=True).drag_c == 0.9
    assert not MPMParams().drag_affine


def _window(dev, affine):
    """test_volume_exact's window case with the drag switch (a jittered 4^3 block with a start state, the layer with
    u, 4 driven + 4 released steps, the polar adjoint)."""
    from physmorph.mpm.function import RolloutSpec
    from physmorph.pipeline.window.layer import layer_relax_data
    rng = np.random.default_rng(150)
    g = np.arange(4, dtype=np.float32) * 0.25
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    x = (X - X.mean(0) + rng.uniform(-0.03, 0.03, X.shape)).astype(np.float32)
    N, Tc = len(x), 4
    prm = MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.9, drag_affine=affine, smoothing=0.955, grid_min=(-4.0,) * 3,
                    nx=16, ny=16, nz=16)
    vol0 = compute_rest_volumes(x, 1.0, prm, dev)
    v0 = rng.normal(0, 0.1, (N, 3)).astype(np.float32)
    C0 = rng.normal(0, 0.5, (N, 3, 3)).astype(np.float32)
    mask, nrm, lnbr, lw = (t.cpu().numpy() for t in layer_relax_data(torch.as_tensor(x), 0.25, k=8, h_sp=2.0))
    layer = (mask, nrm, lnbr, lw, 1.0 / Tc, None, None)
    spec = RolloutSpec(x0=x, m=1.0, lam=800.0, mu=400.0, prm=prm, T=2 * Tc, v0=v0, C0=C0, device=dev, vol0=vol0,
                       layer=layer, control_steps=Tc, polar_adjoint=True)
    dc = torch.tensor(rng.normal(0, 0.01, (Tc, N, 3, 3)).astype(np.float32), device=dev)
    u = torch.tensor((rng.normal(0, 0.002, N) * mask).astype(np.float32), device=dev)
    return spec, dc, u


def _expand(z):
    return torch.cat((z, torch.zeros_like(z)))


def _loss(out):
    xT, vT = out[0], out[2]
    w = torch.linspace(0.5, 1.5, 3, device=xT.device)
    return (xT * w).pow(2).sum() * 1e2 + (vT * w).sum() * 10.0


def test_the_hand_adjoint_with_the_switch_is_the_generated_one(monkeypatch):
    import physmorph.mpm.traj as TR
    from physmorph.mpm.function import warp_mpm_ext
    spec, dc, u = _window("cpu", True)
    out = {}
    for hand in (True, False):
        monkeypatch.setattr(TR, "HAND_ADJOINTS", hand)
        d, uu = dc.clone().requires_grad_(True), u.clone().requires_grad_(True)
        o = warp_mpm_ext(_expand(d), spec, u_t=uu)
        out[hand] = (o[0].detach(), torch.autograd.grad(_loss(o), (d, uu)))
    torch.testing.assert_close(out[True][0], out[False][0], rtol=0, atol=0)
    for a, b in zip(out[True][1], out[False][1]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5 * float(b.abs().max()))   # as test_runtime_exact (3)


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_a_window_with_the_switch_has_an_exact_adjoint(dev):
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from physmorph.mpm.function import PersistentAdjoint, warp_mpm_ext
    spec, dc, u = _window(dev, True)
    adj = PersistentAdjoint(spec)
    dc, u = dc.clone().requires_grad_(True), u.clone().requires_grad_(True)
    got = adj.apply(_expand(dc), u)
    plain = warp_mpm_ext(_expand(dc), spec, u_t=u)
    tol = 0.0 if dev == "cpu" else 2e-5
    for a, b in zip(got, plain):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol * max(1.0, float(b.detach().abs().max())))
    g = torch.autograd.grad(_loss(got), (dc, u))
    ref = torch.autograd.grad(_loss(plain), (dc, u))
    for a, b in zip(g, ref):
        torch.testing.assert_close(a, b, rtol=max(tol, 1e-6) * 10, atol=1e-6 * max(1.0, float(b.detach().abs().max())))
    # the switch acts in this window: the frozen drag ends elsewhere
    frozen = dataclasses.replace(spec, prm=dataclasses.replace(spec.prm, drag_affine=False))
    off = warp_mpm_ext(_expand(dc.detach()), frozen, u_t=u.detach())
    assert float((off[2] - got[2].detach()).abs().max()) > 1e-6
    for i, gi in enumerate(g):
        assert torch.isfinite(gi).all() and float(gi.abs().max()) > 0
        d = -gi / gi.norm()
        vals = []
        for sign in (1.0, -1.0):
            leaves = [dc.detach(), u.detach()]
            leaves[i] = leaves[i] + sign * 1e-3 * d
            with torch.no_grad():
                vals.append(float(_loss(adj.apply(_expand(leaves[0]), leaves[1]))))
        fd = (vals[0] - vals[1]) / 2e-3
        an = float((gi * d).sum())
        assert fd == pytest.approx(an, rel=0.03, abs=1e-4 * max(1.0, abs(an))), (i, fd, an)
