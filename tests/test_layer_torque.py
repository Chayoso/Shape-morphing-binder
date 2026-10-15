"""D150 (b) (`layer_torque_free`): the outer layer's position update below the grid is also free of the three torque
modes n x v0 (mpm/layer_torque.py), so that sum m dx x v0 = 0 for its displacement in every step.

(1) the relaxation (kernels' nine-mode projection): with the torque modes, every step's layer displacement has no
    torque at v0, and still no net translation or rotation; with the six rigid modes alone it has;
(2) u's projection (window.solve.free_of_rigid's modes, layer_torque.u_modes): the projected u moves the layer with no
    translation, rotation or torque at v0; without v0 the modes are the six of D94, as they were;
(3) the adjoint with the torque modes: the hand-written one against Warp's generated one, and a window (persistent
    graphs, the plain bridge) against finite differences on dFc and u.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from physmorph.mpm import layer_torque as LT
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes
from physmorph.pipeline.window.layer import layer_relax_data as _layer_relax_data


def _slab(n_side=12, layers=4, sp=0.25, seed=3):
    rng = np.random.default_rng(seed)
    g = np.arange(n_side) * sp
    X, Y, Z = np.meshgrid(g, np.arange(layers) * sp, g, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    x += rng.uniform(-0.05, 0.05, x.shape).astype(np.float32) * sp
    x -= x.mean(0)
    x[:, 1] += np.where(x[:, 1] > x[:, 1].max() - 0.5 * sp, 0.3 * sp * np.sin(9. * x[:, 0]), 0.).astype(np.float32)
    return x, sp


def _v0(x):
    """A smooth velocity field that is neither rigid nor uniform."""
    return np.stack((np.sin(3 * x[:, 0] + 2 * x[:, 2]), 0.5 * np.cos(2 * x[:, 0]), np.sin(3 * x[:, 2]) + 0.4 * x[:, 1]),
                    1).astype(np.float32)


def _layer(x, sp, v0, torque):
    mask, nrm, nbr, w = (t.numpy() for t in _layer_relax_data(torch.as_tensor(x), sp, k=8, h_sp=2.0))
    nv = LT.torque_modes(torch.as_tensor(mask), torch.as_tensor(nrm), torch.as_tensor(v0)).numpy() if torque else None
    return (mask, nrm, nbr, w, 1.0 / 30, None, None, None, nv), mask, nrm


def _steps(torque, T=6):
    x, sp = _slab()
    v0 = _v0(x)
    layer, mask, nrm = _layer(x, sp, v0, torque)
    prm = MPMParams(dx=0.75, dt=1.0 / 120.0, drag=0.0, smoothing=0.955, grid_min=(-6.0,) * 3, nx=16, ny=16, nz=16)
    tr = Trajectory(x, 1.0, 0.0, 0.0, prm, T, v0=v0, device="cpu", requires_grad=False,
                    vol0=compute_rest_volumes(x, 1.0, prm, "cpu"), layer=layer)
    tr.rollout()
    out = []
    for t in range(T):
        d = tr.x[t + 1].numpy().astype(np.float64) - tr.xu[t + 1].numpy().astype(np.float64)   # the layer's own step
        r = x.astype(np.float64) - x.astype(np.float64).mean(0)
        gross = np.linalg.norm(d, axis=1).sum()
        out.append((np.linalg.norm(d.sum(0)) / gross,
                    np.linalg.norm(np.cross(r, d).sum(0)) / (np.linalg.norm(r, axis=1) * np.linalg.norm(d, axis=1)).sum(),
                    np.linalg.norm(np.cross(d, v0).sum(0)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(v0, axis=1)).sum(),
                    gross))
    return np.array(out)


def test_the_relaxation_puts_no_torque_in_at_the_start_velocity():
    on, off = _steps(True), _steps(False)
    assert (on[:, 3] > 0).all() and (off[:, 3] > 0).all()             # it relaxed something in every step
    assert on[:, 0].max() < 1e-4 and on[:, 1].max() < 1e-3            # still no net translation or rotation (D98)
    assert off[:, 0].max() < 1e-4 and off[:, 1].max() < 1e-3
    assert on[:, 2].max() < 1e-4, on[:, 2]                            # D150 (b): sum dx x v0 = 0 in every step
    assert off[:, 2].max() > 1e-2, off[:, 2]                          # the six rigid modes alone leave a torque


def test_u_projected_on_the_nine_modes_has_no_torque_at_v0():
    x, sp = _slab()
    v0 = torch.as_tensor(_v0(x)).double()
    mask, nrm = (t.double() for t in _layer_relax_data(torch.as_tensor(x), sp, k=8, h_sp=2.0)[:2])
    g = mask * torch.rand(len(x), dtype=torch.float64, generator=torch.Generator().manual_seed(1)).round()   # a u gate
    xt = torch.as_tensor(x).double()
    r = xt - xt.mean(0)
    six = LT.u_modes(g, nrm, r)
    assert torch.equal(six, torch.cat((g[:, None] * nrm, g[:, None] * torch.linalg.cross(r, nrm, dim=1)), 1).double())
    A = LT.u_modes(g, nrm, r, v0)
    assert A.shape[1] == 9
    u = torch.randn(len(x), dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    u = u - A @ (torch.linalg.pinv(A.T @ A) @ (A.T @ u))               # solve.free_of_rigid
    du = (g * u)[:, None] * nrm
    scale = (du.norm(dim=1) * (r.norm(dim=1) + v0.norm(dim=1))).sum()
    for v in (du.sum(0), torch.linalg.cross(r, du, dim=1).sum(0), torch.linalg.cross(du, v0, dim=1).sum(0)):
        assert float(v.norm()) < 1e-10 * float(scale)


def _window(dev, torque):
    """A jittered 4^3 block with a start velocity and affine field, the layer with u and the torque modes at v0, 4
    driven + 4 released steps, the polar adjoint (test_volume_exact's window case, less the bonds)."""
    from physmorph.mpm.function import RolloutSpec
    rng = np.random.default_rng(151)
    g = np.arange(4, dtype=np.float32) * 0.25
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    x = (X - X.mean(0) + rng.uniform(-0.03, 0.03, X.shape)).astype(np.float32)
    N, Tc = len(x), 4
    prm = MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.9, smoothing=0.955, grid_min=(-4.0,) * 3, nx=16, ny=16, nz=16)
    vol0 = compute_rest_volumes(x, 1.0, prm, dev)
    v0 = rng.normal(0, 0.3, (N, 3)).astype(np.float32)
    C0 = rng.normal(0, 0.2, (N, 3, 3)).astype(np.float32)
    mask, nrm, lnbr, lw = (t.cpu().numpy() for t in _layer_relax_data(torch.as_tensor(x), 0.25, k=8, h_sp=2.0))
    ref = (rng.normal(0, 0.005, N) * mask).astype(np.float32)
    nv = LT.torque_modes(torch.as_tensor(mask), torch.as_tensor(nrm), torch.as_tensor(v0)).numpy() if torque else None
    layer = (mask, nrm, lnbr, lw, 1.0 / Tc, None, ref, None, nv)
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
    return (xT * w).pow(2).sum() * 1e2 + (vT * w).sum()


def test_the_hand_adjoint_with_the_torque_modes_is_the_generated_one(monkeypatch):
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
def test_a_window_with_the_torque_modes_has_an_exact_adjoint(dev):
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
    six, *_ = _window(dev, False)                                      # the torque modes change the window's end
    off = warp_mpm_ext(_expand(dc.detach()), six, u_t=u.detach())
    assert float((off[0] - got[0].detach()).abs().max()) > 1e-7
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
