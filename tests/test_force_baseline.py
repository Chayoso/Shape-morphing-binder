"""B13 (`--baseline force` / `force_mf`, physmorph/mpm/force.py): an external force per particle in the driven steps in
place of dFc, which is held at zero; everything else of the pipeline as it is.

(1) the momentum-free projection: no net force and no net torque at the step's positions, equal to the float64
    reference (force.rigid_free);
(2) the hand adjoints of the force's velocity increment, plain and projected (the projection's dependence on the
    positions included), against float64 autograd of the same map;
(3) a window (the persistent graphs, the production path) against central finite differences on the force and u, both
    arms, CPU and CUDA;
(4) momentum: the grid's linear and angular momentum at every step; the plain force adds dt sum f and dt sum x x f, the
    momentum-free force adds neither, to float rounding, with and without the elastic stress;
(5) a zero force is no force: the rollout bit for bit (CPU) and the gradients to rounding;
(6) in a run (CUDA): dFc stays zero, the force acts, the records; the default run has no force.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm import force as FK
from physmorph.mpm.function import PersistentAdjoint
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes
from test_volume_exact import _expand, _window_case


def _dev_or_skip(dev):
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")


def _one_step(dev, mf, rg, N=3000, seed=13):
    """A cloud away from the origin (the projection centres on its own mean) with velocities and a force: a stand-in
    trajectory with one driven step for force.attach / force.kick."""
    rng = np.random.default_rng(seed)
    x = (rng.uniform(-1.0, 1.0, (N, 3)) * np.array([1.0, 0.6, 1.4]) + np.array([3.0, -2.0, 1.0])).astype(np.float32)
    v = rng.normal(0, 0.1, (N, 3)).astype(np.float32)
    f = (rng.normal(0, 1.0, (N, 3)) + np.array([0.3, -0.2, 0.5])).astype(np.float32)
    tr = SimpleNamespace(device=dev, N=N, requires_grad=rg, control_steps=1,
                         x=[wp.array(x, dtype=wp.vec3, device=dev, requires_grad=rg)],
                         v=[wp.array(v, dtype=wp.vec3, device=dev, requires_grad=rg)])
    fa = wp.array(f, dtype=wp.vec3, device=dev, requires_grad=rg)
    FK.attach(tr, ([fa], mf, 0.05))
    return tr, x, v, f, fa


# ---- (1) the projection ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_the_projection_has_no_net_force_or_torque_and_is_the_reference(dev):
    _dev_or_skip(dev)
    tr, x, v, f, _ = _one_step(dev, True, False)
    vk = FK.kick(tr, 0).numpy().astype(np.float64)
    fo = tr.force_fo[0].numpy().astype(np.float64)
    ref = FK.rigid_free(torch.as_tensor(f), torch.as_tensor(x)).numpy()
    assert np.abs(fo - ref).max() < 1e-6 * np.abs(f).max()
    assert np.allclose(vk, v + 0.05 * fo, rtol=0, atol=1e-6)
    r = x.astype(np.float64) - x.astype(np.float64).mean(0)
    gross = np.linalg.norm(fo, axis=1).sum()
    assert np.linalg.norm(fo.sum(0)) < 1e-6 * gross
    assert np.linalg.norm(np.cross(r, fo).sum(0)) < 1e-6 * (np.linalg.norm(r, axis=1) * np.linalg.norm(fo, axis=1)).sum()
    # the force itself has both (the projection removes something), and what is left is most of it
    assert np.linalg.norm(f.sum(0)) > 0.1 * np.linalg.norm(f, axis=1).sum()
    assert np.linalg.norm(fo) > 0.5 * np.linalg.norm(f)


# ---- (2) the hand adjoints ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
@pytest.mark.parametrize("mf", [False, True])
def test_the_increments_adjoint_is_float64_autograds(dev, mf):
    _dev_or_skip(dev)
    tr, x, v, f, fa = _one_step(dev, mf, True)
    W = np.random.default_rng(3).normal(0, 1.0, x.shape).astype(np.float32)
    tape = wp.Tape()
    with tape:
        vk = FK.kick(tr, 0)
    tape.backward(grads={vk: wp.array(W, dtype=wp.vec3, device=dev)})
    got = [a.grad.numpy().astype(np.float64) for a in (tr.x[0], tr.v[0], fa)]
    X, V, F = (torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (x, v, f))
    out = V + 0.05 * (FK.rigid_free(F, X) if mf else F)
    gs = torch.autograd.grad((out * torch.as_tensor(W, dtype=torch.float64)).sum(), (X, V, F), allow_unused=True)
    ref = [np.zeros(x.shape) if g is None else g.numpy() for g in gs]
    for name, a, b in zip(("x", "v", "f"), got, ref):
        scale = max(np.abs(b).max(), 1e-12)
        assert np.abs(a - b).max() <= 1e-5 * scale, (name, float(np.abs(a - b).max()), scale)
    if mf:
        assert np.abs(ref[0]).max() > 1e-4 * np.abs(ref[2]).max()      # the positions' part is not negligible
    else:
        assert np.abs(got[0]).max() == 0.0


# ---- (3) a window against finite differences ------------------------------------------------------------------------
def _force_case(dev, mf, c=0.05):
    spec, dc, u = _window_case(dev)
    spec = dataclasses.replace(spec, force=(mf, c), track_geom=False)
    Tc, N = dc.shape[0], dc.shape[1]
    f = torch.tensor(np.random.default_rng(31).normal(0, 1.0, (Tc, N, 3)).astype(np.float32), device=dev)
    return spec, torch.zeros_like(dc), u, f


def _loss(out):
    xT, FT, vT, _, V = (o.double() for o in out)
    w = torch.linspace(0.5, 1.5, 3, dtype=torch.float64, device=xT.device)
    return (xT * w).pow(2).sum() * 1e2 + V.pow(2).sum() * 1e2 + FT.pow(2).sum() * 1e-1 + (vT * w).sum()


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
@pytest.mark.parametrize("mf", [False, True])
def test_a_window_with_the_force_has_an_exact_adjoint(dev, mf):
    _dev_or_skip(dev)
    spec, dc0, u, f = _force_case(dev, mf)
    adj = PersistentAdjoint(spec)
    f, u = f.clone().requires_grad_(True), u.clone().requires_grad_(True)
    out = adj.apply(_expand(dc0), u, f)
    g = torch.autograd.grad(_loss(out), (f, u))
    # the force acts: without it the same u ends elsewhere
    free = adj.apply(_expand(dc0), u.detach(), torch.zeros_like(f))
    assert float((free[0] - out[0].detach()).abs().max()) > 1e-4
    for i, gi in enumerate(g):
        assert torch.isfinite(gi).all() and float(gi.abs().max()) > 0
        d = -gi / gi.norm()
        h = 1e-1 if i == 0 else 1e-3
        vals = []
        for sign in (1.0, -1.0):
            leaves = [f.detach(), u.detach()]
            leaves[i] = leaves[i] + sign * h * d
            with torch.no_grad():
                vals.append(float(_loss(adj.apply(_expand(dc0), leaves[1], leaves[0]))))
        fd = (vals[0] - vals[1]) / (2 * h)
        an = float((gi * d).sum())
        rel = 1e-3 if dev == "cpu" else 0.03                    # CUDA: the atomics' replay noise
        assert fd == pytest.approx(an, rel=rel, abs=1e-4 * max(1.0, abs(an))), (i, fd, an)


# ---- (4) momentum ---------------------------------------------------------------------------------------------------
def _grid_momenta(tr, prm, T):
    """Per step: the grid's linear momentum sum_i p_i and angular momentum sum_i x_i x p_i after P2G (float64)."""
    ii, jj, kk = np.meshgrid(np.arange(prm.nx), np.arange(prm.ny), np.arange(prm.nz), indexing="ij")
    xi = (np.stack([ii, jj, kk], -1).reshape(-1, 3) * prm.dx + np.array(prm.grid_min)).astype(np.float64)
    P, L = [], []
    for t in range(T):
        p = tr.gmom[t].numpy().astype(np.float64)
        P.append(p.sum(0))
        L.append(np.cross(xi, p).sum(0))
    return np.array(P), np.array(L)


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
@pytest.mark.parametrize("elastic", [False, True])
def test_the_momentum_free_force_injects_no_momentum(dev, elastic):
    _dev_or_skip(dev)
    g = np.arange(6, dtype=np.float32) * 0.25
    x = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    rng = np.random.default_rng(149)
    x = (x - x.mean(0) + rng.uniform(-0.03, 0.03, x.shape) + np.array([0.3, -0.2, 0.1])).astype(np.float32)
    N, T, m, c = len(x), 8, 0.7, 0.05
    prm = MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.0, smoothing=0.955, grid_min=(-6.0,) * 3, nx=24, ny=24, nz=24)
    vol0 = compute_rest_volumes(x, m, prm, dev)
    lam, mu = (800.0, 400.0) if elastic else (0.0, 0.0)
    fs = rng.normal(0, 1.0, (T, N, 3)).astype(np.float32) + np.array([0.5, 0.2, -0.3], np.float32)
    out = {}
    for mf in (False, True):
        f = [wp.array(fs[t], dtype=wp.vec3, device=dev, requires_grad=True) for t in range(T)]
        tr = Trajectory(x, m, lam, mu, prm, T, device=dev, requires_grad=True, vol0=vol0, control_steps=T,
                        force=(f, mf, c))
        tr.rollout()
        P, L = _grid_momenta(tr, prm, T)
        xs = [tr.x[t].numpy().astype(np.float64) for t in range(T)]
        incs = [m * c * (FK.rigid_free(torch.as_tensor(fs[t]), torch.as_tensor(xs[t])).numpy() if mf
                         else fs[t].astype(np.float64)) for t in range(T)]
        gross = np.cumsum([np.linalg.norm(i, axis=1).sum() for i in incs])
        gross_ang = np.cumsum([(np.linalg.norm(xs[t] - xs[t].mean(0), axis=1)
                                * np.linalg.norm(incs[t], axis=1)).sum() for t in range(T)])
        out[mf] = (P, L, incs, xs, gross, gross_ang)
    tol = 1e-5 if elastic else 2e-6
    # the plain force: each step adds its impulse, linear and angular
    P, L, incs, xs, gross, gross_ang = out[False]
    P_ref = np.cumsum([i.sum(0) for i in incs], axis=0)
    L_ref = np.cumsum([np.cross(xs[t], incs[t]).sum(0) for t in range(T)], axis=0)
    assert np.linalg.norm(P - P_ref, axis=1).max() < tol * gross[-1]
    assert np.linalg.norm(L - L_ref, axis=1).max() < 10 * tol * np.abs(L_ref).max() + tol * gross_ang[-1]
    assert np.linalg.norm(P[-1]) > 0.05 * gross[-1]
    # the momentum-free force: none, at any step
    P, L, _, _, gross, gross_ang = out[True]
    assert (np.linalg.norm(P, axis=1) < tol * gross).all(), (np.linalg.norm(P, axis=1) / gross)
    assert (np.linalg.norm(L, axis=1) < tol * gross_ang).all(), (np.linalg.norm(L, axis=1) / gross_ang)


# ---- (5) a zero force is no force -----------------------------------------------------------------------------------
@pytest.mark.parametrize("mf", [False, True])
def test_a_zero_force_is_no_force(mf):
    spec, dc, u = _window_case("cpu")
    dc, u = dc.clone().requires_grad_(True), u.clone().requires_grad_(True)
    plain = PersistentAdjoint(spec).apply(_expand(dc), u)
    forced = PersistentAdjoint(dataclasses.replace(spec, force=(mf, 0.05))).apply(
        _expand(dc), u, torch.zeros(dc.shape[0], dc.shape[1], 3))
    for a, b in zip(plain, forced):
        assert torch.equal(a, b)
    ga = torch.autograd.grad(_loss(plain), (dc, u))
    gb = torch.autograd.grad(_loss(forced), (dc, u))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7 * float(a.abs().max()))


# ---- (6) in a run ---------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the window runs on CUDA")
@pytest.mark.parametrize("baseline", ["", "force", "force_mf"])
def test_in_a_run_dfc_stays_zero_and_the_force_acts(monkeypatch, baseline):
    from physmorph.pipeline import PipelineConfig, run_pipeline
    from physmorph.pipeline.window import setup
    seen = []
    init = setup.Window.__init__

    def recording(self, *a, **kw):
        init(self, *a, **kw)
        seen.append((self.force, self.spec.force, self.tr.force))

    monkeypatch.setattr(setup.Window, "__init__", recording)
    rng = np.random.default_rng(7)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=3, iters=2, animations=3, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=5, reject_stop=0, baseline=baseline)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert seen
    for force, spec_force, tr_force in seen:
        assert (force or None) == (baseline or None) and tr_force == bool(baseline)
        assert (spec_force is None) == (not baseline)
        if baseline:
            assert spec_force[0] == (baseline == "force_mf") and spec_force[1] > 0
    wins = [h for h in res["history"] if h.get("frame_end") and not h.get("null_commit")]
    assert wins
    for h in wins:
        if not baseline:
            assert not any(k.startswith("force_") for k in h)
            continue
        assert h["dfc_absmax"] == 0.0 and h["force_absmax"] > 0 and np.isfinite(h["force_ratio"])
        if baseline == "force_mf":
            assert h["force_dv"] < 1e-9 * max(h["force_gross"], 1e-30)
    with pytest.raises(ValueError):
        PipelineConfig(baseline="force", gap_split="on")
    with pytest.raises(ValueError):
        PipelineConfig(baseline="force", force_scale=0.0)
