"""D149 (`spacing_symmetric`): the minimum spacing's neighbours as the symmetric closure of each particle's nearest
(physmorph.gpu.symmetric_rows), so every pair pushes both of its particles, equally and oppositely.

(1) the closure is symmetric, keeps every listed pair, adds only the reverse of one-sided ones, and pads each row with
    the particle itself up to the widest row;
(2) one step of the push (kernels.k_update) on a cloud whose lists were taken before a compression: with the closure
    sum m dx and sum m x x dx vanish to float rounding; with each particle's own nearest they do not (one-sided pairs
    that came into overlap move one particle only);
(3) the adjoint: k_update_adj on the closure against float64 autograd and central differences of the same map, and a
    whole window (persistent graphs, the plain bridge) against finite differences on dFc and u;
(4) in a run (CUDA): off, the window's lists are the 16 nearest as before (the frozen path); on, their closure, and the
    record's push moves the centre of mass by nothing but rounding.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.gpu import symmetric_rows
from physmorph.mpm import adjoints as ADJ
from physmorph.mpm import kernels as K
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes


def _nearest(x: np.ndarray, k: int = 16) -> torch.Tensor:
    X = torch.as_tensor(x, dtype=torch.float64)
    return torch.cdist(X, X).argsort(1)[:, 1:k + 1]


def _compressed_cloud(n=1000, seed=149):
    """A uniform cloud (pitch 0.2) and the same cloud compressed to half along x with a little noise: pairs that
    neither or only one of them listed come within the spacing."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    y = (x * np.array([0.5, 1.0, 1.0], np.float32) + rng.normal(0, 0.01, x.shape)).astype(np.float32)
    return x, y, 0.9 * (8.0 / n) ** (1.0 / 3.0)


# ---- (1) the closure ------------------------------------------------------------------------------------------------
def test_the_closure_is_symmetric_keeps_every_pair_and_pads_with_self():
    x, _, _ = _compressed_cloud(600)
    nbr = _nearest(x)
    nbr[5, 0] = 5                                                   # a self entry in the input is dropped
    S = symmetric_rows(nbr)
    N = len(x)
    listed = [set(nbr[p].tolist()) - {p} for p in range(N)]
    rows = [set(S[p].tolist()) - {p} for p in range(N)]
    for p in range(N):
        assert rows[p] == listed[p] | {q for q in range(N) if p in listed[q]}
        for q in rows[p]:
            assert p in rows[q]
    deg = np.array([len(r) for r in rows])
    assert S.shape == (N, deg.max()) and S.dtype == torch.int64
    assert deg.max() > 16                                           # the 16 nearest are not symmetric
    for p in range(N):
        real, pad = S[p, :deg[p]], S[p, deg[p]:]
        assert (real != p).all() and (pad == p).all()               # the padding last, and only the padding is p
        assert (real[1:] > real[:-1]).all()                         # ascending, no duplicate


# ---- (2) the push keeps the centre of mass --------------------------------------------------------------------------
def _one_push(x0: np.ndarray, nbr, r: float) -> np.ndarray:
    """x[1] - x[0] of one step at rest with no elasticity: the minimum spacing's push alone (control_steps 1: the
    whole half overlap in the step), float64 of the float32 positions."""
    prm = MPMParams(dx=0.25, dt=1.0 / 120.0, drag=0.0, smoothing=0.955, grid_min=(-3.0,) * 3, nx=24, ny=24, nz=24)
    tr = Trajectory(x0, 1.0, 0.0, 0.0, prm, 1, device="cpu", requires_grad=False,
                    vol0=compute_rest_volumes(x0, 1.0, prm, "cpu"), control_steps=1, spacing=(nbr, r))
    tr.rollout()
    return tr.x[1].numpy().astype(np.float64) - x0.astype(np.float64)


def _imbalance(x0, dx):
    lin = np.linalg.norm(dx.sum(0)) / np.linalg.norm(dx, axis=1).sum()
    rc = x0.astype(np.float64) - x0.astype(np.float64).mean(0)
    ang = np.linalg.norm(np.cross(rc, dx).sum(0)) / (np.linalg.norm(rc, axis=1) * np.linalg.norm(dx, axis=1)).sum()
    return lin, ang


def test_the_closure_keeps_the_centre_of_mass_and_the_listed_rows_do_not():
    x, y, r = _compressed_cloud()
    nbr = _nearest(x)                                               # the lists of the window's start
    own = _one_push(y, nbr, r)
    sym = _one_push(y, symmetric_rows(nbr), r)
    assert np.linalg.norm(sym, axis=1).mean() > 0.05 * r            # the push acts on the compressed cloud
    lin, ang = _imbalance(y, sym)
    assert lin < 1e-6 and ang < 1e-6, (lin, ang)                    # equal masses: sum m dx = sum m x x dx = 0
    lin0, ang0 = _imbalance(y, own)
    assert lin0 > 100 * max(lin, 1e-8) and ang0 > 100 * max(ang, 1e-8), (lin0, ang0, lin, ang)
    # a particle whose closure row adds no overlapping pair is pushed as before (to the order of the sum)
    Y = torch.as_tensor(y, dtype=torch.float64)
    S = symmetric_rows(nbr)
    extra = torch.stack([torch.as_tensor(np.isin(S[p].numpy(), nbr[p].numpy(), invert=True)) for p in range(len(y))])
    L = (Y[:, None, :] - Y[S]).norm(dim=2)
    near = (L < r) & (L > 1e-9)                                     # the padding (L = 0) is no pair
    same = ~(extra & near).any(1).numpy()
    assert same.sum() > 50
    assert np.allclose(own[same], sym[same], rtol=1e-5, atol=3e-7)


# ---- (3) the adjoint ------------------------------------------------------------------------------------------------
def _push_map64(x: torch.Tensor, v: torch.Tensor, S: torch.Tensor, r: float, dt: float, frac: float) -> torch.Tensor:
    """k_update's position map in float64: x + dt v + frac 0.5 sum over the row's pairs nearer than r of
    (r - L) d / L (the padding, at L = 0, adds nothing)."""
    d = x[:, None, :] - x[S]
    L = d.norm(dim=2, keepdim=True)
    on = (L < r) & (L > 1e-9)
    push = torch.where(on, (r - L) * d / L.clamp_min(1e-9), torch.zeros_like(d)).sum(1)
    return x + dt * v + frac * 0.5 * push


def test_the_push_adjoint_on_the_closure_is_exact():
    _, y, r = _compressed_cloud(400)
    S = symmetric_rows(_nearest(y))
    N, Kp = S.shape
    rng = np.random.default_rng(3)
    v = rng.normal(0, 0.1, (N, 3)).astype(np.float32)
    w = rng.normal(0, 1.0, (N, 3)).astype(np.float32)               # the covector at x_out
    dt, s, frac = 1.0 / 240.0, 0.955, 1.0 / 20.0
    dev = "cpu"
    ints = wp.array(S.reshape(-1).int().numpy(), dtype=wp.int32, device=dev)
    rr = wp.array(np.full(N, r, np.float32), dtype=wp.float32, device=dev)
    nb0, rest0, frag0 = (wp.zeros(1, dtype=wp.int32, device=dev), wp.zeros(1, dtype=wp.float32, device=dev),
                         wp.zeros(N, dtype=wp.float32, device=dev))
    adj_x = wp.zeros(N, dtype=wp.vec3, device=dev)
    adj_v = wp.zeros(N, dtype=wp.vec3, device=dev)
    adj_F = wp.zeros(N, dtype=wp.mat33, device=dev)
    adj_Fn = wp.zeros(N, dtype=wp.mat33, device=dev)
    wp.launch(ADJ.k_update_adj, dim=N, inputs=[wp.array(y, dtype=wp.vec3, device=dev), s, dt, nb0, rest0, frag0, 0, frac,
                                                ints, Kp, rr, wp.array(w, dtype=wp.vec3, device=dev),
                                                wp.zeros(N, dtype=wp.mat33, device=dev), adj_x, adj_v, adj_F, adj_Fn],
              device=dev)
    got = adj_x.numpy().astype(np.float64)
    X = torch.as_tensor(y, dtype=torch.float64).requires_grad_(True)
    V = torch.as_tensor(v, dtype=torch.float64)
    W = torch.as_tensor(w, dtype=torch.float64)
    out = _push_map64(X, V, S, r, dt, frac)
    ref = torch.autograd.grad((out * W).sum(), X)[0].numpy()
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-5 * np.abs(ref).max()), float(np.abs(got - ref).max())
    assert np.allclose(adj_v.numpy(), dt * w, rtol=1e-6)
    # central differences of the float64 map along a random direction
    e = torch.as_tensor(rng.normal(0, 1.0, (N, 3)))
    h = 1e-7
    with torch.no_grad():
        fp = float((_push_map64(X + h * e, V, S, r, dt, frac) * W).sum())
        fm = float((_push_map64(X - h * e, V, S, r, dt, frac) * W).sum())
    fd = (fp - fm) / (2 * h)
    an = float((torch.as_tensor(got) * e).sum())
    assert fd == pytest.approx(an, rel=1e-4), (fd, an)
    # the forward kernel is the same map (float32)
    xo, Fo = wp.zeros(N, dtype=wp.vec3, device=dev), wp.zeros(N, dtype=wp.mat33, device=dev)
    Fi = wp.zeros(N, dtype=wp.mat33, device=dev)
    wp.launch(K.k_update, dim=N, inputs=[wp.array(y, dtype=wp.vec3, device=dev), xo, wp.array(v, dtype=wp.vec3, device=dev),
                                         Fi, Fi, Fo, dt, s, nb0, rest0, frag0, 0, frac, ints, Kp, rr], device=dev)
    assert np.allclose(xo.numpy(), out.detach().numpy(), rtol=0, atol=1e-6)


def _window(dev, spacing_rows):
    """test_volume_exact's window case (a jittered 4^3 block with a start state, bonds, the layer with u, 4 driven + 4
    released steps, the polar adjoint), with the minimum spacing on the given rows of its 8 nearest at r = 0.24 (just
    under the lattice's 0.25: most face neighbours overlap)."""
    from scipy.spatial import cKDTree

    from physmorph.mpm.function import RolloutSpec
    from physmorph.pipeline.window.layer import layer_relax_data
    rng = np.random.default_rng(149)
    g = np.arange(4, dtype=np.float32) * 0.25
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    x = (X - X.mean(0) + rng.uniform(-0.03, 0.03, X.shape)).astype(np.float32)
    N, Tc = len(x), 4
    prm = MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.9, smoothing=0.955, grid_min=(-4.0,) * 3, nx=16, ny=16, nz=16)
    vol0 = compute_rest_volumes(x, 1.0, prm, dev)
    v0 = rng.normal(0, 0.1, (N, 3)).astype(np.float32)
    C0 = rng.normal(0, 0.2, (N, 3, 3)).astype(np.float32)
    mask, nrm, lnbr, lw = (t.cpu().numpy() for t in layer_relax_data(torch.as_tensor(x), 0.25, k=8, h_sp=2.0))
    layer = (mask, nrm, lnbr, lw, 1.0 / Tc, None, None)
    snbr = torch.as_tensor(cKDTree(x).query(x, k=9)[1][:, 1:])
    rows = None if spacing_rows is None else spacing_rows(snbr).numpy()        # numpy: copied to the device
    spec = RolloutSpec(x0=x, m=1.0, lam=800.0, mu=400.0, prm=prm, T=2 * Tc, v0=v0, C0=C0, device=dev, vol0=vol0,
                       spacing=None if rows is None else (rows, 0.24), layer=layer, control_steps=Tc,
                       polar_adjoint=True)
    dc = torch.tensor(rng.normal(0, 0.01, (Tc, N, 3, 3)).astype(np.float32), device=dev)
    u = torch.tensor((rng.normal(0, 0.002, N) * mask).astype(np.float32), device=dev)
    return spec, dc, u


def _expand(z):
    return torch.cat((z, torch.zeros_like(z)))


def _loss(out):
    xT, vT = out[0], out[2]
    w = torch.linspace(0.5, 1.5, 3, device=xT.device)
    return (xT * w).pow(2).sum() * 1e2 + (vT * w).sum()


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_a_window_with_the_closure_has_an_exact_adjoint(dev):
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from physmorph.mpm.function import PersistentAdjoint, warp_mpm_ext
    spec, dc, u = _window(dev, symmetric_rows)
    assert int(spec.spacing[0].shape[1]) > 8                       # the 8 nearest of the block are not symmetric
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
    # the push acts in this window: without the spacing the same controls end elsewhere
    free, *_ = _window(dev, None)
    off = warp_mpm_ext(_expand(dc.detach()), free, u_t=u.detach())
    assert float((off[0] - got[0].detach()).abs().max()) > 1e-4
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


# ---- (4) in a run ---------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the window runs on CUDA")
@pytest.mark.parametrize("flag", [False, True])
def test_in_a_run_the_lists_are_the_frozen_paths_or_their_closure(monkeypatch, flag):
    from physmorph import gpu
    from physmorph.pipeline import PipelineConfig, run_pipeline
    from physmorph.pipeline.window import solve
    seen = []
    init = solve.WindowOptimizer.__init__

    def recording(self, *a, **kw):
        init(self, *a, **kw)
        seen.append((self.win.x0.clone(), self.win.spec.spacing[0].clone(), int(self.win.tr.space_K)))

    monkeypatch.setattr(solve.WindowOptimizer, "__init__", recording)
    rng = np.random.default_rng(7)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=3, iters=2, animations=3, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=5, reject_stop=0, min_spacing=0.9, spacing_symmetric=flag)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert seen
    for x0, rows, k in seen:
        own = gpu.knn(x0, 17)[1][:, 1:]
        want = gpu.symmetric_rows(own) if flag else own
        assert torch.equal(rows.long(), want.long()) and k == want.shape[1]
        if not flag:
            assert k == 16                                          # the frozen path: each particle's 16 nearest
    if flag:
        recs = [h for h in res["history"] if h.get("spacing_vcom") is not None]
        assert recs
        for h in recs:                                              # the push moves the centre by rounding only
            assert h["spacing_com"] < 1e-8, h["spacing_com"]
