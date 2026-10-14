"""D147: two bodies on one grid with a field each (physmorph/mpm/bodies.py, contact.py) — the controls keep each body's
momentum, the contact keeps the pair's, the volume gate is the touching condition, the bodies bounce apart, the
two-body adjoint matches finite differences through a contact, and the scene runs end to end (CUDA)."""
import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm import contact as CT
from physmorph.mpm.bodies import Bodies, BodiesAdjoint, body_trajectory
from physmorph.mpm.function import RolloutSpec
from physmorph.mpm.constitutive import lame
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
E, NU = 1.0e4, 0.2                     # c = 37 wu/s at 8 particles a cell of 1 wu: CFL 0.15 at dt 1/240


def _ball(c, r=3.0, h=0.5, seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(-r, r + 1e-9, h)
    p = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    p = p[np.linalg.norm(p, axis=1) <= r]
    p = p + rng.uniform(-0.15 * h, 0.15 * h, p.shape)
    return (p + np.asarray(c)).astype(np.float32)


def _prm(**kw):
    base = dict(dx=1.0, dt=1.0 / 240.0, drag=0.0, grid_min=(-16.0, -10.0, -10.0), nx=32, ny=20, nz=20)
    base.update(kw)
    return MPMParams(**base)


def _specs(xs, vs, prm, T, vol=None):
    lam, mu = lame(E, NU)
    out = []
    for x, v in zip(xs, vs):
        vol0 = compute_rest_volumes(torch.tensor(x, device="cuda"), 1.0, prm, "cuda") if vol is None else vol
        out.append(RolloutSpec(x0=torch.tensor(x, device="cuda"), m=1.0, lam=lam, mu=mu, prm=prm, T=T,
                               v0=torch.tensor(v, device="cuda"), device="cuda", vol0=vol0, polar_adjoint=True,
                               track_geom=False, volume_exact="carried"))
    return out


def _pair(xs, vs, prm, T, dfcs=None, mu=0.0):
    specs = _specs(xs, vs, prm, T)
    trajs = []
    for b, spec in enumerate(specs):
        n = len(xs[b])
        d = torch.zeros(T, n, 3, 3, device="cuda") if dfcs is None else dfcs[b]
        seq = [wp.from_torch(d[t].contiguous(), dtype=wp.mat33) for t in range(T)]
        trajs.append(body_trajectory(spec, seq, requires_grad=False, persistent=False))
    return Bodies(trajs, [0.125, 0.125], mu), specs


def _momentum(tr, t, dx):
    x, v = wp.to_torch(tr.x[t]).double(), wp.to_torch(tr.v[t]).double()
    C = wp.to_torch(tr.C[t]).double().reshape(-1, 3, 3)
    skew = torch.stack((C[:, 2, 1] - C[:, 1, 2], C[:, 0, 2] - C[:, 2, 0], C[:, 1, 0] - C[:, 0, 1]), 1)
    return v.sum(0), (torch.linalg.cross(x, v, dim=1) + dx * dx / 3.0 * skew).sum(0)


def test_controls_keep_each_bodys_momentum():
    """dFc enters the stress alone: no net force (sum w (x_i - x_p) = 0) and a symmetric Kirchhoff stress (APIC keeps
    the angular momentum). Two bodies apart, random controls against none: each body's P and L unchanged."""
    T, prm = 12, _prm()
    xs = [_ball((-6.0, 0.0, 0.0)), _ball((6.0, 0.0, 0.0), seed=1)]
    rng = np.random.default_rng(3)
    vs = [np.tile([0.3, -0.2, 0.1], (len(x), 1)).astype(np.float32) for x in xs]
    dfcs = [torch.tensor(0.01 * rng.standard_normal((T, len(x), 3, 3)), dtype=torch.float32, device="cuda") for x in xs]
    a, _ = _pair(xs, vs, prm, T, dfcs)
    z, _ = _pair(xs, vs, prm, T)
    a.rollout()
    z.rollout()
    for ta, tz in zip(a.trajs, z.trajs):
        Pa, La = _momentum(ta, T, prm.dx)
        Pz, Lz = _momentum(tz, T, prm.dx)
        assert float((Pa - Pz).norm()) <= 1e-5 * float(Pz.norm()) + 1e-6
        assert float((La - Lz).norm()) <= 1e-4 * float(Lz.norm()) + 1e-4
    assert float(wp.to_torch(a.impulse).norm()) == 0.0               # no contact: the bodies are apart


def _collide(T, mu=0.0, offset=0.0, v=1.85):
    prm = _prm()
    xs = [_ball((-3.2, 0.5 * offset, 0.0)), _ball((3.2, -0.5 * offset, 0.0), seed=1)]
    vs = [np.tile([v, 0.0, 0.0], (len(xs[0]), 1)).astype(np.float32),
          np.tile([-v, 0.0, 0.0], (len(xs[1]), 1)).astype(np.float32)]
    pair, _ = _pair(xs, vs, prm, T, mu=mu)
    return pair, prm


def test_contact_keeps_the_pairs_momentum():
    T = 60
    pair, prm = _collide(T, mu=0.5, offset=2.0)
    P0 = sum(_momentum(tr, 0, prm.dx)[0] for tr in pair.trajs)
    L0 = sum(_momentum(tr, 0, prm.dx)[1] for tr in pair.trajs)
    pair.rollout()
    P = sum(_momentum(tr, T, prm.dx)[0] for tr in pair.trajs)
    L = sum(_momentum(tr, T, prm.dx)[1] for tr in pair.trajs)
    J = wp.to_torch(pair.impulse).double()[0]
    assert float(J.norm()) > 0.0                                    # they touched
    assert float((P - P0).norm()) <= 1e-4 * float(_momentum(pair.trajs[0], 0, prm.dx)[0].norm())
    assert float((L - L0).norm()) <= 1e-3 * max(float(L0.norm()), 1.0)


def test_bodies_bounce_apart():
    """Head-on at v_rel / c = 0.1: the centres' relative velocity reverses, and once apart no particle of one body is
    left within a sample spacing of the other."""
    T = 720                                                         # contact lasts about 11 wave crossings (M-b)
    pair, prm = _collide(T)
    pair.rollout()
    x = [wp.to_torch(tr.x[T]) for tr in pair.trajs]
    v = [wp.to_torch(tr.v[T]).mean(0) for tr in pair.trajs]
    assert float(v[1][0] - v[0][0]) > 0.0                           # separating (started at -3.7)
    d = torch.cdist(x[0], x[1]).min()
    assert float(d) > 0.5


def test_volume_gate_is_one_at_a_flat_interface():
    """Two half-spaces of the sample touching at x = 0: the two node volumes add up to the cell at the interface nodes
    (the partition of unity); with a one-cell gap they fall short."""
    prm = _prm()
    h = 0.5
    g = np.arange(-6.0 + 0.5 * h, 6.0, h)
    p = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    left, right = p[p[:, 0] < 0], p[p[:, 0] > 0]
    out = []
    for gap in (0.0, 1.0):
        vols = []
        for q in (left - np.array([gap / 2, 0, 0], np.float32), right + np.array([gap / 2, 0, 0], np.float32)):
            gv = wp.zeros(prm.ngrid, dtype=float, device="cuda")
            xq = wp.array(q, dtype=wp.vec3, device="cuda")
            wp.launch(CT.k_node_volume, dim=len(q), inputs=[xq, wp.ones(len(q), dtype=float, device="cuda"), h ** 3,
                                                           gv, wp.vec3(*prm.grid_min), prm.dx, 1.0 / prm.dx,
                                                           prm.nx, prm.ny, prm.nz], device="cuda")
            vols.append(wp.to_torch(gv).reshape(prm.nx, prm.ny, prm.nz))
        both = (vols[0] > 0) & (vols[1] > 0)
        i = torch.arange(prm.nx, device="cuda")[:, None, None].expand_as(both)
        j = torch.arange(prm.ny, device="cuda")[None, :, None].expand_as(both)
        k = torch.arange(prm.nz, device="cuda")[None, None, :].expand_as(both)
        inner = both & ((j - 10).abs() < 3) & ((k - 10).abs() < 3) & ((i - 16).abs() <= 1)
        out.append((vols[0] + vols[1])[inner])
    assert float((out[0] - 1.0).abs().max()) < 0.02
    assert float(out[1].max()) < 0.95


def test_two_body_adjoint_matches_finite_differences():
    """The pair's tape (BodiesAdjoint) through a contact: dL/d eps along a random control direction against central
    differences of the no-grad rollout."""
    T, prm = 12, _prm()                                              # in contact from the first steps: the window
    xs = [_ball((-2.0, 0.0, 0.0), r=2.0), _ball((2.0, 0.0, 0.0), r=2.0, seed=1)]   # must see the pair touch (M-b P7)
    vs = [np.tile([1.5, 0.0, 0.0], (len(xs[0]), 1)).astype(np.float32),
          np.tile([-1.5, 0.0, 0.0], (len(xs[1]), 1)).astype(np.float32)]
    specs = _specs(xs, vs, prm, T)
    adj = BodiesAdjoint(specs, [0.125, 0.125], 0.5)
    N = sum(len(x) for x in xs)
    D = torch.randn(T, N, 3, 3, generator=torch.Generator().manual_seed(5)).cuda() * 0.01
    W = torch.randn(N, 3, generator=torch.Generator().manual_seed(6)).cuda()

    def loss(eps, grad=False):
        dfc = (eps * D).requires_grad_(grad)
        xT, FT, vT, V, X = adj.apply(dfc, None)
        L = (xT * W).sum() + 0.1 * (V[-3] * W).sum() + 0.05 * (X[T // 2] * W).sum()
        if grad:
            (g,) = torch.autograd.grad(L, dfc)
            return float(L), float((g * D).sum())
        return float(L)

    _, an = loss(0.0, grad=True)
    h = 0.1
    fd = (loss(h) - loss(-h)) / (2 * h)
    assert abs(an - fd) <= 0.05 * abs(fd) + 1e-6, (an, fd)


def test_scene_runs_end_to_end():
    """run_scene on two small clouds thrown toward each other (no gravity, no floor): every window commits, each body is
    recorded in its frame, the pair's momentum is kept, and each body's objective is the frozen one in its frame."""
    from physmorph import gpu
    from physmorph.pipeline import PipelineConfig
    from physmorph.pipeline.scene.runner import run_scene
    from physmorph.pipeline.scene.window import Body
    from physmorph.pipeline.target import build_target, calibrate_units

    rng = np.random.default_rng(7)
    n = 300
    canon = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=3, iters=2, animations=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32)
    bodies, xs, vs = [], [], []
    for b in range(2):
        src = rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32)
        src -= src.mean(0)
        tgt = (rng.uniform(-1.5, 1.5, (n, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
        tgt -= tgt.mean(0)
        tp = build_target(tgt, canon, cfg)
        calibrate_units(tp, gpu.tensor(src), cfg)
        bodies.append(Body(start=b * n, stop=(b + 1) * n, Xq=gpu.tensor(src), tgt=tp, vp=27.0 / n,
                           mass=cfg.mass_ref_n / n))
        xs.append(src + np.array([(-1) ** (b + 1) * 2.0, 0.0, 0.0], np.float32))
        vs.append(np.tile([(-1) ** b * 0.5, 0.0, 0.0], (n, 1)).astype(np.float32))
    world = MPMParams(dx=1.0, drag=0.0, grid_min=(-16.0, -10.0, -10.0), nx=32, ny=20, nz=20)
    x0, v0 = gpu.tensor(np.concatenate(xs)), gpu.tensor(np.concatenate(vs))
    res = run_scene(bodies, x0, v0, world, cfg, windows=2, mu=0.5, log=lambda *_: None)
    hist = res["history"]
    assert len(hist) == 2 and all("bodies" in h and len(h["bodies"]) == 2 for h in hist)
    assert len(res["frames"]) == 1 + 2 * 2 * cfg.T
    P0 = (cfg.mass_ref_n / n) * v0.double().sum(0)
    P = torch.tensor([sum(bp["P"][k] for bp in hist[-1]["bodies"]) for k in range(3)], dtype=torch.float64,
                     device=P0.device)
    assert float((P - P0).norm()) <= 1e-4 * (cfg.mass_ref_n / n) * n * 0.5      # rounding of a body's momentum
    assert all(v == 0 for v in res["guards"].values())
    for h in hist:                                      # the angular momentum's split is recorded for each body
        assert h["L_split"] is not None and len(h["L_split"]) == 2
        for b in h["L_split"]:                          # the updates below the grid do not depend on the point
            assert all(b["own"][k] == b["pair"][k] for k in ("L_u", "L_space", "L_relax"))


def test_one_body_in_a_fixed_frame_is_the_frozen_objective():
    """One body whose frame is fixed at the world's (V0's world arm): the scene's rollout and objective give the frozen
    window's state and terms (the same trajectory and kernels; the grid operation k_grid_op_one is k_grid_op's arithmetic
    without a floor)."""
    from physmorph import gpu
    from physmorph.pipeline import PipelineConfig
    from physmorph.pipeline.scene.objective import SceneObjective
    from physmorph.pipeline.scene.window import Body, SceneWindow
    from physmorph.pipeline.target import build_target, calibrate_units
    from physmorph.pipeline.window.objective import Objective
    from physmorph.pipeline.window.rollout import eval_terms
    from physmorph.pipeline.window.setup import StartState, Window

    rng = np.random.default_rng(11)
    n = 300
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=3, iters=2, animations=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, render_exterior=True, min_spacing=0.9, volume_exact="carried")
    src = rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32)
    src -= src.mean(0)
    tgt = (rng.uniform(-1.5, 1.5, (n, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    tgt -= tgt.mean(0)
    tp = build_target(tgt, prm, cfg)
    x = gpu.tensor(src)
    calibrate_units(tp, x, cfg)
    vol0 = compute_rest_volumes(x, 1.0, prm, "cuda")
    coh = gpu.knn(x, cfg.coh_k + 1)[1][:, 1:]
    bonds = (coh, (x[coh] - x[:, None]).norm(dim=2), torch.zeros(n, device="cuda"))
    start = StartState(x=x, Fp=torch.eye(3, device="cuda").repeat(n, 1, 1))
    leaf = 1e-3 * torch.randn(cfg.T, n, 3, 3, generator=torch.Generator().manual_seed(2)).cuda()
    u = torch.zeros(n, device="cuda")
    win = Window(start, prm, cfg, tp, vol0, bonds)
    e1 = eval_terms(win, Objective(win), leaf, u)
    body = Body(start=0, stop=n, Xq=x, tgt=tp, vp=27.0 / n, mass=cfg.mass_ref_n / n,
                fixed=(torch.zeros(3, device="cuda"), torch.eye(3, device="cuda")))
    swin = SceneWindow(start, prm, cfg, [body], vol0, [bonds], mu=0.5)
    e2 = eval_terms(swin, SceneObjective(swin), leaf, u)
    assert float((e1.xT - e2.xT).abs().max()) < 1e-5
    for k in ("lv", "lr", "lstab"):
        a, b = float(getattr(e1, k)), float(getattr(e2, k))
        assert abs(a - b) <= 1e-4 * max(abs(a), 1e-12), (k, a, b)


def _floor_body(T, u_val=0.0, floor_gap=0.05):
    """One ball whose lowest particle stands floor_gap above the centres' floor (the floor's plane half a pitch lower),
    with its outer layer and a uniform u (outward along the layer's normals: down on the bottom)."""
    from physmorph.pipeline.window.layer import layer_relax_data, layer_spacing
    x = _ball((0.0, 0.0, 0.0))
    vp = 0.125
    floor_c = float(x[:, 1].min()) - floor_gap
    prm = _prm(floor_y=floor_c - 0.5 * vp ** (1.0 / 3.0), f_ext=(0.0, -1.0, 0.0))
    xt = torch.tensor(x, device="cuda")
    sp = layer_spacing(xt)
    mask, nrm, nbr, w = layer_relax_data(xt, sp)
    spec = _specs([x], [np.zeros_like(x)], prm, T)[0]
    spec.layer = (mask, nrm, nbr, w, 1.0 / T, None, None)
    spec.control_steps = T
    u = torch.full((len(x),), u_val, device="cuda")
    return spec, prm, floor_c, u, vp


def test_floor_keeps_positions_above_its_plane():
    """u pushes the bottom layer down (outward); the grid's floor alone stops velocities, not positions: with the
    floor's position constraint every centre ends at or above floor_c, some exactly on it, and the velocities are the
    ones the grid gave (the projection moves positions only)."""
    T = 6
    spec, prm, floor_c, u, vp = _floor_body(T, u_val=0.5)
    seq = [wp.zeros(len(u), dtype=wp.mat33, device="cuda") for _ in range(T)]
    uw = wp.from_torch(u.contiguous(), dtype=wp.float32)
    tr = body_trajectory(spec, seq, uw, requires_grad=False, persistent=False, vp=vp)
    free = body_trajectory(spec, seq, uw, requires_grad=False, persistent=False)        # no position constraint
    for traj in (tr, free):
        Bodies([traj], [vp], 0.5).rollout()
    x, xf = wp.to_torch(tr.x[T]), wp.to_torch(free.x[T])
    assert float(xf[:, 1].min()) < floor_c - 0.05                    # without it u pushes the bottom through
    assert float(x[:, 1].min()) >= floor_c - 1e-6
    assert int((x[:, 1] - floor_c).abs().lt(1e-6).sum()) > 0
    v1, vf1 = wp.to_torch(tr.v[1]), wp.to_torch(free.v[1])            # the first step: the same grid, same velocities
    assert torch.allclose(v1, vf1, atol=1e-5)                      # (the atomics' order aside)


def test_floor_projection_adjoint_is_the_branch_taken():
    """contact.k_floor_position on a tape: the gradient passes unchanged except the vertical one of each projected
    centre, which is zero; against central differences with every centre at least 0.05 off the plane (no branch
    switches under the step)."""
    from physmorph.mpm import contact as C
    rng = np.random.default_rng(4)
    n = 400
    x = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    x[:, 1] = np.where(np.abs(x[:, 1]) < 0.05, np.where(x[:, 1] < 0, -0.05, 0.05), x[:, 1])
    W = rng.standard_normal((n, 3)).astype(np.float32)
    floor_c = 0.0

    def run(xx, grad=False):
        xi = wp.array(xx, dtype=wp.vec3, device="cuda", requires_grad=grad)
        xo = wp.zeros(n, dtype=wp.vec3, device="cuda", requires_grad=grad)
        if not grad:
            wp.launch(C.k_floor_position, dim=n, inputs=[xi, floor_c, xo], device="cuda")
            return float((wp.to_torch(xo).double() * torch.tensor(W, device="cuda").double()).sum()), None
        tape = wp.Tape()
        with tape:
            wp.launch(C.k_floor_position, dim=n, inputs=[xi, floor_c, xo], device="cuda")
        tape.backward(grads={xo: wp.array(W, dtype=wp.vec3, device="cuda")})
        return None, wp.to_torch(xi.grad).cpu().numpy()

    _, g = run(x, grad=True)
    expect = W.copy()
    expect[x[:, 1] < floor_c, 1] = 0.0
    assert np.allclose(g, expect, atol=1e-6)
    d = rng.standard_normal((n, 3)).astype(np.float32)
    h = 1e-3
    fd = (run(x + h * d)[0] - run(x - h * d)[0]) / (2 * h)
    assert abs(fd - float((g * d).sum())) <= 1e-3 * max(1.0, abs(fd))


def test_scene_windows_are_freed_without_the_garbage_collector(monkeypatch):
    """A window's scene window, its pair's trajectories and its tape (BodiesAdjoint) are freed by reference counting:
    with the garbage collector off they are all gone once run_scene returns (D148: a reference cycle through the tape
    kept a 300k window's ~7 GB alive until gc.collect, out of memory after a few windows)."""
    import gc
    import weakref
    from physmorph import gpu
    from physmorph.pipeline import PipelineConfig
    from physmorph.pipeline.scene import runner as RN
    from physmorph.pipeline.scene import window as WN
    from physmorph.pipeline.scene.window import Body
    from physmorph.pipeline.target import build_target, calibrate_units

    refs = []
    adjoint = WN.SceneWindow.adjoint

    def tracked_adjoint(self):
        a = adjoint(self)
        if not any(r() is a for r in refs):
            refs.append(weakref.ref(a))
        return a

    class Tracked(RN.SceneOptimizer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            refs.extend([weakref.ref(self.win), weakref.ref(self.win.tr)] + [weakref.ref(t) for t in self.win.tr.trajs])

    monkeypatch.setattr(WN.SceneWindow, "adjoint", tracked_adjoint)
    monkeypatch.setattr(RN, "SceneOptimizer", Tracked)
    rng = np.random.default_rng(9)
    n = 300
    canon = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=3, iters=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5), render_res=24, dt_res=32)
    bodies, xs, vs = [], [], []
    for b in range(2):
        src = rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32)
        src -= src.mean(0)
        tgt = (rng.uniform(-1.5, 1.5, (n, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
        tgt -= tgt.mean(0)
        tp = build_target(tgt, canon, cfg)
        calibrate_units(tp, gpu.tensor(src), cfg)
        bodies.append(Body(start=b * n, stop=(b + 1) * n, Xq=gpu.tensor(src), tgt=tp, vp=27.0 / n, mass=cfg.mass_ref_n / n))
        xs.append(src + np.array([(-1) ** (b + 1) * 2.0, 0.0, 0.0], np.float32))
        vs.append(np.tile([(-1) ** b * 0.5, 0.0, 0.0], (n, 1)).astype(np.float32))
    world = MPMParams(dx=1.0, drag=0.0, grid_min=(-16.0, -10.0, -10.0), nx=32, ny=20, nz=20)
    gc.collect()
    gc.disable()
    try:
        RN.run_scene(bodies, gpu.tensor(np.concatenate(xs)), gpu.tensor(np.concatenate(vs)), world, cfg, windows=3,
                     mu=0.5, log=lambda *_: None)
        alive = [r() for r in refs if r() is not None]
    finally:
        gc.enable()
    assert refs and not alive, [type(o).__name__ for o in alive]
