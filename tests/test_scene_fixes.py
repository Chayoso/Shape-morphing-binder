"""D147, the fixes of 2026-10-16 (the user's approval after the forensic of the final 300k runs): (1) each body's loss
raster covers its uncontrolled continuation (scene/raster.py), (2) a body whose terms are not finite at the window's
start is left out of the window's objective with zero controls while the other is optimised, (3) a commit whose replay
disagrees is solved again (no simulated time) instead of committing the zero control, up to the single-body patience,
(4) each body's deformation velocity is damped at the single-body drag rate in P2G, keeping each body's linear and
angular momentum (CUDA)."""
import dataclasses

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm import bodies as BD
from physmorph.mpm.bodies import Bodies, BodiesAdjoint, body_trajectory
from physmorph.mpm.constitutive import lame
from physmorph.mpm.function import RolloutSpec
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
E, NU = 1.0e4, 0.2


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


def _spec(x, v, prm, T):
    lam, mu = lame(E, NU)
    xt = torch.tensor(x, device="cuda")
    return RolloutSpec(x0=xt, m=1.0, lam=lam, mu=mu, prm=prm, T=T, v0=torch.tensor(v, device="cuda"), device="cuda",
                       vol0=compute_rest_volumes(xt, 1.0, prm, "cuda"), polar_adjoint=True, track_geom=False,
                       volume_exact="carried")


def _spinning(x, vc, w, s, seed=0):
    """A rigid velocity v_c + w x r plus a deformation part: an expansion s r and a random field."""
    rng = np.random.default_rng(seed)
    r = x - x.mean(0)
    return (np.asarray(vc) + np.cross(np.asarray(w), r) + s * r
            + 0.2 * rng.standard_normal(x.shape)).astype(np.float32)


def _momentum(tr, t, dx):
    x, v = wp.to_torch(tr.x[t]).double(), wp.to_torch(tr.v[t]).double()
    C = wp.to_torch(tr.C[t]).double().reshape(-1, 3, 3)
    skew = torch.stack((C[:, 2, 1] - C[:, 1, 2], C[:, 0, 2] - C[:, 2, 0], C[:, 1, 0] - C[:, 0, 1]), 1)
    return v.sum(0), (torch.linalg.cross(x, v, dim=1) + dx * dx / 3.0 * skew).sum(0)


def _deformation_ke(tr, t):
    from physmorph.pipeline.scene.frame import deformation_velocity
    x, v = wp.to_torch(tr.x[t]), wp.to_torch(tr.v[t])
    return float(deformation_velocity(x, v).double().square().sum())


# ---- (4) the deformation's drag ----------------------------------------------------------------------------------
# The reference: fix 4 as first written (fa0d091), atomic sums to one address with Warp's generated adjoints; the block
# sums with the hand adjoint (mpm/bodies.py) must give the same within float summation order.

@wp.kernel
def _ref_sums(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), m: wp.array(dtype=float),
              sx: wp.array(dtype=wp.vec3), sv: wp.array(dtype=wp.vec3), sm: wp.array(dtype=float)):
    p = wp.tid()
    wp.atomic_add(sx, 0, m[p] * x[p])
    wp.atomic_add(sv, 0, m[p] * v[p])
    wp.atomic_add(sm, 0, m[p])


@wp.kernel
def _ref_moments(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), m: wp.array(dtype=float),
                 sx: wp.array(dtype=wp.vec3), sm: wp.array(dtype=float),
                 sL: wp.array(dtype=wp.vec3), sI: wp.array(dtype=wp.mat33)):
    p = wp.tid()
    r = x[p] - sx[0] / sm[0]
    wp.atomic_add(sL, 0, m[p] * wp.cross(r, v[p]))
    wp.atomic_add(sI, 0, m[p] * (wp.dot(r, r) * wp.identity(n=3, dtype=float) - wp.outer(r, r)))


@wp.kernel
def _ref_apply(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
               sx: wp.array(dtype=wp.vec3), sv: wp.array(dtype=wp.vec3), sm: wp.array(dtype=float),
               sL: wp.array(dtype=wp.vec3), sI: wp.array(dtype=wp.mat33), kappa: float, vd: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    c = sx[0] / sm[0]
    w = wp.inverse(sI[0]) @ sL[0]
    vd[p] = v[p] - kappa * (v[p] - sv[0] / sm[0] - wp.cross(w, x[p] - c))


def _drag_ref(x, v, m, kappa):
    dev = "cuda"
    xa, va = wp.array(x, dtype=wp.vec3, device=dev), wp.array(v, dtype=wp.vec3, device=dev)
    ma = wp.array(m, dtype=float, device=dev)
    sx, sv, sL = (wp.zeros(1, dtype=wp.vec3, device=dev) for _ in range(3))
    sm, sI = wp.zeros(1, dtype=float, device=dev), wp.zeros(1, dtype=wp.mat33, device=dev)
    vd = wp.zeros(len(x), dtype=wp.vec3, device=dev)
    wp.launch(_ref_sums, dim=len(x), inputs=[xa, va, ma, sx, sv, sm], device=dev)
    wp.launch(_ref_moments, dim=len(x), inputs=[xa, va, ma, sx, sm, sL, sI], device=dev)
    wp.launch(_ref_apply, dim=len(x), inputs=[xa, va, sx, sv, sm, sL, sI, kappa, vd], device=dev)
    return wp.to_torch(vd).double().cpu().numpy()


def _drag_once(x, v, m, kappa, lam=None):
    """mpm/bodies.py's forward (block sums), and with lam its hand adjoint: (vd, adj_x, adj_v) as numpy float64."""
    dev, NT = "cuda", BD.DRAG_THREADS
    n = len(x)
    xa, va = wp.array(x, dtype=wp.vec3, device=dev), wp.array(v, dtype=wp.vec3, device=dev)
    ma = wp.array(m, dtype=float, device=dev)
    Z = lambda k, dt: wp.zeros(k, dtype=dt, device=dev)            # noqa: E731
    c, vc, w, M, Ii = Z(1, wp.vec3), Z(1, wp.vec3), Z(1, wp.vec3), Z(1, float), Z(1, wp.mat33)
    pa, pb, pm, pI = Z(NT, wp.vec3), Z(NT, wp.vec3), Z(NT, float), Z(NT, wp.mat33)
    vd = Z(n, wp.vec3)
    wp.launch(BD.k_drag_sums1, dim=NT, inputs=[xa, va, ma, n, pa, pb, pm], device=dev)
    wp.launch(BD.k_drag_centre, dim=1, inputs=[pa, pb, pm, c, vc, M], device=dev)
    wp.launch(BD.k_drag_sums2, dim=NT, inputs=[xa, va, ma, n, c, pa, pI], device=dev)
    wp.launch(BD.k_drag_spin, dim=1, inputs=[pa, pI, w, Ii], device=dev)
    wp.launch(BD.k_drag_apply, dim=n, inputs=[xa, va, c, vc, w, kappa, vd], device=dev)
    out = wp.to_torch(vd).double().cpu().numpy()
    if lam is None:
        return out
    la = wp.array(lam, dtype=wp.vec3, device=dev)
    lam0, g, sA, A = Z(1, wp.vec3), Z(1, wp.vec3), Z(1, wp.vec3), Z(n, wp.vec3)
    ax, av = Z(n, wp.vec3), Z(n, wp.vec3)
    wp.launch(BD.k_drag_adj_sums, dim=NT, inputs=[la, xa, c, n, pa, pb], device=dev)
    wp.launch(BD.k_drag_adj_spin, dim=1, inputs=[pa, pb, Ii, kappa, lam0, g], device=dev)
    wp.launch(BD.k_drag_adj_v, dim=n, inputs=[la, xa, va, ma, c, w, M, lam0, g, kappa, av, A], device=dev)
    wp.launch(BD.k_drag_adj_sumA, dim=NT, inputs=[A, n, pa], device=dev)
    wp.launch(BD.k_drag_adj_total, dim=1, inputs=[pa, sA], device=dev)
    wp.launch(BD.k_drag_adj_x, dim=n, inputs=[A, sA, ma, M, ax], device=dev)
    return out, wp.to_torch(ax).double().cpu().numpy(), wp.to_torch(av).double().cpu().numpy()


def test_deformation_drag_keeps_momentum_and_damps_only_the_deformation():
    x = _ball((3.0, -1.0, 2.0), r=2.0)
    m = np.full(len(x), 0.7, np.float32)
    kappa = 0.9 / 240.0
    v = _spinning(x, (1.0, -2.0, 0.5), (0.3, -0.7, 1.1), 0.4)
    vd = _drag_once(x, v, m, kappa)
    xd, vv, md = x.astype(np.float64), v.astype(np.float64), m.astype(np.float64)[:, None]
    r = xd - (md * xd).sum(0) / md.sum()
    P0, P1 = (md * vv).sum(0), (md * vd).sum(0)
    L0, L1 = (md * np.cross(r, vv)).sum(0), (md * np.cross(r, vd)).sum(0)
    assert np.linalg.norm(P1 - P0) <= 1e-5 * np.linalg.norm(P0)
    assert np.linalg.norm(L1 - L0) <= 1e-5 * np.linalg.norm(L0)
    # the rigid part is untouched, the rest scaled by 1 - kappa
    vc = (md * vv).sum(0) / md.sum()
    Iner = (md[:, 0] * (r * r).sum(1)).sum() * np.eye(3) - (md * r).T @ r
    w = np.linalg.solve(Iner, (md * np.cross(r, vv)).sum(0))
    rig = vc + np.cross(w, r)
    np.testing.assert_allclose(vd - rig, (1 - kappa) * (vv - rig), atol=2e-5)
    # a rigid motion alone is not damped
    vr = (np.asarray([0.5, 0.1, -0.3]) + np.cross(np.asarray([0.2, 0.9, -0.4]), r)).astype(np.float32)
    np.testing.assert_allclose(_drag_once(x, vr, m, kappa), vr, atol=1e-5)


def test_block_sums_give_the_atomic_damping():
    """The block sums (a fixed order) against the first version's atomic sums: the same damped velocities within float
    summation order, on a 40k-sized body far from the origin (the scene's coordinates)."""
    rng = np.random.default_rng(3)
    x = (rng.uniform(-2.5, 2.5, (40000, 3)) + np.array([4.0, -9.0, 2.0])).astype(np.float32)
    m = np.full(len(x), 1.0, np.float32)
    v = _spinning(x, (0.6, 3.0, -0.2), (0.4, -0.3, 0.9), 0.5, seed=4)
    kappa = 0.9 / 240.0
    a, b = _drag_once(x, v, m, kappa), _drag_ref(x, v, m, kappa)
    scale = np.abs(v).max()
    assert np.abs(a - b).max() <= 2e-6 * scale, np.abs(a - b).max()


def test_the_damping_hand_adjoint_is_its_gradient():
    """The hand adjoint against torch's float64 autograd of the same map, vd = v - k (v - v_c - w x r), with respect to
    x and v, for a random cotangent."""
    rng = np.random.default_rng(5)
    x = (rng.uniform(-1.5, 1.5, (3000, 3)) + np.array([2.0, 1.0, -1.0])).astype(np.float32)
    m = rng.uniform(0.5, 1.5, len(x)).astype(np.float32)
    v = _spinning(x, (0.6, 3.0, -0.2), (0.4, -0.3, 0.9), 0.5, seed=6)
    lam = rng.standard_normal(x.shape).astype(np.float32)
    kappa = 0.05                                                    # large, so that every path counts
    _, ax, av = _drag_once(x, v, m, kappa, lam)
    X = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    V = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    mm = torch.tensor(m, dtype=torch.float64)[:, None]
    c = (mm * X).sum(0) / mm.sum()
    r = X - c
    vc = (mm * V).sum(0) / mm.sum()
    L = (mm * torch.linalg.cross(r, V, dim=1)).sum(0)
    Iner = (mm[:, 0] * (r * r).sum(1)).sum() * torch.eye(3, dtype=torch.float64) - (mm * r).T @ r
    w = torch.linalg.solve(Iner, L)
    vd = V - kappa * (V - vc - torch.linalg.cross(w.expand_as(r), r, dim=1))
    gx, gv = torch.autograd.grad((vd * torch.tensor(lam, dtype=torch.float64)).sum(), (X, V))
    np.testing.assert_allclose(av, gv.numpy(), rtol=1e-4, atol=1e-4 * float(gv.abs().max()))
    np.testing.assert_allclose(ax, gx.numpy(), rtol=1e-3, atol=1e-3 * float(gx.abs().max()))


def test_deformation_drag_keeps_each_bodys_momentum_over_a_rollout():
    """Two bodies apart (no contact, gravity or floor), spinning, translating and deforming: with the deformation's drag
    each body's P and L (APIC's affine part included) stay at their start, as without it, and the deformation's
    kinetic energy falls below the undamped run's."""
    T = 48
    prm = _prm()
    xs = [_ball((-7.0, 0.0, 0.0), r=2.5), _ball((7.0, 0.0, 0.0), r=2.5, seed=1)]
    vs = [_spinning(xs[0], (0.4, 0.1, 0.0), (0.0, 0.5, 0.8), 0.6), _spinning(xs[1], (-0.3, 0.0, 0.2), (0.7, 0.0, -0.2),
                                                                        -0.4, seed=1)]
    runs = {}
    for name, rate in (("damped", None), ("free", 0.0)):
        trajs = []
        for x, v in zip(xs, vs):
            spec = _spec(x, v, prm, T)
            seq = [wp.from_torch(torch.zeros(len(x), 3, 3, device="cuda").contiguous(), dtype=wp.mat33)
                   for _ in range(T)]
            trajs.append(body_trajectory(spec, seq, requires_grad=False, persistent=False, drag_rate=rate))
        pair = Bodies(trajs, [0.125, 0.125], 0.5)
        pair.rollout()
        runs[name] = pair
    assert isinstance(runs["damped"].trajs[0], BD.DragTrajectory) and BD.deformation_drag_rate(prm) == 0.9
    for b in range(2):
        tr = runs["damped"].trajs[b]
        P0, L0 = _momentum(tr, 0, prm.dx)
        PT, LT = _momentum(tr, T, prm.dx)
        assert float((PT - P0).norm()) <= 1e-5 * float(P0.norm()) + 1e-6
        assert float((LT - L0).norm()) <= 1e-4 * float(L0.norm()) + 1e-4
        assert _deformation_ke(tr, T) < _deformation_ke(runs["free"].trajs[b], T)


def test_world_drag_leaves_no_deformation_drag():
    """V0 keeps the frozen drag (0.9) on every velocity: no further damping of the deformation."""
    assert BD.deformation_drag_rate(_prm(drag=0.9)) == 0.0
    x = _ball((0.0, 0.0, 0.0), r=2.0)
    spec = _spec(x, np.zeros_like(x), _prm(drag=0.9), 4)
    seq = [wp.from_torch(torch.zeros(len(x), 3, 3, device="cuda").contiguous(), dtype=wp.mat33) for _ in range(4)]
    tr = body_trajectory(spec, seq, requires_grad=False, persistent=False)
    assert not isinstance(tr, BD._DeformationDrag)


def test_deformation_drag_adjoint_matches_finite_differences():
    """The pair's tape with the deformation's drag on spinning, deforming bodies in contact: dL/d eps along a random
    control direction against central differences."""
    T, prm = 12, _prm()
    xs = [_ball((-2.0, 0.0, 0.0), r=2.0), _ball((2.0, 0.0, 0.0), r=2.0, seed=1)]
    vs = [_spinning(xs[0], (1.5, 0.0, 0.0), (0.0, 0.0, 2.0), 0.5), _spinning(xs[1], (-1.5, 0.0, 0.0), (1.5, 0.0, 0.0),
                                                                      -0.5, seed=1)]
    specs = [_spec(x, v, prm, T) for x, v in zip(xs, vs)]
    adj = BodiesAdjoint(specs, [0.125, 0.125], 0.5)
    assert all(isinstance(tr, BD._DeformationDrag) for tr in adj.bodies.trajs)
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


# ---- the small scene of the fixes (1)-(3) ------------------------------------------------------------------------

def _cfg():
    from physmorph.pipeline import PipelineConfig
    return PipelineConfig(T=3, iters=2, animations=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                          render_res=24, dt_res=32)


def _scene(scales=(1.0, 1.0), canon=None, seed=7, n=300):
    """Two clouds, each with its target built on canon (its 'preparation' domain) and a raster over it."""
    from physmorph import gpu
    from physmorph.pipeline.scene.raster import BodyRaster
    from physmorph.pipeline.scene.window import Body
    from physmorph.pipeline.target import build_target, calibrate_units

    rng = np.random.default_rng(seed)
    canon = canon or MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = _cfg()
    bodies, xs, vs = [], [], []
    for b in range(2):
        src = rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32)
        src = (src - src.mean(0)) * scales[b]
        tgt = (rng.uniform(-1.5, 1.5, (n, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
        tgt -= tgt.mean(0)
        tp = build_target(tgt, canon, cfg)
        calibrate_units(tp, gpu.tensor(src), cfg)
        bodies.append(Body(start=b * n, stop=(b + 1) * n, Xq=gpu.tensor(src), tgt=tp, vp=27.0 / n,
                           mass=cfg.mass_ref_n / n, raster=BodyRaster(canon, cfg)))
        xs.append(src + np.array([(-1) ** (b + 1) * 4.0, 0.0, 0.0], np.float32))
        vs.append(np.tile([(-1) ** b * 0.5, 0.0, 0.0], (n, 1)).astype(np.float32))
    world = MPMParams(dx=1.0, drag=0.0, grid_min=(-16.0, -10.0, -10.0), nx=32, ny=20, nz=20)
    return bodies, gpu.tensor(np.concatenate(xs)), gpu.tensor(np.concatenate(vs)), world, cfg


def _window_args(bodies, x0, v0, world, cfg):
    from physmorph import gpu
    from physmorph.pipeline.render_loss import LambdaBalancer
    from physmorph.pipeline.run.state import fragment_mask
    from physmorph.pipeline.scene.solve import SceneTarget
    from physmorph.pipeline.window import StartState

    N = x0.shape[0]
    vol0 = torch.cat([compute_rest_volumes(x0[bd.rows].contiguous(), 1.0, world, "cuda") for bd in bodies])
    bonds = []
    for bd in bodies:
        xb = x0[bd.rows].contiguous()
        coh = gpu.knn(xb, cfg.coh_k + 1)[1][:, 1:]
        bonds.append((coh, (xb[coh] - xb[:, None]).norm(dim=2), fragment_mask(xb, world).float()))
    start = StartState(x=x0.clone(), Fp=torch.eye(3, device="cuda").repeat(N, 1, 1), v=v0.clone())
    tgt = SceneTarget(bodies, 0.5)
    bal = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, None, cap_rel=20.0)
    return start, world, cfg, tgt, bal, vol0, bonds


# ---- (1) the loss raster -----------------------------------------------------------------------------------------

def test_raster_steps_keep_the_cell_and_the_grid_lines():
    from physmorph.pipeline.scene.raster import BodyRaster
    prm = MPMParams(dx=10.96 / 36, nx=36, ny=36, nz=36, grid_min=(-5.48,) * 3)        # the 300k preparation
    cfg = dataclasses.replace(_cfg(), loss_res=56)
    r = BodyRaster(prm, cfg)
    assert (r.um, r.ul) == (9, 14)
    p1, c1 = r.grown(1)
    assert (p1.nx, c1.loss_res) == (54, 84)
    assert abs(p1.dx * p1.nx / c1.loss_res - prm.dx * prm.nx / cfg.loss_res) < 1e-9
    shift = (prm.grid_min[0] - p1.grid_min[0]) / (prm.dx * prm.nx / cfg.loss_res)
    assert abs(shift - round(shift)) < 1e-6                       # whole loss cells
    lo, hi = r.bounds(0)
    assert abs(hi - (5.48 - 10.96 / 56)) < 1e-4 and abs(lo + 5.48) < 1e-6        # the 5.28 wu of the forensic
    assert r.need(-3.2, 3.9) == 0 and r.need(-3.2, 5.0) == 1 and r.need(-3.2, 7.06) == 1 and r.need(-3.2, 8.0) == 2


def test_the_grown_target_is_the_base_target_on_a_larger_grid():
    from physmorph.losses.grid_ot import GridSinkhornLoss
    from physmorph.pipeline.scene.raster import BodyRaster, target_for

    # a preparation cube that holds the target with its margin, as the real one does (the leash): here the targets
    # reach 1.95 wu and the kept cube is [-3, 2.5]
    canon = MPMParams(dx=0.25, nx=24, ny=24, nz=24, grid_min=(-3.0,) * 3)
    bodies, *_ = _scene(scales=(0.6, 0.6), canon=canon)
    body = bodies[0]
    body.base = body.tgt
    g = target_for(body, 1)
    b0 = body.base
    assert abs(g.ldx - b0.ldx) < 1e-6 and g.ldims[0] == b0.ldims[0] + 2 * BodyRaster(canon, _cfg()).ul
    off = int(round(float((b0.lgmin - g.lgmin)[0]) / b0.ldx))
    n0 = b0.ldims[0]
    G = g.grid.reshape(g.ldims)[off:off + n0, off:off + n0, off:off + n0]
    torch.testing.assert_close(G, b0.grid.reshape(b0.ldims), atol=1e-5, rtol=1e-5)
    assert (g.unit_ratio, g.ot_scale) == (b0.unit_ratio, b0.ot_scale) and target_for(body, 1) is g

    def energy(p, y):
        cfg = _cfg()                                               # the Objective's solver settings
        ot = GridSinkhornLoss(p.grid, p.lgmin, p.ldx, p.ldims, eps=p.ldx ** 2, iters=cfg.ot_iters,
                              tol=min(cfg.ot_tol, 1e-3), mass_total=float(p.m.sum()), cuda_blocks=True,
                              support=p.support)
        return float(ot.state_energy(y, p.m))

    # the same transport problem (the same measures on the shared cells, no mass outside): the same energy (the
    # eps-scaling starts from the raster's diameter, which does not change the converged value: 1e-7 on the CPU)
    y = body.Xq * 0.9
    e0, e1 = energy(b0, y), energy(g, y)
    assert np.isfinite(e0) and abs(e1 - e0) <= 1e-4 * abs(e0) + 1e-8
    far = body.Xq * 3.5                                            # beyond the base raster: escaped mass there
    assert not np.isfinite(energy(b0, far)) and np.isfinite(energy(target_for(body, 2), far))


def test_cover_grows_only_the_body_that_leaves():
    from physmorph.pipeline.scene.raster import cover
    from physmorph.pipeline.scene.window import SceneWindow

    canon = MPMParams(dx=0.25, nx=16, ny=16, nz=16, grid_min=(-2.0,) * 3)
    bodies, x0, v0, world, cfg = _scene(scales=(2.0, 0.6), canon=canon)
    start, world, cfg, tgt, bal, vol0, bonds = _window_args(bodies, x0, v0, world, cfg)
    base = [bd.tgt for bd in bodies]
    win = SceneWindow(start, world, cfg, bodies, vol0, bonds, 0.5)
    ks = cover(win)
    assert ks[0] >= 2 and ks[1] == 0
    assert bodies[1].tgt is base[1] and bodies[0].tgt is not base[0] and bodies[0].base is base[0]
    assert win.views[0].tgt is bodies[0].tgt and win.zero_rolled
    lo, hi = bodies[0].raster.bounds(ks[0])
    assert win.raster_extent[0][0] - 2 * canon.dx >= lo and win.raster_extent[0][1] + 2 * canon.dx <= hi
    # the next window on the same base reuses the grown target; a new base (the schedule's) resets the family
    bodies[0].tgt = base[0]
    win2 = SceneWindow(start, world, cfg, bodies, vol0, bonds, 0.5)
    cover(win2)
    assert bodies[0].tgt is win.views[0].tgt


# ---- (2) a body with a non-finite term is left out ---------------------------------------------------------------

def test_a_body_with_a_non_finite_term_is_left_out(monkeypatch):
    from physmorph.pipeline.scene.solve import SceneOptimizer
    from physmorph.pipeline.window.objective import Objective

    bodies, x0, v0, world, cfg = _scene()
    args = _window_args(bodies, x0, v0, world, cfg)
    orig = Objective.transport

    def transport(self, xT):                                       # body 0's transport unsolvable at any state
        e = orig(self, xT)
        return e + float("inf") if getattr(self.win, "b", None) == 0 else e

    monkeypatch.setattr(Objective, "transport", transport)
    opt = SceneOptimizer(*args, 1.0, None, lambda *_: None)
    assert opt.inactive == [0] and opt.obj.active == [False, True]
    res = opt.run(None)
    assert res.commit is not None and not res.stats.get("zero_commit") and res.stats["inactive"] == [0]
    assert res.stats["accepted"] > 0
    r0, r1 = bodies[0].rows, bodies[1].rows
    assert float(opt.dFc.detach()[:, r0].abs().max()) == 0.0 and float(opt.u.detach()[r0].abs().max()) == 0.0
    assert float(opt.dFc.detach()[:, r1].abs().max()) > 0.0


# ---- (3) a replay that disagrees ---------------------------------------------------------------------------------

def test_a_replay_mismatch_is_solved_again_not_committed_as_the_zero_control():
    from physmorph.pipeline.scene.solve import SceneOptimizer

    bodies, x0, v0, world, cfg = _scene()
    args = _window_args(bodies, x0, v0, world, cfg)
    hist = [{"loss": -1.0e9, "alpha_base": 1.0}]                    # the accepted candidate's merit, far below any replay
    opt = SceneOptimizer(*args, 1.0, None, lambda *_: None, retry_mismatch=True)
    opt.accepted = 1
    res = opt.commit(list(hist), 0.0, False, False, None, None)
    assert res.commit is None and res.stats["retry"] == 1 and res.stats["null_reason"] == "commit_replay"
    opt = SceneOptimizer(*args, 1.0, None, lambda *_: None, retry_mismatch=False)
    opt.accepted = 1
    res = opt.commit(list(hist), 0.0, False, False, None, None)
    assert res.commit is not None and res.stats["zero_commit"] == 1 and res.stats["zero_reason"] == "commit_replay"


def test_the_runner_solves_a_mismatched_window_again_without_simulated_time(monkeypatch):
    from physmorph.pipeline.scene import solve as SS
    from physmorph.pipeline.scene.runner import run_scene
    from physmorph.pipeline.window.solve import WindowResult

    bodies, x0, v0, world, cfg = _scene()
    orig, calls = SS.SceneOptimizer.commit, {"n": 0}

    def commit(self, *a, **kw):                                    # the first two attempts' replays disagree
        calls["n"] += 1
        if calls["n"] <= 2 and self.retry_mismatch:
            return WindowResult(commit=None, hist=[], stats={"null_reason": "commit_replay", "retry": 1,
                                                             "dfc": self.dFc.detach()[:self.cfg.T].clone()})
        return orig(self, *a, **kw)

    monkeypatch.setattr(SS.SceneOptimizer, "commit", commit)
    res = run_scene(bodies, x0, v0, world, cfg, windows=2, mu=0.5, log=lambda *_: None)
    hist = res["history"]
    assert len(hist) == 2 and len(res["frames"]) == 1 + 2 * 2 * cfg.T          # no simulated time for the nulls
    assert hist[0]["replay_retries"] == ["commit_replay", "commit_replay"] and hist[1]["replay_retries"] == []
    assert not hist[0].get("zero_commit") or hist[0].get("zero_reason") not in SS.MISMATCH


def test_after_the_patience_the_continuation_is_committed_and_recorded(monkeypatch):
    from physmorph.pipeline.scene import solve as SS
    from physmorph.pipeline.scene.runner import run_scene
    from physmorph.pipeline.window.solve import WindowResult

    bodies, x0, v0, world, cfg = _scene()
    cfg = dataclasses.replace(cfg, patience=2)
    orig = SS.SceneOptimizer.commit

    def commit(self, *a, **kw):                                    # every replay disagrees
        if self.retry_mismatch:
            return WindowResult(commit=None, hist=[], stats={"null_reason": "commit_replay", "retry": 1,
                                                             "dfc": self.dFc.detach()[:self.cfg.T].clone()})
        self.accepted = 1
        return orig(self, [{"loss": -1.0e9, "alpha_base": 1.0}], *a[1:], **kw)

    monkeypatch.setattr(SS.SceneOptimizer, "commit", commit)
    res = run_scene(bodies, x0, v0, world, cfg, windows=1, mu=0.5, log=lambda *_: None)
    h = res["history"][0]
    assert h["replay_retries"] == ["commit_replay"] * 2 and h["zero_commit"] == 1 and h["zero_reason"] == "commit_replay"
    assert len(res["frames"]) == 1 + 2 * cfg.T
