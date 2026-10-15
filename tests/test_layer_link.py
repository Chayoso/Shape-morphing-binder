"""D151 (`layer_link`): the layer's step (u and the relaxation, kernels.k_layer_project) is taken only while it keeps the
particle's nearest frozen neighbour (the minimum spacing's list) within max(berth, its distance before the step); a step
that brings it nearer is always taken. Warp CPU for the trajectory (as tests/test_relax_driven.py): a particle u pushes
out of a slab stays within the berth of its nearest neighbour (and leaves it without the link), an inward u is the old
path bit for bit, the switch off is the old path bit for bit, the hand adjoint is the generated one, and the bridge's
gradient matches central differences while the link withholds a particle's steps. CUDA: the window and the scene hand
the berth (the near band's) to their trajectories, and the one-body scene in a fixed frame is the window with the link."""
import numpy as np
import pytest
import torch
import warp as wp
from scipy.spatial import cKDTree

from physmorph.mpm import traj as TRAJ
from physmorph.mpm.function import RolloutSpec, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes
from physmorph.pipeline.window.layer import layer_relax_data as _layer_relax_data

DEV = "cpu"


def _layer(x, sp, **kw):
    return tuple(t.numpy() for t in _layer_relax_data(torch.as_tensor(x), sp, **kw))


def _slab(n_side=10, layers=4, spacing=0.25, seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(n_side) * spacing
    X, Y, Z = np.meshgrid(g, np.arange(layers) * spacing, g, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    x += rng.uniform(-0.05, 0.05, x.shape).astype(np.float32) * spacing
    x -= x.mean(0)
    return x, spacing


def _params():
    return MPMParams(dx=0.75, dt=1.0 / 120.0, drag=0.0, smoothing=0.955,
                     grid_min=(-6.0, -6.0, -6.0), nx=16, ny=16, nz=16)


def _spacing(x, sp):
    """The minimum spacing's frozen list (16 nearest, as setup.py) with a radius that pushes nothing here."""
    nb = cKDTree(x).query(x, 17)[1][:, 1:].astype(np.int64)
    return nb, 0.05 * sp


def _top(x, mask, nrm):
    top = np.where((mask > 0.5) & (nrm[:, 1] > 0.8) & (np.abs(x[:, 0]) < 0.3) & (np.abs(x[:, 2]) < 0.3))[0]
    return top[len(top) // 2]


def _nearest(x, p, nb):
    return float(np.linalg.norm(x[p] - x[nb[p]], axis=1).min())


def _rollout(x, layer, spacing, u, T=8, Tc=4, lam=0.0, mu=0.0):
    prm = _params()
    vol0 = compute_rest_volumes(x, 1.0, prm, DEV)
    tr = Trajectory(x, 1.0, lam, mu, prm, T, device=DEV, requires_grad=False, vol0=vol0, layer=layer,
                    control_steps=Tc, spacing=spacing, layer_u=wp.array(u, dtype=float, device=DEV))
    tr.rollout()
    return tr


def test_a_particle_u_pushes_out_stays_within_the_berth_of_its_nearest():
    x, sp = _slab()
    mask, nrm, nbr, w = _layer(x, sp, k=8, h_sp=2.0)
    p = _top(x, mask, nrm)
    spacing = _spacing(x, sp)
    nb = spacing[0]
    d0 = _nearest(x, p, nb)
    berth = 1.4 * d0                                     # the particle starts inside it
    T, Tc = 20, 10
    u = np.zeros(len(x), np.float32)
    u[p] = 3.0 * sp                                      # three spacings outward over the driven steps, 0.3 a step
    base = (mask, nrm, nbr, w, 1.0 / Tc, None, None, Tc)
    off = _rollout(x, base, spacing, u, T, Tc)
    on = _rollout(x, base + (berth,), spacing, u, T, Tc)
    assert _nearest(off.x[T].numpy(), p, nb) > berth + 0.3 * sp          # without the link it leaves its neighbours
    for t in range(T + 1):                               # with it the nearest stays within the berth (the neighbours'
        assert _nearest(on.x[t].numpy(), p, nb) <= berth + 0.05 * sp     # own relaxation steps move them a little)
    moved = (on.x[T].numpy()[p] - x[p]) @ nrm[p]
    assert moved > 0.3 * sp                              # the steps up to the berth are taken
    take = np.stack([on.ltake[t + 1].numpy() for t in range(Tc)])
    assert set(np.unique(take)) <= {0.0, 1.0} and take[:, p].min() == 0.0 and take[0, p] == 1.0


def test_an_inward_push_within_the_berth_is_the_old_path_bit_for_bit():
    """Every layer particle pushed in (and elastic): no nearest leaves the berth, every step is taken, the rollout is the
    old path's bit for bit (take = 1 multiplies exactly)."""
    x, sp = _slab(seed=1)
    mask, nrm, nbr, w = _layer(x, sp, k=8, h_sp=2.0)
    spacing = _spacing(x, sp)
    T, Tc = 8, 4
    u = np.where(mask > 0.5, -0.3 * sp, 0.0).astype(np.float32)        # every layer particle pushed in
    base = (mask, nrm, nbr, w, 1.0 / Tc, None, None, Tc)
    berth = float(np.median(cKDTree(x).query(x, 9)[0][:, 8]))           # the sample's median 8th-neighbour distance
    a = _rollout(x, base, spacing, u, T, Tc, lam=800.0, mu=400.0)
    b = _rollout(x, base + (berth,), spacing, u, T, Tc, lam=800.0, mu=400.0)
    for t in range(T + 1):
        assert np.array_equal(a.x[t].numpy(), b.x[t].numpy())


def test_the_switch_off_is_the_old_path_bit_for_bit():
    x, sp = _slab(seed=2)
    mask, nrm, nbr, w = _layer(x, sp, k=8, h_sp=2.0)
    spacing = _spacing(x, sp)
    T, Tc = 8, 4
    u = (np.random.default_rng(4).normal(size=len(x)) * 0.2 * sp).astype(np.float32)
    base = (mask, nrm, nbr, w, 1.0 / Tc, None, None, Tc)
    a = _rollout(x, base, spacing, u, T, Tc, lam=800.0, mu=400.0)
    b = _rollout(x, base + (None,), spacing, u, T, Tc, lam=800.0, mu=400.0)
    assert b.layer_link == 0.0
    for t in range(T + 1):
        assert np.array_equal(a.x[t].numpy(), b.x[t].numpy())
    with pytest.raises(ValueError):                      # the link reads the minimum spacing's list
        _rollout(x, base + (0.3,), None, u, T, Tc)
    from physmorph.pipeline import PipelineConfig
    with pytest.raises(ValueError):
        PipelineConfig(layer_link=True, min_spacing=0.0)
    assert PipelineConfig().layer_link is False


def _spec(x, sp, berth, T, Tc):
    mask, nrm, nbr, w = _layer(x, sp, k=6, h_sp=2.0)
    prm = _params()
    vol0 = compute_rest_volumes(x, 1.0, prm, DEV)
    layer = (mask, nrm, nbr, w, 0.0, None, None, Tc) + ((berth,) if berth is not None else ())   # u only, no relaxation
    return RolloutSpec(x0=x, m=1.0, lam=800.0, mu=400.0, prm=prm, T=T, device=DEV, vol0=vol0, layer=layer,
                       control_steps=Tc, spacing=_spacing(x, sp)), mask, nrm


def test_the_gradients_match_central_differences_while_the_link_withholds_steps(monkeypatch):
    """One top particle a spacing out, beyond the berth (1.2 spacings; every other nearest within it), and u on it only (no
    relaxation): its outward steps are withheld decisively, every other particle's step is taken (zero, or the
    perturbation's, inside the berth); the bridge's gradients (dFc and u) match central differences, the hand adjoint
    matches Warp's generated one, and without the link the particle moves."""
    x, sp = _slab(n_side=5, layers=3)
    T, Tc = 4, 2
    mask, nrm = _spec(x, sp, None, T, Tc)[1:]
    p = int(np.where((mask > 0.5) & (nrm[:, 1] > 0.8) & (np.abs(x[:, 0]) < 0.2) & (np.abs(x[:, 2]) < 0.2))[0][0])
    x = x.copy()
    x[p] += 1.0 * sp * nrm[p]
    nb = _spacing(x, sp)[0]
    others = max(_nearest(x, q, nb) for q in range(len(x)) if q != p)
    berth = 1.2 * sp
    assert others < berth < _nearest(x, p, nb)
    spec, mask, nrm = _spec(x, sp, berth, T, Tc)
    torch.manual_seed(1)
    dfc = (torch.randn(T, len(x), 3, 3) * 2e-2).requires_grad_(True)
    u = torch.zeros(len(x))
    u[p] = 1.5 * sp
    u.requires_grad_(True)
    wvec = torch.randn(len(x), 3)

    def L(d, uu, s=spec):
        xT, FT, vT, FgT, V = warp_mpm_ext(d, s, u_t=uu)
        return (xT.double() * wvec.double()).sum() + 0.1 * (vT.double() * wvec.double()).sum()

    gd, gu = torch.autograd.grad(L(dfc, u), (dfc, u))
    assert abs(float(gu[p])) < 1e-6 * max(float(gu.abs().max()), 1e-12) + 1e-9    # p's steps withheld: no lever
    torch.manual_seed(3)
    for _ in range(3):
        for leaf, g, eps in ((dfc, gd, 1e-3), (u, gu, 1e-3 * sp)):
            d = torch.randn_like(leaf)
            d = d / d.norm()
            with torch.no_grad():
                if leaf is dfc:
                    fd = (L(dfc.detach() + eps * d, u.detach()) - L(dfc.detach() - eps * d, u.detach())) / (2 * eps)
                else:
                    fd = (L(dfc.detach(), u.detach() + eps * d) - L(dfc.detach(), u.detach() - eps * d)) / (2 * eps)
            an = (g * d).sum()
            # 8 %: the float32 rollout's replay noise (test_layer_relax.py's contract for the bridge)
            assert abs(float(fd - an)) <= 8e-2 * max(abs(float(fd)), abs(float(an)), 1e-4), (fd, an)
    # the hand adjoint (adjoints.k_layer_project_take_adj) against Warp's generated one of k_layer_project_take
    monkeypatch.setattr(TRAJ, "HAND_ADJOINTS", False)
    gd2, gu2 = torch.autograd.grad(L(dfc, u), (dfc, u))
    # to rounding, the repo's contract for the hand adjoints (tests/test_runtime_exact.py, test_gap_split.py)
    torch.testing.assert_close(gd, gd2, rtol=1e-4, atol=1e-5 * float(gd2.abs().max()))
    torch.testing.assert_close(gu, gu2, rtol=1e-4, atol=1e-5 * float(gu2.abs().max()))
    monkeypatch.setattr(TRAJ, "HAND_ADJOINTS", True)
    with torch.no_grad():                                 # without the link the particle moves out
        spec_off, _, _ = _spec(x, sp, None, T, Tc)
        x_on = warp_mpm_ext(dfc.detach(), spec, u_t=u.detach())[0].numpy()
        x_off = warp_mpm_ext(dfc.detach(), spec_off, u_t=u.detach())[0].numpy()
    assert (x_off[p] - x_on[p]) @ nrm[p] > 0.5 * sp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the window runs on CUDA")
def test_the_window_hands_the_near_bands_berth_to_its_trajectories(monkeypatch):
    from physmorph.pipeline import PipelineConfig, run_pipeline
    from physmorph.pipeline.window import setup
    seen = []

    class Recording(setup.Trajectory):
        def __init__(self, *a, **kw):
            seen.append(kw["layer"][8] if len(kw["layer"]) > 8 else "absent")
            super().__init__(*a, **kw)

    monkeypatch.setattr(setup, "Trajectory", Recording)
    rng = np.random.default_rng(7)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    for flag in (False, True):
        seen.clear()
        cfg = PipelineConfig(T=3, iters=2, animations=1, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                             render_res=24, dt_res=32, patience=2, min_spacing=0.9, layer_link=flag)
        run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
        if flag:                                          # one berth for the run, the near band's
            assert seen and all(isinstance(s, float) and s > 0 and s == seen[0] for s in seen), seen
        else:
            assert seen and all(s == "absent" for s in seen), seen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the scene runs on CUDA")
def test_the_one_body_scene_with_the_link_is_the_window_with_it():
    """test_scene_bodies' fixed-frame body with the link on in both: the scene's trajectories get the berth and the
    rollout is the window's."""
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
                         render_res=24, dt_res=32, render_exterior=True, min_spacing=0.9, volume_exact="carried",
                         layer_link=True)
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
    u = 0.5 * torch.rand(n, generator=torch.Generator().manual_seed(5)).cuda()     # outward: the link has work
    win = Window(start, prm, cfg, tp, vol0, bonds)
    assert win.tr.layer_link > 0
    e1 = eval_terms(win, Objective(win), leaf, u)
    body = Body(start=0, stop=n, Xq=x, tgt=tp, vp=27.0 / n, mass=cfg.mass_ref_n / n,
                fixed=(torch.zeros(3, device="cuda"), torch.eye(3, device="cuda")))
    swin = SceneWindow(start, prm, cfg, [body], vol0, [bonds], mu=0.5)
    assert all(tr.layer_link == win.tr.layer_link for tr in swin.tr.trajs)
    e2 = eval_terms(swin, SceneObjective(swin), leaf, u)
    assert float((e1.xT - e2.xT).abs().max()) < 1e-5
