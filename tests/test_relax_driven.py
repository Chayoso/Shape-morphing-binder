"""D146 (`relax_driven`): the layer relaxation acts in the driven steps only, the steps u acts in, so the released half of
a window has no position edit. Warp CPU for the trajectory (as tests/test_layer_relax.py): the released steps leave the
layer where the grid's update puts it, the driven steps relax as before, the switch off is the old path bit for bit,
and the bridge's gradients (dFc and u) match central differences with the switch on. CUDA: the window hands the
relaxation's steps to its trajectories."""
import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.function import RolloutSpec, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes
from physmorph.pipeline.window.layer import layer_relax_data as _layer_relax_data

DEV = "cpu"


def _layer(x, sp, **kw):
    return tuple(t.numpy() for t in _layer_relax_data(torch.as_tensor(x), sp, **kw))


def _slab(n_side=10, layers=4, spacing=0.25, seed=0, bump=True):
    rng = np.random.default_rng(seed)
    g = np.arange(n_side) * spacing
    X, Y, Z = np.meshgrid(g, np.arange(layers) * spacing, g, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    x += rng.uniform(-0.05, 0.05, x.shape).astype(np.float32) * spacing
    x -= x.mean(0)
    if bump:                                    # a rough top face: something for the relaxation to do in every step
        top = x[:, 1] > x[:, 1].max() - 0.5 * spacing
        x[:, 1] += np.where(top, 0.3 * spacing * np.sin(9.0 * x[:, 0]) * np.cos(7.0 * x[:, 2]), 0.0).astype(np.float32)
    return x, spacing


def _params():
    return MPMParams(dx=0.75, dt=1.0 / 120.0, drag=0.0, smoothing=0.955,
                     grid_min=(-6.0, -6.0, -6.0), nx=16, ny=16, nz=16)


def _rollout(x, sp, layer, T=8, Tc=4, lam=0.0, mu=0.0):
    prm = _params()
    vol0 = compute_rest_volumes(x, 1.0, prm, DEV)
    tr = Trajectory(x, 1.0, lam, mu, prm, T, device=DEV, requires_grad=False, vol0=vol0, layer=layer, control_steps=Tc)
    tr.rollout()
    return tr


def test_the_released_steps_leave_the_layer_where_the_grid_puts_it():
    x, sp = _slab()
    mask, nrm, nbr, w = _layer(x, sp, k=8, h_sp=2.0)
    T, Tc = 8, 4
    on = _rollout(x, sp, (mask, nrm, nbr, w, 1.0 / Tc, None, None, Tc), T, Tc)
    off = _rollout(x, sp, (mask, nrm, nbr, w, 1.0 / Tc), T, Tc)
    lay = mask > 0.5
    for t in range(T):
        moved_on = np.abs(on.x[t + 1].numpy() - on.xu[t + 1].numpy())[lay].max()     # the projection's own edit
        moved_off = np.abs(off.x[t + 1].numpy() - off.xu[t + 1].numpy())[lay].max()
        assert moved_off > 1e-6 * sp                                                    # the old path: every step
        if t < Tc:
            assert moved_on > 1e-6 * sp                                                 # driven: relaxed as before
        else:
            assert moved_on == 0.0                                                      # released: the identity
    # the driven half is the old path's, step for step
    for t in range(Tc + 1):
        assert np.array_equal(on.x[t].numpy(), off.x[t].numpy())


def test_the_switch_off_is_the_old_path_bit_for_bit():
    x, sp = _slab(seed=2)
    mask, nrm, nbr, w = _layer(x, sp, k=8, h_sp=2.0)
    T, Tc = 8, 4
    a = _rollout(x, sp, (mask, nrm, nbr, w, 1.0 / Tc), T, Tc, lam=800.0, mu=400.0)
    b = _rollout(x, sp, (mask, nrm, nbr, w, 1.0 / Tc, None, None, None), T, Tc, lam=800.0, mu=400.0)
    c = _rollout(x, sp, (mask, nrm, nbr, w, 1.0 / Tc, None, None, T), T, Tc, lam=800.0, mu=400.0)
    for t in range(T + 1):
        assert np.array_equal(a.x[t].numpy(), b.x[t].numpy())
        assert np.array_equal(a.x[t].numpy(), c.x[t].numpy())
    with pytest.raises(ValueError):
        _rollout(x, sp, (mask, nrm, nbr, w, 1.0 / Tc, None, None, T + 1), T, Tc)


def test_a_held_offset_needs_no_pulse_when_u_and_the_relaxation_share_their_steps():
    """The hold model of D146: a bump held by u against the relaxation. With the relaxation in all steps, holding the
    bump's offset across the window needs a larger u and the bump goes out and back inside the window; with the
    relaxation in the driven steps only, u equal to the relaxation's pull holds it in place step by step."""
    x, sp = _slab(n_side=12, seed=3, bump=False)
    mask, nrm, nbr, w = _layer(x, sp, k=8, h_sp=2.0)
    top = np.where((mask > 0.5) & (nrm[:, 1] > 0.8) & (np.abs(x[:, 0]) < 0.3) & (np.abs(x[:, 2]) < 0.3))[0]
    p = top[len(top) // 2]
    xb = x.copy()
    xb[p] += 0.4 * sp * nrm[p]
    T, Tc = 20, 10
    prm = _params()
    vol0 = compute_rest_volumes(xb, 1.0, prm, DEV)

    def path(steps, u_p):
        u = np.zeros(len(xb), np.float32)
        u[p] = u_p
        tr = Trajectory(xb, 1.0, 0.0, 0.0, prm, T, device=DEV, requires_grad=False, vol0=vol0,
                        layer=(mask, nrm, nbr, w, 1.0 / Tc, None, None, steps), control_steps=Tc,
                        layer_u=wp.array(u, dtype=float, device=DEV))
        tr.rollout()
        return np.array([(tr.x[t].numpy()[p] - xb[p]) @ nrm[p] for t in range(T + 1)])

    def held(steps):
        """u that brings the bump back to its start at the window's end (bisection; the end is monotone in u)."""
        lo, hi = 0.0, 4.0 * sp
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if path(steps, mid)[-1] < 0 else (lo, mid)
        return 0.5 * (lo + hi)

    u_all, u_drv = held(None), held(Tc)
    exc_all = np.abs(path(None, u_all)).max()
    exc_drv = np.abs(path(Tc, u_drv)).max()
    assert u_drv < 0.6 * u_all                          # the hold costs u less than in the old path
    assert exc_drv < 0.25 * exc_all                     # and the in-window excursion nearly goes


def test_the_gradients_match_central_differences_with_the_switch_on():
    x, sp = _slab(n_side=5, layers=3)
    prm = _params()
    mask, nrm, nbr, w = _layer(x, sp, k=6, h_sp=2.0)
    T, Tc = 4, 2
    vol0 = compute_rest_volumes(x, 1.0, prm, DEV)
    spec = RolloutSpec(x0=x, m=1.0, lam=800.0, mu=400.0, prm=prm, T=T, device=DEV, vol0=vol0,
                       layer=(mask, nrm, nbr, w, 1.0 / Tc, None, None, Tc), control_steps=Tc)
    torch.manual_seed(1)
    dfc = (torch.randn(T, len(x), 3, 3) * 2e-2).requires_grad_(True)
    u = (torch.randn(len(x)) * 0.05 * sp).requires_grad_(True)
    wvec = torch.randn(len(x), 3)

    def L(d, uu):
        xT, FT, vT, FgT, V = warp_mpm_ext(d, spec, u_t=uu)
        return (xT.double() * wvec.double()).sum() + 0.1 * (vT.double() * wvec.double()).sum()

    gd, gu = torch.autograd.grad(L(dfc, u), (dfc, u))
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the window runs on CUDA")
def test_the_window_hands_the_relaxations_steps_to_its_trajectories(monkeypatch):
    from physmorph.pipeline import PipelineConfig, run_pipeline
    from physmorph.pipeline.window import setup
    seen = []

    class Recording(setup.Trajectory):
        def __init__(self, *a, **kw):
            seen.append(kw["layer"][7] if len(kw["layer"]) > 7 else "absent")
            super().__init__(*a, **kw)

    monkeypatch.setattr(setup, "Trajectory", Recording)
    rng = np.random.default_rng(7)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    for flag, want in ((False, None), (True, 3)):
        seen.clear()
        cfg = PipelineConfig(T=3, iters=2, animations=1, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                             render_res=24, dt_res=32, patience=2, relax_driven=flag)
        run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
        assert seen and all(s == want for s in seen), (flag, seen)
