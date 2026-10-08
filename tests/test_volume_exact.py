"""D129, --volume_exact: the stress reads the volume of the unsmoothed deformation history.

The smoothing keeps (1 - s) of every step's deformation increment in F, so the stress read J = det F ~ 1 where the
motion's own volume was ~ 4 (D128). With the flag the tracked J_{t+1} = J_t det(F_new) / det(F_t) supplies the volume
and the smoothed F + dFc the shape: F_eff = (J / det F)^(1/3) (F + dFc) (mpm/kernels.k_stress_vx, k_volume_update).

(1) flag off: the old trajectory and its adjoint bit for bit, against a reference written by the code before the
    change (tests/data/volume_exact_off_ref.npz, repo_r104 = 0894f7d; `_window_case` builds the case for both);
(2) flag on: a pure dilation drives J to det of the motion's accumulated deformation while the smoothed F lags, J is
    det F exactly when nothing is smoothed (the control included), and the stress then resists the dilation;
(3) flag on: the adjoint (persistent graphs and the plain bridge alike) against finite differences on dFc and u, from
    a start whose J is not det F;
(4) J is carried across a commit: a split rollout equals the whole one, and in the pipeline each window starts from
    the last commit's promoted J (the archive keeps it at F's frames).
Only numpy / torch / warp / physmorph names that existed before D129 are imported at module level (the reference is
written by importing `_window_case` from this file into the old code).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp
from scipy.spatial import cKDTree

from physmorph.mpm.function import RolloutSpec, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes

REF = Path(__file__).parent / "data" / "volume_exact_off_ref.npz"


def _cuda_or_skip(dev):
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")


def _window_case(dev="cpu"):
    """A small window in the pipeline's configuration (D105 recipe's MPM side): a jittered 4^3 block with a start
    state (F, Fp isochoric, v, C), the material bonds (two particles flagged), the outer layer's relaxation with a
    reference and the u leaf, the minimum spacing, 4 driven + 4 released steps, the polar adjoint. Returns
    (spec, dc (Tc,N,3,3), u (N,)) as torch tensors on dev."""
    from physmorph.pipeline.window.layer import layer_relax_data
    rng = np.random.default_rng(129)
    g = np.arange(4, dtype=np.float32) * 0.25
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    x = (X - X.mean(0) + rng.uniform(-0.03, 0.03, X.shape)).astype(np.float32)
    N, Tc = len(x), 4
    prm = MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.9, smoothing=0.955, grid_min=(-4.0,) * 3, nx=16, ny=16, nz=16)
    vol0 = compute_rest_volumes(x, 1.0, prm, dev)
    sym = rng.normal(0, 0.03, (N, 3, 3))
    F0 = (np.eye(3) + 0.5 * (sym + sym.transpose(0, 2, 1))).astype(np.float32)
    a = rng.normal(0, 0.05, (N, 3, 3))
    a = 0.5 * (a + a.transpose(0, 2, 1))
    a -= np.trace(a, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3.0
    w_, V_ = np.linalg.eigh(a)                          # Fp = exp(traceless sym): det 1
    Fp = (V_ @ (np.exp(w_)[..., None] * V_.transpose(0, 2, 1))).astype(np.float32)
    v0 = rng.normal(0, 0.1, (N, 3)).astype(np.float32)
    C0 = rng.normal(0, 0.2, (N, 3, 3)).astype(np.float32)
    nbr = cKDTree(x).query(x, k=7)[1][:, 1:].astype(np.int32)
    rest = (1.05 * np.linalg.norm(x[nbr] - x[:, None], axis=2)).astype(np.float32)
    frag = np.zeros(N, np.float32)
    frag[[0, 37]] = 1.0
    mask, nrm, lnbr, lw = (t.cpu().numpy() for t in layer_relax_data(torch.as_tensor(x), 0.25, k=8, h_sp=2.0))
    ref = (rng.normal(0, 0.005, N) * mask).astype(np.float32)
    layer = (mask, nrm, lnbr, lw, 1.0 / Tc, None, 0.0, None, ref)
    snbr = cKDTree(x).query(x, k=9)[1][:, 1:].astype(np.int32)
    spec = RolloutSpec(x0=x, m=1.0, lam=800.0, mu=400.0, prm=prm, T=2 * Tc, Fp=Fp, v0=v0, F0=F0, C0=C0,
                       device=dev, vol0=vol0, bond_nbr=nbr, bond_rest=rest, bond_frag=frag,
                       spacing=(snbr, 0.2), layer=layer, bond_history=True, control_steps=Tc, polar_adjoint=True)
    dc = torch.tensor(rng.normal(0, 0.01, (Tc, N, 3, 3)).astype(np.float32), device=dev)
    u = torch.tensor((rng.normal(0, 0.002, N) * mask).astype(np.float32), device=dev)
    return spec, dc, u


def _expand(z):
    return torch.cat((z, torch.zeros_like(z)))


def _window_loss(out):
    xT, FT, vT, FgT, V = out
    w = torch.linspace(0.5, 1.5, 3, device=xT.device)
    return (xT * w).pow(2).sum() * 1e2 + V.pow(2).sum() * 1e2 + FgT.pow(2).sum() * 1e-1 + (vT * w).sum()


def window_reference(dev="cpu") -> dict:
    """The outputs of the plain bridge on `_window_case` and the window loss's gradient on dc and u."""
    spec, dc, u = _window_case(dev)
    dc, u = dc.clone().requires_grad_(True), u.clone().requires_grad_(True)
    out = warp_mpm_ext(_expand(dc), spec, u_t=u)
    g_dc, g_u = torch.autograd.grad(_window_loss(out), (dc, u))
    names = ("xT", "FT", "vT", "FgT", "V")
    return {**{k: o.detach().cpu().numpy() for k, o in zip(names, out)},
            "g_dc": g_dc.cpu().numpy(), "g_u": g_u.cpu().numpy()}


# ---- (1) flag off: the old path bit for bit -------------------------------------------------------------------
def test_flag_off_reproduces_the_old_trajectory_and_adjoint_exactly():
    ref = np.load(REF)
    new = window_reference("cpu")
    for k, v in new.items():
        assert np.array_equal(v, ref[k]), (k, float(np.abs(v - ref[k]).max()))


def test_flag_off_has_no_volume_state():
    spec, dc, _ = _window_case("cpu")
    assert spec.volume_exact is False and spec.J0 is None
    from physmorph.mpm import kernels as K
    tr = Trajectory(spec.x0, 1.0, 800.0, 400.0, spec.prm, 2, device="cpu", requires_grad=False, vol0=spec.vol0)
    assert tr.J is None and tr.stress_kernel is K.k_stress
    with pytest.raises(ValueError):
        Trajectory(spec.x0, 1.0, 800.0, 400.0, spec.prm, 2, device="cpu", requires_grad=False, vol0=spec.vol0,
                   J0=np.ones(len(spec.x0), np.float32))


# ---- (2) flag on: the tracked volume and the stress that reads it ---------------------------------------------
def _block(n_side=6, h=0.25):
    g = np.arange(n_side, dtype=np.float32) * h
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return (X - X.mean(0)).astype(np.float32)


def _dilating(x, kappa):
    return (kappa * x).astype(np.float32), np.tile(kappa * np.eye(3, dtype=np.float32), (len(x), 1, 1))


def test_a_pure_dilation_drives_J_while_the_smoothed_F_lags():
    """No stress (lam = mu = 0), no drag: the block expands about uniformly, x ~ s x0. J follows the motion's volume,
    det Fg = prod det(I + dt C), exactly, and is the block's own s^3 (about (1 + kappa t)^3); the smoothed F keeps
    4.5 % of it."""
    x = _block()
    prm = MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.0, smoothing=0.955, grid_min=(-6.0,) * 3, nx=24, ny=24, nz=24)
    kappa, T = 1.5, 40
    v0, C0 = _dilating(x, kappa)
    tr = Trajectory(x, 1.0, 0.0, 0.0, prm, T, v0=v0, C0=C0, device="cpu", requires_grad=False,
                    vol0=compute_rest_volumes(x, 1.0, prm, "cpu"), track_geom=True, volume_exact=True)
    tr.rollout()
    for t in (1, 10, T):
        J, Fg = tr.J[t].numpy(), tr.Fg[t].numpy()
        assert np.allclose(J, np.linalg.det(Fg), rtol=1e-4), t
    J, detF = tr.J[T].numpy(), np.linalg.det(tr.F[T].numpy())
    xT = tr.x[T].numpy()
    s3 = float((xT * x).sum() / (x * x).sum()) ** 3                 # the block's own volume ratio (1.99)
    assert np.allclose(J, s3, rtol=0.01), (J.min(), J.max(), s3)
    assert abs(s3 / (1.0 + kappa * T * prm.dt) ** 3 - 1.0) < 0.03    # about the ballistic (1 + kappa t)^3 = 1.953
    # the smoothed F: ln det F = (1 - s) ln J along the same motion
    assert np.allclose(np.log(detF), (1.0 - prm.smoothing) * np.log(J), rtol=0.05)
    assert detF.max() < 1.04 < J.min()


def test_without_smoothing_J_is_det_F_control_included():
    """s = 0: F_{t+1} = (I + dt C)(F_t + dFc_t), so the unsmoothed history's det is det F at every step, and J (which
    takes the control's volume change at its full size) equals it."""
    x = _block(5)
    rng = np.random.default_rng(3)
    prm = MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.0, smoothing=0.0, grid_min=(-6.0,) * 3, nx=24, ny=24, nz=24)
    T = 12
    dfc = [wp.array(rng.normal(0, 0.01, (len(x), 3, 3)).astype(np.float32), dtype=wp.mat33, device="cpu")
           for _ in range(T)]
    v0, C0 = _dilating(x, 0.8)
    tr = Trajectory(x, 1.0, 800.0, 400.0, prm, T, v0=v0, C0=C0, dFc=dfc, device="cpu", requires_grad=False,
                    vol0=compute_rest_volumes(x, 1.0, prm, "cpu"), volume_exact=True)
    tr.rollout()
    for t in range(1, T + 1):
        assert np.allclose(tr.J[t].numpy(), np.linalg.det(tr.F[t].numpy()), rtol=2e-5, atol=1e-6), t


def test_the_stress_resists_the_dilation_it_now_sees():
    """The same dilating block with elasticity: the old path reads det F <= 1.03 and the block keeps expanding (the
    motion's volume det Fg 1.76 after 40 steps); with the exact volume the pressure stops the dilation within a few
    steps (det Fg peaks near 1.14) and turns it back."""
    x = _block()
    prm = MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.0, smoothing=0.955, grid_min=(-6.0,) * 3, nx=24, ny=24, nz=24)
    kappa, T = 1.5, 40
    v0, C0 = _dilating(x, kappa)
    vol0 = compute_rest_volumes(x, 1.0, prm, "cpu")
    out = {}
    for vx in (False, True):
        tr = Trajectory(x, 1.0, 2000.0, 1000.0, prm, T, v0=v0, C0=C0, device="cpu", requires_grad=False,
                        vol0=vol0, track_geom=True, volume_exact=vx)
        tr.rollout()
        Jg = [float(np.median(np.linalg.det(tr.Fg[t].numpy()))) for t in range(T + 1)]
        out[vx] = (max(Jg), Jg[T])
        if vx:
            assert np.allclose(tr.J[T].numpy(), np.linalg.det(tr.Fg[T].numpy()), rtol=1e-4)
    (peak_off, end_off), (peak_on, end_on) = out[False], out[True]
    assert end_off == peak_off > 1.5, out                            # unresisted: still growing at the end
    assert peak_on - 1.0 < 0.3 * (peak_off - 1.0), out
    assert end_on < peak_on, out                                     # and turned back


# ---- (3) flag on: the adjoint against finite differences ------------------------------------------------------
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_volume_exact_adjoint_matches_finite_differences(dev):
    _cuda_or_skip(dev)
    from physmorph.mpm.function import PersistentAdjoint
    spec, dc, u = _window_case(dev)
    N = len(spec.x0)
    J0 = np.random.default_rng(5).uniform(1.05, 1.3, N).astype(np.float32)      # not det F0: the volume was carried
    spec = dataclasses.replace(spec, volume_exact=True, J0=J0)
    adj = PersistentAdjoint(spec)
    dc, u = dc.clone().requires_grad_(True), u.clone().requires_grad_(True)
    got = adj.apply(_expand(dc), u)
    plain = warp_mpm_ext(_expand(dc), spec, u_t=u)
    tol = 0.0 if dev == "cpu" else 2e-5
    for a, b in zip(got, plain):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol * max(1.0, float(b.detach().abs().max())))
    g = torch.autograd.grad(_window_loss(got), (dc, u))
    ref = torch.autograd.grad(_window_loss(plain), (dc, u))
    for a, b in zip(g, ref):
        torch.testing.assert_close(a, b, rtol=max(tol, 1e-6) * 10, atol=1e-6 * max(1.0, float(b.detach().abs().max())))
    # the flag changes the dynamics: the same controls without it end elsewhere
    off = warp_mpm_ext(_expand(dc.detach()), dataclasses.replace(spec, volume_exact=False, J0=None), u_t=u.detach())
    assert float((off[0] - got[0]).abs().max()) > 1e-5
    for i, gi in enumerate(g):
        assert torch.isfinite(gi).all() and float(gi.abs().max()) > 0
        d = -gi / gi.norm()
        vals = []
        for sign in (1.0, -1.0):
            leaves = [dc.detach(), u.detach()]
            leaves[i] = leaves[i] + sign * 1e-3 * d
            with torch.no_grad():
                vals.append(float(_window_loss(adj.apply(_expand(leaves[0]), leaves[1]))))
        fd = (vals[0] - vals[1]) / 2e-3
        an = float((gi * d).sum())
        assert fd == pytest.approx(an, rel=0.03, abs=1e-4 * max(1.0, abs(an))), (i, fd, an)


# ---- (4) J carried across a commit ----------------------------------------------------------------------------
def test_a_split_rollout_carries_J_like_F():
    """T steps, then T more from the first part's end state (x, v, C, F, Fg and J), equal the 2T steps at once."""
    x = _block(5)
    rng = np.random.default_rng(11)
    prm = MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.9, smoothing=0.955, grid_min=(-6.0,) * 3, nx=24, ny=24, nz=24)
    T = 6
    dfc = rng.normal(0, 0.02, (2 * T, len(x), 3, 3)).astype(np.float32)
    v0, C0 = _dilating(x, 1.2)
    vol0 = compute_rest_volumes(x, 1.0, prm, "cpu")
    J0 = rng.uniform(1.1, 1.4, len(x)).astype(np.float32)
    kw = dict(device="cpu", requires_grad=False, vol0=vol0, track_geom=True, volume_exact=True)
    seq = lambda a, b: [wp.array(dfc[t], dtype=wp.mat33, device="cpu") for t in range(a, b)]  # noqa: E731
    whole = Trajectory(x, 1.0, 800.0, 400.0, prm, 2 * T, v0=v0, C0=C0, dFc=seq(0, 2 * T), J0=J0, **kw)
    whole.rollout()
    first = Trajectory(x, 1.0, 800.0, 400.0, prm, T, v0=v0, C0=C0, dFc=seq(0, T), J0=J0, **kw)
    first.rollout()
    tail = Trajectory(first.x[T].numpy(), 1.0, 800.0, 400.0, prm, T, v0=first.v[T].numpy(), C0=first.C[T].numpy(),
                      F0=first.F[T].numpy(), Fg0=first.Fg[T].numpy(), J0=first.J[T].numpy(), dFc=seq(T, 2 * T), **kw)
    tail.rollout()
    for a, b in ((whole.x[2 * T], tail.x[T]), (whole.J[2 * T], tail.J[T]), (whole.F[2 * T], tail.F[T])):
        assert np.allclose(a.numpy(), b.numpy(), rtol=1e-6, atol=1e-7)
    assert float(np.abs(tail.J[T].numpy() - np.linalg.det(tail.F[T].numpy())).max()) > 0.05


@pytest.fixture(scope="module")
def clouds():
    rng = np.random.default_rng(7)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    return src, tgt


@pytest.mark.parametrize("vx", [False, True])
def test_each_window_starts_from_the_last_commits_volume(clouds, monkeypatch, vx):
    """In the pipeline (CUDA): the first window starts at J = 1 (None), every later one from the promoted J of the last
    accepted commit (an outer-rejected one rolls it back); the archive keeps J at F's frames; J is not det F."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import physmorph.pipeline.run.runner as runner_mod
    from physmorph.pipeline import PipelineConfig, run_pipeline
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=3, iters=2, animations=4, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=10, c2f_event=False, volume_exact=vx)
    optimize, calls = runner_mod.optimize_window, []

    def window(*args, **kw):
        start = args[0]
        res = optimize(*args, **kw)
        calls.append({"J": None if start.J is None else start.J.clone(),
                      "end_J": None if res.commit is None or res.commit.end_J is None else res.commit.end_J.clone(),
                      "end_F": None if res.commit is None else res.commit.end_F.clone()})
        return res

    monkeypatch.setattr(runner_mod, "optimize_window", window)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert all(v == 0 for v in res["guards"].values())
    assert calls[0]["J"] is None
    if not vx:
        assert all(c["J"] is None and c["end_J"] is None for c in calls) and res["frames"].J is None
        return
    by_anim = {h["animation"]: h for h in res["history"] if "animation" in h}
    carried = None                                      # the J the next window must start from (None = 1)
    seen = 0
    for a, c in enumerate(calls):
        if carried is None:
            assert c["J"] is None
        else:
            assert torch.equal(c["J"], carried), a
            seen += 1
        h = by_anim.get(a, {})
        if c["end_J"] is not None and h.get("frame_end"):                 # accepted and kept
            carried = c["end_J"]
            assert np.array_equal(res["frames"].J[h["frame_end"] - 1], carried.cpu().numpy())
            assert float((carried - torch.linalg.det(c["end_F"])).abs().max()) > 1e-4
            assert {"Jx_min", "Jx_p50", "Jx_p99", "Jx_max"} <= set(h)
    assert seen >= 1, "no window started from a carried volume"
    idx, _ = res["frames"].archive_F()
    assert len(res["frames"].archive_J()) == len(idx)
