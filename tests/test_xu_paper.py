"""Xu et al. as published on this simulator (D118, PipelineConfig.xu_protocol "paper"): the paper's durations, the
episode (driven steps only, one control layer held over its first paper timestep, no bonds), the step rule (Adam on
the normalised gradient, a line search that halves until the loss decreases), every episode committed and scored at
its driven end, also when its line search fails; and the default path (xu_protocol "ours") untouched."""
import dataclasses

import numpy as np
import pytest
import torch

import physmorph.pipeline.run.xu_runner as xu_runner
import physmorph.pipeline.window.xu_episode as xe
from physmorph import gpu
from physmorph.losses.volumetric import d_vol_xu
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline import PipelineConfig, run_pipeline
from physmorph.pipeline.target import build_target, calibrate_units
from physmorph.pipeline.window import StartState, Window


@pytest.fixture(scope="module")
def prm():
    return MPMParams(dx=1.0, nx=32, ny=32, nz=32)


@pytest.fixture(scope="module")
def clouds():
    rng = np.random.default_rng(5)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    return src, tgt


def _cfg(**kw):
    base = dict(T=4, iters=2, animations=3, loss_res=12, render_views=2, render_elevs=(0.0, 0.5), render_res=24,
                dt_res=32, patience=2, c2f_event=False, render_exterior=False, render_weight_scale=0.0,
                mass_ref_n=300,                    # a unit dynamics mass: the small body moves within one episode
                baseline="xu", xu_form="tvcg", xu_protocol="paper")
    base.update(kw)
    return PipelineConfig(**base)


def test_the_protocol_is_the_published_method_and_not_the_default():
    assert PipelineConfig().xu_protocol == "ours"
    with pytest.raises(ValueError):
        PipelineConfig(xu_protocol="paper")                               # not a baseline
    with pytest.raises(ValueError):
        PipelineConfig(baseline="xu", xu_protocol="paper")                # the C++ copy's loss, not the paper's
    for b in ("xu", "xu_spray"):
        for form in ("tvcg", "paper"):
            assert PipelineConfig(baseline=b, xu_form=form, xu_protocol="paper").xu_protocol == "paper"


def test_the_durations_and_steps_are_the_papers():
    assert xe.XU.layout(1.0 / 240.0) == (20, 2)       # 10 timesteps of 1/120 s = 1/12 s; a layer = one 1/120 s step
    assert xe.XU.layout(1.0 / 120.0) == (10, 1)       # at the paper's own dt
    assert xe.XU.episodes(420) == 42 and xe.XU.episodes(950) == 95   # Table II Sphere to Bunny, Table IV D to Dragon
    assert (xe.XU.iters, xe.XU.passes) == (4, 3)      # Table II: 4 gradient descent iterations (3 optimization passes)
    assert (xe.XU.alpha, xe.XU.beta1) == (1e-3, 0.9)  # Fig. 8: Adam learning rate 1e-3, beta1 0.90


def test_adam_reads_the_normalised_gradient():
    st = xe.AdamState(torch.zeros(5), torch.zeros(5))
    g = torch.tensor([3.0, -4.0, 0.0, 1e-6, 2.0])
    d, m, v = xe.adam_direction(g, st)
    gh = g / g.norm()
    torch.testing.assert_close(d, gh / (gh.abs() + xe.XU.eps))           # first step: m^ = g^, v^ = g^2
    d10, _, _ = xe.adam_direction(10.0 * g, st)
    torch.testing.assert_close(d10, d)                                   # the gradient's scale does not enter
    assert torch.equal(st.m, torch.zeros(5))                              # not committed


def test_the_line_search_halves_until_the_loss_decreases():
    calls = []

    def loss(x):
        calls.append(float(x))
        return float((x - 1.0) ** 2)

    x0, d = torch.tensor(0.0), torch.tensor(-1.0)                         # the step x - a d = a
    a, L, n = xe.bisection(loss, x0, d, L0=1.0, alpha=4.0, ls_iters=10)
    assert (a, L, n) == (1.0, 0.0, 3) and calls == [4.0, 2.0, 1.0]       # 9 > 1, 1 = 1 (not below), 0 < 1
    calls.clear()
    a, L, n = xe.bisection(loss, x0, -d, L0=1.0, alpha=4.0, ls_iters=5)  # an ascent direction: no trial decreases
    assert a is None and n == 5 and calls == [-4.0, -2.0, -1.0, -0.5, -0.25]


def _target(prm, clouds, cfg):
    src = gpu.tensor(clouds[0])
    tgt = build_target(clouds[1], prm, cfg)
    calibrate_units(tgt, src, cfg)
    return src, tgt, compute_rest_volumes(src, 1.0, prm, cfg.device)


def test_an_episode_drives_every_step_with_one_layer_and_no_bonds(prm, clouds):
    cfg = _cfg()
    src, tgt, vol0 = _target(prm, clouds, cfg)
    N = len(src)
    win = xe.EpisodeWindow(StartState(x=src, Fp=torch.eye(3, device="cuda").repeat(N, 1, 1)), prm, cfg, tgt, vol0,
                           steps=20, layer=2)
    assert (win.Tc, win.T, win.tr.control_steps, win.spec.control_steps) == (1, 20, 2, 2)
    assert win.tr.bonds is None and win.spec.bond_nbr is None
    leaf = torch.randn(1, N, 3, 3, device="cuda", requires_grad=True)
    dc = win.expand(leaf)
    assert dc.shape == (20, N, 3, 3)
    assert torch.equal(dc[0], leaf[0]) and torch.equal(dc[1], leaf[0]) and not bool(dc[2:].any())
    w = torch.randn_like(dc)
    g, = torch.autograd.grad((dc * w).sum(), leaf)
    torch.testing.assert_close(g[0], w[0] + w[1])                       # one variable over the layer's steps


def test_the_default_window_is_settled_transports(prm, clouds):
    cfg = PipelineConfig(T=4, loss_res=12, render_views=2, render_elevs=(0.0, 0.5), render_res=24, dt_res=32,
                         render_exterior=False)
    assert cfg.xu_protocol == "ours"
    src, tgt, vol0 = _target(prm, clouds, cfg)
    N = len(src)
    nbr = gpu.knn(src, cfg.coh_k + 1)[1][:, 1:]
    rest = (src[nbr] - src[:, None, :]).norm(dim=2)
    win = Window(StartState(x=src, Fp=torch.eye(3, device="cuda").repeat(N, 1, 1)), prm, cfg, tgt, vol0,
                 (nbr, rest, torch.zeros(N, device="cuda")))
    assert (win.Tc, win.T, win.tr.control_steps, win.spec.control_steps) == (cfg.T, 2 * cfg.T, cfg.T, cfg.T)
    assert win.tr.bonds and win.spec.bond_nbr is not None
    leaf = torch.randn(cfg.T, N, 3, 3, device="cuda")
    assert torch.equal(win.expand(leaf), torch.cat((leaf, torch.zeros_like(leaf)), dim=0))


def test_the_default_run_never_takes_the_papers_path(prm, clouds, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("the paper protocol ran")

    monkeypatch.setattr(xu_runner, "run_xu_paper", refuse)
    for cfg in (PipelineConfig(T=3, iters=2, animations=1, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                               render_res=24, dt_res=32, patience=2, c2f_event=False),
                PipelineConfig(T=3, iters=2, animations=1, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                               render_res=24, dt_res=32, patience=2, c2f_event=False, render_weight_scale=0.0,
                               baseline="xu_spray", xu_form="paper")):        # D112's baseline in our protocol
        res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
        assert any(h.get("frame_end") for h in res["history"]) or any(h.get("null_commit") for h in res["history"])
    with pytest.raises(AssertionError, match="paper protocol ran"):
        run_pipeline(*clouds, prm, _cfg(), log=lambda *_: None)


@pytest.mark.parametrize("baseline", ["xu", "xu_spray"])
def test_every_episode_is_kept_and_scored_at_its_driven_end(prm, clouds, baseline):
    cfg = _cfg(baseline=baseline)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    steps, layer = xe.XU.layout(prm.dt)
    hist = res["history"]
    assert len(hist) == cfg.animations and not any(h.get("null_commit") for h in hist)
    assert [h["frame_end"] for h in hist] == [1 + steps * (a + 1) for a in range(cfg.animations)]  # no released steps
    assert res["deliver_n"] == len(res["frames"]) == 1 + steps * cfg.animations and res["truncation"] is None
    assert all(h["steps"] == steps and h["layer"] == layer and len(h["passes"]) == xe.XU.passes for h in hist)
    assert sum(h["accepted"] for h in hist) > 0
    assert all(h["accepted"] <= xe.XU.passes * xe.XU.iters for h in hist)
    if baseline == "xu":                          # the loss is read at the episode's last (driven) frame
        tgt = build_target(clouds[1], prm, cfg)
        for h in hist:
            x = gpu.tensor(res["frames"].x[h["frame_end"] - 1])
            L = float(d_vol_xu(x, tgt.m * cfg.xu_mass, *tgt.xu, **cfg.xu_kw()))
            assert L == pytest.approx(h["loss"], rel=1e-4) and L == pytest.approx(h["transport_energy"], rel=1e-4)


def test_an_episode_is_kept_when_its_line_search_fails(prm, clouds, monkeypatch):
    monkeypatch.setattr(xe, "bisection", lambda loss, x, d, L0, alpha, ls_iters: (None, float("inf"), ls_iters))
    cfg = _cfg(animations=2)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    steps, _ = xe.XU.layout(prm.dt)
    hist = res["history"]
    assert len(hist) == 2 and all(h["accepted"] == 0 and h["ls_failed"] == xe.XU.passes for h in hist)
    assert all(h["ctrl_absmax"] == 0.0 for h in hist)                     # the control stays the last accepted: zero
    assert res["deliver_n"] == len(res["frames"]) == 1 + 2 * steps       # time still advances
    # a body at rest without control stays, up to the float32 stress at F = I (about 1e-7 mu; 4.6e-5 here, against a
    # particle spacing of 0.45)
    drift = max(float(np.abs(x - clouds[0]).max()) for x in res["frames"].x)
    assert drift < 1e-3, f"an uncontrolled body at rest moved {drift:.3g}"
