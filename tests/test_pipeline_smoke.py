"""End-to-end smoke tests of the settled-transport pipeline on a small cloud (CUDA), and unit
tests of its pieces: PCGrad, the frame store, window acceptance and the rest gate."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, run_pipeline

GUARDS = {"clamped", "nan_x", "nan_state", "F_reset", "F_flip", "F_invert_steps"}


@pytest.fixture(scope="module")
def prm():
    return MPMParams(dx=1.0, nx=32, ny=32, nz=32)


@pytest.fixture(scope="module")
def clouds():
    rng = np.random.default_rng(7)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    return src, tgt


def _cfg(**kw):
    base = dict(T=3, iters=2, animations=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                render_res=24, dt_res=32, patience=2, c2f_event=False)
    base.update(kw)
    return PipelineConfig(**base)


def _windows(res):
    return [h for h in res["history"] if h.get("frame_end") and not h.get("null_commit")]


def test_render_run_is_live_and_frames_are_promoted_states(prm, clouds):
    src, tgt = clouds
    cfg = _cfg()
    seen = []
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None,
                       on_iter=lambda _i, _x, _F, tele: seen.append(tele))
    assert set(res["guards"]) == GUARDS and all(v == 0 for v in res["guards"].values())
    wins = _windows(res)
    assert wins, "no window was accepted"
    for r in wins:
        assert r["lambda"] > 0 and r["d_render"] is not None and np.isfinite(r["selection_merit"])
        for k in ("loss", "d_vol", "kin", "v_mean", "move", "Jmin_traj", "transport_energy", "g_share"):
            assert k in r
        # the archived window end IS the promoted state, and each window adds 2T frames
        assert np.isfinite(res["frames"].x[r["frame_end"] - 1]).all()
    assert wins[0]["frame_end"] == 1 + 2 * cfg.T
    assert seen and seen[-1]["_grad_render"].shape == (len(src), 3)
    assert np.abs(seen[-1]["_grad_render"]).sum() > 0
    assert 1 <= res["deliver_n"] <= len(res["frames"])


def test_render_off_twin_keeps_the_render_telemetry(prm, clouds):
    src, tgt = clouds
    res = run_pipeline(src, tgt, prm, _cfg(render_weight_scale=0.0, animations=3),   # >1: a tiny cloud can null a window
                       log=lambda *_: None)
    wins = _windows(res)
    assert wins and all(r["lambda"] == 0.0 and r["g_share"] == 0.0 for r in wins)
    assert all(r["d_render"] is not None for r in wins)


def test_pcgrad_projection_math():
    from physmorph.pipeline.window.solve import pcgrad
    gp = [torch.tensor([1.0, 0.0, 0.0])]
    out = pcgrad(gp, [torch.tensor([-2.0, 1.0, 0.0])])          # cos < 0 vs gp
    assert abs(float((out[0] * gp[0]).sum())) < 1e-6            # conflicting component removed
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    out2 = pcgrad(gp, [torch.tensor([0.5, 3.0, 0.0])])           # cos > 0: untouched
    assert torch.allclose(out2[0], torch.tensor([0.5, 3.0, 0.0]))


def test_frame_store_keeps_F_at_the_stride_and_window_ends():
    from physmorph.pipeline.run.state import FrameStore
    x0 = torch.zeros(5, 3, device="cuda")
    store = FrameStore(x0, stride=4)
    xs = [x0 + k for k in range(1, 6)]                            # steps 1..5 of a 6-step window
    Fs = [torch.eye(3, device="cuda").repeat(5, 1, 1) * (1 + k) for k in range(1, 6)]
    store.add_window(xs, Fs, x0 + 6, Fs[-1] * 10)
    assert len(store) == 7 and set(store.F) == {0, 4, 6}
    store.hold()
    idx, F = store.archive_F()
    assert idx == [0, 4, 7] and np.allclose(F[-1], store.F[6])
    store.truncate(1)
    assert len(store) == 1 and set(store.F) == {0}


def test_selection_rejects_a_merit_runaway_and_stops_on_repeated_rejects():
    from physmorph.pipeline.run.selection import Selection
    sel = Selection(PipelineConfig(reject_stop=3, patience=10))
    disp = torch.ones(6)
    rec = {"selection_merit": 1.0, "transport_energy": 1.0}
    reject, brake, improved = sel.judge(rec, {"phys": 1.0}, disp)
    assert not reject and improved
    assert not sel.accepted(rec, 0, disp, improved)
    stops = []
    for _ in range(3):
        rec = {"selection_merit": 1.2, "transport_energy": 1.0}   # +20 %: beyond the 5 % brake
        reject, brake, _ = sel.judge(rec, {"phys": 1.0}, disp)
        assert reject and brake
        stops.append(sel.rejected(rec, brake, replay_rel=0.0))
    assert stops == [False, False, True]                           # reject_stop
    assert rec["replay"] == 1                                      # the same merit again


@pytest.mark.parametrize('velocity,rejected_velocity', [(.04, 0.), (.01, 99.)])
def test_rest_gate_uses_delivered_commit_not_rejected_trials(velocity, rejected_velocity):
    from scripts.pipeline_run import eval_gates
    res = dict(guards={}, deliver_n=2, history=[
        dict(frame_end=2, v_mean=velocity),
        dict(frame_end=3, v_mean=99.),
        dict(null_commit=1, outer_rejected=1, v_mean=rejected_velocity),
    ])
    met = dict(jitter_rel=0., bbox_diag=1., hole_frac=0., hole_frac_tgt=0., outside_max=0., stray_max=0.)
    gates = eval_gates(res, met, SimpleNamespace(dt=.1), T=1)
    assert gates['drift_rel'] == pytest.approx(.1 * velocity)
    assert gates['G3_rest'] == (velocity < .03)
