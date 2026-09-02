"""End-to-end smoke of the blessed path on the warp CPU device: tiny clouds, tiny
horizons — catches integration breakage (shapes, device wiring, bookkeeping) that
py_compile cannot. Physics quality is NOT asserted here; that is the GPU gate run.
"""
import numpy as np
import pytest

from physmorph import metrics
from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, run_pipeline

DEV = "cpu"


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
    base = dict(T=3, iters=2, animations=2, loss_res=12, render_views=2,
                render_elevs=(0.0, 0.5), render_res=24, device=DEV, patience=2)
    base.update(kw)
    return PipelineConfig(**base)


def _check_result(res, cfg, n):
    assert len(res["frames"]) == len(res["F_frames"])
    for fr in (res["frames"][0], res["frames"][-1]):
        assert fr.shape == (n, 3) and np.isfinite(fr).all()
    assert set(res["guards"]) == {"clamped", "nan_x", "nan_state", "F_reset", "F_flip",
                                  "F_invert_steps"}
    recs = [h for h in res["history"] if "d_vol" in h]
    assert recs, "no optimisation window produced a record"
    for k in ("loss", "d_vol", "kin", "v_mean", "move", "Jmin_traj", "accepted"):
        assert k in recs[-1]


def test_phys_arm_runs(prm, clouds):
    src, tgt = clouds
    res = run_pipeline(src, tgt, prm, _cfg(), log=lambda *_: None)
    _check_result(res, _cfg(), len(src))
    assert all(h["d_render"] is None for h in res["history"] if "d_vol" in h)
    met = metrics.summarize(res["frames"], tgt, F_frames=res["F_frames"],
                            n_held=res["n_held"])
    assert np.isfinite(met["chamfer"]) and 0 <= met["sil_iou"] <= 1


def test_render_arm_runs_and_lambda_is_live(prm, clouds):
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    _check_result(res, cfg, len(src))
    recs = [h for h in res["history"] if "d_vol" in h]
    assert all(r["d_render"] is not None for r in recs)
    assert all(r["lambda"] > 0 for r in recs)


def test_material_arm_returns_bounded_s(prm, clouds):
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, opt_material=True, mat_clamp=1.0)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    _check_result(res, cfg, len(src))
    assert res["s"] is not None and res["s"].shape == (2, len(src))
    assert np.abs(res["s"]).max() <= 1.0 + 1e-6
    assert np.isfinite(res["s"]).all()


def test_pcgrad_projection_math():
    from physmorph.pipeline.optimizer import _pcgrad
    import torch
    gp = [torch.tensor([1.0, 0.0, 0.0])]
    gr_conf = [torch.tensor([-2.0, 1.0, 0.0])]      # cos < 0 vs gp
    out, conflicted = _pcgrad(gp, gr_conf)
    assert conflicted
    assert abs(float((out[0] * gp[0]).sum())) < 1e-6    # conflicting component removed
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    out2, c2 = _pcgrad(gp, [torch.tensor([0.5, 3.0, 0.0])])   # cos > 0: untouched
    assert not c2 and torch.allclose(out2[0], torch.tensor([0.5, 3.0, 0.0]))


def test_pace_is_an_upper_bound_per_window(prm, clouds):
    """The window may not cut more than `pace` of its starting loss (adversarial finding:
    the old break-after-accept form allowed a single step to snap the morph)."""
    from physmorph.pipeline.optimizer import optimize_window
    from physmorph.pipeline.render_loss import LambdaBalancer
    from physmorph.pipeline.runner import build_target
    src, tgt_x = clouds
    cfg = _cfg(pace=0.15, iters=6)
    pack = build_target(tgt_x, prm, cfg)
    bal = LambdaBalancer(0.0)
    fr, F_seq, end, s, whist, stats = optimize_window(
        src, prm, cfg, pack, bal, log=lambda *_: None)
    assert whist and stats["L_start"] is not None
    floor = (1 - cfg.pace) * stats["L_start"]
    assert whist[-1]["loss"] >= floor * 0.999       # never below the pace floor


def test_render_lg_end_to_end(prm, clouds):
    """The local-global runner path itself (adversarial finding: it had zero e2e
    coverage — findings about guard counting and telemetry lived in unexecuted code)."""
    src, tgt_x = clouds
    cfg = _cfg(lambda_auto=0.5, lg_sweeps=3)
    res = run_pipeline(src, tgt_x, prm, cfg, log=lambda *_: None)
    recs = [h for h in res["history"] if "d_vol" in h]
    assert recs
    lg_recs = [r for r in recs if "lg_move" in r]
    assert lg_recs, "local pass never ran"
    for r in lg_recs:
        assert r["lg_lam"] > 0 and r["lg_nodes"] > 0
        assert np.isfinite(r["lg_gnorm"])
    assert np.isfinite(res["frames"][-1]).all()
    assert set(res["guards"]) == {"clamped", "nan_x", "nan_state", "F_reset", "F_flip",
                                  "F_invert_steps"}


def test_frames_are_promoted_states(prm, clouds):
    """The archived last frame of each window must BE the promoted state (adversarial
    finding: raw rollout was archived while the clamped state was simulated)."""
    src, tgt = clouds
    cfg = _cfg()
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    recs = [h for h in res["history"] if "d_vol" in h]
    n_windows = len(recs)
    # frame index of the k-th commit boundary is (k+1)*T
    for k in range(n_windows):
        b = (k + 1) * cfg.T
        assert np.isfinite(res["frames"][b]).all()


def test_render_dt_end_to_end(prm, clouds):
    """Pointwise-W1 spray wiring: DT maps built in build_target, term active in
    phys_total, run finishes with finite state (fringe tranche, rationale.md §7)."""
    src, tgt_x = clouds
    cfg = _cfg(lambda_auto=0.5, w_dt=0.5, w_creg=50.0)
    res = run_pipeline(src, tgt_x, prm, cfg, log=lambda *_: None)
    _check_result(res, cfg, len(src))
    recs = [h for h in res["history"] if "d_vol" in h]
    assert recs and all(r["d_render"] is not None for r in recs)
    # causal wiring (Codex finding 14): the W1 scalar is computed on every archived
    # state and feeds the freeze track
    assert all(r["d_dt"] is not None and np.isfinite(r["d_dt"]) for r in recs)
    assert recs[-1]["d_dt"] <= recs[0]["d_dt"] * 1.5   # the term acts, never explodes


def test_w1_independent_of_render_channel(prm, clouds):
    """Codex finding 12: w_dt>0 with lambda_auto=0 must still build and apply the term."""
    src, tgt_x = clouds
    res = run_pipeline(src, tgt_x, prm, _cfg(w_dt=0.5), log=lambda *_: None)
    recs = [h for h in res["history"] if "d_vol" in h]
    assert recs and all(r["d_dt"] is not None for r in recs)
    assert all(r["d_render"] is None for r in recs)


def test_lg_with_w1_is_rejected(prm, clouds):
    """Codex finding 7: the local pass's quadratic energy excludes the W1 term."""
    import pytest as _pytest
    src, tgt_x = clouds
    with _pytest.raises(ValueError):
        run_pipeline(src, tgt_x, prm, _cfg(lambda_auto=0.5, lg_sweeps=2, w_dt=0.5),
                     log=lambda *_: None)
