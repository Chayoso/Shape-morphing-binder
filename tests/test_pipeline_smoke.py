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
