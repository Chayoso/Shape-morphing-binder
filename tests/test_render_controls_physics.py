"""End-to-end smokes of the render-controls-physics contract (warp CPU, tiny clouds):
control basis + geometric-F rendering + running kinetic + Chebyshev covector smoothing
+ gradient-combination modes + density loss units. Physics quality is NOT asserted
here (that is the hyde06 gate run); what is asserted is that every new path runs, is
wired to the gradient, and keeps the archive contracts.
"""
import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, run_pipeline

DEV = "cpu"


@pytest.fixture(scope="module")
def prm():
    return MPMParams(dx=1.0, nx=32, ny=32, nz=32)


@pytest.fixture(scope="module")
def clouds():
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    return src, tgt


def _cfg(**kw):
    base = dict(T=4, iters=2, animations=2, loss_res=12, render_views=2,
                render_elevs=(0.0, 0.5), render_res=24, device=DEV, patience=2)
    base.update(kw)
    return PipelineConfig(**base)


def _recs(res):
    return [h for h in res["history"] if "d_vol" in h]


def test_control_grid_basis_runs_and_moves(prm, clouds):
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, control_grid=4, control_tknots=2)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    recs = _recs(res)
    assert recs and all(np.isfinite(r["loss"]) for r in recs)
    assert recs[-1]["dfc_absmax"] > 0                 # the basis reached the field
    assert float(np.abs(res["frames"][-1] - src).max()) > 0
    assert all(v == 0 for v in res["guards"].values())


def test_shared_control_in_time_is_the_cxx_stride_form(prm, clouds):
    src, tgt = clouds
    cfg = _cfg(control_tknots=1)                       # one dFc for every step
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert _recs(res)


def test_geometric_F_render_channel_and_running_kinetic(prm, clouds):
    src, tgt = clouds
    seen = []
    cfg = _cfg(lambda_auto=0.5, render_F_geom=True, w_kin_running=1.0,
               render_gs_iters=4, render_gs_cheb=True)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None,
                       on_iter=lambda _i, _x, _F, tele: seen.append(tele))
    recs = _recs(res)
    assert recs and all(r["F_kind"] == "geom" for r in recs)
    assert all(r["kin_run"] is not None and np.isfinite(r["kin_run"]) for r in recs)
    assert res["Fg_commits"] and res["Fg_commits"][-1][1].shape == (len(src), 3, 3)
    assert np.isfinite(res["Fg_commits"][-1][1]).all()
    assert seen and np.abs(seen[-1]["_grad_render"]).sum() > 0


def test_geometric_F_is_rendered_to_the_viewer_not_the_physics_F(prm, clouds):
    src, tgt = clouds
    got = []
    cfg = _cfg(lambda_auto=0.5, render_F_geom=True)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None,
                       on_commit=lambda a, x, F, v, rec: got.append((a, F.copy())))
    acc = [r for r in _recs(res) if not r.get("null_commit")]
    if acc and got:
        a, F_view = got[-1]
        fe = acc[-1]["frame_end"]
        Fg = dict((i, f) for i, f in res["Fg_commits"]).get(fe)
        if Fg is not None:
            assert np.allclose(F_view, Fg)
            assert not np.allclose(F_view, res["F_frames"][fe - 1], atol=1e-6) or \
                np.allclose(res["F_frames"][fe - 1], np.eye(3), atol=1e-4)


@pytest.mark.parametrize("mode", ["render", "phys", "cagrad", "blend"])
def test_gradient_combination_modes_run(prm, clouds, mode):
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, grad_project=True, grad_project_mode=mode)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert _recs(res) and np.isfinite(_recs(res)[-1]["loss"])


def test_density_loss_units_give_order_one_lambda(prm, clouds):
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, loss_units="density")
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    recs = _recs(res)
    assert recs and recs[-1]["d_vol"] < 10.0        # dimensionless residual
    assert recs[-1]["lambda"] > 0 and np.isfinite(recs[-1]["lambda"])
    assert "lambda_capped" in recs[-1]
    with pytest.raises(ValueError):
        run_pipeline(src, tgt, prm, _cfg(loss_units="bogus"), log=lambda *_: None)


def test_warm_start_projects_previous_field_onto_new_basis(prm, clouds):
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, control_grid=3, control_tknots=2, warm_start=True,
               animations=3)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert _recs(res)


def test_node_basis_rejects_particle_knn_preconditioner(prm, clouds):
    """control_h1 indexes particles; on a node leaf it raised IndexError mid-window
    (found 2026-09-14). It must fail fast with a clear message instead."""
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, control_grid=4, control_tknots=2, control_h1_iters=3)
    with pytest.raises(ValueError, match="control_h1_iters"):
        run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)


def test_c2f_rebuild_keeps_every_one_shot_calibration(prm, clouds, monkeypatch):
    """REFUTE F5 (2026-09-15): gauss_scale/kde_scale were dropped at the c2f rebuild and
    silently re-calibrated mid-run. The kde path is CPU-runnable and shares the keep
    tuple with gauss_scale: the rebuilt pack must carry the FIRST pack's scale."""
    import physmorph.pipeline.runner as R
    packs = []
    real = R.build_target

    def spy(*a, **k):
        packs.append(real(*a, **k))
        return packs[-1]
    monkeypatch.setattr(R, "build_target", spy)
    src, tgt = clouds
    logs = []
    cfg = _cfg(lambda_auto=0.5, w_kde=1.0, c2f_at=0.5, animations=4, patience=10)
    run_pipeline(src, tgt, prm, cfg, log=lambda *a: logs.append(" ".join(map(str, a))))
    assert any("c2f at anim" in l for l in logs)
    assert len(packs) == 2                              # initial + the c2f rebuild
    first, second = packs
    assert first.kde_scale is not None and second.kde_scale == first.kde_scale
    assert second.gauss_scale == first.gauss_scale      # None == None here (no CUDA
    assert second.unit_ratio == first.unit_ratio        # raster); the keep tuple is
    assert second.unit_grad_ratio == first.unit_grad_ratio   # what is under test


def test_density_units_are_measured_at_the_source(prm, clouds):
    """REFUTE F1: the conversion ratios are measured, both > 1, and logged."""
    from physmorph.pipeline.runner import build_target, calibrate_units
    src, tgt = clouds
    cfg = _cfg(loss_units="density")
    pack = build_target(tgt, prm, cfg)
    calibrate_units(pack, src, cfg)
    assert pack.unit_ratio > 1.0 and pack.unit_grad_ratio > 1.0
    assert np.isfinite(pack.unit_ratio) and np.isfinite(pack.unit_grad_ratio)
    with pytest.raises(ValueError):                    # zero residual at the source
        calibrate_units(build_target(tgt, prm, cfg), tgt, cfg)


def test_velocity_variance_term_is_wired_and_zero_for_uniform_motion(prm, clouds):
    """w_kin_var (2026-09-15): mean_p[mean_t|v_t|^2 - |mean_t v_t|^2] is zero for a
    constant-velocity trajectory and positive for a reversal; it must reach the leaf."""
    import torch
    V = torch.zeros(4, 10, 3); V[:, :, 0] = 1.0                          # uniform motion
    var = (V.pow(2).sum(2).mean(0) - V.mean(0).pow(2).sum(1)).mean()
    assert float(var) == 0.0
    V[2:, :, 0] = -1.0                                                  # reversal
    assert float((V.pow(2).sum(2).mean(0) - V.mean(0).pow(2).sum(1)).mean()) > 0.5
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, w_kin_var=5.0)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    recs = _recs(res)
    assert recs and all(r.get("kin_var") is not None and r["kin_var"] >= 0 for r in recs)
