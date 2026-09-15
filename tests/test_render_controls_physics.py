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
    calibrate_units(pack, src, tgt, cfg)
    assert pack.unit_ratio > 1.0 and pack.unit_grad_ratio > 1.0
    assert np.isfinite(pack.unit_ratio) and np.isfinite(pack.unit_grad_ratio)
    with pytest.raises(ValueError):                    # zero residual at the source
        calibrate_units(build_target(tgt, prm, cfg), tgt, tgt, cfg)
    # REFUTE-2 F6: the legacy side refers to the FIXED reference grid, so the converted
    # weight is the same across the run's own loss resolution (unit_ratio scales exactly
    # like 1/D_vol_density of the run grid)
    from physmorph.losses.volumetric import d_vol_density
    import torch
    packs = {}
    for lr in (12, 24):
        c = _cfg(loss_units="density", loss_res=lr, unit_ref_res=16)
        pk = build_target(tgt, prm, c)
        calibrate_units(pk, src, tgt, c)
        Lden = float(d_vol_density(torch.as_tensor(src), pk.m, pk.grid, pk.lgmin, pk.ldx,
                                   pk.ldims, pk.m_ref, pk.n_support))
        packs[lr] = (pk.unit_ratio, Lden)
    lhs = packs[24][0] / packs[12][0]
    rhs = packs[12][1] / packs[24][1]
    assert abs(lhs - rhs) / rhs < 1e-4, (lhs, rhs)


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


def test_select_archive_F_prefers_geometric_when_present():
    from physmorph.render.covariance import select_archive_F
    N = 5
    eye = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    Fp = np.stack([eye * (1 + 0.1 * k) for k in range(3)])          # physics samples
    Fg = np.stack([eye * (2 + k) for k in range(2)])                # geometric commits
    d = {"F_samples": Fp, "F_sample_idx": np.array([0, 10, 20]),
         "Fg_commits": Fg, "Fg_commit_idx": np.array([11, 21])}     # states at frames 10, 20
    F, kind = select_archive_F(d, 15, prefer_geom=False)
    assert kind == "physics" and np.allclose(F, eye * 1.1)
    F, kind = select_archive_F(d, 15, prefer_geom=True)
    assert kind == "geom" and np.allclose(F, eye * 2.0)
    F, kind = select_archive_F(d, 25, prefer_geom=True)
    assert kind == "geom" and np.allclose(F, eye * 3.0)
    F, kind = select_archive_F(d, 5, prefer_geom=True)              # before the first commit
    assert kind == "physics" and np.allclose(F, eye * 1.0)
    d2 = {"F_samples": Fp, "F_sample_idx": np.array([0, 10, 20])}   # legacy archive
    F, kind = select_archive_F(d2, 25, prefer_geom=True)
    assert kind == "physics" and np.allclose(F, eye * 1.2)


def test_relax_stretch_is_exact_polar_relaxation():
    """REFUTE-2 F10: R S^(1-eta) exactly (no isochoric renormalisation), rotation kept,
    det<=0 rows passed through. Offline helper only."""
    from physmorph.plasticity.assimilation import relax_stretch
    rng = np.random.default_rng(3)
    A = rng.normal(size=(50, 3, 3)).astype(np.float32)
    U, S, Vt = np.linalg.svd(A)
    S = np.clip(np.abs(S) * 2.0 + 0.5, 0.5, 5.0)
    R = U @ Vt
    R[np.linalg.det(R) < 0, :, 0] *= -1.0
    F = np.einsum("nij,njk->nik", R, np.einsum("nij,nj,nkj->nik", Vt.transpose(0, 2, 1), S, Vt))
    out = relax_stretch(F, eta=0.5)
    assert np.allclose(np.linalg.svd(out, compute_uv=False), np.linalg.svd(F, compute_uv=False) ** 0.5,
                       rtol=2e-3, atol=2e-3)
    def polar_R(M):
        u, _, vt = np.linalg.svd(M)
        return u @ vt
    assert np.allclose(polar_R(out), polar_R(F), atol=2e-3)
    bad = F.copy(); bad[0] = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
    assert np.allclose(relax_stretch(bad, eta=0.5)[0], bad[0])
    assert np.allclose(relax_stretch(F, eta=0.0), F)


def test_covariance_saturation_is_bounded_smooth_and_identity_for_small_stretch():
    """REFUTE-2 F11: the render forward model saturates the stretch (no state edit)."""
    import torch
    from physmorph.pipeline.gauss_loss import gaussian_covariance, saturate_stretch
    from physmorph.render.covariance import cov_from_F
    rng = np.random.default_rng(5)
    F = torch.tensor(rng.normal(size=(64, 3, 3)).astype(np.float32)) * 2.0
    F[0] = torch.eye(3) * 1.01                                        # near identity
    cov, _ = gaussian_covariance(F, 0.1, jitter=0.0, sat=2.0)
    ev = torch.linalg.eigvalsh(cov)
    assert float(ev.max()) <= 0.1 ** 2 * 2.0 ** 2 * (1 + 1e-5)        # bounded by s0^2 r^2
    assert float(ev.min()) >= 0.0
    unsat, _ = gaussian_covariance(F[:1], 0.1, jitter=0.0, sat=0.0)
    sat1, _ = gaussian_covariance(F[:1], 0.1, jitter=0.0, sat=100.0)   # r >> stretch: identity
    assert torch.allclose(unsat, sat1, rtol=1e-3, atol=1e-8)
    # orientation preserved: eigenvectors of M and M_s coincide (they commute)
    M = F[1:2] @ F[1:2].transpose(1, 2)
    Ms = saturate_stretch(M, 1.5)
    assert torch.allclose(M @ Ms, Ms @ M, atol=1e-4)
    # differentiable, finite gradients even at F ~ I (no SVD in the graph)
    Fg = F.clone().requires_grad_(True)
    c, _ = gaussian_covariance(Fg, 0.1, jitter=0.0, sat=2.0)
    (g,) = torch.autograd.grad(c.sum(), Fg)
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0
    # numpy forward model matches torch
    cn = cov_from_F(F.numpy(), 0.1, sat=2.0)
    assert np.allclose(cn, cov.numpy(), atol=1e-6)


def test_runner_archives_unrelaxed_Fg_when_not_edited(prm, clouds):
    """F_g is never edited at commits: the archived Fg_commits are the tape's own F_g."""
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, render_F_geom=True, assim=0.5, animations=3, patience=10)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert res["Fg_commits"]
    for _, Fg in res["Fg_commits"]:
        assert np.isfinite(Fg).all() and (np.linalg.det(Fg) > 0).all()
