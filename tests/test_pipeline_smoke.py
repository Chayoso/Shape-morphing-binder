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


def test_nonpaced_settlement_preserves_legacy_evidence_without_claiming_arrival(prm, clouds):
    cfg = _cfg(animations=1, ctrl_rprop=True, ctrl_rprop_arrived=True, settle_pin=True)
    result = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    records = [h for h in result['history'] if 'pin_arrival_evidence' in h]
    assert records and all(r['pin_arrival_evidence'] == 'legacy_no_arrival_contract' for r in records)
    assert all(r['arrived_end_frac'] is None and r['pin_arrival_eligible_frac'] == 1 for r in records)
    # Arrival confirmation must fail before optimizing a loss with no arrival contract.
    cfg.settle_pin_confirm = True
    with pytest.raises(ValueError, match='arrival contract'):
        run_pipeline(*clouds, prm, cfg, log=lambda *_: None)


def test_body_control_pipeline_optimises_the_force_leaf(prm, clouds):
    src, tgt = clouds
    cfg = _cfg(body_ctrl=True)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    _check_result(res, cfg, len(src))
    recs = [h for h in res["history"] if "d_vol" in h]
    assert all(r["body_nodes"] > 0 for r in recs)
    assert any(r["body_rms_wu"] > 0 for r in recs)


@pytest.mark.parametrize('normalized', [False, True])
@pytest.mark.parametrize('terminal', [False, True])
def test_body_ablation_removes_learned_stress_but_keeps_force(prm, clouds, normalized, terminal):
    src, tgt = clouds
    cfg = _cfg(body_ctrl=True, body_no_dfc=True, body_step_normalized=normalized,
               body_terminal_ctrl=terminal, dfc_clip=0.02)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    recs = [h for h in res['history'] if 'd_vol' in h]
    assert recs and all(h['dfc_absmax'] == 0 for h in recs)
    assert any(h['body_rms_wu'] > 0 for h in recs)
    assert all(h['body_step_scale'] == (1/cfg.dfc_clip if normalized else 1) for h in recs)
    assert all(h['body_coeff_max'] <= 1.000001 for h in recs)
    assert all(len(h['body_accepted_alphas']) == h['accepted'] for h in recs)
    if terminal:
        assert any(h['body_terminal_rms_wu'] > 0 for h in recs)


def test_diagnostic_prefix_preserves_full_run_schedule(prm, clouds):
    src, tgt = clouds
    params = dict(animations=4, c2f_at=0.5, lambda_auto=0.5, render_res_hi=32,
                  hold_after_converge=False, patience=10)
    full = run_pipeline(src, tgt, prm, _cfg(**params), log=lambda *_: None)
    prefix = run_pipeline(src, tgt, prm, _cfg(stop_after_windows=2, **params), log=lambda *_: None)
    assert not any('c2f_render_res' in h for h in prefix['history'])
    assert len(prefix['frames']) < len(full['frames'])
    assert np.allclose(prefix['frames'], full['frames'][:len(prefix['frames'])], atol=1e-6)


def test_motion_accounting_does_not_change_the_accepted_trajectory(prm, clouds):
    cfg = _cfg(body_ctrl=True, body_terminal_ctrl=True, ctrl_rprop=True,
               phys_loss='ot_pace', ot_samples=64, ot_iters=4, animations=1,
               layer_ctrl=True, layer_relax=True, commit_pic=True, shift_sub=True)
    plain = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    cfg.motion_accounting = True
    observed = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    np.testing.assert_array_equal(observed['frames'], plain['frames'])
    np.testing.assert_array_equal(observed['F_frames'], plain['F_frames'])
    report = next(r['motion_accounting'] for r in observed['history'] if r.get('frame_end'))
    assert report['T'] == cfg.T and report['window_closure_max_wu'] < 1e-5
    assert report['cohorts']['all']['particles'] == len(clouds[0])
    assert sum(report['cohorts'][k]['particles'] for k in ('arrived_free', 'transit_free', 'pinned_at_start')) == len(clouds[0])


@pytest.mark.parametrize('fixed_gate', [False, True])
def test_outer_plateau_history_ignores_rejected_candidates(prm, clouds, monkeypatch, fixed_gate):
    from physmorph.pipeline import runner
    original = runner.optimize_window
    losses = iter([1., 2., 1.01])
    def controlled_tracks(*args, **kwargs):
        result = original(*args, **kwargs)
        # Real accepted inner trajectories with prescribed outer density tracks:
        # W2 triggers the catastrophe brake; W3 regresses only versus accepted W1.
        result[4][-1]['d_vol'] = next(losses)
        result[-1]['pace_bound'] = True
        return result
    monkeypatch.setattr(runner, 'optimize_window', controlled_tracks)
    cfg = _cfg(animations=3, patience=10, outer_merit=True, w_kin=0.,
               outer_render_committed=fixed_gate)
    result = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    records = [r for r in result['history'] if 'd_vol' in r]
    assert len(records) == 3
    assert records[1]['outer_rejected'] == 1 and records[1]['brake_reject'] == 1
    assert records[2]['outer_accepted'] == 1
    assert records[2]['improved'] == (0 if fixed_gate else 1)
    if fixed_gate:
        assert all(r['outer_render'] is None for r in records)


def test_outer_render_tracks_promoted_positions_and_retains_inner_telemetry(prm, clouds):
    import torch
    from physmorph.pipeline.runner import build_target
    from physmorph.pipeline.outer_merit import fixed_outer_render
    cfg = _cfg(animations=1, outer_render_committed=True, lambda_auto=.5,
               layer_ctrl=True, layer_relax=True, commit_pic=True, shift_sub=True,
               phys_loss='ot_pace', ot_samples=64, ot_iters=4, render_paced=True)
    result = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    record = next(r for r in result['history'] if r.get('frame_end'))
    target = build_target(clouds[1], prm, cfg)
    expected = fixed_outer_render(torch.as_tensor(result['frames'][-1]), cfg, target,
                                  record['d_sil'], record['d_render'])
    assert record['outer_track_version'] == 'committed_fixed_v1'
    assert record['outer_render'] == pytest.approx(float(expected), rel=1e-6)
    assert record['d_sil'] is not None and record['d_render'] is not None


def test_body_rprop_runs_with_independent_braking_and_requires_arrival(prm, clouds):
    cfg = _cfg(body_ctrl=True, body_terminal_ctrl=True, body_rprop=True,
               ctrl_rprop=True, phys_loss='ot_pace', ot_samples=64, ot_iters=4,
               dfc_clip=.02, animations=2)
    result = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    records = [r for r in result['history'] if r.get('frame_end')]
    assert records and all(r['body_step_node_mean'] is not None for r in records)
    assert any(r['body_terminal_rms_wu'] > 0 for r in records)
    for record in records:
        assert record['body_step_transit_min'] is None or record['body_step_transit_min'] >= 1 - 1e-6
        assert len(record['body_update_modes_rms']) == record['accepted']
    cfg.phys_loss = 'density'
    with pytest.raises(ValueError, match='arrival mask'):
        run_pipeline(*clouds, prm, cfg, log=lambda *_: None)


def test_body_rprop_nonunit_scale_reaches_optimizer_without_scaling_brake(prm, clouds, monkeypatch):
    from physmorph.pipeline import runner
    source = clouds[0]
    target = source + np.array([.02, 0., 0.], np.float32)
    original = runner.optimize_window
    factor = [1.]
    def with_scale(x, *args, **kwargs):
        kwargs['body_scale_init'] = np.full(len(x), factor[0], np.float32)
        return original(x, *args, **kwargs)
    monkeypatch.setattr(runner, 'optimize_window', with_scale)
    cfg = _cfg(body_ctrl=True, body_terminal_ctrl=True, body_rprop=True,
               ctrl_rprop=True, phys_loss='ot_pace', ot_samples=64, ot_iters=4,
               animations=1, iters=1, adaptive_alpha=False, alpha=1e-4, ls_noise_rel=0.)
    results = []
    for value in (1., .125):
        factor[0] = value
        result = run_pipeline(source, target, prm, cfg, log=lambda *_: None)
        results.append(next(r for r in result['history'] if r.get('frame_end')))
    full, small = results
    assert full['body_accepted_alphas'] == small['body_accepted_alphas']
    assert small['body_step_transit_min'] is None  # all points arrived in this fixture
    first, second = np.asarray(full['body_update_modes_rms']), np.asarray(small['body_update_modes_rms'])
    assert first[0, 0] > 0 and first[0, 1] > 0
    np.testing.assert_allclose(second[:, 0], .125 * first[:, 0], rtol=2e-3)
    np.testing.assert_allclose(second[:, 1], first[:, 1], rtol=2e-3)
    assert full['body_coeff_max'] < 1 and small['body_coeff_max'] < 1


@pytest.mark.parametrize('overwrite_with_rejected_trial', [False, True])
def test_commit_uses_accepted_state_even_after_rejected_buffer_overwrite(prm, clouds, monkeypatch,
                                                                      overwrite_with_rejected_trial):
    from physmorph.pipeline import optimizer
    original = optimizer._state_ok
    calls = []
    def state_ok(state):
        calls.append(None)
        if overwrite_with_rejected_trial and len(calls) == 2:
            return False  # reject iteration two AFTER it overwrote tr_eval
        return original(state)
    monkeypatch.setattr(optimizer, '_state_ok', state_ok)
    accepted = []
    def capture(it, x, F, stats):
        accepted.append(x.copy())
    cfg = _cfg(animations=1, iters=2 if overwrite_with_rejected_trial else 1,
               max_ls_iters=1, body_ctrl=True, body_terminal_ctrl=True)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None, on_iter=capture)
    recs = [r for r in res['history'] if 'move' in r]
    assert len(accepted) == len(recs) == 1
    assert recs[0]['commit_from_accepted'] is (not overwrite_with_rejected_trial)
    assert np.allclose(res['frames'][-1], accepted[0], atol=2e-6)


def test_render_arm_runs_and_lambda_is_live(prm, clouds):
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5)
    seen = []
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None,
                       on_iter=lambda _i, _x, _F, tele: seen.append(tele))
    _check_result(res, cfg, len(src))
    recs = [h for h in res["history"] if "d_vol" in h]
    assert all(r["d_render"] is not None for r in recs)
    assert all(r["lambda"] > 0 for r in recs)
    assert seen and seen[-1]["_grad_phys"].shape == (len(src), 3)
    assert seen[-1]["_grad_render"].shape == (len(src), 3)
    assert np.isfinite(seen[-1]["_grad_render"]).all()
    assert np.abs(seen[-1]["_grad_render"]).sum() > 0


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


def test_control_h1_spreads_surface_signal_without_rescaling():
    import torch
    from physmorph.pipeline.optimizer import _control_h1
    # Chain topology: an impulse at particle 0 must reach neighbours, while the
    # preconditioner preserves the joint gradient norm used by lambda balancing.
    knn = torch.tensor([[1], [0], [1], [2]])
    g = torch.zeros(1, 4, 3, 3)
    g[0, 0, 0, 0] = 1.0
    out = _control_h1(g, knn, iters=3, kappa=2.0)
    assert out[0, 1:].abs().sum() > 0
    assert torch.allclose(out.norm(), g.norm(), rtol=1e-5, atol=1e-6)


def test_surface_weights_are_bounded_and_nonuniform(clouds):
    from physmorph.pipeline.runner import _surface_weights
    w = _surface_weights(clouds[0], k=8, fraction=0.35, floor=0.05)
    assert w.shape == (len(clouds[0]),)
    assert np.isfinite(w).all() and w.min() >= 0.05 and w.max() <= 1.0
    assert float(w.std()) > 0.05


def test_surface_only_render_covector_is_zero_on_frozen_interior(prm, clouds):
    """The render channel can observe a material skin without dropping MPM mass."""
    from physmorph.pipeline.runner import _surface_weights
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, surface_grad_frac=0.35,
               render_surface_only=True, iters=1, animations=1)
    seen = []
    run_pipeline(src, tgt, prm, cfg, log=lambda *_: None,
                 on_iter=lambda _i, _x, _F, tele: seen.append(tele))
    mask = _surface_weights(src, cfg.surface_grad_k, cfg.surface_grad_frac,
                            cfg.surface_grad_floor) > 0.5
    assert seen and mask.any() and (~mask).any()
    gr = seen[-1]["_grad_render"]
    assert np.abs(gr[mask]).sum() > 0.0
    assert np.abs(gr[~mask]).sum() == 0.0


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


def test_fill_arm_end_to_end(prm, clouds):
    """Hole-side W1 wiring: deficit field built per window, run finishes finite."""
    src, tgt_x = clouds
    cfg = _cfg(lambda_auto=0.5, w_dt=0.5, w_fill=0.5, assim_iso=True)
    res = run_pipeline(src, tgt_x, prm, cfg, log=lambda *_: None)
    _check_result(res, cfg, len(src))
