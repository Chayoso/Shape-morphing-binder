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


def test_phase_velocity_cost_allows_progress_but_penalizes_released_drift():
    from physmorph.pipeline.optimizer import _velocity_variance
    v = torch.tensor([[[1., 0., 0.]], [[3., 0., 0.]],
                      [[0., 0., 0.]], [[0., 0., 0.]]], requires_grad=True)
    value = _velocity_variance(v, 2)
    assert float(value) == pytest.approx(.5)
    expected_grad = torch.tensor([[[-.5, 0., 0.]], [[.5, 0., 0.]],
                                  [[0., 0., 0.]], [[0., 0., 0.]]])
    torch.testing.assert_close(torch.autograd.grad(value, v)[0], expected_grad)
    assert float(_velocity_variance(v)) == pytest.approx(1.5)
    constant_phases = torch.cat((torch.ones(20, 2, 3), torch.zeros(20, 2, 3)))
    assert float(_velocity_variance(constant_phases, 20)) == 0.
    assert float(_velocity_variance(constant_phases)) == pytest.approx(.75)
    # A constant released velocity is drift, not rest, even with zero variance.
    drift = torch.ones(4, 1, 3, requires_grad=True)
    cost = _velocity_variance(drift, 2)
    assert float(cost) == pytest.approx(1.5)
    grad, = torch.autograd.grad(cost, drift)
    torch.testing.assert_close(grad[:2], torch.zeros(2, 1, 3))
    torch.testing.assert_close(grad[2:], torch.full((2, 1, 3), .5))
    assert float(_velocity_variance(drift)) == 0.


def test_settled_phase_objective_agrees_between_gradient_search_and_replay(prm, clouds, monkeypatch):
    import physmorph.pipeline.runner as runner
    optimize, windows = runner.optimize_window, []
    def checked(*args, **kwargs):
        logs = []
        kwargs['log'] = lambda msg: logs.append(msg)
        result = optimize(*args, **kwargs)
        assert not any('commit rollout failed' in msg for msg in logs)
        windows.append(result)
        return result
    monkeypatch.setattr(runner, 'optimize_window', checked)
    # This tiny fixture needs one more backtrack with the released kinetic cost.
    cfg = _cfg(solver_mode='settled_transport', lambda_auto=.5, grad_project=True,
               pace=0., layer_ctrl=True, warm_start=True, loss_units='density',
               w_kin_var=200., ot_iters=1600, max_ls_iters=12)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert len(windows) == 2
    assert all(w[-1]['accepted'] > 0 for w in windows)
    assert all(v == 0 for v in res['guards'].values())


@pytest.mark.parametrize('kde_weight', [0., .1])
def test_settled_delivery_merit_uses_current_geometry(prm, clouds, monkeypatch,
                                                    kde_weight):
    import physmorph.pipeline.runner as runner
    import physmorph.pipeline.optimizer as opt
    from scipy.spatial import cKDTree
    optimize, checked = runner.optimize_window, []

    def check(x, params, cfg, pack, bal, **kw):
        result = optimize(x, params, cfg, pack, bal, **kw)
        frames, _, end, _, _, stats = result
        assert stats['accepted'] > 0
        assert 'selection_merit' in stats
        xt = torch.as_tensor(frames[-1])
        vt = torch.as_tensor(end['v'])
        # All control/motion priors are disabled in this fixture. Independently
        # evaluate the final physical state, full DT and current nearest target.
        expected = pack.ot_scale * float(pack.grid_ot.state_energy(xt, pack.m, vt, cfg.T * params.dt))
        expected += bal.lam * float(opt.d_render(xt, pack.sils, pack.views,
            cfg.render_res, pack.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray))
        expected += cfg.w_dt / pack.unit_ratio * float(opt.d_w1(
            xt, pack.m, pack.dt3, pack.dtgmin, pack.dtdx, pack.dtdims))
        distance = cKDTree(pack.pts.numpy()).query(frames[-1])[0]
        expected += cfg.w_nn / pack.unit_ratio * float(np.sum(
            pack.m.numpy() * np.maximum(distance - cfg.nn_berth_k * pack.nn_spacing, 0.)))
        if cfg.w_kde:
            nbr = opt.kde_assign(xt, pack.pts, cfg.kde_k)
            expected += cfg.w_kde * pack.kde_scale * float(opt.d_kde(
                xt, pack.pts, nbr, pack.kde_h, pack.kde_rho_ref))
        assert stats['selection_merit'] == pytest.approx(expected, rel=2e-5, abs=1e-8)
        checked.append(stats['selection_merit'])
        return result

    monkeypatch.setattr(runner, 'optimize_window', check)
    cfg = _cfg(solver_mode='settled_transport', lambda_auto=.5, grad_project=True,
               animations=1, pace=0., loss_units='density', w_kin=0., w_ctrl=0., w_box=0.,
               w_dt=.2, dt_res=16, w_nn=.2, nn_far_k=1000.,
               w_kde=kde_weight)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert _recs(res)[0]['selection_merit'] == checked[0]


@pytest.mark.parametrize('solver_mode', ['legacy', 'settled_transport'])
def test_accepted_step_memory_is_scaled_and_only_used_in_settled_mode(prm, clouds, monkeypatch,
                                                                    solver_mode):
    import physmorph.pipeline.runner as runner
    optimize, first_steps = runner.optimize_window, []

    def checked(*args, **kw):
        pack = args[3]
        pack.settled_step = 1e-4
        kw['alpha_scale'] = .5
        result = optimize(*args, **kw)
        assert result[-2]
        first_steps.append(result[-2][0]['alpha'])
        if solver_mode == 'settled_transport':
            assert pack.settled_step == pytest.approx(result[-2][-1]['alpha'] / .5)
        else:
            assert pack.settled_step == 1e-4
        return result

    monkeypatch.setattr(runner, 'optimize_window', checked)
    cfg = _cfg(solver_mode=solver_mode, lambda_auto=.5, grad_project=True,
               animations=1, iters=1, pace=0., loss_units='density', adaptive_alpha=False)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert all(v == 0 for v in res['guards'].values())
    if solver_mode == 'settled_transport':
        assert 0 < first_steps[0] <= 5.5e-5 * (1 + 1e-12)
    else:
        assert first_steps[0] > 1e-3


@pytest.mark.parametrize('solver_mode', ['legacy', 'settled_transport'])
def test_replay_noise_floor_uses_loss_units_only_in_settled_mode(prm, clouds, monkeypatch,
                                                               solver_mode):
    import physmorph.pipeline.optimizer as opt
    import physmorph.pipeline.runner as runner
    variance, optimize = opt._velocity_variance, runner.optimize_window
    calls, noise, ratios = 0, [], []

    def noisy_variance(*args, **kw):
        nonlocal calls
        calls += 1
        value = variance(*args, **kw)
        # Reproduce a small mismatch between the two calibration rollouts on
        # deterministic CPU hardware. Physical noise is identical in both units.
        return value + 1e-6 if calls == 2 else value

    def measured(*args, **kw):
        # Isolate the replay floor from OT's separate gradient calibration.
        # The zero-control reference now has zero loss in either unit system.
        args[3].ot_scale = 0.
        ratios.append(args[3].unit_ratio)
        result = optimize(*args, **kw)
        noise.append(result[-1]['replay_rel'])
        return result

    monkeypatch.setattr(opt, '_velocity_variance', noisy_variance)
    monkeypatch.setattr(runner, 'optimize_window', measured)
    for units in ['legacy', 'density']:
        calls = 0
        cfg = _cfg(solver_mode=solver_mode, lambda_auto=.5, grad_project=True,
                   animations=1, iters=1, pace=0., loss_units=units,
                   phys_loss='ot_pace', ot_grid=True, ot_debias=True,
                   w_kin_var=200., w_box=0., w_ctrl=0., replay_calibrate=True)
        run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    # 200 * 1e-6 in physical loss units, independent of density conversion.
    # Legacy's existing literal-floor calibration intentionally stays unchanged.
    assert ratios[1] > 1.
    density_noise = 2e-4 if solver_mode == 'settled_transport' else 2e-4 / ratios[1]
    assert noise == pytest.approx([2e-4, density_noise], rel=2e-4)


@pytest.mark.parametrize('replay_noise,expect_commit', [(2e-6, True), (1., False)])
def test_settled_replay_calibration_distinguishes_noise_from_regression(prm, clouds, monkeypatch,
                                                                      replay_noise, expect_commit):
    import physmorph.pipeline.optimizer as opt
    import physmorph.pipeline.runner as runner
    variance = opt._velocity_variance
    optimize, packs = runner.optimize_window, []
    committed_step = False
    calls = 0

    def cached_window(*args, **kw):
        pack = args[3]
        pack.settled_step = .02  # Does not constrain the normal initial step.
        packs.append(pack)
        return optimize(*args, **kw)

    def accepted(*args):
        nonlocal committed_step
        committed_step = True

    def corrupt_final_replay(*args, **kw):
        nonlocal calls
        calls += 1
        value = variance(*args, **kw)
        if committed_step:
            return value + replay_noise
        return value + 1e-6 if calls == 2 else value

    monkeypatch.setattr(opt, '_velocity_variance', corrupt_final_replay)
    monkeypatch.setattr(runner, 'optimize_window', cached_window)
    cfg = _cfg(solver_mode='settled_transport', lambda_auto=.5, grad_project=True,
               animations=1, iters=1, pace=0., loss_units='density',
               w_kin_var=200., replay_calibrate=True, max_ls_iters=12)
    res = run_pipeline(*clouds, prm, cfg, on_iter=accepted, log=lambda *_: None)
    assert committed_step
    assert any(r.get('frame_end') for r in res['history']) == expect_commit
    assert any(r.get('null_commit') for r in res['history']) != expect_commit
    assert (packs[0].settled_step is not None) == expect_commit
    if not expect_commit:
        assert all(np.array_equal(frame, clouds[0]) for frame in res['frames'])


@pytest.mark.parametrize('c2f', [False, True])
def test_settled_delivers_best_merit_even_inside_legacy_tolerance(prm, clouds, monkeypatch, c2f):
    import physmorph.pipeline.runner as runner
    optimize, windows = runner.optimize_window, []

    def scored(*args, **kw):
        result = optimize(*args, **kw)
        assert result[-1]['accepted'] > 0
        # Resolution changes make the first score incomparable. The last two
        # differ by less than legacy tol, but the earlier one is still better.
        result[-1]['selection_merit'] = ([1e-6, .999, 1.001] if c2f
                                        else [1., .999, 1.001])[len(windows)]
        windows.append(result)
        return result

    monkeypatch.setattr(runner, 'optimize_window', scored)
    cfg = _cfg(solver_mode='settled_transport', lambda_auto=.5, grad_project=True,
               animations=3, patience=10, pace=0., loss_units='density',
               best_truncate=True, c2f_at=1/3 if c2f else 0., render_res_hi=28)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert len(windows) == 3
    assert res['truncation']['best_animation'] == 2
    np.testing.assert_array_equal(res['frames'][res['deliver_n'] - 1], windows[1][0][-1])


@pytest.mark.parametrize('solver_mode', ['legacy', 'settled_transport'])
@pytest.mark.parametrize('c2f', [False, True])
def test_delivery_merit_plateau_respects_mode_and_resolution(prm, clouds, monkeypatch,
                                                             solver_mode, c2f):
    import physmorph.pipeline.runner as runner
    optimize, windows = runner.optimize_window, []

    def scored(*args, **kw):
        result = optimize(*args, **kw)
        assert result[-1]['accepted'] > 0
        i = len(windows)
        # Raw render keeps improving, but the fixed-weight objective plateaus.
        # A resolution rebuild starts a new, incomparable score scale.
        result[-2][-1]['d_render'] = result[-2][-1]['d_sil'] = 1. / (i + 1)
        result[-1]['selection_merit'] = 10. if c2f and i >= 2 else 1.
        windows.append(result)
        return result

    monkeypatch.setattr(runner, 'optimize_window', scored)
    cfg = _cfg(solver_mode=solver_mode, lambda_auto=.5, grad_project=True,
               animations=4, patience=1, pace=0., loss_units='density',
               c2f_at=.5 if c2f else 0., render_res_hi=28)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    expected = 2 if solver_mode == 'settled_transport' and not c2f else 4
    assert len(windows) == len(_recs(res)) == expected
    if c2f:
        assert any('c2f_render_res' in r for r in res['history'])


def test_settled_nonfinite_delivery_merit_cannot_commit(prm, clouds, monkeypatch):
    import physmorph.pipeline.optimizer as opt
    import physmorph.pipeline.runner as runner
    optimize, seeds = runner.optimize_window, []

    def tracked(*args, **kw):
        seeds.append((kw['s_init'], kw['dfc_init']))
        return optimize(*args, **kw)

    monkeypatch.setattr(runner, 'optimize_window', tracked)
    monkeypatch.setattr(opt, 'd_nn_band_current',
                        lambda x, *args, **kw: x.new_tensor(float('nan')))
    cfg = _cfg(solver_mode='settled_transport', lambda_auto=.5, grad_project=True,
               animations=2, patience=3, pace=0., loss_units='density', w_nn=.2,
               opt_material=True, warm_start=True)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert any(r.get('null_commit') for r in res['history'])
    assert all(np.array_equal(frame, clouds[0]) for frame in res['frames'])
    assert len(seeds) == 2 and all(s is None and dc is None for s, dc in seeds)


@pytest.mark.parametrize('solver_mode', ['legacy', 'settled_transport'])
def test_delivery_merit_excludes_null_padding_only_in_settled_mode(prm, clouds, monkeypatch, solver_mode):
    import physmorph.pipeline.runner as runner
    optimize, windows = runner.optimize_window, []

    def one_commit(*args, **kw):
        if not windows:
            result = optimize(*args, **kw)
            assert result[-1]['accepted'] > 0
            windows.append(result)
            return result
        fr, fs, end, s, _, stats = windows[0]
        return fr, fs, end, s, [], dict(stats, accepted=0, grad_converged=False)

    monkeypatch.setattr(runner, 'optimize_window', one_commit)
    cfg = _cfg(solver_mode=solver_mode, lambda_auto=.5, grad_project=True,
               animations=3, patience=10, pace=0., loss_units='density')
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    accepted = next(r for r in res['history'] if r.get('frame_end'))
    padding = 0 if solver_mode == 'settled_transport' else 2
    assert len(res['frames']) == accepted['frame_end'] + padding
    expected = accepted['frame_end'] if solver_mode == 'settled_transport' else len(res['frames'])
    assert res['deliver_n'] == expected


@pytest.mark.parametrize('solver_mode', ['legacy', 'settled_transport'])
def test_null_retry_does_not_insert_simulation_time_in_settled_motion(prm, clouds, monkeypatch,
                                                                     solver_mode):
    import physmorph.pipeline.runner as runner
    optimize, windows = runner.optimize_window, []
    calls = 0

    def retry_once(*args, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            fr, fs, end, s, _, stats = windows[-1]
            return fr, fs, end, s, [], dict(stats, accepted=0, grad_converged=False)
        result = optimize(*args, **kw)
        assert result[-1]['accepted'] > 0
        windows.append(result)
        return result

    monkeypatch.setattr(runner, 'optimize_window', retry_once)
    cfg = _cfg(solver_mode=solver_mode, lambda_auto=.5, grad_project=True,
               animations=3, patience=10, pace=0., loss_units='density')
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    expected = list(windows[0][0])
    if solver_mode == 'legacy':
        expected.append(expected[-1])
    expected.extend(windows[1][0][1:])
    np.testing.assert_array_equal(np.stack(res['frames']), np.stack(expected))
    assert len(res['frames']) == len(res['F_frames'])
    null = next(r for r in res['history'] if r.get('null_commit'))
    assert bool(null.get('no_simulated_time')) == (solver_mode == 'settled_transport')


@pytest.mark.parametrize('term', ['w_fill', 'w_jdens'])
def test_settled_delivery_merit_rejects_changing_optional_targets(prm, clouds, term):
    cfg = _cfg(solver_mode='settled_transport', lambda_auto=.5, grad_project=True,
               pace=0., dt_res=16, jdens_res=16, **{term: .1})
    with pytest.raises(ValueError, match='fixed delivery merit'):
        run_pipeline(*clouds, prm, cfg, log=lambda *_: None)


def test_spatial_transport_quadrature_runs_actual_control_optimization(prm, clouds, monkeypatch):
    import physmorph.losses.ot as ot

    original, blurs = ot.grid_transport_displacement, []
    loss_call, values = ot.GridSinkhornLoss.__call__, []

    def checked(x, mass, target, origin, dx, dims, **kw):
        blurs.append((kw['eps'], dx))
        return original(x, mass, target, origin, dx, dims, **kw)

    monkeypatch.setattr(ot, 'grid_transport_displacement', checked)
    def checked_loss(self, current):
        out = loss_call(self, current)
        values.append(float(out.detach()))
        return out

    monkeypatch.setattr(ot.GridSinkhornLoss, '__call__', checked_loss)
    src, tgt = clouds
    cfg = _cfg(lambda_auto=.5, solver_mode='settled_transport', grad_project=True, pace=0.,
               ot_debias=True, ot_handoff=True, ot_samples=128, ot_iters=400,
               loss_units='density', layer_ctrl=True, layer_gate_ot=True, dfc_clip=.02)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert blurs and all(eps >= dx ** 2 for eps, dx in blurs)
    assert len(values) > cfg.animations * cfg.iters
    assert np.isfinite(values).all()
    assert any(h.get('accepted', 0) > 0 for h in res['history'])
    assert np.max(np.abs(res['frames'][-1] - src)) > 1e-5
    assert all(v == 0 for v in res['guards'].values())


def test_settled_transport_delivers_release_tail_and_keeps_calibration(prm, clouds, monkeypatch):
    import physmorph.pipeline.runner as runner
    from physmorph.mpm.traj import Trajectory
    from physmorph.losses.ot import GridSinkhornLoss
    optimize, run, energy = runner.optimize_window, Trajectory.run, GridSinkhornLoss.state_energy
    windows, cached, steps = [], [], []
    outer = False
    active_bal = None

    def checked_run(tr):
        if active_bal is not None and len(windows) > 0:
            # A rejected first window rolls back the balancer, not the calibration.
            assert active_bal.lam == cached[0]
        return run(tr)

    def checked_window(x, params, cfg, pack, bal, **kw):
        nonlocal outer, active_bal
        outer, active_bal = False, bal
        steps.append(getattr(pack, 'settled_step', None))
        assert cfg.ot_grid and cfg.pbr_target_mode == 'matched'
        if windows:
            kw['dfc_init'] = np.full((cfg.T, len(x), 3, 3), 1e-4, np.float32)
        result = optimize(x, params, cfg, pack, bal, **kw)
        assert len(result[0]) == 2 * cfg.T + 1
        assert result[-1]['dfc'].shape == (cfg.T, len(x), 3, 3)
        cached.append(bal.lam)
        windows.append(result)
        outer, active_bal = True, None
        return result

    def reject_first_outer(self, x, *args, **kw):
        if outer and len(windows) == 1:
            return x.new_tensor(float('inf'))
        return energy(self, x, *args, **kw)

    monkeypatch.setattr(Trajectory, 'run', checked_run)
    monkeypatch.setattr(runner, 'optimize_window', checked_window)
    monkeypatch.setattr(GridSinkhornLoss, 'state_energy', reject_first_outer)
    cfg = _cfg(solver_mode='settled_transport', lambda_auto=.5, grad_project=True,
               pace=0., layer_ctrl=True, warm_start=True, loss_units='density',
               outer_merit=True, animations=3, reject_stop=4, patience=4)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert len(windows) >= 2 and cached[0] > 0
    assert all(v == cached[0] for v in cached)
    assert any(r.get('outer_rejected') for r in res['history'])
    assert any(r.get('accepted', 0) > 0 for r in res['history'])
    assert all(v == 0 for v in res['guards'].values())
    assert steps[:2] == [None, None]  # Rejected first window cannot donate a step.
    assert len(steps) >= 3 and steps[2] is not None and steps[2] > 0


def test_settled_recipe_reports_effective_options_without_changing_defaults():
    import dataclasses
    cfg = PipelineConfig(solver_mode='settled_transport')
    reported = dataclasses.asdict(cfg)
    assert reported['ot_grid'] and reported['ot_debias']
    assert reported['phys_loss'] == 'ot_pace'
    assert reported['pbr_target_mode'] == 'matched'
    baseline = PipelineConfig()
    assert not baseline.ot_grid and baseline.pbr_target_mode == 'surface'


@pytest.mark.parametrize('solver_mode', ['legacy', 'settled_transport'])
@pytest.mark.parametrize('transport,merit', [(2., .8), (.8, 2.)])
def test_outer_brake_uses_delivery_cost_only_for_settled_mode(
        prm, clouds, monkeypatch, solver_mode, transport, merit):
    import physmorph.pipeline.runner as runner
    from physmorph.losses.ot import GridSinkhornLoss
    optimize, energy = runner.optimize_window, GridSinkhornLoss.state_energy
    windows, outer = [], False

    def scored(*args, **kwargs):
        nonlocal outer
        outer = False
        result = optimize(*args, **kwargs)
        assert result[-1]['accepted'] > 0
        result[-1]['selection_merit'] = merit if windows else 1.
        windows.append(result)
        outer = True
        return result

    def primary(self, x, *args, **kwargs):
        if outer:
            return x.new_tensor(transport if len(windows) == 2 else 1.)
        return energy(self, x, *args, **kwargs)

    monkeypatch.setattr(runner, 'optimize_window', scored)
    monkeypatch.setattr(GridSinkhornLoss, 'state_energy', primary)
    cfg = _cfg(solver_mode=solver_mode, lambda_auto=.5, grad_project=True,
               pace=0., loss_units='density', outer_merit=True,
               phys_loss='ot_pace', ot_grid=True, ot_debias=True)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    accepted = [h for h in res['history'] if 'frame_end' in h]
    should_accept = merit < 1. if solver_mode == 'settled_transport' else transport < 1.
    assert len(accepted) == (2 if should_accept else 1)
    assert bool(_recs(res)[-1].get('brake_reject')) != should_accept
    assert all(v == 0 for v in res['guards'].values())


def test_settled_delivery_progress_cannot_freeze_on_stalled_raw_tracks(prm, clouds, monkeypatch):
    import physmorph.pipeline.runner as runner
    from physmorph.losses.ot import GridSinkhornLoss
    optimize, energy = runner.optimize_window, GridSinkhornLoss.state_energy
    windows, outer = [], False

    def scored(*args, **kwargs):
        nonlocal outer
        outer = False
        result = optimize(*args, **kwargs)
        assert result[-1]['accepted'] > 0
        result[-1]['selection_merit'] = .8 ** len(windows)
        result[-2][-1].update(d_sil=1., d_render=1., kin=1.)
        windows.append(result)
        outer = True
        return result

    def primary(self, x, *args, **kwargs):
        return x.new_tensor(1.) if outer else energy(self, x, *args, **kwargs)

    monkeypatch.setattr(runner, 'optimize_window', scored)
    monkeypatch.setattr(GridSinkhornLoss, 'state_energy', primary)
    cfg = _cfg(solver_mode='settled_transport', lambda_auto=.5, grad_project=True,
               animations=4, patience=1, pace=0., loss_units='density', outer_merit=True)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert len(windows) == 4
    assert len([h for h in res['history'] if 'frame_end' in h]) == 4


@pytest.mark.parametrize('change', ['resolution', 'render_weight'])
@pytest.mark.parametrize('eject', [False, True])
def test_settled_outer_cost_starts_new_epoch_when_calibration_changes(prm, clouds, monkeypatch, change, eject):
    import physmorph.pipeline.runner as runner
    optimize, windows = runner.optimize_window, []

    def scored(*args, **kwargs):
        result = optimize(*args, **kwargs)
        assert result[-1]['accepted'] > 0
        result[-1]['selection_merit'] = 10. if windows else 1e-6
        if change == 'render_weight':
            # An initially zero render gradient leaves calibration pending;
            # a later positive calibration changes the objective's scale.
            result[-2][-1]['lambda'] = .5 if windows else 0.
        windows.append(result)
        return result

    monkeypatch.setattr(runner, 'optimize_window', scored)
    if eject:
        counts = iter([0, 0, 1, 0])  # candidate/start at each window
        monkeypatch.setattr(runner, '_iso_count', lambda *args: next(counts))
    cfg = _cfg(solver_mode='settled_transport', lambda_auto=.5, grad_project=True,
               pace=0., loss_units='density', outer_merit=True,
               eject_veto=eject,
               c2f_at=.5 if change == 'resolution' else 0., render_res_hi=28)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    accepted = [h for h in res['history'] if 'frame_end' in h]
    if eject:
        assert len(accepted) == 1
        assert _recs(res)[-1]['eject_reject'] == 1
        return
    assert len(accepted) == 2
    assert [h['selection_epoch'] for h in accepted] == [0, 1]
    assert accepted[1]['outer_gain'] is None
    assert res['deliver_n'] == accepted[1]['frame_end']


@pytest.mark.parametrize('bad,outer_merit', [(float('nan'), True), (float('inf'), False)])
def test_settled_runner_rejects_nonfinite_window_cost(prm, clouds, monkeypatch, bad, outer_merit):
    import physmorph.pipeline.runner as runner
    optimize = runner.optimize_window

    def corrupted(*args, **kwargs):
        result = optimize(*args, **kwargs)
        assert result[-1]['accepted'] > 0
        result[-1]['selection_merit'] = bad
        return result

    monkeypatch.setattr(runner, 'optimize_window', corrupted)
    cfg = _cfg(solver_mode='settled_transport', lambda_auto=.5, grad_project=True,
               animations=1, pace=0., loss_units='density', outer_merit=outer_merit)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert not any('frame_end' in h for h in res['history'])
    assert all(np.array_equal(x, clouds[0]) for x in res['frames'])


@pytest.mark.parametrize('bad', [float('nan'), 31.])
def test_settled_trajectory_rejects_midpoint_escape_even_with_safe_endpoint(prm, bad):
    from physmorph.pipeline.optimizer import _positions_in_domain
    x = torch.zeros(9, 3, 3)
    assert _positions_in_domain(x, prm)
    x[4, 0, 0] = bad
    assert not _positions_in_domain(x, prm)


def test_transport_calibration_survives_render_resolution_change(prm, clouds, monkeypatch):
    import physmorph.pipeline.runner as runner
    build, packs = runner.build_target, []
    optimize, starts = runner.optimize_window, []
    def tracked(*args, **kw):
        pack = build(*args, **kw)
        packs.append(pack)
        return pack
    def checked_window(*args, **kw):
        starts.append(getattr(args[3], 'settled_step', None))
        return optimize(*args, **kw)
    monkeypatch.setattr(runner, 'build_target', tracked)
    monkeypatch.setattr(runner, 'optimize_window', checked_window)
    cfg = _cfg(lambda_auto=.5, solver_mode='settled_transport',
               ot_samples=128, loss_units='density', c2f_at=.5, render_res_hi=28,
               grad_project=True, pace=0.)
    run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert len(packs) == 2
    assert packs[0].ot_scale == packs[1].ot_scale
    assert packs[0].grid_ot is packs[1].grid_ot
    assert all(p.settled_scale is not None for p in packs)
    assert packs[0].settled_scale is not packs[1].settled_scale
    assert all(getattr(p, 'settled_step', None) is not None for p in packs)
    assert starts == [None, None]  # A new render resolution needs a fresh search.


@pytest.mark.parametrize('outer_merit', [True, False])
def test_unsolved_transport_cannot_initialize_an_accepted_commit(prm, clouds, monkeypatch, outer_merit):
    import physmorph.pipeline.runner as runner
    from physmorph.losses.ot import GridSinkhornLoss
    optimize, energy = runner.optimize_window, GridSinkhornLoss.state_energy
    outer = False
    def checked_window(*args, **kwargs):
        nonlocal outer
        outer = False
        result = optimize(*args, **kwargs)
        outer = True
        return result
    def failed_outer_energy(self, x, *args, **kwargs):
        return x.new_tensor(float('inf')) if outer else energy(self, x, *args, **kwargs)
    monkeypatch.setattr(runner, 'optimize_window', checked_window)
    monkeypatch.setattr(GridSinkhornLoss, 'state_energy', failed_outer_energy)
    cfg = _cfg(animations=1, lambda_auto=.5, solver_mode='settled_transport',
               grad_project=True, pace=0., loss_units='density', outer_merit=outer_merit)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert any(h.get('outer_rejected') for h in res['history'])
    assert all(np.array_equal(frame, clouds[0]) for frame in res['frames'])


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


def test_material_coherence_prior_is_zero_for_affine_motion_and_reaches_the_leaf(prm, clouds):
    """w_coh: Laplacian of the window displacement over frozen source neighbours —
    zero for any affine motion (translation, rotation, uniform stretch), positive when one
    particle leaves its neighbours; wired to the control through the MPM adjoint."""
    import torch
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(9)
    x0 = rng.uniform(-1, 1, (400, 3)).astype(np.float32)
    nbr = torch.as_tensor(cKDTree(x0).query(x0, k=9)[1][:, 1:])
    x0t = torch.as_tensor(x0)
    A = torch.tensor([[1.3, 0.2, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.1]])
    xT = x0t @ A.T + torch.tensor([0.3, -0.2, 0.1])
    u = xT - x0t
    lap = (u - u[nbr].mean(1)).pow(2).sum(1).mean()
    # affine field: the neighbour mean of u equals u at the neighbourhood centroid, so the
    # residual is only the centroid offset times (A - I) — small for a symmetric kNN set
    assert float(lap) < 5e-3
    xT2 = xT.clone(); xT2[0] += torch.tensor([1.0, 0.0, 0.0])          # one particle races ahead
    u2 = xT2 - x0t
    assert float((u2 - u2[nbr].mean(1)).pow(2).sum(1).mean()) > 1e-3
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, w_coh=10.0)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert _recs(res) and np.isfinite(_recs(res)[-1]["loss"])


def test_bond_stretch_bound_is_one_sided_and_frontier_mask_dilates(prm, clouds):
    """w_bond: zero for rigid motion and for compression, positive only for stretch beyond
    (1+s0) of the window-start length; vol_frontier: the target is restricted to cells
    within one loss cell of the current occupancy."""
    import torch
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(12)
    x0 = rng.uniform(-1, 1, (300, 3)).astype(np.float32)
    nbr = cKDTree(x0).query(x0, k=9)[1][:, 1:]
    nbr_t = torch.as_tensor(nbr); x0t = torch.as_tensor(x0)
    sp = float(np.median(cKDTree(x0).query(x0, k=2)[0][:, 1]))
    d_src = np.linalg.norm(x0[nbr] - x0[:, None, :], axis=2)
    w_b = torch.as_tensor(np.exp(-d_src ** 2 / (2 * (2 * sp) ** 2)).astype(np.float32))
    lmax = (x0t[nbr_t] - x0t[:, None, :]).norm(dim=2) * 1.3

    def bond(xT):
        d = (xT[nbr_t] - xT[:, None, :]).norm(dim=2)
        return float((w_b * torch.relu(d - lmax).pow(2)).sum(1).mean() / sp ** 2)
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    assert bond(x0t @ R.T + 0.4) == 0.0                       # rigid motion
    assert bond(x0t * 0.6) == 0.0                              # compression is free
    assert bond(x0t * 1.2) == 0.0                              # stretch within s0 (1.3)
    assert bond(x0t * 2.0) > 0.0                               # beyond s0: penalised
    xr = x0t.clone(); xr[0] += torch.tensor([1.0, 0, 0])       # one runaway particle
    assert bond(xr) > 0.0
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, w_bond=10.0, vol_frontier=True)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert _recs(res) and np.isfinite(_recs(res)[-1]["loss"])
    # frontier mask: a 3^3 dilation of the occupancy on the loss grid
    occ = torch.zeros(1, 1, 6, 6, 6); occ[0, 0, 2, 2, 2] = 1.0
    m = torch.nn.functional.max_pool3d(occ, 3, stride=1, padding=1)
    assert int(m.sum()) == 27 and float(m[0, 0, 2, 2, 2]) == 1.0 and float(m[0, 0, 0, 0, 0]) == 0.0
