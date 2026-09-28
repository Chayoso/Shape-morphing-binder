"""CPU-only proposal driver arithmetic, fixed cohorts and actual callback wiring."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from scripts.probes.variance_proposal_audit import (
    analyze_proposals, native_alpha, pin_report, prepare_metrics, trajectory_summary, run_audited,
)
from scripts.probes.variance_gradient_audit import updated_balancer


def path_fixture():
    x0 = torch.tensor([[0., 0., 0.], [0., 1., 0.]])
    X = torch.stack([x0+torch.tensor([1., 0., 0.]), x0+torch.tensor([2., 0., 0.])])
    return dict(x0=x0, pins=torch.tensor([False, False]), positions=X,
                promoted=x0.clone(), physical_v=torch.zeros_like(X))


def test_final_phase_separates_raw_motion_pic_jump_and_net_drift():
    data = path_fixture()
    report = trajectory_summary(data, data, ~data['pins'], dt=.5, spacing=.25)
    assert report['raw_all']['wu']['rms'] == 1
    assert report['raw_interior']['sp']['rms'] == 4
    assert report['raw_final']['wu']['rms'] == 1
    assert report['pic_jump']['wu']['rms'] == 2
    assert report['saved_final']['wu']['rms'] == 1
    assert report['raw_path']['net_over_path']['median'] == 1
    assert report['saved_path']['net_over_path']['median'] == 0
    assert report['raw_path']['reversal']['fraction'] == 0
    assert report['saved_path']['reversal']['fraction'] == 1
    assert report['physical_velocity_wu_per_s']['max'] == 0
    assert report['raw_minus_dt_physical_v']['wu']['rms'] == 1
    assert report['endpoint_closure_max_wu'] == 0


def test_fixed_cohort_and_empty_are_not_reselected_from_small_motion():
    data = path_fixture()
    data['pins'][1] = True
    data['positions'][:, 1] += 100
    report = trajectory_summary(data, data, ~data['pins'], dt=.5, spacing=.25)
    assert report['particles'] == 1 and report['raw_all']['wu']['rms'] == 1
    assert not pin_report(data, data)['raw_exact']
    empty = trajectory_summary(data, data, torch.zeros(2, dtype=torch.bool), .5, .25)
    assert empty['particles'] == 0 and empty['saved_all']['wu']['rms'] is None
    assert empty['saved_path']['reversal']['fraction'] is None
    assert empty['saved_path']['net_over_path']['count'] == 0


def test_constant_velocity_path_has_no_pic_jump_or_advection_residual():
    data = path_fixture()
    data['promoted'] = data['positions'][-1].clone()
    data['physical_v'][:] = torch.tensor([2., 0., 0.])
    report = trajectory_summary(data, data, ~data['pins'], .5, .25)
    assert report['pic_jump']['wu']['max'] == 0
    assert report['raw_minus_dt_physical_v']['wu']['max'] == 0
    assert report['saved_path']['net_over_path']['median'] == 1


def audit_fixture(u_only=False):
    from physmorph.pipeline.grad_combine import pcgrad
    p = [torch.tensor([2., 0.]), torch.tensor([1.])]
    q = [torch.tensor([0., 3.]), torch.tensor([1.])]
    r = [torch.tensor([-2., 1.]), torch.tensor([2.])]
    state = dict(alpha_lam=.5, ema=.3, cap=1.2, cap_rel=2., lam=.4, capped=False)
    weight, _ = updated_balancer(state, p, pcgrad(p, r)[0])
    source = np.random.default_rng(8).uniform(-1., 1., (12, 3)).astype(np.float32)
    reference, mask, metadata = prepare_metrics(source, source, 'cpu')
    x0 = torch.from_numpy(source)
    positions = torch.stack([x0+.01, x0+.02])
    pins = torch.zeros(12, dtype=torch.bool)
    pins[0] = True
    positions[:, pins] = x0[pins]
    initial = dict(x0=x0, pins=pins, positions=positions,
                   promoted=positions[-1].clone(), physical_v=torch.zeros_like(positions))
    calls = []
    def evaluate(values, observable):
        calls.append((observable, [value.clone() for value in values]))
        shift = float(values[0].sum())*.001
        X = positions.clone()
        X[:, ~pins, 0] += shift
        return dict(controls_delta=[-.1*value for value in values], positions=X,
            promoted=X[-1].clone(), physical_v=torch.zeros_like(X),
            F=torch.eye(3).reshape(1, 9).repeat(12, 1), v=torch.zeros(12, 3), C=torch.zeros(12, 3, 3),
            state_ok=True, Jmin=1., merits=dict(physical=1.+shift, geometric=2.+shift),
            predicted_decrease=.1, required_decrease=.01, first_trial_merit_ok=False,
            alpha=.1, **{'lambda': weight}, observable=observable, stats_restore_exact=True)
    audit = dict(same_graph=True, alternate_forward_count=0, mode='render', grad_h1=False,
        gradients=dict(physical=p, geometric=q, render_raw=r, transport=[torch.tensor([.1, -.2]), torch.tensor([.7])]),
        leaf_names=['stress', 'surface_u'], balancer_state=state, layer_u_render_only=u_only,
        trial_settings=dict(alpha=.1, **{'lambda': weight}, initial_merits=dict(physical=1., geometric=2.),
                            unscaled_alpha=.1, adaptive_alpha=False, target_norm=1., min_alpha_scale=.1),
        trial_reference=initial, trial_evaluate=evaluate, win_index=23, N=12, T=2, dt=.5)
    return audit, reference, mask, metadata, calls


@pytest.mark.parametrize('u_only', [False, True])
def test_only_two_fixed_lambda_composites_are_evaluated_in_aba_order(u_only):
    audit, reference, mask, metadata, calls = audit_fixture(u_only)
    before = deepcopy(audit['balancer_state'])
    report = analyze_proposals(audit, reference, mask, metadata)
    assert [name for name, _ in calls] == ['physical', 'geometric', 'physical']
    p, q, r, t = [torch.cat(audit['gradients'][key]) for key in ('physical', 'geometric', 'render_raw', 'transport')]
    # Independent dense joint projection; q does not conflict with r in this fixture.
    rp = r-torch.dot(p, r)/torch.dot(p, p)*p
    weight = audit['trial_settings']['lambda']
    a, b = p+weight*rp+t, q+weight*r+t
    if u_only:
        a[-1], b[-1] = weight*rp[-1], weight*r[-1]
    torch.testing.assert_close(torch.cat(calls[0][1]), a, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(calls[1][1]), b, rtol=0, atol=0)
    assert all(torch.equal(x, y) for x, y in zip(calls[0][1], calls[2][1]))
    assert report['comparisons']['A_to_A_repeat']['controls_exact'] and report['contract_valid']
    assert not any(value['first_trial_merit_ok'] for value in report['trials'].values())
    assert audit['balancer_state'] == before
    assert report['trials']['A']['geometry']['target_fixed']
    assert report['trials']['B']['motion']['all_window_start_free']['particles'] == 11
    assert not report['primitive_gate_passed']


def test_changed_trial_gain_fails_instead_of_comparing_different_steps():
    audit, reference, mask, metadata, calls = audit_fixture()
    evaluate = audit['trial_evaluate']
    def changed(*args, **kwargs):
        trial = evaluate(*args, **kwargs)
        trial['alpha'] *= 2
        return trial
    audit['trial_evaluate'] = changed
    with pytest.raises(RuntimeError, match='Trial gain changed'):
        analyze_proposals(audit, reference, mask, metadata)


def test_native_adaptive_alpha_uses_composite_norm_but_does_not_add_trial():
    settings = dict(unscaled_alpha=.2, adaptive_alpha=True, target_norm=2., min_alpha_scale=.1)
    assert native_alpha([torch.tensor([3., 4.])], settings) == pytest.approx(.08)
    assert native_alpha([torch.tensor([300., 400.])], settings) == pytest.approx(.02)
    assert native_alpha([torch.zeros(2)], settings) == pytest.approx(.2)
    settings['adaptive_alpha'] = False
    assert native_alpha([torch.tensor([300., 400.])], settings) == pytest.approx(.2)


def test_unsafe_scalar_state_is_nullable_without_erasing_failure():
    audit, reference, mask, metadata, calls = audit_fixture()
    evaluate = audit['trial_evaluate']
    def unsafe(*args, **kwargs):
        trial = evaluate(*args, **kwargs)
        if kwargs['observable'] == 'geometric':
            trial.update(Jmin=float('nan'), state_ok=False)
            trial['merits']['geometric'] = float('inf')
        return trial
    audit['trial_evaluate'] = unsafe
    report = analyze_proposals(audit, reference, mask, metadata)
    assert report['trials']['B']['Jmin'] is None
    assert report['trials']['B']['merits']['geometric'] is None
    assert report['trials']['B']['state_ok'] is False
    assert report['contract_valid']  # Invalid proposed state is a result, not a changed experiment.
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('change_controls', [False, True])
def test_repeat_control_gate_is_exact_but_state_repeat_is_descriptive(change_controls):
    audit, reference, mask, metadata, calls = audit_fixture()
    evaluate = audit['trial_evaluate']
    def changed(*args, **kwargs):
        trial = evaluate(*args, **kwargs)
        if len(calls) == 3:
            if change_controls:
                trial['controls_delta'][0][0] += 1e-5
            else:
                trial['positions'][:, 1:, 0] += .001
                trial['promoted'][1:, 0] += .001
        return trial
    audit['trial_evaluate'] = changed
    report = analyze_proposals(audit, reference, mask, metadata)
    assert report['contract_valid'] is (not change_controls)
    if not change_controls:
        assert report['comparisons']['A_to_A_repeat']['state']['positions']['difference_l2'] > 0
    assert report['cohorts']['all_window_start_free']['particles'] == 11


def test_small_real_noninitial_window_preserves_baseline_and_actual_lambda():
    from physmorph.pipeline import PipelineConfig, run_pipeline
    from physmorph.mpm.state import MPMParams
    source = np.random.default_rng(31).uniform(-1.3, 1.3, (96, 3)).astype(np.float32)
    target = (source*[1.2, .9, 1.05]+[.15, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=2, animations=2, stop_after_windows=2,
        loss_res=12, render_views=2, render_elevs=(0., .5), render_res=24,
        device='cpu', patience=5, commit_pic=True, commit_pic_objective=True,
        outer_render_committed=True, body_ctrl=True, body_terminal_ctrl=True,
        layer_ctrl=True, layer_relax=True, lambda_auto=.3,
        grad_project=True, grad_project_mode='render', w_kin=.2, w_kin_var=.7,
        w_ctrl=0., w_box=0., max_ls_iters=1, adaptive_alpha=False, alpha=1e-4, loss_units='density')
    prm = MPMParams(dx=1., nx=32, ny=32, nz=32)
    plain = run_pipeline(source, target, prm, deepcopy(cfg), log=lambda *_: None)
    actual, report = run_audited(source, target, prm, deepcopy(cfg), selected_index=1, log=lambda *_: None)
    assert report['observation_valid'] and report['baseline_lambda_matches_production_exactly']
    assert report['selected_outer_accepted'] and report['selected_commit']['accepted_ordinal'] == 2
    assert report['audit']['comparisons']['A_to_A_repeat']['controls_exact']
    assert report['audit']['leaf_names'] == ['stress', 'body', 'surface_u']
    assert all(trial['prepared_inputs_exact'] and trial['prepared_input_names']
               for trial in report['audit']['trials'].values())
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(np.stack(actual['frames']), np.stack(plain['frames']))
    assert not any(actual['guards'].values())
