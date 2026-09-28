"""Counterfactual direction algebra and actual small CPU pipeline admission."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.probes.variance_gradient_audit import (
    GradientAudit, analyze_gradient_audit, composite, run_audited, updated_balancer,
)


def fixture(mode='render', u_only=False):
    def split(values):
        return [torch.tensor(values[:2], dtype=torch.float32), torch.tensor(values[2:], dtype=torch.float32)]
    physical, geometric, render = split([2., 0., 1.]), split([0., 3., 1.]), split([-2., 1., 2.])
    return dict(gradients=dict(physical=physical, geometric=geometric,
        physical_repeat=[x.clone() for x in physical], geometric_repeat=[x.clone() for x in geometric],
        render_raw=render, transport=split([.1, -.2, .7])), leaf_names=['stress', 'surface_u'],
        balancer_state=dict(alpha_lam=.5, ema=.3, cap=1.2, cap_rel=2., lam=.4, capped=False),
        physical_variance=.2, geometric_variance=.5, effective_weight=2.,
        physics_core=10., geometric_physics_core=10.6,
        N=3, T=20, dt=1/240, win_index=0, mode=mode, grad_h1=False,
        layer_u_render_only=u_only, same_graph=True, alternate_forward_count=0,
        scope='Synthetic identical-prepared-graph fixture')


@pytest.mark.parametrize('u_only', [False, True])
def test_fixed_lambda_alone_does_not_isolate_physical_change(u_only):
    data = fixture(u_only=u_only)
    original = deepcopy(data)
    report = analyze_gradient_audit(data)
    assert report['pcgrad_conflict_baseline'] and not report['pcgrad_conflict_geometric']
    # Independent dense oracle: projection is JOINT across both leaves.
    p, q, r = (torch.cat(data['gradients'][key]) for key in ('physical', 'geometric', 'render_raw'))
    projected = r-torch.dot(p, r)/torch.dot(p, p)*p
    la = .7*.4 + .3*min(.5*float(p.norm())/float(projected.norm()), 1.2)
    lb = .7*.4 + .3*min(.5*float(q.norm())/float(r.norm()), 1.2)
    assert report['lambda_baseline'] == pytest.approx(la, rel=2e-7)
    assert report['lambda_geometric'] == pytest.approx(lb, rel=2e-7)
    transport = torch.cat(data['gradients']['transport'])
    stages = [p+la*projected+transport, q+la*projected+transport,
              q+la*r+transport, q+lb*r+transport]
    if u_only:
        for vector, weight, rend in zip(stages, [la, la, la, lb], [projected, projected, r, r]):
            vector[-1] = weight*rend[-1]
    assert report['directions']['adaptive']['joint']['difference_l2'] == pytest.approx(
        float((stages[-1]-stages[0]).double().norm()), rel=2e-6)
    assert report['stage_changes'][1]['joint']['difference_l2'] > 0
    partition = report['delta_partition']
    assert partition['closure']['difference_max_abs'] < 1e-6
    assert partition['gram_sum'] == pytest.approx(partition['total_delta_squared'], rel=3e-7)
    assert sum(partition['signed_projection_on_total']) == pytest.approx(1., abs=3e-7)
    assert data['balancer_state'] == original['balancer_state']
    for key, tensors in data['gradients'].items():
        for value, before in zip(tensors, original['gradients'][key]):
            assert torch.equal(value, before)
    if u_only:
        assert report['stage_changes'][0]['leaves'][1]['difference_l2'] == 0


def test_off_mode_and_transport_do_not_enter_balancer():
    data = fixture(mode='off')
    a = analyze_gradient_audit(data)
    data['gradients']['transport'] = [1000*x for x in data['gradients']['transport']]
    b = analyze_gradient_audit(data)
    assert a['lambda_baseline'] == b['lambda_baseline']
    assert a['lambda_geometric'] == b['lambda_geometric']
    assert a['stage_changes'][1]['joint']['difference_l2'] == 0
    assert a['directions']['baseline']['joint']['a_l2'] != b['directions']['baseline']['joint']['a_l2']


def test_live_ema_cap_state_is_cloned_not_reinitialized():
    state = dict(alpha_lam=.5, ema=.3, cap=.6, cap_rel=999., lam=.4, capped=False)
    saved = deepcopy(state)
    value, after = updated_balancer(state, [torch.tensor([20.])], [torch.tensor([1.])])
    assert value == pytest.approx(.7*.4+.3*.6)
    assert after['cap'] == .6 and after['capped']
    assert state == saved


def test_zero_directions_and_absent_transport_stay_explicit():
    data = fixture(mode='off')
    for key, values in data['gradients'].items():
        if key != 'transport':
            for value in values:
                value.zero_()
    data['gradients']['transport'] = None
    report = analyze_gradient_audit(data)
    assert report['physical_gradient_change']['joint']['cosine'] is None
    assert report['physical_change_to_max_repeat_l2'] is None
    assert report['delta_partition']['signed_projection_on_total'] == [None]*3
    assert not report['primitive_gate_passed']


@pytest.mark.parametrize('key,value', [('same_graph', False), ('alternate_forward_count', 1),
                                     ('mode', 'blend'), ('grad_h1', True)])
def test_unsupported_or_reprepared_comparison_fails_closed(key, value):
    data = fixture()
    data[key] = value
    with pytest.raises(RuntimeError):
        analyze_gradient_audit(data)


def test_rejected_or_missing_selected_attempt_cannot_be_published_as_accepted():
    audit = GradientAudit(23)
    data = fixture()
    data['win_index'] = 23
    audit.observe(23, data)
    audit.commit(23, None, None, None, dict(outer_rejected=True))
    report = audit.finish(dict(guards={}, history=[]))
    assert report['observed'] and not report['selected_outer_accepted'] and not report['observation_valid']
    assert report['selected_commit']['accepted_ordinal'] is None
    empty = GradientAudit(23).finish(dict(guards={}, history=[]))
    assert not empty['observed'] and not empty['observation_valid']


@pytest.mark.parametrize('u_only', [False, True])
def test_real_second_window_layer_pipeline_confirms_production_lambda_and_admission(u_only):
    from physmorph.pipeline import PipelineConfig, run_pipeline
    from physmorph.mpm.state import MPMParams
    source = np.random.default_rng(31).uniform(-1.3, 1.3, (96, 3)).astype(np.float32)
    target = (source*[1.2, .9, 1.05]+[.15, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=2, animations=2, stop_after_windows=2,
        loss_res=12, render_views=2, render_elevs=(0., .5), render_res=24,
        device='cpu', patience=5, commit_pic=True, commit_pic_objective=True,
        outer_render_committed=True, body_ctrl=True, body_terminal_ctrl=True,
        layer_ctrl=True, layer_relax=True, layer_u_render_only=u_only,
        lambda_auto=.3, grad_project=True, grad_project_mode='render',
        w_kin=.2, w_kin_var=.7, w_ctrl=0., w_box=0., max_ls_iters=1,
        adaptive_alpha=False, alpha=1e-4, loss_units='density')
    prm = MPMParams(dx=1., nx=32, ny=32, nz=32)
    plain = run_pipeline(source, target, prm, deepcopy(cfg), log=lambda *_: None)
    actual, report = run_audited(source, target, prm, deepcopy(cfg), selected_index=1, log=lambda *_: None)
    assert report['observation_valid'] and report['baseline_lambda_matches_production_exactly']
    assert report['selected_attempt'] == report['selected_commit']['accepted_ordinal'] == 2
    audit = report['audit']
    assert audit['balancer_before']['lam'] is not None
    assert audit['leaf_names'] == ['stress', 'body', 'surface_u']
    assert audit['physical_gradient_change']['joint']['difference_l2'] > 0
    assert audit['raw_render_vs_physical']['leaves'][2]['b_l2'] > 0
    assert audit['delta_partition']['closure']['difference_max_abs'] < 1e-6
    np.testing.assert_allclose(np.stack(actual['frames']), np.stack(plain['frames']), rtol=3e-6, atol=3e-6)
    assert not any(actual['guards'].values())
