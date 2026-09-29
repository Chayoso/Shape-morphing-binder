"""Owned immediate-selection boundary, independently of optimizer policy."""
from copy import deepcopy
from types import SimpleNamespace
import gc
import weakref

import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.frozen_body_window import FrozenBodyWindow
from physmorph.pipeline.prepared_reference import PreparedReference
from physmorph.pipeline import window_selection as module


def equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected: equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected): equal(a, b)
    elif torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


class FakeWithdrawal:
    """Same-forward state supplier; gates under test never delegate to its health."""
    def __init__(self, owner):
        for name in ('spec', 'idx', 'weights', 'gate', 'coefficients', 'stress', 'surface_u'):
            setattr(self, name, deepcopy(getattr(owner, name)))
        self.calls, self.closed, self.corrupt = 0, False, None

    def close(self):
        self.closed = True

    def evaluate(self, terminal, displacement):
        self.calls += 1
        n, T = len(self.idx), self.spec.T
        x0 = torch.tensor(self.spec.x0)
        delta = displacement[self.idx[:, 0]]*.1
        delta[torch.tensor(self.spec.pin) > .5] = 0.
        X = torch.stack([x0+(t+1)*delta for t in range(T)])
        I = torch.eye(3).repeat(n, 1, 1)
        F = I.repeat(T, 1, 1, 1)
        V = delta.repeat(T, 1, 1)
        C = torch.zeros(T, n, 3, 3)
        result = dict(x=X[-1].clone(), F=F[-1].reshape(n, 9).clone(), Fg=F[-1].reshape(n, 9).clone(),
            v=V[-1].clone(), C=C[-1].clone(), positions=X, V=V, F_initial=I, F_sequence=F,
            C_sequence=C, coast_X=X[-1].repeat(T+1, 1, 1), coast_V=V[-1].repeat(T+1, 1, 1),
            coast_F=I.reshape(n, 9).repeat(T+1, 1, 1), coast_C=C[-1].repeat(T+1, 1, 1, 1),
            valid=True, pins_exact=True, min_det=1., body_energy=(displacement.square()+terminal.square()).mean())
        if self.corrupt is not None:
            self.corrupt(result)
        return result


def make_context(monkeypatch, **overrides):
    monkeypatch.setattr(module, 'FrozenWithdrawalWindow', FakeWithdrawal)
    prm = MPMParams()
    cfg = PipelineConfig(T=2, body_ctrl=True, body_terminal_ctrl=True, phys_loss='ot_pace',
                         loss_units='density', pace=.2, w_pbr=.2, device='cpu')
    for key, value in overrides.items(): setattr(cfg, key, value)
    spec = SimpleNamespace(prm=prm, T=2, body_ctrl=True, body_modes=2, device='cpu',
        x0=np.array([[0., 0., 0.], [.2, 0., 0.], [0., .2, 0.]], np.float32),
        pin=np.array([1., 0., 0.], np.float32))
    n = len(spec.x0)
    owner = FrozenBodyWindow(spec, SimpleNamespace(idx=torch.arange(n)[:, None], weights=torch.ones(n, 1)),
        torch.ones(n, 1), torch.zeros(n, 6), torch.zeros(2, n, 3, 3), None)
    frames = [spec.x0.copy() for _ in range(3)]
    Fs = [np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)) for _ in range(3)]
    end = dict(F=Fs[-1].copy(), v=np.zeros((n, 3), np.float32), C=np.zeros((n, 3, 3), np.float32),
               Fg=None, n_inv_steps=0, Jmin_traj=1.)
    hist = [dict(loss=10., d_render=2.7, d_pbr=2., d_sil=2.7, d_vol=5., kin=.4,
                 **{'lambda': .2}, grad_norm=.7, alpha=.01, render_work=.4, body_update_modes_rms=[.2, .3],
                 dfc_absmax=.03, s_absmax=None)]
    stats = dict(accepted=2, rejected=1, pace_bound=False, L_start=12.,
        render_influence_steps=[{'render': .3}], commit_from_accepted=True,
        replay_diagnostics=dict(commit_source='accepted_buffer'),
        body_rms_wu=99., body_terminal_rms_wu=88., body_coeff_max=77., body_coeff_saturated_frac=.6,
        g_phys_norm=.9, render_work=.4, step_norm=.2, gx=np.ones((n, 3), np.float32),
        motion_accounting={'stale': 1}, mom_out=None, plan_img=np.ones((n, 3), np.float32),
        arrived_mask=np.array([True, False, True]), arrive_idx=np.arange(n),
        u_final=np.arange(n, dtype=np.float32), dfc=np.ones((2, n, 9), np.float32))
    original = (frames, Fs, end, None, hist, stats)
    reference = PreparedReference({}, {}, {}, .2, 'fixed_test')
    metrics = dict(merit=9.8, physical=9.2, render=3., volume=4., silhouette=2.6, pbr=2.,
                   stored_terminal=.1, stored_running=.2, stored_variance=.3, body_energy=.01,
                   unit_weight=1., lambda_render=.2)
    lease = [True]
    context = module.PreparedWindowSelection(original, owner, reference, lambda _: dict(metrics), lease, cfg, prm, 4)
    return context, original, metrics, owner, lease


def test_original_and_none_are_exact_owned_without_replay(monkeypatch):
    context, original, _, owner, lease = make_context(monkeypatch)
    expected = deepcopy(original)
    choice = context.original()
    observed = context.inspect(choice)
    assert observed['identity'] and observed['window'] == 4
    assert 'V' not in observed['values'] and 'coast_X' not in observed['values']
    observed['values']['positions'].zero_()
    observed['coefficients'].fill_(.8)
    original[0][-1].fill(99.)
    for selected in (None, choice):
        result, report = context.resolve(selected)
        equal(result, expected)
        assert not report['selected'] and report['selected_label'] == 'original'
        result[0][-1].fill(77.)
    assert context._model.calls == 0
    context.close()
    assert context.closed and owner.closed and not lease[0]
    with pytest.raises(RuntimeError, match='expired'): context.original()


def test_selected_forward_refreshes_state_metrics_and_keeps_donor_history(monkeypatch):
    context, original, metrics, _, _ = make_context(monkeypatch)
    coeff = torch.zeros(3, 6)
    coeff[:, :3] = .1
    coeff[:, 3:] = .2
    choice = context.evaluate(coeff, 'candidate')
    inspect = context.inspect(choice)
    assert inspect['eligible'] and not inspect['identity']
    context.certify(choice, dict(passed=True, scope='trusted all-phase raw witness'))
    inspect['values']['F_sequence'].zero_()
    inspect['coefficients'].zero_()
    result, report = context.resolve(choice)
    fr, Fs, end, material, hist, stats = result
    assert report['selected'] and report['selected_label'] == 'candidate'
    assert not np.array_equal(fr[-1], original[0][-1])
    np.testing.assert_array_equal(end['F'], Fs[-1])
    assert end['Fg'] is None and end['n_inv_steps'] == 0 and end['Jmin_traj'] == 1.
    equal(hist, original[4])
    for key in ('accepted', 'rejected', 'render_influence_steps', 'plan_img', 'arrived_mask', 'arrive_idx', 'u_final', 'dfc'):
        equal(stats[key], original[5][key])
    selected = stats['selected_observation']
    assert selected['loss'] == metrics['merit'] and selected['d_render'] == 3.-.2*2.
    assert selected['d_pbr'] == 2. and selected['kin_var'] == .3
    assert selected['dfc_absmax'] == .03 and selected['s_absmax'] is None
    assert selected['alpha'] is None and selected['grad_norm'] is None
    assert stats['g_phys_norm'] is None and stats['render_work'] is None and stats['gx'] is None
    assert stats['commit_from_accepted'] is False and stats['motion_accounting'] is None
    assert stats['replay_diagnostics']['commit_source'] == 'private_selected_forward'
    assert stats['body_rms_wu'] == pytest.approx(.5*np.sqrt(.03))
    assert stats['body_terminal_rms_wu'] == pytest.approx(.5*np.sqrt(.12))
    assert stats['body_coeff_max'] == pytest.approx(np.sqrt(.15))
    assert stats['body_coeff_saturated_frac'] == 0.
    assert report['donor_observation']['alpha'] == .01
    context.close()


@pytest.mark.parametrize('report', [None, {'passed': False}])
def test_uncertified_eligible_candidate_falls_back_exactly(monkeypatch, report):
    context, original, _, _, _ = make_context(monkeypatch)
    choice = context.evaluate(torch.ones(3, 6)*.05, 'eligible')
    assert context.inspect(choice)['eligible']
    if report is not None: context.certify(choice, report)
    result, receipt = context.resolve(choice)
    equal(result, original)
    assert not receipt['selected'] and 'missing_or_failed_raw_certificate' in receipt['failures']
    context.close()


def test_candidate_registry_inspection_certificate_and_later_forwards_are_owned(monkeypatch):
    context, original, _, owner, _ = make_context(monkeypatch)
    first = context.evaluate(torch.ones(3, 6)*.1, 'first')
    snapshot = context.inspect(first)
    context.evaluate(torch.ones(3, 6)*.2, 'second')
    equal(context.inspect(first), snapshot)
    certificate = dict(passed=True, proof={'counts': [3]})
    context.certify(first, certificate)
    certificate['passed'] = False
    certificate['proof']['counts'][0] = 0
    owner.coefficients.fill_(.9)
    owner.spec.x0.fill(3.)
    result, report = context.resolve(first)
    assert report['selected'] and report['raw_certificate']['proof']['counts'] == [3]
    with pytest.raises(ValueError, match='Foreign'): context.resolve(object())
    other, *_ = make_context(monkeypatch)
    with pytest.raises(ValueError, match='Foreign'): other.resolve(first)
    with pytest.raises(ValueError, match='boolean'): context.certify(first, {'passed': 'yes'})
    context.close(); other.close()
    with pytest.raises(RuntimeError, match='expired'): context.inspect(first)


@pytest.mark.parametrize('corruption,reason', [
    ('middle_F', 'post_or_effective_determinant'), ('effective_F', 'post_or_effective_determinant'),
    ('middle_bounds', 'all_step_bounds'), ('middle_C_nan', 'nonfinite_full_state'),
    ('pin_V', 'exact_pins'), ('pin_C', 'exact_pins'), ('pin_X', 'exact_pins'),
    ('coast_C', 'nonfinite_full_state'), ('boundary', 'same_forward_boundary')])
def test_actual_full_state_health_rejects_lying_endpoint_flags(monkeypatch, corruption, reason):
    context, original, _, _, _ = make_context(monkeypatch)
    def corrupt(v):
        if corruption == 'middle_F': v['F_sequence'][0, 1] *= .01
        elif corruption == 'effective_F': context._model.stress[0, 1] = -.999*torch.eye(3)
        elif corruption == 'middle_bounds': v['positions'][0, 1, 0] = 100.
        elif corruption == 'middle_C_nan': v['C_sequence'][0, 1, 0, 0] = float('nan')
        elif corruption == 'pin_V': v['V'][0, 0, 0] = .1
        elif corruption == 'pin_C': v['C_sequence'][0, 0, 0, 0] = .1
        elif corruption == 'pin_X': v['positions'][0, 0, 0] = .1
        elif corruption == 'coast_C': v['coast_C'][1, 1, 0, 0] = float('nan')
        else: v['coast_X'][0, 1, 0] += .1
    context._model.corrupt = corrupt
    choice = context.evaluate(torch.ones(3, 6)*.05, 'bad')
    assert reason in context.inspect(choice)['failures']
    context.certify(choice, {'passed': True})
    result, report = context.resolve(choice)
    assert not report['selected']
    equal(result, original)
    context.close()


@pytest.mark.parametrize('merit,passes', [(9.599999, False), ((1-.2)*12, True), (10., True),
                                       (10.+32*np.finfo(np.float32).eps*10, True), (10.001, False), (float('nan'), False)])
def test_original_merit_ceiling_roundoff_and_strict_pace_floor(monkeypatch, merit, passes):
    context, original, metrics, _, _ = make_context(monkeypatch)
    metrics['merit'] = merit
    choice = context.evaluate(torch.zeros(3, 6), 'identity_replay')
    assert context.inspect(choice)['eligible'] is passes
    context.certify(choice, {'passed': True})
    result, report = context.resolve(choice)
    assert report['selected'] is passes
    if not passes: equal(result, original)
    context.close()


@pytest.mark.parametrize('coeff', [torch.ones(3, 6), torch.zeros(3, 3), torch.full((3, 6), float('nan'))])
def test_invalid_joint_coefficients_never_launch_private_forward(monkeypatch, coeff):
    context, original, _, _, _ = make_context(monkeypatch)
    choice = context.evaluate(coeff, 'invalid')
    assert context._model.calls == 0
    result, _ = context.resolve(choice)
    equal(result, original)
    context.close()


@pytest.mark.parametrize('name', ['commit_pic', 'shift_sub', 'lg_sweeps', 'reattach', 'rest_commit',
    'settle_commit', 'settle_pin_follow', 'settle_pin_yield', 'geometric_rest', 'geometric_variance',
    'render_F_geom', 'use_gauss_loss', 'surface_gs_loss', 'continuity', 'settle_pin_kkt', 'opt_material', 'mom_carry'])
def test_unsupported_scope_rejected_before_private_model(monkeypatch, name):
    with pytest.raises(ValueError, match='does not support'):
        make_context(monkeypatch, **{name: 1})


def test_inactive_render_channels_preserve_none(monkeypatch):
    context, _, _, _, _ = make_context(monkeypatch)
    context._original[4][-1]['d_render'] = None
    context._original[4][-1]['d_pbr'] = None
    choice = context.evaluate(torch.zeros(3, 6), 'inactive')
    context.certify(choice, {'passed': True})
    result, report = context.resolve(choice)
    assert report['selected']
    row = result[5]['selected_observation']
    assert all(row[key] is None for key in ('d_render', 'd_render_total', 'd_pbr', 'd_sil', 'lambda'))
    context.close()


def test_retained_closed_context_releases_numerical_payloads(monkeypatch):
    context, _, _, _, lease = make_context(monkeypatch)
    choice = context.evaluate(torch.zeros(3, 6), 'owned')
    references = [weakref.ref(context._model), weakref.ref(context._reference),
                  weakref.ref(context._original[0][0]), weakref.ref(context._choices[choice]['values']['positions']),
                  weakref.ref(context._evaluate_merit)]
    context.close()
    gc.collect()
    assert context.closed and not lease[0] and all(reference() is None for reference in references)
    assert all(getattr(context, name) is None for name in (
        '_original', '_model', '_owner', '_evaluate_merit', '_reference', '_cfg', '_prm', '_start', '_pins'))
    context.close()


def test_constructor_failure_expires_lease_and_closes_allocated_model(monkeypatch):
    context, original, _, _, _ = make_context(monkeypatch)
    original = deepcopy(original)
    original[5]['L_start'] = float('nan')
    owner = context._owner
    reference = context._reference
    model_references = []
    class WatchedWithdrawal(FakeWithdrawal):
        def __init__(self, owner):
            super().__init__(owner)
            model_references.append(weakref.ref(self))
        def close(self):
            super().close()
            closed.append(True)
    closed, lease = [], [True]
    monkeypatch.setattr(module, 'FrozenWithdrawalWindow', WatchedWithdrawal)
    with pytest.raises(ValueError, match='finite original'):
        module.PreparedWindowSelection(original, owner, reference, lambda _: {}, lease,
                                       context._cfg, context._prm, 4)
    gc.collect()
    assert not lease[0] and owner.closed and closed == [True]
    assert all(reference() is None for reference in model_references)
    context.close()
