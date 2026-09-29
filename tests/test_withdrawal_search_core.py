"""P331 orchestration/math oracles plus one real CPU observer isolation case."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.probes import withdrawal_search_core as core


class Reference:
    def terms(self, x):
        zero = x.sum()*0
        return dict(volume=zero, render=zero, silhouette=zero, pbr=zero)


class Merit:
    def __init__(self, kind='constant', tensor_offset=0.):
        self.kind, self.tensor_offset = kind, tensor_offset
    def value(self, values):
        q = values['test_q']
        return q*q if self.kind == 'quadratic' else 1+(.1*q if self.kind == 'linear' else q*0)
    def __call__(self, values):
        return dict(merit=float(self.value(values).detach()), lambda_render=.3)
    def terms(self, values):
        return dict(merit=self.value(values)+self.tensor_offset)
    def binding_digest(self):
        return 'unchanged-test-binding'


class Model:
    instances = []
    corruption = None
    def __init__(self, owner):
        self.owner, self.calls, self.closed = owner, 0, False
        self.instances.append(self)
    def evaluate(self, terminal, displacement):
        q = terminal[:, 0].sum()+displacement[:, 0].sum()
        self.calls += 1
        values = fake_values(q)
        if self.corruption == 'nonfinite' and self.calls == 1:
            values['coast_C'][1, 0, 0, 0] = float('nan')
            values['valid'] = False
        if self.corruption == 'closure' and self.calls == 2:
            values['positions'] = values['positions']+.001
            values['x'] = values['positions'][-1]
        return values
    def close(self):
        self.closed = True


def fake_values(q):
    T, N, dt = 2, 2, .1
    unit = torch.tensor([1., 0., 0.])
    head = torch.ones(T, N, 3)*.5+q*.01*unit
    V = torch.zeros_like(head)+q*0
    coast = head[-1][None]+torch.arange(T+1)[:, None, None]*dt*(1-q)*unit
    future_V = torch.ones(T, N, 3)*(1-q)*unit
    coast_V = torch.cat((V[-1][None], future_V))
    F = torch.eye(3).reshape(1, 9).repeat(N, 1)
    C = torch.zeros(N, 3, 3)
    return dict(x=head[-1], positions=head, V=V, v=V[-1], F=F, Fg=F.clone(), C=C,
        body_energy=q*q, coast_X=coast, coast_V=coast_V,
        coast_F=F[None].repeat(T+1, 1, 1), coast_Fg=F[None].repeat(T+1, 1, 1),
        coast_C=C[None].repeat(T+1, 1, 1, 1), test_q=q,
        valid=True, pins_exact=True, min_det=1., health=dict(head_valid=True, coast_valid=True))


def packet(kind='constant', *, tensor_offset=0.):
    original = fake_values(torch.tensor(0.))
    evaluator = Merit(kind, tensor_offset)
    coefficients = torch.zeros(1, 6)
    owner = SimpleNamespace(coefficients=coefficients, spec=SimpleNamespace(T=2,
                            prm=SimpleNamespace(dx=1., dt=.1)), closed=False)
    return dict(rollout=owner, controls=dict(body=coefficients.clone()), reference=Reference(),
        evaluate_merit=evaluator, x0=torch.ones(2, 3)*.5,
        pins=torch.zeros(2, dtype=torch.bool), start_arrived=torch.ones(2, dtype=torch.bool),
        dt=.1, lambda_render=.3, history=dict(loss=evaluator(original)['merit'], body_update_modes_rms=[.1, .1]),
        **{k: original[k] for k in ('positions', 'V', 'F', 'C')})


class Callbacks:
    def __init__(self, fail_confirmation=False, raw_error=False):
        self.rows, self.baselines, self.labels = [], [], []
        self.fail_confirmation, self.raw_error = fail_confirmation, raw_error
    def raw(self, label, values, baseline):
        self.labels.append(label)
        if self.raw_error:
            raise RuntimeError('injected raw observer failure')
        if baseline:
            self.baselines.append(label)
        else:
            assert len(self.baselines) == 3
        return dict(passed=not (self.fail_confirmation and label == 'confirm_0'),
                    baseline_ready=len(self.baselines) == 3, scope='test orchestration, no geometric evidence')
    def record(self, label, values, info):
        assert 'label' not in info
        self.rows.append((label, {k: v.detach().clone() for k, v in values.items() if torch.is_tensor(v)},
                          deepcopy(info)))


@pytest.fixture
def fake(monkeypatch):
    Model.instances, Model.corruption = [], None
    monkeypatch.setattr(core, 'FrozenWithdrawalWindow', Model)
    return Model


def test_radius_uses_node_three_vector_RMS_and_exact_joint_projection():
    assert core.joint_trust_radius({'body_update_modes_rms': [3., 4.]}, 9) == 15
    c = torch.tensor([[.6, 0., 0., .8, 0., 0.], [1.2, 0., 0., 1.6, 0., 0.]])
    old = c.clone()
    result = core.project_joint(c)
    assert torch.equal(c, old) and result.data_ptr() != c.data_ptr()
    torch.testing.assert_close(result, torch.tensor([[.6, 0., 0., .8, 0., 0.]]).repeat(2, 1), rtol=0, atol=0)
    assert not torch.equal(result[1, :3], c[1, :3])  # Both modes projected together.


@pytest.mark.parametrize('rms', [None, [], [1.], [-1., 2.], [float('nan'), 1.], [float('inf'), 1.], [1e308, 1e308]])
def test_invalid_radius_metadata_fails_closed(rms):
    with pytest.raises(ValueError):
        core.joint_trust_radius({'body_update_modes_rms': rms}, 2)


def test_success_uses_three_originals_float_merit_then_three_owned_confirmations(fake):
    data, callbacks = packet(tensor_offset=1234.), Callbacks()
    report = core.run_search(data, record=callbacks.record, raw_observe=callbacks.raw)
    assert report['candidate_found'] and report['confirmed'] and not report['adopted']
    assert report['status'] == 'confirmed_candidate_no_adoption'
    assert callbacks.labels == ['baseline_0', 'baseline_1', 'baseline_2', 'candidate_h00', 'confirm_0', 'confirm_1', 'confirm_2']
    assert all(row['constraints']['head_merit'] == 1 for row in report['baselines']+report['proposals'])
    assert report['ceilings']['head_merit'] == 1+core.allowance(1)
    controls = [row[2]['arrays']['coefficients'] for row in callbacks.rows]
    assert all(torch.equal(c, controls[3]) for c in controls[4:])
    assert torch.count_nonzero(controls[3][:, :3]) and torch.count_nonzero(controls[3][:, 3:])
    assert controls[3].double().norm() <= report['initial_radius']+report['linear_trials'][0]['radius_tolerance']
    assert torch.equal(data['controls']['body'], torch.zeros_like(controls[0]))
    assert fake.instances[0].closed and not data['rollout'].closed
    row = report['proposals'][0]['per_id_energy_change']['start_free']
    assert row['geometric']['worsened_ids'] == 0 and row['geometric']['mean_change'] < 0


def test_nonlinear_complete_merit_blocks_all_mean_improvements(fake):
    callbacks = Callbacks()
    report = core.run_search(packet('quadratic'), record=callbacks.record, raw_observe=callbacks.raw)
    assert report['status'] == 'no_candidate_passed' and not report['candidate_found']
    assert len(report['proposals']) == 11 and not report['confirmations']
    assert all(r['objective_decrease']['passed'] and not r['nonlinear_constraints']['head_merit']
               and r['raw']['passed'] for r in report['proposals'])
    assert len(callbacks.rows) == 14  # Failed prepared constraints still raw-observed and archived.


def test_constraint_covector_contains_full_merit_not_prepared_surrogate(fake):
    callbacks = Callbacks()
    core.run_search(packet('linear'), record=callbacks.record, raw_observe=callbacks.raw)
    gradients = callbacks.rows[2][2]['arrays']['constraint_gradients']
    assert torch.count_nonzero(gradients[0]) == 0
    torch.testing.assert_close(gradients[3], torch.tensor([[.1, 0., 0., .1, 0., 0.]]), rtol=0, atol=0)


def test_first_confirmation_failure_stops_without_trying_another_candidate(fake):
    callbacks = Callbacks(fail_confirmation=True)
    report = core.run_search(packet(), record=callbacks.record, raw_observe=callbacks.raw)
    assert report['provisional_candidate_found'] and not report['candidate_found']
    assert report['status'] == 'confirmation_failed_no_adoption'
    assert len(report['proposals']) == 1 and len(report['confirmations']) == 3


@pytest.mark.parametrize('field', ['pins', 'start_arrived'])
def test_invalid_cohort_layout_fails_before_model(fake, field):
    data = packet(); data[field] = data[field].float()
    with pytest.raises(core.SearchFailure, match='fixed cohort') as caught:
        core.run_search(data, record=Callbacks().record, raw_observe=Callbacks().raw)
    assert caught.value.report['status'] == 'error' and not fake.instances


def test_empty_primary_zero_radius_and_empty_secondary_have_explicit_scope(fake):
    data, callbacks = packet(), Callbacks()
    data['pins'].fill_(True)
    assert core.run_search(data, record=callbacks.record, raw_observe=callbacks.raw)['status'] == 'inconclusive_empty_start_free'
    assert not callbacks.rows and not fake.instances
    data = packet(); data['history']['body_update_modes_rms'] = [0., 0.]
    assert core.run_search(data, record=callbacks.record, raw_observe=callbacks.raw)['status'] == 'inconclusive_zero_original_update'
    data = packet(); data['start_arrived'].zero_()
    report = core.run_search(data, record=callbacks.record, raw_observe=callbacks.raw)
    secondary = report['proposals'][0]['per_id_energy_change']['start_arrived_free']
    assert secondary == dict(particles=0, geometric=None, stored=None)


@pytest.mark.parametrize('corruption', ['closure', 'nonfinite'])
def test_bad_original_preserves_actual_witness_and_closes_private_model(fake, corruption):
    fake.corruption = corruption
    callbacks = Callbacks()
    report = core.run_search(packet(), record=callbacks.record, raw_observe=callbacks.raw)
    assert report['status'] == 'baseline_gate_failed' and not report['proposals']
    assert len(callbacks.rows) == (2 if corruption == 'closure' else 1)
    assert fake.instances[0].closed
    if corruption == 'nonfinite':
        assert torch.isnan(callbacks.rows[0][1]['coast_C']).any()


def test_raw_exception_archives_current_generation_and_attaches_incremental_report(fake):
    callbacks = Callbacks(raw_error=True)
    with pytest.raises(core.SearchFailure, match='raw observer failure') as caught:
        core.run_search(packet(), record=callbacks.record, raw_observe=callbacks.raw)
    assert len(callbacks.rows) == 1 and callbacks.rows[0][0] == 'baseline_0'
    assert caught.value.report['baselines'][0]['error']['message'] == 'injected raw observer failure'
    assert caught.value.report['merit_binding_unchanged'] and fake.instances[0].closed


def test_projected_step_must_pass_actual_joint_trust_and_direction_checks(fake, monkeypatch):
    def oversized(gradient, constraints, bounds, radius):
        return torch.ones_like(gradient, dtype=torch.float64)*10, dict(status='test oversized')
    monkeypatch.setattr(core, 'affine_ball_step', oversized)
    callbacks = Callbacks()
    report = core.run_search(packet(), record=callbacks.record, raw_observe=callbacks.raw)
    assert len(report['linear_trials']) == 11 and not report['proposals']
    assert all(r['projected_norm'] > r['radius']+r['radius_tolerance'] for r in report['linear_trials'])
    assert len(callbacks.rows) == 3


def test_negative_but_unresolved_prediction_does_not_run_candidate(fake, monkeypatch):
    def unresolved(gradient, constraints, bounds, radius):
        return -gradient.double()/gradient.double().norm()*1e-9, dict(status='test unresolved')
    monkeypatch.setattr(core, 'affine_ball_step', unresolved)
    callbacks = Callbacks()
    report = core.run_search(packet(), record=callbacks.record, raw_observe=callbacks.raw)
    assert not report['proposals'] and len(callbacks.rows) == 3
    for trial in report['linear_trials']:
        assert 0 < trial['predicted_decrease'] < trial['predicted_decrease_floor']
        assert not trial['prediction_resolved']


def test_real_joint_search_is_observational_to_original_cpu_solver(monkeypatch):
    from test_checkpoint_merit_terms import fixture, run_observed
    from physmorph.pipeline import runner
    source, target, prm, cfg = fixture()
    baseline = runner.run_pipeline(source, target, prm, deepcopy(cfg), log=lambda *_: None)
    reports = []
    def callback(live):
        evidence = Callbacks()  # Raw geometry is intentionally a test stub here.
        report = core.run_search(live, record=evidence.record, raw_observe=evidence.raw)
        assert report['merit_binding_unchanged'] and len(report['baselines']) == 3
        assert all(row['passed'] for row in report['baselines'])
        gradients = evidence.rows[2][2]['arrays']['constraint_gradients']
        assert gradients.shape[0] == 8 and gradients[3].norm() > 0
        reports.append(report)
    actual, _ = run_observed(monkeypatch, callback, cfg=cfg)
    assert reports
    np.testing.assert_array_equal(actual['frames'], baseline['frames'])
    np.testing.assert_array_equal(actual['F_frames'], baseline['F_frames'])
    for a, b in zip(actual['history'], baseline['history']):
        for key in ('loss', 'lambda', 'd_vol', 'd_sil', 'frame_end', 'render_influence_steps'):
            assert a.get(key) == b.get(key)
