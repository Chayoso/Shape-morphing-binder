"""CPU orchestration gates; these do not run the native candidate branch."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig
from scripts.probes import post_assimilation_candidate as probe


def recipe():
    cfg = PipelineConfig(stop_after_windows=21, assim_fp64=True)
    prm = MPMParams()
    return cfg, prm, json.loads(json.dumps(dict(effective_config=asdict(cfg), mpm=asdict(prm))))


def test_recipe_json_roundtrip():
    cfg, prm, protocol = recipe()
    probe.compatible_recipe(cfg, prm, protocol)


@pytest.mark.parametrize('field,value', [('assim_fp64', False), ('stop_after_windows', 20), ('T', 7), ('assim', .123)])
def test_recipe_rejects_unregistered_change(field, value):
    cfg, prm, protocol = recipe()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        probe.compatible_recipe(cfg, prm, protocol)


class Policy:
    def __init__(self, pins, eta=1.):
        self.data = dict(pin=torch.tensor(pins), eta=torch.full((len(pins),), eta),
                         x0=torch.zeros(len(pins), 3))
    def arrays(self):
        return self.data
    def metadata(self):
        return dict(N=len(self.data['pin']))


def test_policy_different_membership_uses_intersection_without_repair():
    estimate, actual = Policy([1., 1., 0., 0.]), Policy([1., 0., 1., 0.], eta=2.)
    row, cohorts = probe.policy_comparison(estimate, actual, torch.tensor([True, False, False, False]))
    assert row['pin_disagreements'] == 2
    assert row['estimated_only_new'] == row['actual_only_new'] == 1
    assert cohorts['common_surviving_free'].tolist() == [False, False, False, True]
    assert not row['array_differences']['eta']['exact']
    assert 'passed' not in row  # Model mismatch is not a fabricated raw-quality verdict.


def test_policy_old_pin_release_fails():
    with pytest.raises(ValueError, match='released old pins'):
        probe.policy_comparison(Policy([1., 0.]), Policy([0., 0.]), torch.tensor([True, False]))


CHECKS = ('health_passed', 'predicted_closure_passed', 'passive_raw_passed',
          'controlled_raw_passed', 'prepared_constraints_passed', 'common_motion_passed')


def decision(**changes):
    args = dict(selected=True, outer_committed=True, successor_committed=True,
                archive_exact=True, guards_clear=True, actual_check=dict.fromkeys(CHECKS, True))
    args.update(changes)
    return probe.branch_decision(**args)


@pytest.mark.parametrize('missing', CHECKS)
def test_every_actual_gate_is_required(missing):
    checks = dict.fromkeys(CHECKS, True)
    checks.pop(missing)
    row = decision(actual_check=checks)
    assert not row['candidate_branch_gate_passed']
    assert row['experimental_branch_rejected']
    assert row['disposition'] == 'experimental_branch_rejected_no_rollback'
    assert row['missing_actual_checks'] == [missing]


def test_late_failure_revokes_successful_disposition():
    report = dict(completed=True, candidate_selected=True, **decision())
    assert report['candidate_branch_gate_passed']
    probe.fail_report(report, OSError('render receipt failed'))
    assert not report['completed'] and not report['candidate_branch_gate_passed']
    assert report['experimental_branch_rejected']
    assert report['disposition'] == 'experimental_branch_rejected_no_rollback'
    assert not report['deliverable_promoted']


def test_no_candidate_does_not_claim_branch_pass():
    row = decision(selected=False)
    assert row['disposition'] == 'original_result_retained'
    assert not row['candidate_branch_gate_passed'] and not row['experimental_branch_rejected']
    report = dict(candidate_selected=False, **row)
    probe.fail_report(report, ValueError('incomplete'))
    assert report['disposition'] == 'driver_incomplete_no_candidate_adoption'


def health_case():
    prm = MPMParams(dx=1., nx=16, ny=16, nz=16, grid_min=(0., 0., 0.))
    return dict(coast_X=torch.full((3, 2, 3), 4.), coast_V=torch.zeros(3, 2, 3),
        coast_C=torch.zeros(3, 2, 3, 3), coast_F=torch.eye(3).repeat(3, 2, 1, 1)), torch.tensor([True, False]), prm


@pytest.mark.parametrize('fault', ['none', 'intermediate_F', 'pin_X', 'pin_V', 'pin_C', 'bounds', 'nan'])
def test_physical_health_checks_intermediate_actual_states(fault):
    coast, pins, prm = health_case()
    if fault == 'intermediate_F':
        coast['coast_F'][1, 1, 0, 0] = -1
    elif fault in ('pin_X', 'pin_V'):
        coast['coast_'+fault[-1]][1, 0, 0] += 1
    elif fault == 'pin_C':
        coast['coast_C'][1, 0, 0, 0] = 1
    elif fault == 'bounds':
        coast['coast_X'][1, 1, 0] = 100
    elif fault == 'nan':
        coast['coast_V'][1, 1, 0] = float('nan')
    row = probe.coast_health(coast, pins, prm)
    assert row['passed'] is (fault == 'none')


def test_output_cap_keeps_bounded_failure_receipt(tmp_path, monkeypatch):
    capture = probe.CandidateCapture(tmp_path, None, None, None)
    monkeypatch.setattr(probe, 'LIMIT', probe.RESERVE)
    assert not capture.finish(dict(completed=True, failure=None))
    row = json.loads((tmp_path/'failure.json').read_text())
    assert not row['completed'] and row['experimental_branch_rejected']
    assert row['failure']['type'] == 'ValueError'


class Quality:
    def __init__(self, source, target, x0, pins):
        self.x0, self.pins = np.array(x0, copy=True), np.array(pins, copy=True)
    def observe(self, *args, **kwargs):
        return {'passed': True}
    def archive_state(self):
        return {'current_pins': self.pins}


@pytest.mark.parametrize('outcome', ['disabled', 'no_candidate', 'failed_confirmation', 'exception'])
def test_selector_preserves_owned_original_without_confirmation(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(probe, 'is_cuda_execution', lambda: True)
    monkeypatch.setattr(probe, 'RawWithdrawalQuality', Quality)
    X = [np.full((2, 3), 4., np.float32) for _ in range(3)]
    F = [np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)) for _ in range(3)]
    end = dict(F=F[-1].copy(), v=np.zeros((2, 3), np.float32), C=np.zeros((2, 3, 3), np.float32))
    original = (X, F, end, None, [{'loss': 2.}], {'arrived_mask': np.array([True, False])})
    choice, estimate = object(), object()
    class Context:
        _reference = SimpleNamespace(tag='current live reference')
        def original(self):
            return choice
        def resolve(self, supplied):
            assert supplied is choice
            return deepcopy(original), {'selected': False}
        def inspect(self, supplied):
            assert supplied is choice
            return {'coefficients': torch.zeros(1, 6)}
        def search_post_assimilation(self, supplied, *, record, raw_observe):
            assert supplied is estimate
            if outcome == 'exception':
                raise ValueError('confirmation callback failed')
            return choice, dict(candidate_found=False, confirmed=False, status=outcome)
    context = Context()
    capture = probe.CandidateCapture(tmp_path, X[0], X[0], estimate, enabled=outcome != 'disabled')
    capture.donor, capture.start = deepcopy(original), {'pin': torch.tensor([1., 0.])}
    if outcome == 'exception':
        with pytest.raises(ValueError, match='confirmation callback failed'):
            capture.select(context)
        assert not (tmp_path/'selected_raw_head.npz').exists()
        assert (tmp_path/'search.json').exists() and (tmp_path/'raw_baseline_envelope.npz').exists()
    else:
        assert capture.select(context) is choice
        assert capture.receipt['no_candidate_original_exact']
        with np.load(tmp_path/'selected_raw_head.npz') as archive:
            assert np.array_equal(archive['X'], np.stack(X))
        if outcome != 'disabled':
            assert capture.reference is not context._reference
            assert capture.quality.pins.tolist() == [True, False]


def test_actual_handoff_zero_new_pins_is_explicit_and_pin_projection_checked(tmp_path):
    capture = probe.CandidateCapture(tmp_path, None, None, None)
    pin = torch.tensor([1., 0.])
    capture.start = {'pin': pin}
    capture.head = dict(x=torch.ones(2, 3), F=torch.eye(3).repeat(2, 1, 1),
                        v=torch.zeros(2, 3), C=torch.zeros(2, 3, 3))
    capture.successor = {key+'0': value.clone() for key, value in capture.head.items()}
    capture.successor['pin'] = pin.clone()
    capture.successor['v0'][0, 0] = 1.
    with pytest.raises(ValueError, match='handoff differs'):
        capture.check_handoff()
    capture.successor['v0'][0, 0] = 0.
    capture.check_handoff()
    assert capture.receipt['handoff']['new_pins'] == 0


def test_individual_worsening_is_reported_despite_aggregate_improvement():
    before = dict(geometric=torch.tensor([10., 1.], dtype=torch.float64), stored=torch.tensor([10., 1.], dtype=torch.float64))
    after = dict(geometric=torch.tensor([5., 2.], dtype=torch.float64), stored=torch.tensor([5., 2.], dtype=torch.float64))
    row = probe.energy_comparison(after, before, {'common': torch.tensor([True, True])})['common']
    assert row['geometric']['mean_change'] < 0
    assert row['geometric']['worsened_ids'] == row['stored']['worsened_beyond_roundoff_ids'] == 1
