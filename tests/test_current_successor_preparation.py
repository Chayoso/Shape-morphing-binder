"""CPU reporting/orchestration tests; no native simulation is run here."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline.config import PipelineConfig
from scripts.probes import current_successor_preparation as probe
from scripts.probes import post_assimilation_candidate as candidate


class State:
    def __init__(self):
        self.data = dict(x0=torch.full((2, 3), 4.), v0=torch.zeros(2, 3),
            C0=torch.zeros(2, 3, 3), F0=torch.eye(3).repeat(2, 1, 1), Fp=torch.eye(3).repeat(2, 1, 1),
            m=torch.ones(2), lam=torch.ones(2), mu=torch.ones(2), eta=torch.ones(2), vol=torch.ones(2),
            pin=torch.tensor([1., 0.]), layer_mask=torch.ones(2), layer_ug=torch.ones(2),
            layer_nbr=torch.tensor([[1], [0]], dtype=torch.int32), layer_nrm=torch.ones(2, 3),
            layer_w=torch.ones(2, 1), bond_nbr=torch.tensor([[1], [0]], dtype=torch.int32),
            bond_frag=torch.zeros(2), bond_rest=torch.ones(2, 1))
        self.meta = dict(T=2, layer_F=False, source_body_control=False)
    def arrays(self):
        return {key: value.clone() for key, value in self.data.items()}
    def metadata(self):
        return deepcopy(self.meta)


def test_only_surface_u_gate_is_qualified_for_passive_zero_control():
    a, b = State(), State()
    b.data['layer_ug'][0] = 0
    b.meta['source_body_control'] = True
    row = probe.compare_prepared(a, b, MPMParams())
    assert row['passed'] and not row['layer_ug']['exact']
    assert row['layer_ug']['different_entries'] == 1
    assert 'layer_ug' not in row['fields']
    b.data['layer_ug'][0] = float('nan')
    assert not probe.compare_prepared(a, b, MPMParams())['passed']


@pytest.mark.parametrize('key', ['vol', 'pin', 'layer_mask', 'bond_frag', 'layer_nbr', 'bond_nbr'])
def test_exact_material_and_discrete_policy_cannot_hide_in_float_tolerance(key):
    a, b = State(), State()
    b.data[key].flatten()[0] += 1 if not b.data[key].is_floating_point() else torch.finfo(torch.float32).eps
    row = probe.compare_prepared(a, b, MPMParams())
    assert not row['passed'] and not row['fields'][key]['passed']


def test_float_geometry_uses_unchanged_closure_not_exact_claim():
    a, b = State(), State()
    b.data['Fp'][0, 0, 0] += torch.finfo(torch.float32).eps
    row = probe.compare_prepared(a, b, MPMParams())
    assert row['passed'] and not row['fields']['Fp']['exact']
    b.data['Fp'][0, 0, 0] += .1
    assert not probe.compare_prepared(a, b, MPMParams())['passed']


def histories():
    actual = dict(previous=torch.zeros(2, 3), scale=torch.ones(2), reversals=torch.ones(2, dtype=torch.int32),
                  frozen=torch.zeros(2, dtype=torch.bool), settled=torch.tensor([True, False]),
                  settled_at=torch.tensor([20, -1], dtype=torch.int32), pins=torch.tensor([True, False]),
                  neighbors=torch.tensor([[1], [0]]))
    expected = {key: value.clone() for key, value in actual.items() if key not in ('previous', 'neighbors')}
    expected['pins'] = expected['pins'].float()
    expected['displacement'] = actual['previous'].clone()
    return expected, actual, deepcopy(actual)


@pytest.mark.parametrize('key', ['previous', 'scale', 'reversals', 'settled_at', 'neighbors'])
def test_history_differences_fail_exactly(key):
    predicted, actual, original = histories()
    assert probe.compare_history(predicted, actual, original)['passed']
    actual[key].flatten()[0] += 1
    assert not probe.compare_history(predicted, actual, original)['passed']


def passing_report():
    return dict(completed=True, bindings_unchanged=True, archives_exact=True, outer_commits=[True, True],
        guards_clear=True, receipt=dict(w20_identity=True, w21_identity=True),
        history_comparison={'passed': True}, prepared_comparison={'passed': True}, coast_comparison={'passed': True})


@pytest.mark.parametrize('key', ['completed', 'bindings_unchanged', 'archives_exact', 'guards_clear',
    'history_comparison', 'prepared_comparison', 'coast_comparison', 'receipt', 'outer_commits'])
def test_missing_final_gate_never_becomes_success(key):
    row = passing_report()
    assert probe.final_gate(row)
    del row[key]
    assert not probe.final_gate(row)


def test_cap_failure_retains_p337_scope(tmp_path, monkeypatch):
    capture = probe.CurrentPreparationCapture(tmp_path)
    monkeypatch.setattr(probe, 'LIMIT', probe.RESERVE)
    assert not capture.finish(dict(passed=True))
    row = json.loads((tmp_path/'failure.json').read_text())
    assert row['scope'] == probe.SCOPE and not row['passed']
    assert row['disposition'] == 'incomplete_original_identity_diagnostic'
    assert row['optimizer_progress']['persisted_optimizer_returns'] == 0


def test_partial_optimizer_receipt_has_owned_current_discretization(tmp_path):
    capture = probe.CurrentPreparationCapture(tmp_path)
    cfg, prm = PipelineConfig(T=20, iters=8), MPMParams(dt=1/240, dx=.3)
    source = np.zeros((12, 3), np.float32)
    capture.configure_progress(cfg, prm, source)
    cfg.T = 99; prm.dx = 99; source[:] = 99
    capture.record_optimizer_return(19, (None, None, None, None, [{'loss': 1.}], {'accepted': 1}))
    row = json.loads((tmp_path/'optimizer_progress.json').read_text())
    assert row['discretization']['N'] == 12 and row['discretization']['T'] == 20
    assert row['discretization']['dx_wu'] == .3
    assert np.all(capture.source == 0)
    receipt = capture.optimizer_progress_receipt()
    assert receipt['persisted_optimizer_returns'] == 1 and receipt['observed_returns_persisted']
    assert receipt['artifact'] == 'optimizer_progress.json'
    assert not capture.finish(dict(passed=False, completed=False, optimizer_progress=receipt))
    assert json.loads((tmp_path/'result.json').read_text())['optimizer_progress'] == receipt


@pytest.mark.parametrize('mutate_original', [False, True])
def test_two_contexts_return_original_and_archive_current_preview(tmp_path, monkeypatch, mutate_original):
    monkeypatch.setattr(probe, 'is_cuda_execution', lambda: True)
    monkeypatch.setattr(candidate, 'is_cuda_execution', lambda: True)
    X = [np.full((2, 3), 4., np.float32) for _ in range(3)]
    Fs = [np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)) for _ in range(3)]
    original = (X, Fs, dict(F=Fs[-1].copy(), v=np.zeros((2, 3), np.float32), C=np.zeros((2, 3, 3), np.float32)),
                None, [{'loss': 1.}], {'arrived_mask': np.array([True, False])})
    predicted, actual, old = histories()
    predicted.update(scale_apply=torch.ones(2), active=torch.ones(2, dtype=torch.bool),
        flip=torch.zeros(2, dtype=torch.bool), newly=torch.zeros(2, dtype=torch.bool), telemetry={})
    class Context:
        def __init__(self, index):
            self.index, self._original, self.choice = index, deepcopy(original), object()
        def original(self):
            return self.choice
        def inspect(self, choice):
            assert choice is self.choice
            return dict(window=self.index, coefficients=torch.zeros(1, 6))
        def resolve(self, choice):
            assert choice is self.choice
            return deepcopy(self._original), {'selected': False}
        def current_admission_history(self):
            return deepcopy(old if self.index == 19 else actual)
        def preview_current_successor(self):
            assert self.index == 19
            if mutate_original:
                self._original[0][-1][0, 0] += 1
            return State(), deepcopy(predicted), {'scope': 'current original only'}
    capture = probe.CurrentPreparationCapture(tmp_path)
    capture.donor = deepcopy(original)
    capture.start = {'pin': torch.tensor([1., 0.])}
    first, second = Context(19), Context(20)
    if mutate_original:
        with pytest.raises(ValueError, match='Preview changed original W20'):
            capture.select(first)
        assert not capture.receipt['w20_identity']
        return
    assert capture.select(first) is first.choice
    capture.w21_donor = deepcopy(original)
    assert capture.select(second) is second.choice
    assert capture.receipt['w20_identity'] and capture.receipt['w21_identity']
    assert 'estimate_scope' not in capture.receipt
    assert capture.receipt['preparation_source'].startswith('Current original head')
    assert capture.receipt['new_pins'] == 0
    assert (tmp_path/'predicted_successor.npz').exists()
    assert (tmp_path/'window_21_admission_history.npz').exists()
    assert capture.estimate is None and not capture.enabled
