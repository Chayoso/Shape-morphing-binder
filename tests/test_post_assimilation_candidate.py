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


class NoArrayRead:
    """Numerical return/lease sentinels must never be copied for scalar reporting."""
    def __array__(self, *args, **kwargs):
        raise AssertionError('Unexpected numerical array download')
    def __deepcopy__(self, memo):
        raise AssertionError('Unexpected numerical state/lease ownership')
    def __float__(self):
        raise AssertionError('Unexpected scalar extraction')


def optimizer_result(index, steps=8):
    history, influence = [], []
    for iteration in range(steps):
        weight = 1.+index/10
        step = dict(iteration=iteration, lambda_render=weight, nominal_render_share=(iteration+1)/10,
            direction_statistics_available=True, render_loss_before=2.+index,
            render_loss_after=2.+index-(iteration+1)/100,
            observed_render_loss_change=-(iteration+1)/100,
            optimization_endpoint_change_rms_wu=.001*(iteration+1),
            channels={'stress': dict(nominal_render_share=.25, accepted_control_delta_norm=.125,
                optimizer_render_direction_dot_delta=-.01, weighted_optimizer_render_direction_dot_delta=-weight*.01)})
        influence.append(step)
        history.append(dict(iter=iteration, loss=3.-iteration/100, d_vol=2., d_render=step['render_loss_after'],
                            **{'lambda': weight}, render_influence=step))
    state = NoArrayRead()
    stats = dict(accepted=steps, rejected=2, g_share=.4, render_influence_steps=influence,
                 render_channels={'stress': {'share': .2}}, dfc=state, _window_selection=state)
    return (state, state, state, state, history, stats)


def test_failure_after_twentieth_optimizer_return_retains_exact_inner_telemetry(tmp_path):
    cfg, prm, _ = recipe()
    capture = probe.CandidateCapture(tmp_path, np.zeros((2, 3)), None, None,
                                     config=asdict(cfg), mpm=asdict(prm))
    results = [optimizer_result(index) for index in range(20)]
    original_histories = deepcopy([result[4] for result in results])
    calls = []
    def original(*args, **kwargs):
        calls.append(kwargs['win_index'])
        return results[kwargs['win_index']]
    wrapped = capture.wrap(original)
    for index in range(19):
        assert wrapped(win_index=index) is results[index]
    # Actual parent wrapper's post-return preparation check fails before selection:
    # this fake optimizer deliberately did not instantiate a prepared Trajectory.
    with pytest.raises(ValueError, match='Expected one ordinary prepared trajectory'):
        wrapped(win_index=19)
    assert calls == list(range(20))
    saved = json.loads((tmp_path/'optimizer_progress.json').read_text())
    assert len(saved['optimizer_returns']) == 20
    assert saved['summary']['inner_accepted_updates_reported'] == 20*8
    assert saved['summary']['recorded_inner_render_steps'] == 20*8
    assert saved['summary']['recorded_inner_history_rows'] == 20*8
    assert saved['discretization']['N'] == 2
    assert saved['discretization']['T'] == cfg.T and saved['discretization']['dt'] == prm.dt
    assert saved['summary']['outer_acceptance'] == 'unknown'
    assert not any('committed' in key for key in saved['summary'])
    for index, row in enumerate(saved['optimizer_returns']):
        assert row['win_index'] == index and row['optimizer_return_ordinal'] == index+1
        assert row['outer_acceptance'] == 'unknown'
        assert row['history'] == original_histories[index] == results[index][4]
        assert row['accepted_render_steps'] == results[index][5]['render_influence_steps']
        assert row['omitted_non_scalar_paths'] == []
        assert 'dfc' not in row['statistics'] and '_window_selection' not in row['statistics']
    # Same raw inner observations as the standard complete-report arithmetic,
    # without assigning its outer-commit scope to this partial receipt.
    from physmorph.pipeline.render_reporting import summarize_render_influence
    complete = summarize_render_influence([dict(animation=i, outer_accepted=True,
        **result[5], **{'lambda': result[4][-1]['lambda']}) for i, result in enumerate(results)], {}, {})
    for name in ('step_nominal_share', 'first_iteration_nominal_share', 'observed_render_loss_change',
                 'endpoint_optimizer_change_rms_wu', 'channels'):
        assert saved['summary'][name] == complete[name]
    assert saved['summary']['adaptive_lambda']['count'] == 160  # Per-update, not per-window.
    assert capture.files['optimizer_progress.json'] == probe.identity(tmp_path/'optimizer_progress.json')
    assert capture.optimizer_progress_receipt()['binding'] == capture.files['optimizer_progress.json']
    assert capture.optimizer_progress_receipt()['observed_returns_persisted']


def test_progress_owns_only_scalar_containers_and_never_downloads_arrays(tmp_path):
    capture = probe.CandidateCapture(tmp_path, None, None, None)
    result = optimizer_result(0, steps=1)
    result[4][0]['optional_array'] = NoArrayRead()
    result[4][0]['cpu_scalar'] = np.float64(3.)
    capture.record_optimizer_return(0, result)
    before = json.loads((tmp_path/'optimizer_progress.json').read_text())
    result[4][0]['loss'] = 999.
    result[5]['render_influence_steps'][0]['channels']['stress']['nominal_render_share'] = 999.
    assert capture.optimizer_returns[0] == before['optimizer_returns'][0]
    assert before['optimizer_returns'][0]['omitted_non_scalar_paths'] == ['.history[0].optional_array']
    assert before['optimizer_returns'][0]['history'][0]['optional_array'] is None
    assert before['optimizer_returns'][0]['history'][0]['cpu_scalar'] == 3.


def test_optimizer_exception_does_not_invent_a_return_or_outer_acceptance(tmp_path):
    capture = probe.CandidateCapture(tmp_path, None, None, None)
    def original(*args, **kwargs):
        raise RuntimeError('ordinary solve did not return')
    with pytest.raises(RuntimeError, match='did not return'):
        capture.wrap(original)(win_index=0)
    assert not (tmp_path/'optimizer_progress.json').exists()
    receipt = capture.optimizer_progress_receipt()
    assert receipt['artifact'] is None and receipt['summary']['optimizer_returns'] == 0
    assert receipt['summary']['adaptive_lambda'] is None
    assert receipt['summary']['outer_acceptance'] == 'unknown'


def test_progress_write_failure_preserves_last_durable_receipt_and_reserve(tmp_path, monkeypatch):
    capture = probe.CandidateCapture(tmp_path, None, None, None)
    capture.record_optimizer_return(0, optimizer_result(0, steps=1))
    path = tmp_path/'optimizer_progress.json'
    previous = path.read_bytes()
    bound = dict(capture.files[path.name])
    # Even the temporary replacement must fit alongside the old receipt.
    monkeypatch.setattr(probe, 'LIMIT', path.stat().st_size+probe.RESERVE)
    with pytest.raises(ValueError, match='reserved12GB'):
        capture.record_optimizer_return(1, optimizer_result(1, steps=1))
    assert path.read_bytes() == previous and capture.files[path.name] == bound
    assert not (tmp_path/'optimizer_progress.json.tmp').exists()
    receipt = capture.optimizer_progress_receipt()
    assert receipt['summary']['optimizer_returns'] == 2 and receipt['persisted_optimizer_returns'] == 1
    assert not receipt['observed_returns_persisted']


def test_inner_zero_update_and_unavailable_directions_stay_explicit(tmp_path):
    capture = probe.CandidateCapture(tmp_path, None, None, None)
    capture.record_optimizer_return(0, optimizer_result(0, steps=0))
    result = optimizer_result(1, steps=1)
    step = result[5]['render_influence_steps'][0]
    step.update(direction_statistics_available=False, nominal_render_share=None,
                observed_render_loss_change=None, render_loss_before=None, render_loss_after=None)
    step['channels'] = {'stress': dict(accepted_control_delta_norm=.125, direction_statistics=None)}
    result[4][0]['lambda'] = None
    capture.record_optimizer_return(1, result)
    saved = json.loads((tmp_path/'optimizer_progress.json').read_text())
    summary = saved['summary']
    assert summary['optimizer_returns'] == 2 and summary['inner_accepted_updates_reported'] == 1
    assert summary['recorded_inner_render_steps'] == 1
    assert summary['step_nominal_share'] is summary['adaptive_lambda'] is summary['observed_render_loss_change'] is None
    assert summary['channels']['stress']['nominal_share'] is None
    assert summary['channels']['stress']['control_delta_norm']['median'] == .125


def test_replacement_directory_sync_precedes_binding_and_failure_stays_explicit(tmp_path, monkeypatch):
    capture = probe.CandidateCapture(tmp_path, None, None, None)
    capture.record_optimizer_return(0, optimizer_result(0, steps=1))
    old_binding = dict(capture.files['optimizer_progress.json'])
    def failed_directory_sync(directory):
        assert directory == tmp_path
        # New bytes have been atomically installed, but metadata is not advanced.
        saved = json.loads((tmp_path/'optimizer_progress.json').read_text())
        assert saved['summary']['optimizer_returns'] == 2
        assert capture.persisted_optimizer_returns == 1
        assert capture.files['optimizer_progress.json'] == old_binding
        assert capture.progress_write_stage == 'replacement_installed_directory_sync_pending'
        raise OSError('directory fsync failed')
    monkeypatch.setattr(probe, 'sync_progress_directory', failed_directory_sync)
    with pytest.raises(OSError, match='directory fsync failed'):
        capture.record_optimizer_return(1, optimizer_result(1, steps=1))
    receipt = capture.optimizer_progress_receipt()
    assert not receipt['observed_returns_persisted'] and receipt['persisted_optimizer_returns'] == 1
    assert receipt['last_write_stage'] == 'replacement_installed_directory_sync_pending'
    assert probe.identity(tmp_path/'optimizer_progress.json') != receipt['binding']


@pytest.mark.parametrize('fail', [False, True])
def test_posix_directory_sync_closes_descriptor_even_on_failure(monkeypatch, fail):
    events = []
    def opened(path, flags):
        events.append(('open', path, flags))
        return 123
    def synced(descriptor):
        events.append(('fsync', descriptor))
        if fail:
            raise OSError('directory error')
    # Only the isolated helper sees the POSIX mock; no Windows Path construction.
    with monkeypatch.context() as scoped:
        scoped.setattr(probe.os, 'name', 'posix')
        scoped.setattr(probe.os, 'O_DIRECTORY', 0x10000, raising=False)
        scoped.setattr(probe.os, 'open', opened)
        scoped.setattr(probe.os, 'fsync', synced)
        scoped.setattr(probe.os, 'close', lambda descriptor: events.append(('close', descriptor)))
        if fail:
            with pytest.raises(OSError, match='directory error'):
                probe.sync_progress_directory('/data/evidence')
        else:
            probe.sync_progress_directory('/data/evidence')
    assert events == [('open', '/data/evidence', probe.os.O_RDONLY | 0x10000), ('fsync', 123), ('close', 123)]
