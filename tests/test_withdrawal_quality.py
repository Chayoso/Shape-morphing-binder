"""P331 raw gates: fixed identities and intermediate phases cannot be hidden."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest
import torch

from scripts.probes import withdrawal_quality as module


def test_views_match_exact_p329_unstaggered_rings():
    expected = [(j*2*np.pi/8, elevation) for elevation in (0., .5, -.5) for j in range(8)]
    assert module.VIEWS == tuple(expected)
    assert module.VIEWS[0][0] == module.VIEWS[8][0] == module.VIEWS[16][0] == 0.


def fixture():
    rng = np.random.default_rng(834)
    dense = rng.uniform(-.06, .06, (80, 3)) + [0., 2.7, 0.]
    sparse = np.array([[-.8, 3.4, 0.], [.8, 3.5, 0.], [0., 3.6, .8], [0., 3.7, -.8]])
    source = np.concatenate((dense, sparse)).astype(np.float32)
    pins = np.zeros(len(source), bool)
    pins[-1] = True
    return source, source.copy(), source.copy(), pins


def trajectory(x0, head=(0., 0.), coast=(0., 0.)):
    # Markers change only particle0.x; a mock geometry reader consumes them.
    states = []
    for marker in head:
        x = x0.copy()
        x[0, 0] = marker
        states.append(x)
    tail = [states[-1].copy()]
    for marker in coast:
        x = x0.copy()
        x[0, 0] = marker
        tail.append(x)
    return dict(positions=np.stack(states), coast_X=np.stack(tail), valid=True, pins_exact=True)


def stub_observer(monkeypatch):
    """Inject recorded raw observations, not a replica of the acceptance code."""
    source, target, x0, pins = fixture()
    x0[0, 0] = 0.
    observer = module.RawWithdrawalQuality(source, target, x0, pins)
    def phase(x):
        marker = int(x[0, 0])
        covered = np.zeros(len(target), bool)
        covered[[0, 1] if marker != 1 else [1, 2]] = True
        result = dict(covered=covered, target_covered=np.int64(2), upper_target_covered=np.int64(2),
            tip_count=np.int64(3), fixed_source_density=np.float64(1.), under_half_count=np.int64(0),
            target_near_fraction=np.float64(2/len(target)), upper_target_near_fraction=np.float64(2/len(target)),
            under_half_fraction=np.float64(0.), iou=np.ones((2, 24), np.float64),
            hole_pixels=np.zeros((2, 24), np.int64), extra_hole_pixels=np.zeros((2, 24), np.int64),
            clipped_centers=np.zeros((2, 24), np.int64))
        if marker == 2:
            result['hole_pixels'][1, 9] = 1
            result['extra_hole_pixels'][1, 9] = 1
        elif marker == 3:
            result['fixed_source_density'] = np.float64(.5)
            result['under_half_count'] = np.int64(1)
        elif marker == 4:
            result['iou'][0, 4] = .99
        elif marker == 5:
            result['clipped_centers'][0, 4] = 1
        elif marker == 6:
            result['tip_count'] = np.int64(2)
        elif marker == 7:
            result['upper_target_covered'] = np.int64(1)
        elif marker == 8:
            result['target_covered'] = np.int64(1)
        return result
    monkeypatch.setattr(observer, '_phase', phase)
    return observer, x0


def establish(observer, x0):
    for i in range(3):
        row = observer.observe('original_'+str(i), trajectory(x0), baseline=True)
        assert row['passed'] and row['baseline_ready'] == (i == 2)


def test_same_coverage_count_cannot_swap_out_stable_target_ids(monkeypatch):
    observer, x0 = stub_observer(monkeypatch)
    establish(observer, x0)
    before = observer.archive_state()
    result = observer.observe('swap', trajectory(x0, head=(1., 0.)))
    assert not result['passed']
    assert result['phases'][1]['target_covered'] == result['phases'][2]['target_covered'] == 2
    assert result['phases'][1]['lost_stable_target_ids'] == 1
    assert result['failing_gates'] == [dict(gate='lost_stable_target_ids', phase=1, count=1)]
    for key, value in before.items():
        np.testing.assert_array_equal(value, observer.archive_state()[key])


@pytest.mark.parametrize('kind', ['head', 'coast'])
@pytest.mark.parametrize('marker,gate', [(2., 'hole_pixels'), (3., 'fixed_source_density'),
    (4., 'iou'), (5., 'clipped_centers'), (6., 'tip_count'),
    (7., 'upper_target_covered'), (8., 'target_covered')])
def test_intermediate_loss_fails_despite_good_head_and_coast_endpoints(monkeypatch, kind, marker, gate):
    observer, x0 = stub_observer(monkeypatch)
    establish(observer, x0)
    values = trajectory(x0, **{kind: (marker, 0.)})
    result = observer.observe('intermediate', values)
    assert not result['passed']
    assert not result['phases'][2]['failing_gates'] and not result['phases'][4]['failing_gates']
    phase = 1 if kind == 'head' else 3
    assert any(entry['gate'] == gate and entry['phase'] == phase for entry in result['failing_gates'])


def test_requires_three_baselines_and_prevents_later_widening(monkeypatch):
    observer, x0 = stub_observer(monkeypatch)
    for i in range(3):
        result = observer.observe('early_'+str(i), trajectory(x0))
        assert not result['passed'] and result['failing_gates'] == [dict(gate='three_baselines_required')]
        observer.observe('baseline_'+str(i), trajectory(x0), baseline=True)
    assert observer.observe('same', trajectory(x0))['passed']
    with pytest.raises(ValueError, match='three'):
        observer.observe('fourth', trajectory(x0, head=(3., 0.)), baseline=True)


def test_intersection_and_per_phase_envelope_use_all_three_originals(monkeypatch):
    observer, x0 = stub_observer(monkeypatch)
    for i, marker in enumerate((0., 1., 0.)):
        observer.observe('baseline_'+str(i), trajectory(x0, head=(marker, 0.)), baseline=True)
    assert observer.observe('phase1_swap', trajectory(x0, head=(1., 0.)))['passed']
    assert not observer.observe('phase3_swap', trajectory(x0, coast=(1., 0.)))['passed']
    state = observer.archive_state()
    np.testing.assert_array_equal(np.flatnonzero(state['stable_covered'][1]), [1])
    np.testing.assert_array_equal(np.flatnonzero(state['stable_covered'][3]), [0, 1])
    # Independent second witness: weaker repeat only widens its own phase.
    other, x0 = stub_observer(monkeypatch)
    for i, marker in enumerate((0., 3., 0.)):
        other.observe('baseline_'+str(i), trajectory(x0, head=(marker, 0.)), baseline=True)
    assert other.observe('same_weaker_phase', trajectory(x0, head=(3., 0.)))['passed']
    assert not other.observe('different_weaker_phase', trajectory(x0, coast=(3., 0.)))['passed']


def test_real_cohort_includes_pins_and_all_views_use_fixed_extent():
    source, target, x0, pins = fixture()
    observer = module.RawWithdrawalQuality(source, target, x0, pins)
    np.testing.assert_array_equal(observer.source_ids, [80, 81, 82, 83])
    assert observer.cohort['includes_pinned_ids'] and observer.cohort['pinned_count'] == 1
    assert observer.cohort['ids_sha256'] == sha256(np.array([80, 81, 82, 83], '<i8').tobytes()).hexdigest()
    original = observer._phase(x0)
    np.testing.assert_array_equal(original['iou'], 1.)
    np.testing.assert_array_equal(original['clipped_centers'], 0)
    assert int(original['target_covered']) == len(target)
    displaced = x0.copy()
    displaced[:, 0] += 100.
    moved = observer._phase(displaced)
    # View0 sees x displacement; view2 looks along that axis and still sees y/z.
    assert moved['clipped_centers'][0, 0] == len(source)
    assert moved['clipped_centers'][0, 2] == 0
    assert moved['iou'][0, 0] == 0. and moved['iou'][0, 2] == 1.
    assert observer.extent == pytest.approx(1.15*np.linalg.norm(target, axis=1).max())
    state = observer.archive_state()
    assert state['source_cohort_pins'].tolist() == [False, False, False, True]
    state['source_ids'][:] = 0
    np.testing.assert_array_equal(observer.source_ids, [80, 81, 82, 83])


def test_real_all_phase_report_is_json_and_retains_no_input_graph(monkeypatch):
    source, target, x0, pins = fixture()
    observer = module.RawWithdrawalQuality(source, target, x0, pins)
    # Read-only Torch inputs are detached for geometry without retaining leaves.
    head = torch.tensor(np.stack([x0]), requires_grad=True)
    coast = torch.tensor(np.stack([x0, x0]), requires_grad=True)
    values = dict(positions=head, coast_X=coast, valid=True, pins_exact=True)
    for i in range(3):
        observer.observe('original_'+str(i), values, baseline=True)
    result = observer.observe('same', values)
    assert result['passed'] and [(r['kind'], r['step']) for r in result['phases']] == [
        ('initial', 0), ('head', 1), ('coast', 1)]
    json.dumps(result, allow_nan=False)
    assert not any(torch.is_tensor(v) for v in vars(observer).values())
    for row in result['phases']:
        assert np.asarray(row['iou']).shape == (2, 24)
        assert not row['failing_gates'] and row['lost_stable_target_ids'] == 0


def test_boundary_phase_count_and_invalid_state_fail_closed(monkeypatch):
    observer, x0 = stub_observer(monkeypatch)
    establish(observer, x0)
    bad = trajectory(x0)
    bad['coast_X'][0, 0, 0] += .1
    with pytest.raises(ValueError, match='boundary'):
        observer.observe('bad_boundary', bad)
    with pytest.raises(ValueError, match='phase count'):
        observer.observe('wrong_steps', trajectory(x0, head=(0.,), coast=(0.,)))
    bad = trajectory(x0)
    bad['coast_X'][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        observer.observe('nonfinite', bad)
    invalid = observer.observe('invalid', dict(trajectory(x0), valid=False))
    assert not invalid['passed'] and invalid['failing_gates'] == [dict(gate='invalid_forward')]


def test_roundoff_is_bounded_and_integer_counts_remain_exact():
    failures = []
    bound = np.array([1.], np.float64)
    unit = 64*np.finfo(np.float64).eps*(1+bound)
    module.RawWithdrawalQuality._compare('density', bound-unit/2, bound, True, failures)
    assert not failures
    module.RawWithdrawalQuality._compare('density', bound-unit*2, bound, True, failures)
    assert failures[0]['gate'] == 'density'
    failures = []
    module.RawWithdrawalQuality._compare('count', np.array([999999]), np.array([1000000]), True, failures)
    assert failures[0]['actual'] == 999999 and failures[0]['bound'] == 1000000
