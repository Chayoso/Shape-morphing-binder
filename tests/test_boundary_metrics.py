"""Independent analytic examples for the Torch boundary-replay report, CPU only."""
import copy
import json
import math

import pytest
import torch

from scripts.probes.boundary_metrics import summarize_boundary


def vectors(x):
    return torch.tensor([[v, 0., 0.] for v in x], dtype=torch.float64)


def make_case(j, last, pic_step, raw_step, *, repeat_step=None, pin=None, layer=None, dt=.5):
    n = len(j)
    raw = torch.zeros(n, 3, dtype=torch.float64)
    promoted = vectors(j)
    pin = torch.zeros(n, dtype=torch.bool) if pin is None else torch.tensor(pin, dtype=torch.bool)
    layer = torch.ones(n, dtype=torch.bool) if layer is None else torch.tensor(layer, dtype=torch.bool)
    data = dict(previous=raw-vectors(last), raw=raw, promoted=promoted, pin=pin, layer_mask=layer)
    starts = dict(pic=promoted.clone(), pic_repeat=promoted.clone(),
                  free_raw=torch.where(pin[:, None], promoted, raw))
    branches = {}
    for name, values in (('pic', pic_step), ('pic_repeat', pic_step if repeat_step is None else repeat_step),
                         ('free_raw', raw_step)):
        step = vectors(values)
        step[pin] = 0
        x0 = starts[name]
        branches[name] = dict(x0=x0, x1=x0+step, pre_layer=x0+step*.75,
                              v1=step*.5/dt, F1=torch.eye(3, dtype=raw.dtype).expand(n, -1, -1).clone(),
                              C1=torch.zeros(n, 3, 3, dtype=raw.dtype))
    return data, branches


def test_changed_endpoint_is_not_an_induced_response_and_components_close():
    data, branches = make_case([1., 1.], [.5, .5], [.25, .25], [.25, .25])
    got = summarize_boundary(data, branches, dt=.5, spacing=.25)['cohorts']['all_free']
    assert got['endpoint_difference']['wu']['rms'] == 1.
    assert got['induced_next_response']['response']['wu']['max'] == 0.
    assert got['alignment_with_negative_j']['global_projection_coefficient'] == 0.
    assert got['alignment_with_negative_j']['global_cosine'] is None
    components = got['branches']['pic']
    assert components['advection']['wu']['mean'] == .125
    assert components['pre_layer_residual']['wu']['mean'] == .0625
    assert components['layer']['wu']['mean'] == .0625
    assert components['total']['sp']['mean'] == 1.
    assert components['component_closure']['wu']['max'] == 0.
    assert got['endpoint_identity_residual']['wu']['max'] == 0.
    assert 'bond' not in components


def test_induced_advection_and_layer_can_cancel_without_zero_component_response():
    data, branches = make_case([1., 2., 3.], [.25]*3, [0.]*3, [0.]*3,
                               pin=[False, False, True], layer=[True, False, True])
    # A moves -j/4 before the layer operator; its layer then undoes this.
    # B has neither movement. Final next displacements are identical, while
    # the two induced components are nonzero and point in opposite directions.
    for arm in ('pic', 'pic_repeat'):
        branches[arm]['v1'] = vectors([-.5, -1., 0.])
        branches[arm]['pre_layer'] = branches[arm]['x0']+.5*branches[arm]['v1']
    report = summarize_boundary(data, branches, dt=.5, spacing=.25)
    for name, count in (('all_free', 2), ('layer_free', 1)):
        got = report['cohorts'][name]
        parts = got['induced_components']
        assert got['induced_next_response']['response']['wu']['max'] == 0.
        assert got['alignment_with_negative_j']['global_cosine'] is None
        assert parts['advection']['wu']['count'] == count
        assert parts['advection']['wu']['mean'] == (.375 if count == 2 else .25)
        assert parts['layer']['sp']['mean'] == (1.5 if count == 2 else 1.)
        for component, sign in (('advection', 1), ('layer', -1)):
            alignment = parts[component]['alignment_with_negative_j']
            assert alignment['global_projection_coefficient'] == sign*.25
            assert alignment['global_cosine'] == sign
        assert parts['pre_layer_residual']['wu']['max'] == 0.
        assert parts['sum_vs_induced_response_closure']['wu']['max'] == 0.
    assert report['pin_checks']['pic']['pre_layer_position_exact']


def test_known_recoil_common_cohorts_alignment_and_reversal_counts():
    data, branches = make_case([1., 2., 0., 3.], [.2]*4,
                               [-.4, -.9, .1, 9.], [.1]*4,
                               pin=[False, False, False, True], layer=[True, False, True, True])
    report = summarize_boundary(data, branches, dt=.5, spacing=.2)
    all_free, thin = report['cohorts']['all_free'], report['cohorts']['layer_free']
    assert all_free['count'] == 3 and thin['count'] == 2
    # Response is [-.5,-1,0], never the final-position difference [+.5,+1,0].
    assert all_free['induced_next_response']['response']['wu']['rms'] == pytest.approx(math.sqrt(1.25/3))
    assert all_free['alignment_with_negative_j']['global_projection_coefficient'] == pytest.approx(.5)
    assert all_free['alignment_with_negative_j']['global_cosine'] == pytest.approx(1.)
    assert all_free['alignment_with_negative_j']['direction_pair_count'] == 2
    for component, fraction in (('advection', .5), ('pre_layer_residual', .25), ('layer', .25)):
        part = all_free['induced_components'][component]
        assert part['wu']['rms'] == pytest.approx(fraction*math.sqrt(1.25/3))
        assert part['alignment_with_negative_j']['global_projection_coefficient'] == pytest.approx(.5*fraction)
    assert all_free['induced_components']['sum_vs_induced_response_closure']['wu']['max'] < 1e-15
    assert all_free['reversals']['pic'] == dict(eligible=3, reversals=2, fraction=2/3)
    assert all_free['reversals']['free_raw']['fraction'] == 0
    assert thin['reversals']['pic']['fraction'] == .5
    assert all_free['reversals']['common_eligible']['pic_only_count'] == 2
    assert all_free['last_raw_step']['wu']['mean'] == pytest.approx(.2)
    assert all_free['last_saved_step']['wu']['mean'] == pytest.approx(1.2)
    assert all_free['boundary_j']['sp']['max'] == 10.
    for name in branches:
        assert report['pin_checks'][name]['position_exact']
        assert report['pin_checks'][name]['pre_layer_position_exact']
        assert report['pin_checks'][name]['v1_zero']
        assert report['pin_checks'][name]['C1_zero']


def test_projection_is_global_least_squares_not_mean_of_row_coefficients():
    data, branches = make_case([1., 2.], [.2, .2], [-.9, .1], [.1, .1])
    got = summarize_boundary(data, branches, .5, .2)['cohorts']['all_free']['alignment_with_negative_j']
    # dot([-1,0],[-1,-2]) / (1+4) = .2, whereas the mean row coefficient is .5.
    assert got['global_projection_coefficient'] == pytest.approx(.2)
    assert got['global_cosine'] == pytest.approx(1/math.sqrt(5))
    assert got['parallel_displacement_wu']['mean'] == pytest.approx(.5)


def test_repeat_noise_and_state_response_units():
    data, branches = make_case([1., 1.], [.2, .2], [-.4, -.4], [.1, .1], repeat_step=[-.3, -.3])
    branches['pic']['F1'][:, 0, 0] += .5
    branches['pic_repeat']['F1'][:, 0, 0] += .7
    branches['pic']['C1'][:, 0, 1] += 2.
    branches['pic_repeat']['C1'][:, 0, 1] += 2.25
    got = summarize_boundary(data, branches, .5, .2)['cohorts']['all_free']
    noise = got['induced_next_response']
    assert noise['response']['wu']['rms'] == pytest.approx(.5)
    assert noise['pic_repeat_noise']['wu']['rms'] == pytest.approx(.1)
    assert noise['rms_signal_to_noise'] == pytest.approx(5.)
    assert noise['fraction_above_pointwise_repeat_and_threshold'] == 1.
    states = got['state_differences']['pic_minus_free_raw']
    assert states['v1']['wu_per_s']['rms'] == pytest.approx(.5)
    assert states['v1']['sp_per_s']['rms'] == pytest.approx(2.5)
    assert states['F1']['frobenius']['rms'] == pytest.approx(.5)
    assert states['C1']['frobenius']['rms'] == 2.
    assert states['F1']['units'] == 'dimensionless' and states['C1']['units'] == '1/s'
    repeat = got['state_differences']['pic_repeat_minus_pic']
    assert repeat['F1']['frobenius']['rms'] == pytest.approx(.2)
    assert repeat['C1']['frobenius']['rms'] == .25


def test_zero_repeat_noise_is_explicit_and_json_has_no_infinity():
    data, branches = make_case([1.], [.2], [-.4], [.1])
    got = summarize_boundary(data, branches, .5, .2)
    json.dumps(got, allow_nan=False)
    noise = got['cohorts']['all_free']['induced_next_response']
    assert noise['noise_status'] == 'zero_measured_repeat_noise'
    assert noise['rms_signal_to_noise'] is None
    assert noise['response']['wu']['rms'] > 0


def test_reversal_requires_both_steps_strictly_above_spacing_threshold():
    eps = .5e-4
    data, branches = make_case([0.]*4, [0., eps, 2*eps, 2*eps],
                               [-2*eps, -2*eps, -eps, -2*eps], [-2*eps]*4)
    got = summarize_boundary(data, branches, .5, .5)
    assert got['direction_threshold_wu'] == eps
    reversal = got['cohorts']['all_free']['reversals']
    assert reversal['pic'] == dict(eligible=1, reversals=1, fraction=1.)
    assert reversal['free_raw'] == dict(eligible=2, reversals=2, fraction=1.)
    assert reversal['common_eligible']['count'] == 1


def test_shared_preceding_reversal_separates_changed_history_from_next_response():
    # The remap reverses the preceding direction, while ID 0 has identical next
    # motion in both arms. Whole-alternative reversal differs even without a
    # next-step response; holding the prior fixed correctly removes that effect.
    data, branches = make_case([-.5, -.5], [.25, .25], [.125, -.125], [.125, .125])
    report = summarize_boundary(data, branches, .5, .25)
    got = report['cohorts']['all_free']['reversals']
    assert 'BOTH prior and next' in report['definitions']['whole_alternative_reversal']
    assert got['pic']['fraction'] == .5
    assert got['free_raw']['fraction'] == 0.
    shared = got['shared_preceding_saved']
    assert shared['pic']['fraction'] == .5
    assert shared['free_raw']['fraction'] == 1.
    assert shared['common_eligible'] == dict(count=2, pic_fraction=.5, free_raw_fraction=1.,
                                             pic_only_count=0, free_raw_only_count=1)
    # Select only ID 0: no induced response and no common-history reversal delta.
    data['layer_mask'][1] = False
    one = summarize_boundary(data, branches, .5, .25)['cohorts']['layer_free']
    assert one['induced_next_response']['response']['wu']['max'] == 0.
    assert one['reversals']['common_eligible']['pic_only_count'] == 1
    assert one['reversals']['shared_preceding_saved']['common_eligible']['pic_only_count'] == 0


@pytest.mark.parametrize('all_pinned', [False, True])
def test_empty_cohorts_are_absence_not_zero_motion_evidence(all_pinned):
    data, branches = make_case([1., 2.], [.2, .2], [.1, .1], [.1, .1],
                               pin=[all_pinned]*2, layer=[False, False])
    got = summarize_boundary(data, branches, .5, .2)
    empty = got['cohorts']['all_free' if all_pinned else 'layer_free']
    assert empty['count'] == 0
    assert empty['boundary_j']['wu']['rms'] is None
    assert empty['reversals']['pic'] == dict(eligible=0, reversals=0, fraction=None)
    assert empty['induced_next_response']['noise_status'] == 'empty_cohort'
    assert empty['state_differences']['pic_minus_free_raw']['F1']['frobenius']['count'] == 0
    assert empty['induced_components']['advection']['wu']['rms'] is None
    assert empty['induced_components']['layer']['alignment_with_negative_j']['global_cosine'] is None
    assert empty['induced_components']['sum_vs_induced_response_closure']['wu']['max'] is None
    json.dumps(got, allow_nan=False)


def test_pin_violations_reported_and_initial_intervention_enforced():
    data, branches = make_case([1., 2.], [.2, .2], [.1, .1], [.1, .1], pin=[True, False])
    branches['pic']['x1'][0, 1] += .125
    branches['pic']['v1'][0, 1] += 1.
    branches['pic']['C1'][0, 0, 0] += 1.
    got = summarize_boundary(data, branches, .5, .2)
    assert not got['pin_checks']['pic']['position_exact']
    assert got['pin_checks']['pic']['max_position_drift_wu'] == .125
    assert not got['pin_checks']['pic']['v1_zero'] and not got['pin_checks']['pic']['C1_zero']
    # Pin row is intentionally held at promoted even in free_raw.
    branches['free_raw']['x0'][0] = data['raw'][0]
    with pytest.raises(ValueError, match='fixed boundary intervention'):
        summarize_boundary(data, branches, .5, .2)


def test_matrix_layouts_no_mutation_no_autograd_or_tensor_output():
    data, branches = make_case([1., 2.], [.2, .2], [.1, .1], [.1, .1])
    branches['pic_repeat']['F1'] = branches['pic_repeat']['F1'].reshape(-1, 9)
    branches['free_raw']['C1'] = branches['free_raw']['C1'].reshape(-1, 9)
    data['previous'].requires_grad_()
    originals = copy.deepcopy((data, branches))
    result = summarize_boundary(data, branches, .5, .2)
    json.dumps(result, allow_nan=False)
    for key in data:
        assert torch.equal(data[key], originals[0][key])
    for arm in branches:
        for key in branches[arm]:
            assert torch.equal(branches[arm][key], originals[1][arm][key])
    assert data['previous'].grad is None


@pytest.mark.parametrize('dt,spacing', [(0., .2), (.5, 0.), (float('nan'), .2), (.5, float('inf'))])
def test_invalid_units_fail_closed(dt, spacing):
    data, branches = make_case([1.], [.2], [.1], [.1])
    with pytest.raises(ValueError):
        summarize_boundary(data, branches, dt, spacing)


def test_invalid_tensor_contracts_fail_closed():
    data, branches = make_case([1.], [.2], [.1], [.1])
    with pytest.raises(ValueError, match='branches must'):
        summarize_boundary(data, {'pic': branches['pic']}, .5, .2)
    data['layer_mask'] = data['layer_mask'].float()
    with pytest.raises(ValueError, match='dtype'):
        summarize_boundary(data, branches, .5, .2)
    data['layer_mask'] = data['layer_mask'].bool()
    branches['pic']['v1'][0, 0] = torch.nan
    with pytest.raises(ValueError, match='finite'):
        summarize_boundary(data, branches, .5, .2)
