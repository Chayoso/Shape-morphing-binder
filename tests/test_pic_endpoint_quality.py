"""CPU raw-geometry oracles, independent of the rendering/loss path."""
import json

import numpy as np
import pytest

from physmorph.metrics import target_extent, sil_iou, hole_frac
from scripts.probes.pic_endpoint_quality import prepare_reference, endpoint_quality


@pytest.fixture
def cloud():
    axes = (np.linspace(-.4, .4, 5), np.linspace(2.1, 2.9, 5), np.linspace(-.4, .4, 5))
    return np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)


def dense_distance(a, b):
    return np.linalg.norm(a[:, None]-b[None, :], axis=-1)


def test_identity_and_missing_top_material_have_distinct_raw_quality(cloud):
    ref = prepare_reference(cloud, cloud)
    same = endpoint_quality(cloud, ref)
    truncated = cloud.copy()
    truncated[truncated[:, 1] > 2.3, 1] = 2.1
    bad = endpoint_quality(truncated, ref)
    assert same['chamfer'] == 0 and same['sil_iou'] == 1.
    assert same['target_near_frac'] == 1. and same['top_target_near_frac'] == 1.
    assert same['source_gap_sp'] == same['target_gap_sp'] == dict(median=0., p95=0., max=0.)
    assert bad['chamfer'] > same['chamfer'] and bad['sil_iou'] < same['sil_iou']
    assert bad['top_target_near_frac'] < same['top_target_near_frac']
    assert bad['target_gap_sp']['max'] > 0 and bad['tip_n'] < same['tip_n']
    assert bad['top_n'] == 0 and bad['top_density'] is None and bad['top_under_half'] is None
    assert bad['top_target_n'] > 0  # Fixed target cohort survives the candidate's loss of its top.
    assert same['finite'] and same['target_fixed']
    json.dumps(bad, allow_nan=False)


def test_dense_oracle_matches_existing_distance_density_and_threshold_definitions(cloud):
    source = (cloud-np.array([0., 2.5, 0.]))*1.7
    ref = prepare_reference(source, cloud)
    target_pairs = dense_distance(cloud, cloud)
    target_knn = np.sort(target_pairs, axis=1)
    expected_radius = np.median(target_knn[:, 8])
    assert ref['radius'] == pytest.approx(expected_radius)
    assert ref['radius'] != pytest.approx(2*ref['source_spacing'])
    assert ref['target_spacing'] == pytest.approx(np.median(target_knn[:, 1]))
    assert ref['source_spacing'] == pytest.approx(np.median(np.sort(dense_distance(source, source), axis=1)[:, 1]))
    x = cloud.copy()
    x[::3, 0] += .73
    result = endpoint_quality(x, ref)
    pair = dense_distance(x, cloud)
    sd, td = pair.min(axis=1), pair.min(axis=0)
    top, top_target = x[:, 1] > 2.3, cloud[:, 1] > 2.3
    counts = (dense_distance(x[top], x) <= expected_radius).sum(axis=1)-1
    ts = ref['target_spacing']
    assert result['chamfer'] == pytest.approx(sd.mean()+td.mean())
    assert result['target_near_frac'] == pytest.approx((td <= 2*ts).mean())
    assert result['top_target_near_frac'] == pytest.approx((td[top_target] <= 2*ts).mean())
    assert result['source_out_far_frac'] == pytest.approx((sd > 4.5*ts).mean())
    assert result['top_density'] == pytest.approx(counts.mean()/8)
    assert result['top_under_half'] == pytest.approx((counts < 4).mean())
    for key, values in (('source_gap_sp', sd/ts), ('target_gap_sp', td/ts), ('top_target_gap_sp', td[top_target]/ts)):
        assert result[key] == pytest.approx(dict(median=np.median(values), p95=np.percentile(values, 95), max=values.max()))


def test_shared_projection_extent_does_not_autofit_translated_candidate(cloud):
    ref = prepare_reference(cloud, cloud)
    extent = target_extent(cloud)
    moved = cloud+np.array([20., 20., 20.])
    result = endpoint_quality(moved, ref)
    assert result['extent'] == ref['extent'] == extent
    assert result['sil_iou'] == sil_iou(moved, cloud, extent)
    assert result['hole_frac'] == hole_frac(moved, extent)
    assert result['sil_iou'] < .1 and result['source_out_far_frac'] == 1.
    assert result['target_near_frac'] == 0.


def test_reference_owns_target_and_tip_without_mutating_input(cloud):
    original = cloud.copy()
    ref = prepare_reference(cloud, cloud)
    expected = endpoint_quality(original, ref)
    np.testing.assert_array_equal(cloud, original)
    cloud[:] += 100
    np.testing.assert_array_equal(ref['target'], original)
    np.testing.assert_array_equal(ref['tip'], original[original[:, 1].argmax()])
    assert endpoint_quality(original, ref) == expected


def test_empty_target_top_and_candidate_top_are_null_safe(cloud):
    cloud = cloud-np.array([0., 4., 0.])
    ref = prepare_reference(cloud, cloud)
    result = endpoint_quality(cloud, ref)
    assert result['top_n'] == result['top_target_n'] == 0
    assert result['top_density'] is result['top_under_half'] is None
    assert result['top_target_near_frac'] is result['top_target_gap_sp'] is None
    assert result['target_near_frac'] == 1. and result['chamfer'] == 0
    json.dumps(result, allow_nan=False)


def test_tip_uses_strict_quarter_unit_radius_and_first_argmax():
    axis = np.array([0., .125, .25])
    xx, zz = np.meshgrid(axis, axis, indexing='ij')
    target = np.stack([xx.ravel(), np.full(9, 2.5), zz.ravel()], axis=1)
    ref = prepare_reference(target, target)
    np.testing.assert_array_equal(ref['tip'], target[0])
    x = ref['tip']+np.array([[0., 0., 0.], [.25, 0., 0.], [0., .25, 0.]])
    result = endpoint_quality(x, ref)
    assert result['tip_n'] == 1  # The two points at exactly .25 are excluded.
    assert result['top_density'] == pytest.approx((2+1+1)/3/8)
    assert result['top_under_half'] == 1.


@pytest.mark.parametrize('which', ['source', 'target', 'endpoint'])
def test_nonfinite_inputs_fail_closed(cloud, which):
    damaged = cloud.copy()
    damaged[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        if which == 'endpoint':
            endpoint_quality(damaged, prepare_reference(cloud, cloud))
        else:
            prepare_reference(damaged if which == 'source' else cloud, damaged if which == 'target' else cloud)


def test_invalid_reference_and_backend_mismatch_fail_closed(cloud):
    with pytest.raises(ValueError, match='N >= 9'):
        prepare_reference(cloud, cloud[:8])
    with pytest.raises(ValueError, match='spacing'):
        prepare_reference(np.ones((9, 3)), cloud)
    with pytest.raises(ValueError, match='spacing'):
        prepare_reference(cloud, np.ones((9, 3)))
    ref = prepare_reference(cloud, cloud)
    with pytest.raises(ValueError, match='N >= 1'):
        endpoint_quality(np.empty((0, 3)), ref)
    ref['cuda'] = True
    with pytest.raises(ValueError, match='same compute backend'):
        endpoint_quality(cloud, ref)
