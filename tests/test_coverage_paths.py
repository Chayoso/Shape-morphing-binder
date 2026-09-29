"""Archive-only geometry diagnostics: identity, set semantics and closure."""
import numpy as np
import pytest

from scripts.probes.coverage_paths import coverage_sets, follow_paths, analyze


def test_losses_and_gains_are_not_the_net_count_and_boundary_is_inclusive():
    distances = np.array([[.5, .6, .4, .7], [.5, .6, .6, .7], [.5, .6, .4, .7],
                          [.6, .5, .4, .7]])
    got = coverage_sets(distances, .5)
    np.testing.assert_array_equal(got['lost'], [[1, 0, 0, 0]] * 3)
    np.testing.assert_array_equal(got['gained'], [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0]])
    np.testing.assert_array_equal(got['ambiguous'], [0, 0, 1, 0])
    np.testing.assert_array_equal(got['selected'], [0, 1, 2])
    assert got['covered'][3].sum() == got['covered'][0].sum()
    with pytest.raises(ValueError, match='Invalid distance'):
        coverage_sets(distances * np.nan, .5)


def test_nearest_identity_switch_is_not_fixed_material_motion():
    x0 = np.zeros((6, 3)); x0[:, 0] = [1, 2, 3, 4, 5, 9]
    X = np.broadcast_to(x0, (4, 2, 6, 3)).copy()
    X[:, 0, 5, 0] = .1  # Not among any endpoint's four suppliers.
    V = np.diff(np.concatenate((np.broadcast_to(x0, (4, 1, 6, 3)), X), axis=1), axis=1) / .25
    ids = np.broadcast_to(np.arange(4), (4, 1, 4))
    pins = np.array([1, 0, 0, 0, 0, 0], dtype=bool)
    arrived = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    result = follow_paths(x0, X, V, np.zeros((1, 3)), ids, pins, arrived, .5, .25)
    np.testing.assert_array_equal(result['material_ids'], np.arange(4))
    np.testing.assert_array_equal(result['cohort'], [0, 1, 2, 2])
    np.testing.assert_array_equal(result['nearest_ids'][:, :, 0], [[0, 5, 0]] * 4)
    np.testing.assert_array_equal(result['nearest_outside_global_endpoint_union'][:, :, 0], [[0, 1, 0]] * 4)
    np.testing.assert_array_equal(result['nearest_outside_target_endpoint_set'][:, :, 0], [[0, 1, 0]] * 4)
    np.testing.assert_array_equal(result['occupancy'][:, :, 0], [[0, 1, 0]] * 4)
    assert np.all(result['geometric_V'] == 0)
    assert np.all(result['fixed_distance'][:, :, 0, 0] == 1)
    assert result['positions'].shape[1] == 3 and result['V'].shape[1] == 2  # No invented v0.

    # Target B's endpoint supplier may still be outside target A's witness set.
    two_ids = np.broadcast_to([[0, 1, 2, 3], [5, 4, 3, 2]], (4, 2, 4))
    both = follow_paths(x0, X, V, np.array([[0, 0, 0], [10, 0, 0]]), two_ids,
                        pins, arrived, .5, .25)
    assert not both['nearest_outside_global_endpoint_union'].any()
    np.testing.assert_array_equal(both['nearest_outside_target_endpoint_set'][:, :, 0], [[0, 1, 0]] * 4)


def test_empty_selection_is_supported():
    x0 = np.arange(18, dtype=float).reshape(6, 3)
    X = np.broadcast_to(x0, (4, 2, 6, 3))
    result = follow_paths(x0, X, X * 0, np.empty((0, 3)), np.empty((4, 0, 4), dtype=int),
                          np.zeros(6, dtype=bool), np.zeros(6, dtype=bool), .5, .25)
    assert result['nearest_ids'].shape == (4, 3, 0)
    assert result['fixed_distance'].shape == (4, 3, 0, 16)


def test_real_distance_closure_and_pins_cannot_be_overridden():
    cloud = np.array([[0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0], [4, 3, 0]], dtype=float)
    obs = dict(x0=cloud, source=cloud, target=cloud, plan=cloud,
               pins=np.array([1, 0, 0, 0, 0], dtype=bool), start_arrived=np.ones(5, dtype=bool), dt=.25)
    states = [dict(positions=np.broadcast_to(cloud, (2, 5, 3)).copy(), V=np.zeros((2, 5, 3))) for _ in range(4)]
    row = dict(geometry=dict(target_near_frac=1., upper_target_near_frac=1.))
    result, endpoints, paths = analyze(obs, states, [row] * 4)
    assert result['covered'] == [5] * 4 and result['targets'] == []
    with pytest.raises(ValueError, match='coverage failed closure'):
        analyze(obs, states, [dict(geometry=dict(target_near_frac=.8, upper_target_near_frac=1.))] * 4)
    states[-1]['positions'][0, 0, 0] += .01
    with pytest.raises(ValueError, match='Pinned path changed'):
        analyze(obs, states, [row] * 4)
