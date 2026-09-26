"""Arrival evidence must distinguish geometric arrival from legacy eligibility."""
import numpy as np
import pytest

from physmorph.pipeline.settlement import accepted_arrivals, pin_arrival_evidence


def test_accepted_arrival_detects_departure_and_new_arrival():
    x = np.array([[0.2, 0, 0], [0.01, 0, 0]], np.float32)
    assert accepted_arrivals(x, np.zeros_like(x), 0.1, [True, False]).tolist() == [False, True]
    assert not accepted_arrivals(x, None, None).any()


def test_confirmed_arrival_keeps_transit_protection_until_both_endpoints_arrive():
    x = np.array([[.2, 0, 0], [.01, 0, 0], [.02, 0, 0]], np.float32)
    start = [True, False, True]
    result = accepted_arrivals(x, np.zeros_like(x), .1, start, require_start=True)
    assert result.tolist() == [False, False, True]
    assert start == [True, False, True]
    for images, radius, mask in [(None, None, start), (x, .1, None), (x, .1, [True])]:
        with pytest.raises(ValueError, match='window-start mask'):
            accepted_arrivals(x, images, radius, mask, require_start=True)


@pytest.mark.parametrize('confirm', [False, True])
def test_pin_evidence_separates_endpoint_arrival_from_confirmed_eligibility(confirm):
    x = np.array([[.2, 0, 0], [.01, 0, 0], [.02, 0, 0]], np.float32)
    start = np.array([True, False, True])
    evidence = pin_arrival_evidence(x, np.zeros_like(x), .1, start, require_start=confirm)
    expected = [False, False, True] if confirm else [False, True, True]
    assert evidence.eligible.tolist() == expected
    assert evidence.telemetry() == {
        'arrived_end_frac': 2 / 3,
        'pin_arrival_evidence': 'accepted_full_plan',
        'pin_arrival_eligible_frac': sum(expected) / 3,
    }
    assert start.tolist() == [True, False, True]


@pytest.mark.parametrize('missing', ['images', 'radius', 'both'])
def test_legacy_pin_eligibility_never_claims_geometric_arrival(missing):
    x = np.zeros((3, 3), np.float32)
    images = None if missing in ('images', 'both') else x
    radius = None if missing in ('radius', 'both') else .1
    evidence = pin_arrival_evidence(x, images, radius)
    assert evidence.eligible.all()
    assert evidence.telemetry() == {
        'arrived_end_frac': None,
        'pin_arrival_evidence': 'legacy_no_arrival_contract',
        'pin_arrival_eligible_frac': 1.0,
    }
    # A supplied legacy mask is copied, never broadened or modified by the helper.
    start = np.array([False, True, False])
    evidence = pin_arrival_evidence(x, images, radius, start)
    assert evidence.eligible.tolist() == start.tolist()
    evidence.eligible[:] = True
    assert start.tolist() == [False, True, False]


@pytest.mark.parametrize('missing', ['images', 'radius', 'start', 'shape'])
def test_confirmed_pin_evidence_rejects_incomplete_contract(missing):
    x = np.zeros((3, 3), np.float32)
    start = None if missing == 'start' else ([True] if missing == 'shape' else [True] * 3)
    with pytest.raises(ValueError, match='window-start mask'):
        pin_arrival_evidence(x, None if missing == 'images' else x,
                             None if missing == 'radius' else .1, start, require_start=True)


def test_pin_evidence_excludes_nonfinite_positions_and_keeps_boundary_arrival():
    x = np.array([[.1, 0, 0], [np.nan, 0, 0], [0, 0, 0]], np.float32)
    images = np.zeros_like(x)
    images[2, 0] = np.inf
    evidence = pin_arrival_evidence(x, images, .1)
    assert evidence.eligible.tolist() == [True, False, False]
    assert evidence.end_fraction == 1 / 3
