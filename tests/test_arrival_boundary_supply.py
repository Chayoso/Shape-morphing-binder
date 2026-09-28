"""CPU geometry/event reference cases; no pipeline execution."""
import hashlib

import numpy as np
import pytest

from scripts.probes.arrival_boundary_supply import boundary_summary, verified_bytes
from scripts.probes.transient_supply import frame_sample


def test_pic_losses_gains_and_endpoint_bracket_have_distinct_denominators():
    # ID0 loses support due PIC; ID1 gains; ID2 was already unsupported at start.
    counts = np.array([[4, 4, 0], [4, 0, 0], [0, 4, 4]])
    gaps = np.array([[2., 2., 50.], [1., 4., 999.], [3., 2., 1.]])
    result = boundary_summary(counts, gaps)
    assert result['source_pic_loss']['count'] == 1
    assert result['source_pic_loss']['eligible_ids'] == 1
    assert result['source_pic_gain']['count'] == 2
    assert result['source_pic_gain']['eligible_ids'] == 2
    assert result['source_raw_deficit_between_supported_endpoints']['count'] == 1
    assert result['source_raw_deficit_between_supported_endpoints']['eligible_ids'] == 1
    assert result['source_raw_deficit_between_supported_endpoints']['zero_count_ids'] == 1
    assert result['target_pic_loss']['maximum_gap_sp'] == 3.
    assert result['target_pic_gain']['maximum_gap_sp'] == 999.
    # Unfinished target ID2 is explicitly excluded from endpoint-bracket severity.
    bracket = result['target_raw_deficit_between_covered_endpoints']
    assert bracket['count'] == bracket['eligible_ids'] == 1
    assert bracket['maximum_gap_sp'] == bracket['p95_gap_sp'] == 4.


def test_exact_thresholds_and_empty_event_populations():
    result = boundary_summary(np.full((3, 2), 4), np.full((3, 2), 2.))
    assert result['source_pic_loss']['count'] == result['target_pic_loss']['count'] == 0
    assert result['source_pic_loss']['minimum_count'] is None
    assert result['target_pic_loss']['maximum_gap_sp'] is None
    assert result['source_pic_gain']['eligible_ids'] == 0
    assert result['source_pic_gain']['fraction'] is None
    empty = boundary_summary(np.empty((3, 0)), np.empty((3, 0)))
    assert empty['states']['raw_T']['target_coverage'] is None
    assert empty['target_raw_deficit_between_covered_endpoints']['fraction'] is None


def test_actual_raw_cloud_supply_recovers_at_promoted_endpoint():
    start = np.array([[0., 0, 0], [.1, 0, 0], [-.1, 0, 0], [0, .1, 0], [0, -.1, 0]])
    raw = start.copy()
    raw[:, 0] = np.arange(5)+3
    values = [frame_sample(x, np.array([0]), np.array([[0., 0, 0]]), .25, .1)
              for x in (start, raw, start)]
    result = boundary_summary(np.stack([v[0] for v in values]), np.stack([v[1] for v in values]))
    assert result['states']['previous_start']['source_density'] == .5
    assert result['states']['raw_T']['source_density'] == 0.
    assert result['states']['promoted']['target_coverage'] == 1.
    assert result['target_pic_gain']['count'] == 1
    assert result['target_raw_deficit_between_covered_endpoints']['maximum_gap_sp'] == 30.


def test_bad_three_state_samples_fail():
    with pytest.raises(ValueError, match='three-state'):
        boundary_summary(np.zeros((2, 1)), np.zeros((2, 1)))
    with pytest.raises(ValueError, match='three-state'):
        boundary_summary(np.zeros((3, 1)), np.full((3, 1), float('nan')))


def test_payload_hash_and_size_precede_parsing(tmp_path):
    path = tmp_path/'evidence.npz'
    data = b'opaque array bytes, not parsed by verification'
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert verified_bytes(path, digest, len(data)) == data
    with pytest.raises(ValueError, match='hash/size'):
        verified_bytes(path, digest, len(data)+1)
    path.write_bytes(data+b'x')
    with pytest.raises(ValueError, match='hash/size'):
        verified_bytes(path, digest)
