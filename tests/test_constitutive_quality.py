"""P300 provenance/scope guard tests; no production pipeline or GPU execution."""
from copy import deepcopy
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest

from scripts.probes import constitutive_quality as probe


@pytest.fixture
def manifests():
    old = {'physmorph/compute.py': 'unchanged'}
    new = dict(old)
    for name, pair in probe.APPROVED_BLOBS.items():
        old[name], new[name] = pair
    return (dict(root='old', digest=probe.SOURCE_DIGESTS[0], files=old),
            dict(root='new', digest=probe.SOURCE_DIGESTS[1], files=new),
            [dict(probe.HELPERS) for _ in range(3)])


def test_exact_treatment_manifest_accepted(manifests):
    result = probe.validate_snapshots(*manifests)
    assert result['changed_files'] == sorted(probe.APPROVED_BLOBS)
    assert 'not numerical source equivalence' in result['interpretation']


@pytest.mark.parametrize('change', ['digest', 'membership', 'extra_change', 'approved_blob', 'same_core', 'helper'])
def test_source_guard_rejects_unapproved_bytes(manifests, change):
    old, new, helpers = deepcopy(manifests)
    if change == 'digest':
        new['digest'] = 'not-the-run-source'
    elif change == 'membership':
        new['files']['physmorph/unreviewed.py'] = 'extra'
    elif change == 'extra_change':
        new['files']['physmorph/compute.py'] = 'different'
    elif change == 'approved_blob':
        new['files']['physmorph/mpm/constitutive.py'] = 'another-adjoint'
    elif change == 'same_core':
        new['files'] = dict(old['files'])
    else:
        helpers[2]['scripts/probes/quality_compare.py'] = 'loaded-different-helper'
    with pytest.raises(ValueError):
        probe.validate_snapshots(old, new, helpers)


@pytest.mark.parametrize('arm', [0, 1])
@pytest.mark.parametrize('name', list(probe.APPROVED_BLOBS))
def test_each_approved_blob_is_bound_to_its_arm(manifests, arm, name):
    old, new, helpers = deepcopy(manifests)
    (old, new)[arm]['files'][name] = 'unapproved-replacement'
    with pytest.raises(ValueError, match='blob pair'):
        probe.validate_snapshots(old, new, helpers)


def test_treatment_direction_and_helper_manifest_count_cannot_be_rewritten(manifests):
    old, new, helpers = manifests
    with pytest.raises(ValueError, match='source digest'):
        probe.validate_snapshots(new, old, helpers)
    for shortened in (helpers[:2], helpers+[helpers[0]]):
        with pytest.raises(ValueError, match='helper manifests'):
            probe.validate_snapshots(old, new, shortened)


@pytest.mark.parametrize('origin', [0, 1, 2])
@pytest.mark.parametrize('name', list(probe.HELPERS))
def test_every_driver_and_helper_is_bound_in_both_trees_and_loaded_code(manifests, origin, name):
    old, new, helpers = deepcopy(manifests)
    helpers[origin][name] = 'same-source-but-different-probe'
    with pytest.raises(ValueError, match='Driver or numerical helper bytes'):
        probe.validate_snapshots(old, new, helpers)


def test_snapshot_binds_paths_and_full_bytes(tmp_path):
    root = tmp_path/'snapshot'
    (root/'physmorph/mpm').mkdir(parents=True)
    a, b = b'alpha\n', b'beta\n'
    (root/'physmorph/z.py').write_bytes(a)
    (root/'physmorph/mpm/a.py').write_bytes(b)
    expected = hashlib.sha256(b'physmorph/mpm/a.py\0'+b+b'physmorph/z.py\0'+a).hexdigest()
    assert probe.snapshot(root)['digest'] == expected
    (root/'physmorph/z.py').write_bytes(a+b'# changed\n')
    assert probe.snapshot(root)['digest'] != expected


@pytest.fixture
def recipe():
    config = dict(T=20, loss_res=36, commit_pic=True, archive_stride=1, body_ctrl=True,
                  lambda_auto=.5, geometric_rest=False, alpha=.02)
    metadata = dict(arms={probe.ARM: dict(config=config)},
                    provenance=dict(mpm=dict(dt=1/240, dx=.3062907543956724)))
    fields = set(config)
    path = Path('/data/original/target_reference.npz')
    expected = probe.expected_config(metadata, fields, path)
    return metadata, fields, path, expected


def test_exact_original_cap6_recipe_accepted(recipe):
    metadata, fields, path, cfg = recipe
    mpm = metadata['provenance']['mpm']
    assert probe.validate_recipe([cfg, deepcopy(cfg)], [mpm, deepcopy(mpm)], metadata, fields, path) == cfg


@pytest.mark.parametrize('change', ['alpha', 'both_alpha', 'cap', 'both_cap', 'variance', 'dt',
                                   'target_reference', 'release'])
def test_recipe_guard_rejects_policy_or_discretization_change(recipe, change):
    metadata, fields, path, cfg = recipe
    configs = [deepcopy(cfg), deepcopy(cfg)]
    mpms = [deepcopy(metadata['provenance']['mpm']) for _ in range(2)]
    if change == 'alpha':
        configs[1]['alpha'] = .03
    elif change == 'both_alpha':
        for config in configs:
            config['alpha'] = .03
    elif change == 'cap':
        configs[1]['stop_after_windows'] = 24
    elif change == 'both_cap':
        for config in configs:
            config['stop_after_windows'] = 8
    elif change == 'variance':
        configs[1]['geometric_variance'] = True
    elif change == 'dt':
        mpms[1]['dt'] = 1/120
    elif change == 'target_reference':
        for config in configs:
            config['target_reference'] = '/data/different_reference.npz'
    else:
        for config in configs:
            config['settle_pin_yield'] = True
    with pytest.raises(ValueError):
        probe.validate_recipe(configs, mpms, metadata, fields, path)


def test_run_hash_and_internal_metadata_must_match():
    arm = dict(config={'same': True}, history=[], guards={'nan': 0})
    meta = dict(code_sha256='actual', provenance=dict(code_hash='actual', mpm={'dt': 1}),
                mpm={'dt': 1}, **deepcopy(arm), arms={probe.ARM: deepcopy(arm)})
    probe.validate_run_binding(meta, 'actual')
    with pytest.raises(ValueError, match='own frozen source'):
        probe.validate_run_binding(meta, 'rewritten')
    meta['arms'][probe.ARM]['config']['same'] = False
    with pytest.raises(ValueError, match='config'):
        probe.validate_run_binding(meta, 'actual')


@pytest.mark.parametrize('arm_index', [0, 1])
@pytest.mark.parametrize('change', ['code_hash', 'provenance_hash', 'other_arm_hash',
                                   'mpm', 'history', 'guards', 'fired_guard'])
def test_each_run_binding_rejects_relabelled_or_inconsistent_evidence(arm_index, change):
    digest = probe.SOURCE_DIGESTS[arm_index]
    arm = dict(config={'same': True}, history=[dict(animation=0)], guards={'nan': 0})
    meta = dict(code_sha256=digest, provenance=dict(code_hash=digest, mpm={'dt': 1/240}),
                mpm={'dt': 1/240}, **deepcopy(arm), arms={probe.ARM: deepcopy(arm)})
    if change == 'code_hash':
        meta['code_sha256'] = 'other-tree'
    elif change == 'provenance_hash':
        meta['provenance']['code_hash'] = 'other-tree'
    elif change == 'other_arm_hash':
        meta['code_sha256'] = meta['provenance']['code_hash'] = probe.SOURCE_DIGESTS[1-arm_index]
    elif change == 'mpm':
        meta['provenance']['mpm']['dt'] = 1/120
    elif change == 'history':
        meta['arms'][probe.ARM]['history'][0]['animation'] = 1
    elif change == 'guards':
        meta['arms'][probe.ARM]['guards']['nan'] = 1
    else:
        meta['guards']['nan'] = meta['arms'][probe.ARM]['guards']['nan'] = 1
    with pytest.raises(ValueError):
        probe.validate_run_binding(meta, digest)


def test_source_arrays_are_byte_bound(monkeypatch):
    source = np.zeros((300000, 3), np.float32)
    target = np.ones_like(source)
    monkeypatch.setattr(probe, 'SOURCE_SHA', probe.array_identity(source))
    monkeypatch.setattr(probe, 'TARGET_SHA', probe.array_identity(target))
    probe.validate_inputs(source, target)
    source[23, 1] = .0001
    with pytest.raises(ValueError, match='array hashes'):
        probe.validate_inputs(source, target)
    with pytest.raises(ValueError, match='float32'):
        probe.array_identity(target.astype(np.float64))


def test_native_input_roles_counts_and_target_bytes_are_bound(monkeypatch):
    source = np.zeros((300000, 3), np.float32)
    target = np.ones_like(source)
    monkeypatch.setattr(probe, 'SOURCE_SHA', probe.array_identity(source))
    monkeypatch.setattr(probe, 'TARGET_SHA', probe.array_identity(target))
    with pytest.raises(ValueError, match='array hashes'):
        probe.validate_inputs(target, source)
    with pytest.raises(ValueError, match='N300k'):
        probe.validate_inputs(source[:-1], target)
    with pytest.raises(ValueError, match='N300k'):
        probe.array_identity(source.reshape(300000, 1, 3))
    target[32, 2] = np.nextafter(np.float32(1), np.float32(2))
    with pytest.raises(ValueError, match='array hashes'):
        probe.validate_inputs(source, target)


def test_copied_suffix_is_excluded_but_actual_stopping_scope_retained():
    history = [dict(animation=0, frame_end=21), dict(animation=1, frame_end=41),
               dict(animation=2, held=True)]
    scope = probe.scope_history(history, delivered=42, archived=42)
    assert scope['delivered'] == 41
    assert scope['physical_indices'] == list(range(41))
    assert scope['endpoint_scope']['held_suffix_excluded'] == 1
    assert scope['endpoint_scope']['actual_accepted_commits'] == 2
    assert scope['interior_hold_frames'] == 0


def test_delivery_truncation_never_relabels_later_acceptance_geometry():
    history = [dict(animation=i, frame_end=21+20*i) for i in range(3)]
    scope = probe.scope_history(history, delivered=41, archived=61)
    assert scope['endpoint_scope']['actual_accepted_commits'] == 3
    assert scope['endpoint_scope']['delivered_accepted_commits'] == 2
    assert scope['endpoint_scope']['actual_last_accepted_frame'] == 60
    assert scope['endpoint_scope']['later_accepted_geometry_measured'] is False


@pytest.mark.parametrize('records', [[dict(animation=0, frame_end=19)],
                                     [dict(animation=6, frame_end=21)],
                                     [dict(animation=0, frame_end=21), dict(animation=1, frame_end=30)]])
def test_malformed_physical_history_is_rejected(records):
    with pytest.raises(ValueError):
        probe.scope_history(records, delivered=41, archived=41)


def test_empty_or_short_motion_is_inconclusive_not_zero():
    assert probe.matching_motion_status(0, 6) == 'inconclusive_empty_common_free_cohort'
    assert probe.matching_motion_status(12, 2) == 'inconclusive_fewer_than_three_common_commits'
    assert probe.matching_motion_status(12, 6) == 'available_descriptive'


def test_zero_future_physical_pin_observations_excludes_copied_frame(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_CACHE_PATH', str(tmp_path/'warp'))
    quality = importlib.import_module('scripts.probes.quality_compare')
    frames = np.zeros((22, 2, 3), dtype=np.float32)
    # A copied suffix must not grant a physical observation to a newly pinned ID.
    frames[-1, 0, 0] = 9
    run = dict(source=frames[0], frames=frames, pins=np.array([True, False]),
               pin_at=np.array([1, -1]), records=[dict(animation=0, frame_end=21)], delivered=21)
    report = quality.scoped_pin_motion(run, 1.)
    assert report['admitted_points'] == 1
    assert report['checked_particles'] == report['moved_particles_exact'] == 0
    assert report['per_particle_max_drift_sp'] is None


def test_phase_summary_detects_boundary_jump_on_same_ids(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_CACHE_PATH', str(tmp_path/'warp'))
    frames = np.zeros((61, 1, 3), dtype=np.float32)
    # W2 advances +1 in phases1..19 and returns19 at phase20. W3 repeats.
    for offset in (20, 40):
        frames[offset:offset+20, 0, 0] = np.arange(20)
    run = dict(frames=frames, records=[dict(frame_end=end) for end in (21, 41, 61)])
    result = probe.phase_summary(run, np.array([0]), 3, 1.)
    assert result['first_19_sp']['rms'] == 1
    assert result['final_phase_sp']['rms'] == 19
    assert result['reversal_groups']['into_final_phase']['fraction'] == 1
    assert result['reversal_groups']['from_final_to_next_phase1']['fraction'] == 1
    assert result['net_over_path']['max'] == 0
    assert result['window1_excluded']


def test_file_evidence_hashes_actual_bytes(tmp_path):
    path = tmp_path/'evidence.json'
    path.write_bytes(b'{"complete":true}\n')
    record = probe.file_record(path)
    assert record['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record['bytes'] == path.stat().st_size
