"""P320 metadata/clock guards only; no simulation or modeled quality outcomes."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

from test_quality_compare import probe


MODE = 'layer_projection_off_full'


def config_pair():
    a = dict(layer_ctrl=True, layer_relax=True, stop_after_windows=300, animations=300,
             T=20, iters=8, loss_res=36, commit_pic=False, commit_pic_objective=False,
             shift_sub=False, outer_render_committed=True, body_ctrl=True,
             motion_accounting=True, lambda_auto=.5, render_until=0,
             compute_backend='cuda', archive_stride=1, phys_loss='auto')
    return a, dict(a, layer_ctrl=False, layer_relax=False)


def metadata_run(delivered=63):
    # Three accepted rollouts, with an interior null hold and rejected solve.
    # The event shares an animation number with its following actual solve.
    history = [dict(animation=0, accepted=2, frame_end=21),
               dict(animation=1, c2f_render_res=96),
               dict(animation=1, null_commit=1),
               dict(animation=2, accepted=2, outer_rejected=1),
               dict(animation=3, accepted=2, frame_end=42),
               dict(animation=4, accepted=2, frame_end=62),
               dict(animation=5, grad_converged=1), dict(animation=6, held=1)]
    frames = np.broadcast_to(np.zeros((1, 1, 3), np.float32), (63, 300000, 3))
    records = [r for r in history if r.get('frame_end') and r['frame_end'] <= delivered]
    truncation = None if delivered == 63 else dict(
        frames_kept=delivered, frames_dropped=63-delivered,
        best_animation=int(records[-1]['animation'])+1)
    return dict(source=frames[0], target=frames[0], frames=frames, delivered=delivered,
                records=records, arm=dict(history=history, config=config_pair()[0],
                    deliver_n=delivered, n_held=1, truncation=truncation))


def test_layer_contract_has_exact_two_flag_change_and_native_discretization(probe):
    a, b = config_pair()
    assert probe.checked_config_changes(a, b, MODE) == {
        'layer_ctrl': [True, False], 'layer_relax': [True, False]}
    prm = dict(dt=1/240, dx=.3062907543956724)
    probe.checked_mpm_parameters(prm, dict(prm), MODE)
    with pytest.raises(ValueError, match='dt=1/240'):
        probe.checked_mpm_parameters(dict(prm, dt=1/120), dict(prm, dt=1/120), MODE)
    with pytest.raises(ValueError, match='mismatch'):
        probe.checked_mpm_parameters(prm, dict(prm, dx=.3), MODE)


@pytest.mark.parametrize('key,value', [
    ('stop_after_windows', 60), ('stop_after_windows', 24), ('animations', 60),
    ('T', 19), ('iters', 7), ('loss_res', 72), ('commit_pic', True),
    ('commit_pic_objective', True), ('shift_sub', True), ('outer_render_committed', False),
    ('body_ctrl', False), ('motion_accounting', False), ('lambda_auto', 0.),
    ('lambda_auto', float('inf')), ('render_until', 299), ('surface_gs_loss', True),
    ('render_F_geom', True), ('geometric_rest', True), ('geometric_variance', True),
    ('reattach', True), ('settle_commit', True), ('assim_consensus', True),
    ('settle_pin_yield', True), ('settle_pin_follow', True), ('settle_pin_kkt', True),
    ('lg_sweeps', 1), ('w_grow', 1), ('local_dress_iters', 1)])
def test_layer_contract_rejects_shared_out_of_scope_recipe(probe, key, value):
    a, b = config_pair()
    a[key] = b[key] = value
    with pytest.raises(ValueError, match=MODE):
        probe.checked_config_changes(a, b, MODE)


@pytest.mark.parametrize('change', [dict(layer_ctrl=True), dict(layer_relax=True),
                                  dict(layer_ctrl=None), dict(lambda_auto=.6),
                                  dict(new_unreviewed_flag=True)])
def test_layer_contract_rejects_partial_or_additional_treatment(probe, change):
    a, b = config_pair()
    with pytest.raises(ValueError, match=MODE):
        probe.checked_config_changes(a, dict(b, **change), MODE)


def test_layer_archive_is_mmap_safe_and_does_not_read_F_payload(probe, tmp_path):
    prefix = tmp_path/'arm'
    path = str(prefix)+'_render_full_dt_iso_nn.npz'
    np.savez(path, frames=np.zeros((2, 3, 3), np.float32), deliver_n=2,
             F_samples=np.array([{'must_not_be_unpickled': True}], dtype=object))
    assert probe.checked_layer_archive(prefix) == 2
    np.savez_compressed(path, frames=np.zeros((2, 3, 3), np.float32), deliver_n=2)
    with pytest.raises(ValueError, match='uncompressed'):
        probe.checked_layer_archive(prefix)
    np.savez(path, frames=np.zeros((2, 3, 3), np.float32), deliver_n=2.)
    with pytest.raises(ValueError, match='integer scalar'):
        probe.checked_layer_archive(prefix)
    with zipfile.ZipFile(path, 'a') as archive:
        with pytest.warns(UserWarning, match='Duplicate name'):
            archive.writestr('frames.npy', b'duplicate must be rejected before parsing')
    with pytest.raises(ValueError, match='one uncompressed'):
        probe.checked_layer_archive(prefix)


def test_delivery_clock_preserves_own_endpoints_and_common_retained_scope(probe):
    a, b = metadata_run(), metadata_run(42)
    probe.checked_layer_delivery(a, 63)
    probe.checked_layer_delivery(b, 42)
    assert a['hold_pairs'] == [(20, 21), (61, 62)]
    assert len(a['actual_records']) == len(b['actual_records']) == 3
    assert len(a['records']) == 3 and len(b['records']) == 2
    assert a['delivery_scope']['delivery_rows_after_last_accepted'] == 1
    assert b['delivery_scope']['actual_last_accepted_frame'] == 61
    assert b['delivery_scope']['delivered_last_accepted_frame'] == 41
    assert len(b['delivery_scope']['attempts']) == 6
    assert b['delivery_scope']['c2f_events'] == [dict(animation=1, c2f_render_res=96)]
    scoped, description = probe.scoped_runs([a, b], MODE)
    assert scoped[0] is a and scoped[1] is b
    assert description['configured_windows'] == 300
    assert description['common_accepted_commits'] == 2
    assert [len(run['records']) for run in scoped] == [3, 2]
    clock = probe.accepted_raw_indices(a['records'], 20)
    assert clock == [0, *range(1, 21), *range(22, 62)]
    assert 21 not in clock and 62 not in clock
    trace = probe.render_reference_history(b, MODE)
    assert [r['attempt'] for r in trace if r['accepted']] == [1, 4, 5]
    assert [r['attempt'] for r in trace if r['delivered']] == [1, 4]
    assert trace[2]['accepted'] is False


@pytest.mark.parametrize('mutation', [
    lambda r: r['arm']['history'][0].update(outer_rejected=1),
    lambda r: r['arm']['history'][0].update(outer_accepted=0),
    lambda r: r['arm']['history'][0].update(accepted=0),
    lambda r: r['arm']['history'][0].update(frame_end=20),
    lambda r: r['arm']['history'][1].update(frame_end=21),
    lambda r: r['arm']['history'][2].update(frame_end=22),
    lambda r: r['arm']['history'][3].update(animation=1),
    lambda r: r['arm']['history'][-1].update(null_commit=1),
    lambda r: r['arm'].update(n_held=0),
    lambda r: r['arm'].update(deliver_n=62),
    lambda r: r.update(delivered=62),
    lambda r: r.update(records=r['records'][:-1]),
    lambda r: r.update(frames=r['frames'][:-1]),
    lambda r: r.update(source=r['source'][:-1])])
def test_corrupt_clock_or_admission_scope_fails_closed(probe, mutation):
    run = metadata_run()
    mutation(run)
    with pytest.raises(ValueError, match='P320'):
        probe.checked_layer_delivery(run, 63)


def test_delivery_clamp_midwindow_and_unrecorded_truncation_fail(probe):
    with pytest.raises(ValueError, match='clamping'):
        probe.checked_layer_delivery(metadata_run(), 100)
    for delivered in (30, 42):
        run = metadata_run(delivered)
        if delivered == 42:
            run['arm']['truncation'] = None
        with pytest.raises(ValueError, match='truncation'):
            probe.checked_layer_delivery(run, delivered)


def test_one_retained_commit_is_valid_scope_but_has_no_common_motion(probe):
    run = metadata_run(21)
    probe.checked_layer_delivery(run, 21)
    assert len(run['records']) == 1 and len(run['actual_records']) == 3
    assert probe.cohort_motion(run, np.array([0, 1]), 1, .03) is None
    assert probe.cohort_motion(run, np.array([], dtype=int), 3, .03) is None


def test_post_delivery_pin_admission_does_not_change_common_free_mask(probe):
    run = metadata_run(42)
    probe.checked_layer_delivery(run, 42)
    run['pins'] = np.zeros(300000, bool)
    run['pin_at'] = np.full(300000, -1, np.int64)
    run['pins'][:3] = True
    run['pin_at'][:3] = [1, 4, 5]
    assert probe.admitted_mask(run).nonzero()[0].tolist() == [0, 1]
    assert probe.admitted_mask(run, through_attempt=1).nonzero()[0].tolist() == [0]


def test_phase_clock_excludes_null_hold_and_post_delivery_commit(probe):
    spec = importlib.util.spec_from_file_location(
        'phase_layer_test', Path(__file__).parents[1]/'scripts/probes/raw_phase.py')
    phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase)
    run = metadata_run(42)
    probe.checked_layer_delivery(run, 42)
    indices = phase.phase_frame_indices(run['records'], 1, 2, 20)
    assert indices == [[20, *range(22, 42)]]
    with pytest.raises(ValueError, match='interval'):
        phase.phase_frame_indices(run['records'], 1, 3, 20)


@pytest.mark.parametrize('sampled,changed_artifact', [(0, False), (2, False), (2, True)])
def test_phase_empty_or_short_scope_is_inconclusive_but_artifact_mismatch_rejects(
        probe, tmp_path, monkeypatch, sampled, changed_artifact):
    spec = importlib.util.spec_from_file_location(
        'phase_layer_short_test', Path(__file__).parents[1]/'scripts/probes/raw_phase.py')
    phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase)
    runs = []
    for name in ('baseline', 'candidate'):
        prefix = str(tmp_path/name)
        bindings = {}
        for suffix in ('.json', '.npz', '_render_full_dt_iso_nn.npz'):
            path = Path(prefix+suffix)
            path.write_bytes((name+suffix).encode())
            bindings[suffix] = probe.bind_layer_artifact(path)
        runs.append(dict(prefix=prefix, artifact_bindings=bindings,
                         artifact_hashes={suffix: row['sha256'] for suffix, row in bindings.items()},
                         meta=dict(provenance=dict(code_hash='identical_core'))))
    quality = Path(sys.modules['scripts.probes.quality_compare'].__file__)
    report = dict(intervention=MODE, probe_sha256=hashlib.sha256(quality.read_bytes()).hexdigest(),
                  code_hash='identical_core', analysis_scope={'kind': 'full runs'},
                  mpm={'dt': 1/240}, n=300000, T=20, native_spacing=.035,
                  cohorts={'common_endpoint_free_both': dict(eligible_count=sampled,
                      sampled_count=sampled, ids_sha256='cohort', arms={'baseline': None, 'candidate': None})})
    for name, run in zip(('baseline', 'candidate'), runs):
        report[name] = dict(prefix=run['prefix'], artifact_hashes=dict(run['artifact_hashes']))
    reference, output = tmp_path/'quality.json', tmp_path/'phase.json'
    reference.write_text(json.dumps(report))
    if changed_artifact:
        runs[1]['artifact_hashes']['.json'] = 'modified'
    monkeypatch.setattr(phase, 'checked_runs', lambda *args: (runs, {}))
    def no_cuda(*args):
        raise AssertionError('Inconclusive metadata case entered numerical CUDA analysis')
    monkeypatch.setattr(phase, 'cuda_execution', no_cuda)
    if changed_artifact:
        with pytest.raises(ValueError, match='artifacts changed'):
            phase.phase_audit(Path(runs[0]['prefix']), Path(runs[1]['prefix']), reference, output)
        assert not output.exists()
    else:
        phase.phase_audit(Path(runs[0]['prefix']), Path(runs[1]['prefix']), reference, output)
        result = json.loads(output.read_text())
        assert result['status'] == 'inconclusive'
        assert result['reason'] == ('empty_common_free_cohort' if sampled == 0
                                    else 'fewer_than_three_common_commits')


def test_artifact_verification_detects_content_change_even_with_restored_stat(probe, tmp_path):
    prefix = str(tmp_path/'run')
    path = Path(prefix+'.json')
    path.write_bytes(b'original')
    before = path.stat()
    binding = probe.bind_layer_artifact(path)
    run = dict(prefix=prefix, artifact_bindings={'.json': binding})
    probe.verify_layer_artifacts([run])
    path.write_bytes(b'modified')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert probe.artifact_stat(path) == binding['stat']
    with pytest.raises(ValueError, match='changed during analysis'):
        probe.verify_layer_artifacts([run])


def test_artifact_change_while_hashing_fails(probe, tmp_path, monkeypatch):
    path = tmp_path/'artifact'
    path.write_bytes(b'before')
    original = probe.file_digest
    def mutate_after_hash(path):
        digest = original(path)
        Path(path).write_bytes(b'after-longer')
        return digest
    monkeypatch.setattr(probe, 'file_digest', mutate_after_hash)
    with pytest.raises(ValueError, match='changed while hashing'):
        probe.bind_layer_artifact(path)


def test_nonempty_stationary_cohort_has_zero_motion_but_no_path_share(probe):
    spec = importlib.util.spec_from_file_location(
        'phase_stationary_test', Path(__file__).parents[1]/'scripts/probes/raw_phase.py')
    phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase)
    moves = np.zeros((20, 2, 3), np.float32)
    normals = np.array([[1., 0., 0.], [0., 1., 0.]], np.float32)
    tangent1 = np.array([[0., 1., 0.], [0., 0., 1.]], np.float32)
    tangent2 = np.cross(normals, tangent1)
    result = phase.motion_summary(moves, normals, tangent1, tangent2, .035, include_rms=True)
    for values in result.values():
        assert values is not None and set(values.values()) == {0.}
    total = np.linalg.norm(moves, axis=-1).sum()
    assert phase.phase_path_share(total, total, MODE) is None
    assert phase.phase_path_share(np.float32(1.), np.float32(3.), MODE) == float(np.float32(1.)/np.float32(3.))
