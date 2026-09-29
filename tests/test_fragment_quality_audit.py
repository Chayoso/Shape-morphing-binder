"""P327 exact-mode provenance and full-clock scope; CPU metadata only."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from test_quality_compare import probe
from test_layer_projection_audit import config_pair, metadata_run


MODE = 'fragment_adjoint_retained_full'


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def bind(probe, run, suffix):
    value = probe.bind_layer_artifact(run['prefix']+suffix)
    run['artifact_bindings'][suffix] = value
    run['artifact_hashes'][suffix] = value['sha256']


def refresh_links(probe, run):
    """Keep parent hashes consistent to exercise deeper semantic rejection."""
    prefix = run['prefix']
    for suffix in ('.fragment_protocol.json', '.protocol.json'):
        bind(probe, run, suffix)
    trace = json.loads(Path(prefix+'.rest_trace.json').read_bytes())
    trace['protocol_sha256'] = run['artifact_hashes']['.protocol.json']
    write_json(prefix+'.rest_trace.json', trace)
    bind(probe, run, '.rest_trace.json')
    activity = json.loads(Path(prefix+'.fragment_activity.json').read_bytes())
    activity.update(protocol_sha256=run['artifact_hashes']['.fragment_protocol.json'],
                    rest_trace_sha256=run['artifact_hashes']['.rest_trace.json'])
    write_json(prefix+'.fragment_activity.json', activity)
    bind(probe, run, '.fragment_activity.json')


@pytest.fixture
def evidence(probe, tmp_path):
    code_root = Path(__file__).parents[1]
    inputs_dir = tmp_path/'inputs'; inputs_dir.mkdir()
    config = config_pair()[0]
    prm = dict(dt=1/240, dx=.3062907543956724)
    source = dict(arms={'render_full_dt_iso_nn': dict(config=config)}, provenance=dict(mpm=prm))
    write_json(inputs_dir/'source.json', source)
    # Opaque I/O fixtures; no source geometry or physical result is modeled.
    for name in ('source_render_full_dt_iso_nn.npz', 'target_reference.npz'):
        (inputs_dir/name).write_bytes(name.encode())
    inputs = {str(p): probe.file_digest(p) for p in inputs_dir.iterdir()}
    wrapper_names = ('scripts/probes/fragment_adjoint_compare.py', 'docs/fragment_adjoint_p327.md')
    producer_names = [p.relative_to(code_root).as_posix() for p in sorted((code_root/'physmorph').rglob('*.py'))]
    producer_names += ['scripts/probes/full_horizon.py', 'scripts/probes/gpu_pipeline.py',
                      'scripts/probes/coverage_paths.py', 'scripts/ops/run_p303_probe.sh',
                      'scripts/ops/cuda_python.py', 'docs/full_horizon_p316.md']
    code = lambda names: {str(code_root/name): probe.file_digest(code_root/name) for name in names}
    runs = []
    for mode in ('legacy', 'retained'):
        run = metadata_run(63 if mode == 'legacy' else 42)
        probe.checked_layer_delivery(run, run['delivered'])
        run['prefix'] = str(tmp_path/mode)
        run['meta'] = dict(provenance=dict(mpm=prm))
        run['arm']['config'].update(target_reference=str(inputs_dir/'target_reference.npz'), geometric_variance=False)
        run['artifact_bindings'], run['artifact_hashes'] = {}, {}
        for suffix in ('.json', '.npz', '_render_full_dt_iso_nn.npz', '.render_influence.json', '.render_influence.md'):
            Path(run['prefix']+suffix).write_bytes((mode+suffix).encode())
            bind(probe, run, suffix)
        output_hashes = {run['prefix']+suffix: digest for suffix, digest in run['artifact_hashes'].items()}
        Path(run['prefix']+'.log').write_bytes(b'bound process log')
        bind(probe, run, '.log')
        write_json(run['prefix']+'.protocol.json', dict(arm='raw', code=code(producer_names),
            inputs=inputs, source_config=config, mpm=prm))
        write_json(run['prefix']+'.fragment_protocol.json', dict(mode=mode, code=code(wrapper_names),
            no_withdrawal_objective=True))
        folder = Path(run['prefix']+'_cohorts'); folder.mkdir()
        attempts, cursor = [], 0
        for row in run['arm']['history']:
            if row.get('held') or 'c2f_render_res' in row:
                continue
            sidecar = f"attempt_{row['animation']:03d}.npz"
            (folder/sidecar).write_bytes(sidecar.encode())
            committed = bool(row.get('frame_end') and not row.get('null_commit') and not row.get('outer_rejected'))
            entry = dict(animation=row['animation'], sidecar=sidecar, sha256=probe.file_digest(folder/sidecar),
                start_frame=cursor, committed=committed, outer_rejected=bool(row.get('outer_rejected')),
                null_commit=bool(row.get('null_commit')), grad_converged=bool(row.get('grad_converged')))
            if committed:
                cursor = row['frame_end']-1
            elif row.get('null_commit') and not row.get('outer_rejected'):
                cursor += 1
            entry['end_frame'] = cursor
            attempts.append(entry)
        write_json(run['prefix']+'.rest_trace.json', dict(inputs_code_unchanged=True,
            result_sha256=run['artifact_hashes']['.json'], output_sha256=output_hashes,
            actual_archive_frames=63, deliver_n=run['delivered'], actual_last_accepted=4,
            accepted_attempts=[0, 3, 4], attempts=attempts))
        model = dict(model=0, attempt=0, N=300000, T=20, device='cuda:0',
            reverse_unique_buffers=1 if mode == 'legacy' else 20,
            retained_allocations=20, observation_buffers=20)
        write_json(run['prefix']+'.fragment_activity.json', dict(mode=mode, code_unchanged=True,
            models=[model], forwards=[dict(forward=0, model=0, attempt=0)]))
        refresh_links(probe, run)
        runs.append(run)
    return runs


def test_same_config_mode_is_explicit_and_preserves_full_clock(probe, evidence):
    a = config_pair()[0]
    assert probe.checked_config_changes(a, dict(a), MODE) == {}
    assert MODE in probe.FULL_INTERVENTIONS and MODE in probe.FULL_RAW_SCOPE
    assert probe.LAYER_FULL != MODE
    probe.checked_fragment_evidence(evidence)
    assert [r['fragment_evidence']['mode'] for r in evidence] == ['legacy', 'retained']
    runs, scope = probe.scoped_runs(evidence, MODE)
    assert runs is evidence and scope['common_accepted_commits'] == 2
    assert [len(r['actual_records']) for r in runs] == [3, 3]
    assert [len(r['records']) for r in runs] == [3, 2]
    assert probe.accepted_raw_indices(runs[1]['records'], 20) == [0, *range(1, 21), *range(22, 42)]
    assert [r['attempt'] for r in probe.render_reference_history(runs[1], MODE) if r['delivered']] == [1, 4]


@pytest.mark.parametrize('key,value', [('layer_ctrl', False), ('layer_relax', False),
    ('stop_after_windows', 60), ('T', 19), ('commit_pic', True), ('shift_sub', True),
    ('geometric_variance', True), ('lambda_auto', 0.)])
def test_shared_out_of_scope_recipe_is_rejected(probe, key, value):
    config = dict(config_pair()[0], **{key: value})
    with pytest.raises(ValueError, match=MODE):
        probe.checked_config_changes(config, dict(config), MODE)


def test_any_config_difference_or_discretization_difference_rejects(probe):
    config = config_pair()[0]
    with pytest.raises(ValueError, match=MODE):
        probe.checked_config_changes(config, dict(config, lambda_auto=.6), MODE)
    with pytest.raises(ValueError, match='dt=1/240'):
        probe.checked_mpm_parameters({'dt': 1/120}, {'dt': 1/120}, MODE)


@pytest.mark.parametrize('suffix,change', [
    ('.fragment_activity.json', lambda x: x.update(mode='legacy')),
    ('.fragment_activity.json', lambda x: x.update(code_unchanged=False)),
    ('.fragment_activity.json', lambda x: x['models'][0].update(reverse_unique_buffers=1)),
    ('.fragment_activity.json', lambda x: x['models'][0].update(N=40000)),
    ('.fragment_activity.json', lambda x: x['forwards'][0].update(attempt=1)),
    ('.fragment_activity.json', lambda x: x.update(forwards=[])),
    ('.fragment_protocol.json', lambda x: x.update(mode='legacy')),
    ('.fragment_protocol.json', lambda x: x.update(no_withdrawal_objective=False)),
    ('.fragment_protocol.json', lambda x: x['code'].update({next(iter(x['code'])): '0'*64})),
    ('.protocol.json', lambda x: x.update(arm='baseline')),
    ('.protocol.json', lambda x: x['mpm'].update(dt=1/120)),
    ('.protocol.json', lambda x: x['source_config'].update(lambda_auto=.6)),
    ('.rest_trace.json', lambda x: x.update(actual_last_accepted=3)),
    ('.rest_trace.json', lambda x: x.update(accepted_attempts=[0, 3])),
    ('.rest_trace.json', lambda x: x['attempts'][2].update(committed=True)),
    ('.rest_trace.json', lambda x: x['attempts'][2].update(outer_rejected=False)),
    ('.rest_trace.json', lambda x: x['attempts'][1].update(null_commit=False)),
    ('.rest_trace.json', lambda x: x['attempts'][3].update(start_frame=20)),
    ('.rest_trace.json', lambda x: x['attempts'][3].update(end_frame=42)),
    ('.rest_trace.json', lambda x: x['attempts'][0].update(sidecar='../escape.npz')),
    ('.rest_trace.json', lambda x: x['attempts'][0].update(sidecar='attempt_004.npz',
                                                        sha256=x['attempts'][4]['sha256'])),
    ('.rest_trace.json', lambda x: x['output_sha256'].pop(next(iter(x['output_sha256'])))),
])
def test_rehashed_but_invalid_producer_evidence_rejects(probe, evidence, suffix, change):
    run = evidence[1]
    path = run['prefix']+suffix
    data = json.loads(Path(path).read_bytes()); change(data); write_json(path, data)
    refresh_links(probe, run)
    with pytest.raises(ValueError, match='P327'):
        probe.checked_fragment_evidence(evidence)


def test_native_input_mismatch_and_recipe_override_reject(probe, evidence):
    path = Path(evidence[0]['prefix']+'.protocol.json')
    data = json.loads(path.read_bytes())
    native = next(p for p in data['inputs'] if p.endswith('target_reference.npz'))
    Path(native).write_bytes(b'changed')
    with pytest.raises(ValueError, match='native input'):
        probe.checked_fragment_evidence(evidence)


def test_same_unbound_recipe_override_in_both_arms_still_rejects(probe, evidence):
    # Equal configurations alone are insufficient: bind the immutable recipe.
    for run in evidence:
        run['arm']['config']['w_kin'] = 123.
    with pytest.raises(ValueError, match='original raw recipe'):
        probe.checked_fragment_evidence(evidence)


def test_rehashed_activity_cannot_point_to_another_protocol(probe, evidence):
    run = evidence[1]
    path = Path(run['prefix']+'.fragment_activity.json')
    activity = json.loads(path.read_bytes())
    activity['protocol_sha256'] = evidence[0]['artifact_hashes']['.fragment_protocol.json']
    write_json(path, activity)
    bind(probe, run, '.fragment_activity.json')
    with pytest.raises(ValueError, match='binding mismatch'):
        probe.checked_fragment_evidence(evidence)


@pytest.mark.parametrize('suffix', ['.fragment_activity.json', '.fragment_protocol.json',
                                  '_cohorts/attempt_000.npz', '.log'])
def test_after_analysis_tamper_detected_even_with_same_stat(probe, evidence, suffix):
    probe.checked_fragment_evidence(evidence)
    path = Path(evidence[1]['prefix']+suffix)
    before = path.stat(); original = path.read_bytes()
    path.write_bytes(b'X'+original[1:])
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='changed during analysis'):
        probe.verify_layer_artifacts(evidence)


def test_mode_propagates_to_inconclusive_phase_without_gpu(probe, evidence, tmp_path, monkeypatch):
    probe.checked_fragment_evidence(evidence)
    spec = importlib.util.spec_from_file_location('fragment_phase_test', Path(__file__).parents[1]/'scripts/probes/raw_phase.py')
    phase = importlib.util.module_from_spec(spec); spec.loader.exec_module(phase)
    code_hash = 'same_checked_core'
    for run in evidence:
        run['meta']['provenance']['code_hash'] = code_hash
    quality = Path(sys.modules['scripts.probes.quality_compare'].__file__)
    report = dict(intervention=MODE, probe_sha256=probe.file_digest(quality), code_hash=code_hash,
        analysis_scope={'kind': 'full runs'}, mpm={'dt': 1/240}, n=300000, T=20, native_spacing=.035,
        cohorts={'common_endpoint_free_both': dict(sampled_count=0, arms={'baseline': None, 'candidate': None})})
    for label, run in zip(('baseline', 'candidate'), evidence):
        report[label] = dict(prefix=run['prefix'], artifact_hashes=run['artifact_hashes'])
    reference, output = tmp_path/'quality.json', tmp_path/'phase.json'
    write_json(reference, report)
    monkeypatch.setattr(phase, 'checked_runs', lambda *args: (evidence, {}))
    monkeypatch.setattr(phase, 'cuda_execution', lambda *a: pytest.fail('empty metadata scope entered CUDA'))
    phase.phase_audit(Path(evidence[0]['prefix']), Path(evidence[1]['prefix']), reference, output)
    assert json.loads(output.read_bytes())['status'] == 'inconclusive'
    assert phase.phase_path_share(np.float32(0), np.float32(0), MODE) is None
