"""CPU-only checks of strict CLI boundaries; no simulation or CUDA invocation."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from physmorph.input_assets import load_target_reference, load_input_reference, point_hash, file_hash


@pytest.mark.parametrize('configured', [None, '/data/test/cache/warp'])
def test_package_configures_warp_cache_before_initialization(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv('WARP_CACHE_PATH', raising=False)
    else:
        monkeypatch.setenv('WARP_CACHE_PATH', configured)
    state = SimpleNamespace(kernel_cache_dir='existing-local-default')
    seen = []
    fake = SimpleNamespace(config=state, init=lambda: seen.append(state.kernel_cache_dir))
    monkeypatch.setitem(sys.modules, 'warp', fake)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'physmorph' / '__init__.py'))
    assert seen == [configured or 'existing-local-default']


def _reference(path, target, *, schema=1, count=None):
    metadata = {'schema': schema, 'n': len(target) if count is None else count,
                'target_sha256': point_hash(target), 'disc_ref_factor': 1.5}
    np.savez(path, normals=np.ones_like(target), weights=np.ones(len(target)),
             spacing=.2, metadata=json.dumps(metadata))


def test_reference_validates_exact_input_and_records_loaded_bytes(tmp_path):
    target = np.arange(18, dtype=np.float32).reshape(6, 3)
    path = tmp_path / 'reference.npz'
    _reference(path, target)
    loaded = load_target_reference(path, target, 1.5)
    assert loaded['provenance']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded['provenance']['path'] == str(path.resolve())
    with pytest.raises(ValueError, match='does not match'):
        load_target_reference(path, target + 1, 1.5)
    with pytest.raises(ValueError, match='does not match'):
        load_target_reference(path, target, 1.)
    _reference(path, target, schema=2)
    with pytest.raises(ValueError, match='schema or point count'):
        load_target_reference(path, target, 1.5)
    _reference(path, target, count=7)
    with pytest.raises(ValueError, match='schema or point count'):
        load_target_reference(path, target, 1.5)


@pytest.fixture
def input_bundle(tmp_path):
    source_path, target_path = tmp_path / 'source.obj', tmp_path / 'target.obj'
    source_path.write_text('source mesh bytes')
    target_path.write_text('target mesh bytes')
    source = np.arange(18, dtype=np.float32).reshape(6, 3)
    target = source * np.float32(1.2)
    metadata = {'schema': 1, 'input_schema': 1, 'n': 6,
                'source_sha256': point_hash(source), 'target_sha256': point_hash(target),
                'disc_ref_factor': 1.5, 'preparation': {
                    'source_mesh_sha256': file_hash(source_path),
                    'target_mesh_sha256': file_hash(target_path),
                    'seed': 1, 'sampler': 'stratified', 'sample': 'volume'}}
    values = dict(src=source, tgt=target, source_volume=2., target_volume=2.,
                  normals=np.ones_like(target), weights=np.ones(6), spacing=.2,
                  metadata=json.dumps(metadata))
    path = tmp_path / 'input.npz'
    np.savez(path, **values)
    options = dict(source_path=source_path, target_path=target_path, seed=1,
                   sampler='stratified', sample='volume')
    return path, values, options


def test_bundle_validates_identity_and_is_also_shading_reference(input_bundle):
    path, values, options = input_bundle
    bundle = load_input_reference(path, 6, **options)
    np.testing.assert_array_equal(bundle['source'], values['src'])
    np.testing.assert_array_equal(bundle['target'], values['tgt'])
    assert bundle['volumes'] == (2., 2.)
    reference = load_target_reference(path, bundle['target'], 1.5)
    assert reference['provenance']['sha256'] == bundle['provenance']['sha256'] == file_hash(path)
    for key, value in (('seed', 2), ('sampler', 'replacement'), ('sample', 'shell')):
        with pytest.raises(ValueError, match='sampling'):
            load_input_reference(path, 6, **{**options, key: value})
    options['target_path'].write_text('different target mesh')
    with pytest.raises(ValueError, match='mesh inputs conflict'):
        load_input_reference(path, 6, **options)


@pytest.mark.parametrize('change,message', [
    ({'src': np.zeros((6, 3), np.float32)}, 'point hashes'),
    ({'tgt': np.zeros((5, 3), np.float32)}, 'shape'),
    ({'tgt': np.full((6, 3), np.nan, np.float32)}, 'finite'),
    ({'src': np.zeros((6, 3), np.float64)}, 'float32'),
    ({'source_volume': 0.}, 'positive'),
    ({'target_volume': np.inf}, 'finite'),
])
def test_bundle_rejects_invalid_inputs(input_bundle, change, message):
    path, values, options = input_bundle
    np.savez(path, **{**values, **change})
    with pytest.raises(ValueError, match=message):
        load_input_reference(path, 6, **options)


@pytest.mark.parametrize('backend', ['legacy', 'cuda'])
def test_cli_gates_use_selected_backend(monkeypatch, backend):
    import physmorph.compute as compute
    from scripts import pipeline_run as cli
    state = {'active': False}
    @contextmanager
    def context(device):
        assert device == 'cuda:0'
        state['active'] = True
        try:
            yield
        finally:
            state['active'] = False
    def gate(*args, **kwargs):
        assert state['active'] == (backend == 'cuda')
        return {'pass': True}
    monkeypatch.setattr(compute, 'cuda_execution', context)
    monkeypatch.setattr(cli, 'gate1_plumbing', gate)
    monkeypatch.setattr(cli, 'gate1_channels', gate)
    assert cli.gate1_checks(None, None, backend, 'cuda:0') == {
        'G1a': {'pass': True}, 'G1b': {'pass': True}}
    assert not state['active']


def test_strict_cli_rejects_cpu_numerical_viewer():
    from scripts.pipeline_run import validate_cli_backend
    for port, directory in ((8080, ''), (0, 'packets')):
        with pytest.raises(ValueError, match='CPU numerical live viewer'):
            validate_cli_backend(SimpleNamespace(compute_backend='cuda', live_port=port,
                                                 live_dir=directory))
        validate_cli_backend(SimpleNamespace(compute_backend='legacy', live_port=port,
                                             live_dir=directory))
    validate_cli_backend(SimpleNamespace(compute_backend='cuda', live_port=0, live_dir=''))


def test_metric_timing_excludes_trailing_nulls_but_preserves_delivered_motion():
    from scripts.pipeline_run import delivered_metric_timing
    history = [{'frame_end': 3}, {'frame_end': 4, 'null_commit': True},
               {'frame_end': 7}, {'frame_end': 8, 'null_commit': True}]
    assert delivered_metric_timing(9, 9, history) == (9, 2, [2, 6])
    assert delivered_metric_timing(9, 7, history) == (7, 0, [2, 6])
    assert delivered_metric_timing(9, 6, history) == (6, 0, [2])
    assert delivered_metric_timing(9, 8, history) == (8, 1, [2, 6])
    assert delivered_metric_timing(2, 2, [{'frame_end': 2, 'null_commit': True}]) == (2, 1, [])


@pytest.fixture
def launchers(tmp_path):
    bash = (r'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash'))
    if not bash or not Path(bash).is_file():
        pytest.skip('Bash is needed for launcher argument tests')
    repo = tmp_path / 'repo'
    ops = repo / 'scripts' / 'ops'
    ops.mkdir(parents=True)
    original = Path(__file__).resolve().parents[1] / 'scripts' / 'ops'
    for name in ('run_gpu_pipeline.sh', 'run_body_control.sh'):
        shutil.copyfile(original / name, ops / name)
    (ops / 'hyde06_env.sh').write_text(
        'export REPO=$PHYSMORPH_RUN_REPO\nexport OUT=$REPO/output\n'
        'export PY=$REPO/fake_python.sh\nexport RECIPE="--animations 300"\ncd "$REPO"\n')
    (ops / 'gpu_env.sh').write_text('export STRICT_GPU_ENV=1\n')
    fake = repo / 'fake_python.sh'
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n'
                    'printf "%s\\n" "${STRICT_GPU_ENV:-0}" > "$CAPTURE.env"\n')
    fake.chmod(0o755)
    target = tmp_path / 'inputs' / 'target reference.npz'
    target.parent.mkdir()
    target.write_bytes(b'prepared input placeholder')
    capture = repo / 'capture'
    env = dict(os.environ, CAPTURE=capture.as_posix())
    env.pop('PHYSMORPH_TARGET_REFERENCE', None)
    env.pop('PHYSMORPH_INPUT_REFERENCE', None)
    def run(name, *args):
        return subprocess.run([bash, (ops / name).as_posix(), *args], env=env, cwd=tmp_path,
                              capture_output=True, text=True)
    return run, capture, target


def test_strict_launcher_forces_cuda_after_user_arguments(launchers):
    run, capture, _ = launchers
    result = run('run_gpu_pipeline.sh', '0', '--compute_backend', 'legacy', '--n', '300000')
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines()[-2:] == ['--compute_backend', 'cuda']
    assert Path(str(capture) + '.env').read_text().strip() == '1'
    assert run('run_gpu_pipeline.sh').returncode == 2
    assert run('run_gpu_pipeline.sh', '1').returncode == 2


def test_active_body_launcher_requires_input_and_has_explicit_legacy_choice(launchers):
    run, capture, target = launchers
    args = ('0', 'c291_test', 'bunny', '300000', '6', 'body_terminal')
    assert run('run_body_control.sh', *args).returncode == 2
    assert not capture.exists()
    result = run('run_body_control.sh', *args, '--target-reference', 'inputs/target reference.npz')
    assert result.returncode == 0, result.stderr
    argv = capture.read_text().splitlines()
    assert argv[:2] == ['scripts/ops/cuda_python.py', 'scripts/pipeline_run.py']
    assert argv[-2:] == ['--compute_backend', 'cuda']
    reference = argv[argv.index('--target_reference') + 1]
    assert reference.startswith('/') and reference.endswith('/inputs/target reference.npz')
    assert argv[argv.index('--animations') + 1] == '300'
    assert argv[argv.index('--stop_after_windows') + 1] == '6'
    result = run('run_body_control.sh', '0', 'c291_legacy', *args[2:], '--legacy-comparison')
    assert result.returncode == 0, result.stderr
    argv = capture.read_text().splitlines()
    assert argv[0] == 'scripts/pipeline_run.py'
    assert argv[argv.index('--compute_backend') + 1] == 'legacy'
    assert Path(str(capture) + '.env').read_text().strip() == '0'


def test_active_launcher_uses_input_bundle_for_both_geometry_and_shading(launchers):
    run, capture, _ = launchers
    result = run('run_body_control.sh', '0', 'c291_bundle', 'bunny', '300000', '1',
                 'body_terminal', '--input-reference', 'inputs/target reference.npz')
    assert result.returncode == 0, result.stderr
    argv = capture.read_text().splitlines()
    assert argv[argv.index('--input_reference') + 1] == argv[argv.index('--target_reference') + 1]
    assert argv[-2:] == ['--compute_backend', 'cuda']
