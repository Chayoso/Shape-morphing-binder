"""P329 protocol, witness preservation and actual CPU observer isolation."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, runner
from scripts.probes.prepared_withdrawal import (
    Capture, closure, coast_losses, identity, passed_report, private_closure,
    safe_json, validate_recipe, verify_bindings,
)


def recipe():
    cfg = PipelineConfig(T=20, iters=8, loss_res=36, device='cuda:0', compute_backend='cuda',
        body_ctrl=True, body_terminal_ctrl=True, layer_ctrl=True, layer_relax=True,
        lambda_auto=.5, render_res=64, loss_units='density', commit_pic=False, shift_sub=False)
    return dict(config=asdict(cfg), mpm=asdict(MPMParams(dx=.3062907543956724, dt=1/240)))


def test_recipe_only_changes_cap_and_rejects_scope_drift():
    source = recipe()
    original = deepcopy(source)
    cfg, prm = validate_recipe(source)
    assert source == original and cfg.stop_after_windows == 20
    expected = deepcopy(source['config']); expected['stop_after_windows'] = 20
    assert asdict(cfg) == expected and asdict(prm) == source['mpm']


@pytest.mark.parametrize('key,value', [('T', 21), ('iters', 16), ('commit_pic', True),
    ('commit_pic_objective', True), ('shift_sub', True), ('geometric_rest', True),
    ('render_F_geom', True), ('surface_gs_loss', True), ('layer_relax', False),
    ('lambda_auto', 0.), ('compute_backend', 'cpu')])
def test_bad_recipe_rejected_before_numerics(key, value):
    source = recipe(); source['config'][key] = value
    with pytest.raises(ValueError): validate_recipe(source)


def test_changed_artifact_bytes_cannot_reuse_binding(tmp_path):
    path = tmp_path/'recipe.json'; path.write_text('{"value":1}')
    bindings = {str(path): identity(path)}
    verify_bindings(bindings)
    path.write_text('{"value":2}')  # Same byte length; digest still rejects it.
    with pytest.raises(ValueError, match='changed'): verify_bindings(bindings)


def test_closure_has_declared_units_and_keeps_nonfinite_failure():
    reference = torch.tensor([0., 1., -2.])
    scale = 1/(20/240)
    tolerance = 32*torch.finfo(reference.dtype).eps*(scale+reference.abs())
    assert closure(reference+tolerance*.5, reference, scale)['passed']
    failure = closure(reference+tolerance*2, reference, scale)
    assert not failure['passed'] and failure['max_tolerance_ratio'] > 1
    nonfinite = closure(torch.tensor([float('nan'), 1., -2.]), reference, scale)
    assert not nonfinite['passed'] and not nonfinite['finite']
    assert safe_json(nonfinite)['max_abs'] is None


def test_future_cost_includes_first_displacement_and_excludes_boundary_velocity():
    X = torch.tensor([[[0., 0., 0.]], [[1., 0., 0.]], [[3., 0., 0.]]], requires_grad=True)
    V = torch.tensor([[[999., 0., 0.]], [[2., 0., 0.]], [[3., 0., 0.]]], requires_grad=True)
    values = dict(coast_X=X, coast_V=V)
    losses = coast_losses(values, torch.tensor([True]), .5)
    assert float(losses['geometric_step_mean_square']) == 10.
    assert float(losses['stored_speed_mean_square']) == 6.5
    gx, = torch.autograd.grad(losses['geometric_step_mean_square'], X)
    gv, = torch.autograd.grad(losses['stored_speed_mean_square'], V)
    assert gx[0].abs().sum() > 0 and torch.count_nonzero(gv[0]) == 0
    assert coast_losses(values, torch.tensor([False]), .5) is None


def test_archive_budget_fails_before_writing(tmp_path, monkeypatch):
    import scripts.probes.prepared_withdrawal as probe
    capture = Capture(tmp_path, .1)
    monkeypatch.setattr(probe, 'LIMIT', 1_048_576+12)
    with pytest.raises(ValueError, match='3GB'):
        capture.save('too_large.npz', {'array': torch.ones(4)})
    assert not list(tmp_path.iterdir())


def test_status_never_infers_acceptance_or_missing_state_identity():
    checkpoint = dict(measurement_passed=True, optimizer_state_after_callback_exact=True,
        original_return_preserved=dict(x=True, v=True, F=True, C=True),
        committed_original_endpoint=dict(x=True, F=True, v=True))
    report = dict(failure=None, bindings_unchanged=True, outer_accepted=True, checkpoint=checkpoint)
    assert passed_report(report, dict(guards=dict(inverted=0)))
    for field in ('original_return_preserved', 'committed_original_endpoint'):
        invalid = deepcopy(report); invalid['checkpoint'][field] = {}
        assert not passed_report(invalid, dict(guards={}))
    assert not passed_report(dict(report, outer_accepted=False), dict(guards={}))
    assert not passed_report(report, None)
    assert not passed_report(dict(report, failure={'type': 'OutOfMemory'}), None)
    assert not passed_report(report, dict(guards={'inverted': 1}))


@pytest.mark.parametrize('field', ['volume', 'silhouette', 'pbr', 'render', 'weighted_render'])
def test_nonfinite_coast_observation_cannot_be_serialized_into_a_pass(tmp_path, field):
    capture = Capture(tmp_path, .1)
    scalar_gate = {'value': {'passed': True}}
    row = dict(particles=1, losses={'geometric': 1.}, gradients={'geometric': {'terminal': {'finite': True}}})
    capture.report = dict(T=20, errors=[], merit_binding_unchanged=True,
        original_closure=scalar_gate, joint_closure=scalar_gate, private_joint_closure=scalar_gate,
        scalar_closure=scalar_gate, original_scalar_closure=scalar_gate,
        original_health={'valid': True}, joint_health={'valid': True}, cohorts={'start_free': row},
        coast_prepared_observations=[dict(phase=phase, volume=1., silhouette=1., pbr=1., render=1.,
                                         weighted_render=.1) for phase in (0, 10, 20)])
    assert capture.measurement_passed()
    capture.report['coast_prepared_observations'][1][field] = float('nan')
    assert not capture.measurement_passed()
    saved = safe_json(capture.report)
    assert saved['coast_prepared_observations'][1][field] is None


@pytest.mark.parametrize('bad_C', [False, True])
def test_actual_observer_preserves_original_pipeline_and_failed_C_witness(tmp_path, monkeypatch, bad_C):
    source = np.random.default_rng(27).uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (source*[1.2, .85, 1.05]+[.1, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=1, animations=1, loss_res=12, render_views=2,
        render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
        outer_render_committed=True, body_ctrl=True, body_terminal_ctrl=True,
        lambda_auto=.3, w_kin=.2, w_kin_var=.3, w_ctrl=.001, w_jvol=.5, w_box=0.,
        max_ls_iters=1, adaptive_alpha=False, alpha=1e-4, replay_calibrate=False,
        phys_loss='ot_pace', loss_units='density', ot_samples=128, ot_iters=20,
        render_paced=True, layer_ctrl=True, layer_relax=True)
    prm = MPMParams(dx=1., nx=32, ny=32, nz=32)
    baseline = runner.run_pipeline(source, target, prm, deepcopy(cfg), log=lambda *_: None)
    capture = Capture(tmp_path, .1, window=0, iteration=1)
    original = runner.optimize_window
    if bad_C:
        def observe(index, packet):
            # Corrupt an owned observer reference, never the production state.
            capture.observe(index, dict(packet, C=packet['C']+.01))
        def wrapped(*args, **kwargs):
            return original(*args, on_checkpoint=observe, checkpoint_iterations=(1,),
                            checkpoint_rollout=True, checkpoint_merit=True, **kwargs)
    else:
        wrapped = capture.wrap(original)
    monkeypatch.setattr(runner, 'optimize_window', wrapped)
    actual = runner.run_pipeline(source, target, prm, deepcopy(cfg), on_commit=capture.commit, log=lambda *_: None)
    np.testing.assert_array_equal(actual['frames'], baseline['frames'])
    np.testing.assert_array_equal(actual['F_frames'], baseline['F_frames'])
    assert actual['guards'] == baseline['guards'] and capture.outer_accepted
    assert capture.report is not None and capture.report['measurement_passed'] is not bad_C
    assert capture.report['joint_closure']['C']['passed'] is not bad_C
    assert (tmp_path/'joint_state.npz').exists() and (tmp_path/'coast_gradients.npz').exists()
    assert json.loads((tmp_path/'checkpoint.json').read_text())['measurement_passed'] is not bad_C
    assert all(identity(tmp_path/name) == value for name, value in capture.sidecars.items())
    with np.load(tmp_path/'accepted_head.npz', allow_pickle=False) as saved:
        assert saved['positions'].shape == (3, 160, 3) and saved['C'].shape == (160, 3, 3)
    with np.load(tmp_path/'joint_state.npz', allow_pickle=False) as saved:
        values = {key: torch.from_numpy(saved[key]) for key in ('positions', 'V', 'F', 'C', 'Fg')}
        independent = private_closure(values, tmp_path/'private_head.npz', type('Spec', (), {'T': 3, 'prm': prm})())
        assert all(row['passed'] for row in independent.values())
    if not bad_C:
        assert capture.report['optimizer_state_after_callback_exact']
        assert all(capture.report['original_return_preserved'].values())
        assert all(capture.report['committed_original_endpoint'].values())
