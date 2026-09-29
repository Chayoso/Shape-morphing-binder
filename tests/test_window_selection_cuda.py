"""Opt-in hyde06 P332 handoff gates; no separate CUDA-prefix equality claim."""
from copy import deepcopy
import os

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.compute import cuda_module, is_cuda_execution, to_host
from physmorph.mpm.withdrawal import OwnedWithdrawal
from physmorph.pipeline import optimizer, runner
from test_checkpoint_merit_terms import fixture
from test_window_selection_pipeline import recipe


pytestmark = pytest.mark.skipif(
    os.environ.get('PHYSMORPH_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
    reason='Requires explicit PHYSMORPH_CUDA_TESTS=1 on hyde06')


def aligned():
    assert is_cuda_execution()
    stream = torch.cuda.current_stream().cuda_stream
    assert stream == wp.get_stream('cuda:0').cuda_stream
    assert stream == cuda_module().cuda.get_current_stream().ptr


def owned(value):
    if torch.is_tensor(value): return value.detach().clone()
    if isinstance(value, dict): return {key: owned(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return type(value)(owned(item) for item in value)
    if hasattr(value, '__cuda_array_interface__'): return value.copy()
    return deepcopy(value)


def exact(actual, expected):
    """Exact comparisons only within one run's actual owned state lineage."""
    aligned()
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected: exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected): exact(a, b)
    elif torch.is_tensor(expected) or hasattr(expected, '__cuda_array_interface__'):
        a, b = torch.as_tensor(actual, device='cuda:0'), torch.as_tensor(expected, device='cuda:0')
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(expected, np.ndarray):
        # Metadata arrays, if present, are uploaded for the comparison.
        exact(torch.as_tensor(actual, device='cuda:0'), torch.as_tensor(expected, device='cuda:0'))
    else:
        assert actual == expected


def run_cuda(monkeypatch, source, target, prm, cfg, select, observe_prepared=None):
    cfg = deepcopy(cfg)
    cfg.device, cfg.compute_backend = 'cuda:0', 'cuda'
    prepared, donors, metadata = [], [], []
    original, constructor = runner.optimize_window, optimizer.Trajectory

    def wrapped(*args, **kwargs):
        aligned()
        trajectories = []
        def remember(*pos, **kw):
            tr = constructor(*pos, **kw)
            trajectories.append(tr)
            return tr
        with monkeypatch.context() as patch:
            patch.setattr(optimizer, 'Trajectory', remember)
            result = original(*args, **kwargs)
        assert len(trajectories) == 1
        capture = OwnedWithdrawal.capture(trajectories[0], 0)
        prepared.append({key: torch.as_tensor(value, device='cuda:0').detach().clone()
                         for key, value in capture.arrays().items()})
        metadata.append(dict(capture.metadata(), window=int(kwargs['win_index'])))
        donors.append(owned((*result[:5], {key: value for key, value in result[5].items()
                                           if key != '_window_selection'})))
        # Null/gradient-stopped successors still prepared a real trajectory.
        # Their pin evidence must not depend on selection-context eligibility.
        if observe_prepared is not None:
            observe_prepared(prepared, donors, metadata)
        return result

    def selector(ctx):
        aligned()
        return select(ctx, prepared, donors, metadata)

    with monkeypatch.context() as patch:
        patch.setattr(runner, 'optimize_window', wrapped)
        result = runner.run_pipeline(source, target, prm, cfg, log=lambda *_: None,
                                     select_window=selector)
    return result


def test_identity_owned_inspection_preserves_actual_successor_and_new_pin_lineage(monkeypatch):
    source, target, prm, cfg = recipe()  # N160,T3,dt1/240,dx1,loss12^3,iters2; four windows.
    contexts, archives, witnesses = [], [], []
    prepared_windows = []
    def observe_prepared(prepared, donors, metadata):
        aligned()
        index = len(prepared)-1
        prepared_windows.append(metadata[index]['window'])
        assert metadata[index]['layer_present'] and metadata[index]['bonds_present']
        if index:
            current, previous = prepared[index], donors[index-1]
            pinned = current['pin'] > .5
            if previous[4]:
                exact(current['x0'], previous[0][-1])
                exact(current['F0'], previous[2]['F'])
                exact(current['v0'][~pinned], torch.as_tensor(previous[2]['v'], device='cuda:0')[~pinned])
                exact(current['C0'][~pinned], torch.as_tensor(previous[2]['C'], device='cuda:0')[~pinned])
            assert not bool(current['v0'][pinned].any()) and not bool(current['C0'][pinned].any())
            witnesses.append(dict(window=metadata[index]['window'], pinned=int(pinned.sum()),
                newly_pinned=int((pinned & ~(prepared[index-1]['pin'] > .5)).sum()),
                assimilation_changed=not torch.equal(current['Fp'], prepared[index-1]['Fp'])))

    def select(ctx, prepared, donors, metadata):
        choice = ctx.original()
        inspection = ctx.inspect(choice)
        index = inspection['window']
        assert index == metadata[-1]['window'] and inspection['coefficients'].is_cuda
        inspection['values']['positions'].zero_()
        inspection['values']['F_sequence'].zero_()
        inspection['coefficients'].fill_(99.)
        identity, report = ctx.resolve(choice)
        exact(identity, donors[-1])
        assert not report['selected'] and ctx._model.adjoint is None
        # Declared evidence boundary: compare published frames after run_pipeline returns.
        archives.append(dict(window=index, positions=np.stack([to_host(x) for x in identity[0][1:]]),
                             F=np.stack([to_host(x) for x in identity[1][1:]])))
        contexts.append(ctx)
        return choice
    result = run_cuda(monkeypatch, source, target, prm, cfg, select, observe_prepared)
    assert len(contexts) >= 2 and all(ctx.closed for ctx in contexts)
    assert len(prepared_windows) >= 3
    assert any(row['newly_pinned'] > 0 for row in witnesses), 'Actual new pin admission is required'
    assert any(row['assimilation_changed'] for row in witnesses), 'Actual assimilation is required'
    assert not any(result['guards'].values())
    rows = {row['animation']: row for row in result['history'] if 'window_selection' in row}
    for archive in archives:
        row = rows[archive['window']]
        assert not row.get('outer_rejected') and not row['window_selection']['selected']
        start = row['frame_end']-cfg.T
        np.testing.assert_array_equal(np.stack(result['frames'][start:start+cfg.T]), archive['positions'])
        np.testing.assert_array_equal(np.stack(result['F_frames'][start:start+cfg.T]), archive['F'])


def test_changed_private_candidate_reaches_same_forward_handoff_and_next_start(monkeypatch):
    source, target, prm, cfg = fixture()
    cfg.animations, cfg.layer_k, cfg.w_pbr, cfg.settle_pin = 2, 8, .2, False
    contexts, chosen, archives = [], [], []
    def select(ctx, prepared, donors, metadata):
        original = ctx.inspect(ctx.original())
        index = original['window']
        if index:
            prior = chosen[index-1]['values']
            for key, field in (('x0', 'x'), ('v0', 'v'), ('C0', 'C'), ('F0', 'F')):
                wanted = prior[field].reshape_as(prepared[index][key])
                exact(prepared[index][key], wanted)
            assert not torch.equal(prepared[index]['Fp'], prepared[index-1]['Fp'])
        coefficients = original['coefficients']*(1+1e-5)
        assert not torch.equal(coefficients, original['coefficients'])
        choice = ctx.evaluate(coefficients, 'cuda_tiny_changed_body')
        observed = ctx.inspect(choice)
        assert observed['eligible'], observed['failures']
        assert observed['values']['positions'].is_cuda and observed['values']['health']['same_forward']
        assert ctx._model.adjoint.g_fwd is not None
        assert not torch.equal(observed['values']['positions'], original['values']['positions'])
        ctx.certify(choice, dict(passed=True, scope='API-only CUDA fixture; not production raw-quality evidence'))
        selected, report = ctx.resolve(choice)
        assert report['selected'] and selected[2]['Fg'] is None
        exact(selected[4], donors[index][4])
        exact(selected[5]['render_influence_steps'], donors[index][5]['render_influence_steps'])
        exact(torch.stack([torch.as_tensor(x, device='cuda:0') for x in selected[0][1:]]), observed['values']['positions'])
        exact(torch.stack([torch.as_tensor(F, device='cuda:0') for F in selected[1][1:]]), observed['values']['F_sequence'])
        for key in ('F', 'v', 'C'):
            actual = torch.as_tensor(selected[2][key], device='cuda:0')
            exact(actual, observed['values'][key].reshape_as(actual))
        observation = selected[5]['selected_observation']
        assert observation['loss'] == observed['metrics']['merit']
        assert observation['d_render'] == observed['metrics']['render']-cfg.w_pbr*observed['metrics']['pbr']
        assert observation['grad_norm'] is None and observation['alpha'] is None
        archives.append(dict(positions=to_host(observed['values']['positions']),
            F=to_host(observed['values']['F_sequence']), metrics=deepcopy(observed['metrics']),
            donor_accepted=donors[index][5]['accepted'], donor_iters=len(donors[index][4])))
        chosen.append(observed)
        contexts.append(ctx)
        return choice
    result = run_cuda(monkeypatch, source, target, prm, cfg, select)
    assert len(chosen) == 2 and all(ctx.closed for ctx in contexts)
    assert not any(result['guards'].values())
    for index, archive in enumerate(archives):
        row = result['history'][index]
        assert row['window_selection']['selected'] and not row.get('outer_rejected')
        assert row['F_kind'] == 'physics' and not row['commit_from_accepted']
        assert row['loss'] == archive['metrics']['merit'] and row['kin'] == archive['metrics']['stored_terminal']
        assert row['accepted'] == archive['donor_accepted'] and row['iters'] == archive['donor_iters']
        start = index*cfg.T+1
        np.testing.assert_array_equal(np.stack(result['frames'][start:start+cfg.T]), archive['positions'])
        np.testing.assert_array_equal(np.stack(result['F_frames'][start:start+cfg.T]), archive['F'])


def test_cuda_selector_exception_expires_context_and_final_merit_lease(monkeypatch):
    source, target, prm, cfg = fixture()
    contexts, evaluators = [], []
    def select(ctx, *_):
        assert ctx.inspect(ctx.original())['coefficients'].is_cuda
        contexts.append(ctx)
        evaluators.append(ctx._evaluate_merit)
        raise RuntimeError('CUDA selector fixture interruption')
    with pytest.raises(RuntimeError, match='CUDA selector fixture interruption'):
        run_cuda(monkeypatch, source, target, prm, cfg, select)
    assert len(contexts) == 1 and contexts[0].closed
    with pytest.raises(RuntimeError, match='expired'): contexts[0].original()
    with pytest.raises(RuntimeError, match='expired'): evaluators[0]({})


def test_cuda_auto_loss_resolves_before_context_validation(monkeypatch):
    source, target, prm, cfg = fixture()
    cfg.phys_loss = 'auto'
    seen = []
    def select(ctx, *_):
        assert ctx._cfg.phys_loss == 'ot_pace' and ctx._cfg.ot_handoff and ctx._cfg.ot_debias
        seen.append(ctx)
        return ctx.original()
    result = run_cuda(monkeypatch, source, target, prm, cfg, select)
    assert len(seen) == 1 and seen[0].closed and not any(result['guards'].values())
