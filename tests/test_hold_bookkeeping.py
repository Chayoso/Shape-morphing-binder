"""Copied suffixes are archive metadata, never evidence of simulated rest."""
import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from physmorph import metrics
from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, run_pipeline, runner
from physmorph.pipeline import dressing


@pytest.mark.parametrize('with_dressing', [False, True])
@pytest.mark.parametrize('hold', [False, True])
def test_frozen_copy_counts_preserve_archive_and_loop(monkeypatch, with_dressing, hold):
    # Enter the real runner's frozen branch without a numerical optimization.
    # Dressing is an archive observer here; no appearance solve is exercised.
    source = np.array([[-.5, -.5, -.5], [.5, .5, .5]], np.float32)
    calls, covered, configured = [], [], []
    gauss = SimpleNamespace(configure_source=lambda *args: configured.append(args))
    monkeypatch.setattr(runner, 'build_target', lambda *args, **kw: SimpleNamespace(
        gauss=gauss if with_dressing else None))
    monkeypatch.setattr(runner, '_surface_weights',
                        lambda x, *args: np.ones(len(x), np.float32))

    class ArchiveObserver:
        def __init__(self, *args):
            pass

        def cover_frames(self, count):
            covered.append(count)

        def export(self):
            return {'covered': covered.copy()}

    monkeypatch.setattr(dressing, 'DressState', ArchiveObserver)

    def converged_window(x, *args, **kwargs):
        calls.append(x.copy())
        return [], [], {}, None, [], {'grad_converged': True}

    monkeypatch.setattr(runner, 'optimize_window', converged_window)
    cfg = PipelineConfig(device='cpu', animations=4, T=2, iters=1,
                         hold_after_converge=hold, persistent_rest_volume=False,
                         local_dress_iters=int(with_dressing),
                         gauss_children=2 if with_dressing else 1,
                         render_surface_only=with_dressing,
                         surface_grad_frac=1. if with_dressing else 0.)
    result = run_pipeline(source, source.copy(), MPMParams(), cfg, log=lambda *_: None)
    copies = (cfg.animations - 1 if with_dressing else 1) if hold else 0
    assert len(calls) == 1  # Frozen iterations never invoke another rollout.
    np.testing.assert_array_equal(calls[0], source)
    assert result['converged'] and result['truncation'] is None
    assert result['deliver_n'] == len(result['frames']) == len(result['F_frames']) == 1 + copies
    assert result['n_held'] == copies
    assert result['history'] == [dict(animation=0, grad_converged=1, render_target_kind=None)] + [
        dict(animation=i+1, held=1) for i in range(copies)]
    for x, F in zip(result['frames'], result['F_frames']):
        np.testing.assert_array_equal(x, source)
        np.testing.assert_array_equal(F, np.broadcast_to(np.eye(3, dtype=np.float32), (len(source), 3, 3)))
    if with_dressing:
        # Same cover_frames/continue path, including the final archive closure.
        counts = list(range(2, cfg.animations+1)) if hold else [1] * (cfg.animations-1)
        assert covered == counts + [1 + copies]
        assert len(configured) == 1
    else:
        assert covered == configured == []  # Break after the first frozen iteration.

    # Appending the held suffix to a known moving tail must not dilute its metric.
    preceding = source - np.array([.25, 0., 0.], np.float32)
    tail = [preceding, *result['frames']]
    report = metrics.jitter(tail, n_held=result['n_held'])
    assert report['jitter_abs'] == pytest.approx(.25)
    if copies:
        assert metrics.jitter(tail, n_held=0)['jitter_abs'] < report['jitter_abs']
        assert not np.shares_memory(result['frames'][0], result['frames'][-1])
        assert not np.shares_memory(result['F_frames'][0], result['F_frames'][-1])


def test_full_quality_scope_is_delivery_scoped_without_changing_data(tmp_path, monkeypatch):
    previous_cache = wp.config.kernel_cache_dir
    monkeypatch.setenv('WARP_CACHE_PATH', str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        'hold_quality_compare', Path(__file__).parents[1]/'scripts/probes/quality_compare.py')
    probe = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(probe)
    finally:
        wp.config.kernel_cache_dir = previous_cache
    # A later accepted commit exists in history but lies outside the delivery.
    records = [dict(animation=0, frame_end=3), dict(animation=1, frame_end=5)]
    runs = [dict(records=records[:1], delivered=3, arm=dict(history=records))]
    for mode in probe.FULL_INTERVENTIONS:
        scoped, description = probe.scoped_runs(runs, mode)
        assert scoped is runs and scoped[0]['records'] == records[:1]
        assert scoped[0]['arm']['history'] == records
        assert description['kind'] == 'full runs'
        assert 'delivery' in description['endpoint_claim']
        assert 'truncation' in description['endpoint_claim']
        assert 'stopping point' not in description['endpoint_claim']


def test_attempt_count_retains_rejected_null_and_converged_calls():
    from scripts.probes.render_influence import count_optimizer_attempts

    history = [dict(animation=0, frame_end=3),
               dict(animation=1, c2f_render_res=96),
               dict(animation=1, outer_rejected=True),
               dict(animation=2, null_commit=1),
               dict(animation=3, grad_converged=1),
               dict(animation=4, held=1),
               dict(note='non-optimizer metadata')]
    before = deepcopy(history)
    assert count_optimizer_attempts(history) == 4
    assert count_optimizer_attempts(history, through_animation=0) == 1
    assert count_optimizer_attempts(history, through_animation=1) == 2
    assert count_optimizer_attempts(history, through_animation=2) == 3
    assert count_optimizer_attempts(history, through_animation=4) == 4
    assert count_optimizer_attempts([]) == 0
    assert history == before and len(history) == 7
