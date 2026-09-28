"""Actual small CPU pipelines and falsification of the independent merit audit."""
import numpy as np
import pytest
import torch

from physmorph.pipeline import PipelineConfig
from physmorph.mpm.state import MPMParams
from scripts.probes.geometric_variance_integration import (
    independent_variances, objective_check, run_audited,
)


@pytest.mark.parametrize('geometric', [False, True])
@pytest.mark.parametrize('reject_last', [False, True])
def test_real_cap1_driver_reads_the_accepted_path_and_objective(geometric, reject_last, monkeypatch):
    from physmorph.pipeline import optimizer
    calls = []
    original = optimizer._state_ok
    def state_ok(state):
        calls.append(None)
        return False if reject_last and len(calls) == 2 else original(state)
    monkeypatch.setattr(optimizer, '_state_ok', state_ok)
    source = np.random.default_rng(27).uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (source*[1.2, .85, 1.05]+[.1, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=2, animations=1, stop_after_windows=1, loss_res=12,
        render_views=2, render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
        commit_pic=True, commit_pic_objective=True, geometric_variance=geometric,
        outer_render_committed=True, body_ctrl=True, body_terminal_ctrl=True, lambda_auto=.3,
        w_kin=.2, w_kin_var=.7, w_ctrl=0., w_box=0., max_ls_iters=1,
        adaptive_alpha=False, alpha=1e-4, loss_units='density')
    prm = MPMParams(dx=1., nx=32, ny=32, nz=32)
    result, report, audit = run_audited(source, target, prm, cfg, log=lambda *_: None)
    assert report['bounded_integration_passed']
    assert not report['primitive_gate_passed']
    assert report['full_archive_exact']
    assert report['windows'][0]['inner_accepted_iterations'] > 0
    assert report['windows'][0]['commit_source'] == ('replay' if reject_last else 'accepted_buffer')
    assert report['windows'][0]['active_pin_count'] == 0
    assert report['windows'][0]['active_pin_check_vacuous']
    assert report['windows'][0]['objective']['expected_contribution'] > 0
    assert report['windows'][0]['objective']['nonzero_participation_resolved']
    assert report['windows'][0]['selected_observable'] == ('geometric' if geometric else 'physical')
    assert bool(report['gradient_forwards']) is geometric
    assert all(row['terminal_exact'] for row in report['gradient_forwards'])
    # Audit owns snapshots even after the returned pipeline archive is changed.
    saved = audit.owned['promoted'].clone()
    result['frames'][-1].fill(0.)
    assert torch.equal(audit.owned['promoted'], saved)


def test_independent_variance_includes_projection_and_cannot_use_physical_V():
    start = torch.zeros(2, 3)
    positions = torch.zeros(3, 2, 3)
    promoted = torch.ones(2, 3)
    observed = independent_variances(start, positions, promoted, torch.zeros_like(positions), .5)
    assert float(observed['physical']) == 0.
    assert float(observed['geometric']) == pytest.approx(8/3)


def test_driver_rejects_wrong_objective_coefficient():
    # Reported weight disagrees with the actual prepared objective.
    audit = dict(observed_variance=.25, effective_weight=2.,
        evaluate=lambda value: 3.+value, accepted_merit=3.25, final_merit=3.25,
        replay_tolerance=1e-6)
    with pytest.raises(RuntimeError, match='recorded variance weight'):
        objective_check(audit, torch.tensor(.25, dtype=torch.float64), torch.tensor(1.))
