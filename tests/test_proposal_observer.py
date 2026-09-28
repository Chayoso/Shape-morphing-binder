"""Prepared proposal replay must not advance or corrupt the production solve."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from physmorph.pipeline import PipelineConfig, run_pipeline
from physmorph.mpm.state import MPMParams
from physmorph.pipeline.grad_combine import pcgrad
from scripts.probes.variance_gradient_audit import composite, updated_balancer


@pytest.mark.parametrize('units', ['legacy', 'density'])
@pytest.mark.parametrize('body', [False, True])
def test_trial_replays_restore_controls_moments_and_production(monkeypatch, units, body):
    from physmorph.pipeline import runner, optimizer
    source = np.random.default_rng(31).uniform(-1.3, 1.3, (96, 3)).astype(np.float32)
    target = (source*[1.2, .9, 1.05]+[.15, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=2, animations=2, loss_res=12, render_views=2,
        render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
        commit_pic=True, commit_pic_objective=True, outer_render_committed=True,
        body_ctrl=body, body_terminal_ctrl=body, layer_ctrl=True, layer_relax=True,
        lambda_auto=.3, grad_project=True, grad_project_mode='render', mom_carry=0. if body else .7,
        w_kin=.2, w_kin_var=.7, w_ctrl=0., w_box=0., max_ls_iters=2,
        adaptive_alpha=True, alpha=1e-4, loss_units=units)
    prm = MPMParams(dx=1., nx=32, ny=32, nz=32)
    plain = run_pipeline(source, target, prm, deepcopy(cfg), log=lambda *_: None)
    original, state_ok = runner.optimize_window, optimizer._state_ok
    expired, observations = [], []
    raise_in_trial = [False]

    def checked_state(state):
        if raise_in_trial[0]:
            raise RuntimeError('injected trial check failure')
        return state_ok(state)

    monkeypatch.setattr(optimizer, '_state_ok', checked_state)

    def observe(window, audit):
        assert window == 1
        settings = audit['trial_settings']
        values, names = audit['gradients'], audit['leaf_names']
        p, q, r = (values[k] for k in ('physical', 'geometric', 'render_raw'))
        rp, _ = pcgrad(p, r)
        rq, _ = pcgrad(q, r)
        lam, _ = updated_balancer(audit['balancer_state'], p, rp)
        assert lam == settings['lambda'] and settings['alpha'] > 0
        ga = composite(p, rp, lam, values['transport'], names, audit['layer_u_render_only'])
        gb = composite(q, rq, lam, values['transport'], names, audit['layer_u_render_only'])
        evaluate = audit['trial_evaluate']
        a = evaluate(ga, observable='physical')
        b = evaluate(gb, observable='geometric')
        # Fail after candidate controls are installed, then prove restoration by replay.
        raise_in_trial[0] = True
        with pytest.raises(RuntimeError, match='injected trial'):
            evaluate(gb, observable='geometric')
        raise_in_trial[0] = False
        again = evaluate(ga, observable='physical')
        assert all(torch.equal(x, y) for x, y in zip(a['controls_delta'], again['controls_delta']))
        for key in ('positions', 'promoted', 'physical_v', 'F', 'v', 'C'):
            torch.testing.assert_close(a[key], again[key], atol=2e-6, rtol=2e-5)
        for value in (a, b, again):
            assert value['stats_restore_exact'] and value['alpha'] == settings['alpha']
            assert value['lambda'] == lam and set(value['merits']) == {'physical', 'geometric'}
            before = settings['initial_merits'][value['observable']]
            after = value['merits'][value['observable']]
            expected_gate = value['state_ok'] and after <= before-value['required_decrease']
            assert value['first_trial_merit_ok'] == expected_gate
        assert sum(float((x-y).square().sum()) for x, y in
                   zip(a['controls_delta'], b['controls_delta'])) > 0
        expired.append(evaluate)
        observations.append(a)

    def wrapped(*args, **kwargs):
        if kwargs['win_index'] == 1:
            kwargs.update(on_gradient_audit=observe, audit_proposals=True)
        result = original(*args, **kwargs)
        if kwargs['win_index'] == 1:
            with pytest.raises(RuntimeError, match='expired'):
                expired[-1]([], observable='physical')
        return result

    monkeypatch.setattr(runner, 'optimize_window', wrapped)
    audited = run_pipeline(source, target, prm, deepcopy(cfg), log=lambda *_: None)
    assert len(observations) == 1 and not any(audited['guards'].values())
    np.testing.assert_allclose(np.stack(plain['frames']), np.stack(audited['frames']), atol=3e-6, rtol=3e-6)
    assert [row.get('frame_end') for row in plain['history']] == [row.get('frame_end') for row in audited['history']]
    for a, b in zip(plain['history'], audited['history']):
        if 'lambda' in a:
            assert a['lambda'] == pytest.approx(b['lambda'], rel=2e-5, abs=1e-10)
