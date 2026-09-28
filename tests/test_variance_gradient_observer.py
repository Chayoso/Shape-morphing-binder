"""The prepared-graph audit must leave the production objective and step intact."""
import copy

import numpy as np
import pytest
import torch

from physmorph.pipeline import PipelineConfig, run_pipeline
from physmorph.mpm.state import MPMParams


@pytest.mark.parametrize('units', ['legacy', 'density'])
@pytest.mark.parametrize('mode', ['render', 'off'])
@pytest.mark.parametrize('layer', [False, True])
def test_observer_is_owned_and_does_not_change_step(monkeypatch, units, mode, layer):
    from physmorph.pipeline import runner
    source = np.random.default_rng(31).uniform(-1.3, 1.3, (96, 3)).astype(np.float32)
    target = (source * [1.2, .9, 1.05] + [.15, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=2, animations=1, loss_res=12, render_views=2,
        render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
        commit_pic=True, commit_pic_objective=True, geometric_variance=False,
        body_ctrl=True, body_terminal_ctrl=True, lambda_auto=.3,
        layer_ctrl=layer, layer_relax=layer,
        grad_project=True, grad_project_mode=mode,
        w_kin=.2, w_kin_var=.7, w_ctrl=0., w_box=0., max_ls_iters=1,
        adaptive_alpha=False, alpha=1e-4, loss_units=units)
    prm = MPMParams(dx=1., nx=32, ny=32, nz=32)
    plain = run_pipeline(source, target, prm, copy.deepcopy(cfg), log=lambda *_: None)
    original = runner.optimize_window
    audits = []

    def observe(window, audit):
        assert window == 0 and audit['same_graph'] and audit['alternate_forward_count'] == 0
        assert audit['mode'] == mode and audit['T'] == 3 and audit['N'] == len(source)
        assert audit['balancer_state']['lam'] is None
        assert audit['geometric_physics_core']-audit['physics_core'] == pytest.approx(
            audit['effective_weight']*(audit['geometric_variance']-audit['physical_variance']),
            rel=3e-5, abs=2e-6*max(abs(audit['physics_core']), 1e-10))
        gradients = audit['gradients']
        assert ('surface_u' in audit['leaf_names']) == layer
        for name, values in gradients.items():
            if values is not None:
                assert len(values) == len(audit['leaf_names'])
                assert all(not value.requires_grad and torch.isfinite(value).all() for value in values)
        for name in ('physical', 'geometric'):
            for initial, repeat in zip(gradients[name], gradients[name+'_repeat']):
                assert initial.data_ptr() != repeat.data_ptr()
                torch.testing.assert_close(initial, repeat, atol=2e-7, rtol=2e-5)
        difference = sum(float((a-b).square().sum()) for a, b in
                         zip(gradients['physical'], gradients['geometric']))
        assert difference > 0
        audits.append(copy.deepcopy(audit))
        # Owned observer payloads may be retained/mutated without corrupting solver scratch.
        for values in gradients.values():
            if values is not None:
                for value in values:
                    value.fill_(float('nan'))
        audit['balancer_state']['lam'] = 1e30

    def wrapped(*args, **kwargs):
        return original(*args, on_gradient_audit=observe, **kwargs)

    monkeypatch.setattr(runner, 'optimize_window', wrapped)
    audited = run_pipeline(source, target, prm, copy.deepcopy(cfg), log=lambda *_: None)
    assert len(audits) == 1
    assert plain['guards'] == audited['guards'] and not any(audited['guards'].values())
    assert len(plain['frames']) == len(audited['frames'])
    np.testing.assert_allclose(np.stack(plain['frames']), np.stack(audited['frames']), atol=2e-6, rtol=2e-6)
    assert [bool(r.get('frame_end')) for r in plain['history']] == [bool(r.get('frame_end')) for r in audited['history']]
    for a, b in zip(plain['history'], audited['history']):
        assert a['lambda'] == pytest.approx(b['lambda'], rel=2e-5, abs=1e-9)


@pytest.mark.parametrize('change', [dict(geometric_variance=True), dict(T=1),
    dict(w_kin_var=0.), dict(w_kin_var=float('nan')), dict(grad_h1=True),
    dict(grad_project_mode='blend'), dict(lambda_auto=0.)])
def test_unsupported_audit_fails_before_target_access(change):
    from physmorph.pipeline.optimizer import optimize_window
    from physmorph.pipeline.render_loss import LambdaBalancer
    cfg = PipelineConfig(commit_pic=True, commit_pic_objective=True, T=3, w_kin_var=.7,
                         grad_project=True, grad_project_mode='render')
    for key, value in change.items():
        setattr(cfg, key, value)
    with pytest.raises(ValueError, match='gradient audit'):
        optimize_window(np.zeros((4, 3), np.float32), MPMParams(), cfg, None,
                        LambdaBalancer(cfg.lambda_auto), on_gradient_audit=lambda *_: None)
