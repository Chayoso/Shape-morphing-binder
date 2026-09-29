"""Read-only reference observer through real CPU MPM/optimizer/outer promotion."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, runner


@pytest.mark.parametrize('layer', [False,True])
def test_reference_observer_preserves_actual_cpu_pipeline_and_expires(monkeypatch,layer,tmp_path):
    source = np.random.default_rng(27).uniform(-1.5, 1.5, (160,3)).astype(np.float32)
    target = (source*[1.2,.85,1.05]+[.1,0,0]).astype(np.float32)
    cfg = PipelineConfig(T=3,iters=2,animations=2,loss_res=12,render_views=2,
        render_elevs=(0.,.5),render_res=24,device='cpu',patience=5,
        outer_render_committed=True,body_ctrl=True,body_terminal_ctrl=True,
        lambda_auto=.3,w_kin=.2,w_kin_var=0.,w_ctrl=0.,w_box=0.,max_ls_iters=1,
        adaptive_alpha=False,alpha=1e-4,replay_calibrate=False,phys_loss='ot_pace',
        loss_units='density',ot_samples=128,ot_iters=20,render_paced=True,
        layer_ctrl=layer,layer_relax=layer)
    prm = MPMParams(dx=1.,nx=32,ny=32,nz=32)
    baseline = runner.run_pipeline(source,target,prm,deepcopy(cfg),log=lambda *_:None)
    original = runner.optimize_window
    captured = []
    closures = []

    def observe(index, packet, common):
        terms = packet['reference'].terms(packet['xT'])
        shared = common(packet['xT'])
        merit = float(terms['volume']+packet['lambda_render']*terms['render'])+shared['value']
        assert merit == pytest.approx(packet['expected']['merit'],rel=3e-6,abs=1e-8)
        assert bool(torch.isfinite(shared['gradient']).all())
        # Mutation of an owned observer copy must never modify production state.
        packet['positions'].fill_(123.)
        packet['pins'].fill_(True)
        captured.append(index)
        closures.append((common,packet['xT']))

    def wrap(*args,**kwargs):
        result = original(*args,on_reference=observe,**kwargs)
        with pytest.raises(RuntimeError,match='expired'):
            closures[-1][0](closures[-1][1])
        return result

    monkeypatch.setattr(runner,'optimize_window',wrap)
    observed = runner.run_pipeline(source,target,prm,deepcopy(cfg),log=lambda *_:None)
    assert captured == [0,1]
    np.testing.assert_array_equal(observed['frames'],baseline['frames'])
    np.testing.assert_array_equal(observed['F_frames'],baseline['F_frames'])
    assert observed['guards'] == baseline['guards']
    for a,b in zip(observed['history'],baseline['history']):
        for key in ('loss','lambda','d_vol','d_sil','frame_end'):
            assert a.get(key) == b.get(key)

    checkpoints = []
    models = []
    callback_packets = []

    def checkpoint(index,packet):
        assert packet['optimizer_state_exact']
        assert packet['gradient_replay']['merit'] == pytest.approx(packet['history']['loss'],rel=3e-6,abs=1e-8)
        assert set(packet['scalar_gradient']) == ({'stress','body','surface_u'} if layer else {'stress','body'})
        if layer:
            assert packet['scalar_gradient']['surface_u']['norm'] > 0
            assert bool((packet['controls']['surface_u'] != 0).any())
        model = packet['rollout']
        terminal = packet['controls']['body'][:,3:].detach().clone().requires_grad_()
        values = model.evaluate(terminal)
        assert values['valid']
        torch.testing.assert_close(values['positions'],packet['positions'],rtol=1e-6,atol=2e-7)
        torch.testing.assert_close(values['V'],packet['V'],rtol=1e-5,atol=2e-7)
        speed = values['v'].square().mean()+((values['positions'][-1]-values['positions'][-2])/packet['dt']).square().mean()
        derivative, = torch.autograd.grad(speed,terminal)
        assert bool(torch.isfinite(derivative).all()) and float(derivative.norm()) > 0
        if index==0 and packet['iteration']==1:
            from physmorph.pipeline.frozen_body_window import FrozenBodyWindow
            from dataclasses import asdict
            path = tmp_path/'window.npz'
            model.save(path,dict(reference=asdict(packet['reference']),positions=packet['positions']))
            restored,observations = FrozenBodyWindow.load(path,'cpu')
            restored_terminal = terminal.detach().clone().requires_grad_()
            replay = restored.evaluate(restored_terminal)
            torch.testing.assert_close(replay['positions'],packet['positions'],rtol=0,atol=0)
            assert torch.equal(observations['positions'],packet['positions'])
            from physmorph.pipeline.prepared_reference import PreparedReference
            restored_ref = PreparedReference(**observations['reference'])
            for key,value in packet['reference'].terms(packet['positions'][-1]).items():
                torch.testing.assert_close(restored_ref.terms(replay['x'])[key],value,rtol=0,atol=0)
            displacement = restored.coefficients[:,:3].detach().clone().requires_grad_()
            fixed_terminal = restored_terminal.detach().clone()
            explicit = restored.evaluate(fixed_terminal,displacement)
            jacobian, = torch.autograd.grad(explicit['x'][:,0].mean(),displacement)
            assert float(jacobian.norm())>0
            direction = jacobian/jacobian.norm()
            samples = []
            with torch.no_grad():
                for sign in (-1,1):
                    shifted = restored.evaluate(fixed_terminal,displacement+sign*.01*direction)
                    assert shifted['valid']
                    samples.append(float(shifted['x'][:,0].mean()))
            assert (samples[1]-samples[0])/.02 == pytest.approx(float(jacobian.norm()),rel=.03,abs=1e-5)
            assert torch.equal(fixed_terminal,restored_terminal.detach())
            from physmorph.pipeline.affine_braking import geometric_running
            selected = ~packet['pins']
            moving = restored.evaluate(fixed_terminal,displacement)
            running = geometric_running(moving['positions'],packet['x0'],packet['dt'],selected)
            running_grad, = torch.autograd.grad(running,displacement)
            direction = running_grad/running_grad.norm()
            values = []
            with torch.no_grad():
                for sign in (-1,1):
                    shifted = restored.evaluate(fixed_terminal,displacement+sign*.001*direction)
                    values.append(float(geometric_running(shifted['positions'],packet['x0'],packet['dt'],selected)))
            assert (values[1]-values[0])/.002 == pytest.approx(float(running_grad.norm()),rel=.03,abs=1e-5)
            restored.close()
        model.coefficients.fill_(0.)
        models.append(model)
        callback_packets.append(packet)
        checkpoints.append((index,packet['iteration']))
        packet['positions'].fill_(-123.)
        packet['C'].fill_(-123.)
        packet['controls']['stress'].fill_(999.)
        assert packet['history'].get('render_influence') is not None
        packet['history']['render_influence'].clear()

    def with_checkpoints(*args,**kwargs):
        return original(*args,on_checkpoint=checkpoint,checkpoint_iterations=(1,2),checkpoint_rollout=True,**kwargs)

    monkeypatch.setattr(runner,'optimize_window',with_checkpoints)
    checkpointed = runner.run_pipeline(source,target,prm,deepcopy(cfg),log=lambda *_:None)
    assert checkpoints == [(0,1),(0,2),(1,1),(1,2)]
    assert all(p['optimizer_state_after_callback_exact'] for p in callback_packets)
    for model in models:
        with pytest.raises(RuntimeError,match='expired'):
            model.evaluate(torch.zeros_like(model.coefficients[:,3:]))
    np.testing.assert_array_equal(checkpointed['frames'],baseline['frames'])
    np.testing.assert_array_equal(checkpointed['F_frames'],baseline['F_frames'])
    for a,b in zip(checkpointed['history'],baseline['history']):
        for key in ('loss','lambda','d_vol','d_sil','frame_end'):
            assert a.get(key) == b.get(key)
        assert a.get('render_influence_steps')
        assert a['render_influence_steps'] == b['render_influence_steps']
