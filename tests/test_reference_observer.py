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
        lambda_auto=.3,w_kin=.2,w_kin_var=.3,w_ctrl=.001,w_jvol=.5,w_box=0.,max_ls_iters=1,
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
    merit_evaluators = []

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
        for expired in merit_evaluators:
            with pytest.raises(RuntimeError,match='expired'):
                expired(values)
        evaluate = packet['evaluate_merit']
        binding = evaluate.binding_digest()
        merit_evaluators.append(evaluate)
        base_merit = evaluate(values)
        assert base_merit['merit']==base_merit['physical']+base_merit['lambda_render']*base_merit['render']
        assert base_merit['merit']==pytest.approx(packet['history']['loss'],rel=3e-6,abs=1e-8)
        assert base_merit['unit_weight']!=1.
        shifted_energy = values['body_energy']+5.
        changed = evaluate(dict(values,body_energy=shifted_energy))
        expected = base_merit['unit_weight']*cfg.w_ctrl*float(shifted_energy-values['body_energy'])
        assert changed['merit']-base_merit['merit']==pytest.approx(expected,rel=.003,abs=2e-9)
        population = values['V'].var(dim=0,correction=0).sum(-1).mean()
        assert base_merit['stored_variance']==pytest.approx(float(population),rel=3e-5,abs=1e-10)
        shifted_v = values['V'].detach().clone();shifted_v[0,:,0]+=.3
        varied = evaluate(dict(values,V=shifted_v))
        expected = base_merit['unit_weight']*cfg.w_kin_var*float(
            shifted_v.var(dim=0,correction=0).sum(-1).mean()-population)
        assert varied['merit']-base_merit['merit']==pytest.approx(expected,rel=.003,abs=2e-9)
        shifted_f = values['F']*1.05
        distorted = evaluate(dict(values,F=shifted_f))
        j0,j1 = [torch.linalg.det(f.reshape(-1,3,3)) for f in (values['F'],shifted_f)]
        expected = base_merit['unit_weight']*cfg.w_jvol*float(((j1-1)*j1.log()-(j0-1)*j0.log()).mean())
        assert distorted['merit']-base_merit['merit']==pytest.approx(expected,rel=.003,abs=2e-9)
        with pytest.raises(ValueError,match='Inconsistent'):
            evaluate(dict(values,v=values['v']+.1))
        with pytest.raises(ValueError,match='body_energy'):
            evaluate(dict(values,body_energy=values['body_energy']*float('nan')))
        # Fail after losses_of clears its mutable render cache, then recover.
        import physmorph.pipeline.optimizer as optimizer_module
        def failed_render(*args,**kwargs):
            raise RuntimeError('injected render failure')
        with monkeypatch.context() as nested:
            nested.setattr(optimizer_module,'d_render',failed_render)
            with pytest.raises(RuntimeError,match='injected'):
                evaluate(values)
        assert evaluate(values)==base_merit
        assert evaluate.binding_digest()==binding
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
            from physmorph.pipeline.support_braking import support_values
            support_forward = restored.evaluate(fixed_terminal,displacement)
            witness_ids = torch.nonzero(selected).flatten()[:1]
            support_q = support_forward['x'][witness_ids].detach().clone()+torch.tensor([[.1,.15,.2]])
            support_value = support_values(support_forward['x'],support_q,witness_ids,1.)[0]
            support_grad, = torch.autograd.grad(support_value,displacement)
            assert float(support_grad.norm())>1e-6
            direction = support_grad/support_grad.norm()
            samples = []
            for sign in (-1,1):
                shifted = restored.evaluate(fixed_terminal,displacement+sign*.003*direction)
                assert shifted['valid']
                samples.append(float(support_values(shifted['x'],support_q,witness_ids,1.)[0].detach()))
            assert (samples[1]-samples[0])/.006==pytest.approx(float(support_grad.norm()),rel=.04,abs=1e-6)
            restored.close()
            # The paired component diagnostic must linearize a real MPM window
            # only once, with a live original-merit evaluator and exact ownership.
            from scripts.probes.running_braking_repair import repair
            local = dict(packet,trial05_terminal=model.coefficients[:,3:].detach().clone()*.5)
            base_dir = tmp_path/'shared_base';base_dir.mkdir()
            origin_dir = tmp_path/'shared_origin';origin_dir.mkdir()
            # Bunny-specific upper-y quality cohorts do not exist in this toy cloud.
            # Stub only that report; forward, losses, all gradients and merit stay real.
            from test_remainder_braking import synthetic_summary
            with monkeypatch.context() as quality:
                quality.setattr('scripts.probes.running_braking_repair.summarize',synthetic_summary)
                common = repair(model,local,source,target,base_dir,baseline_only=True,prepare_support=True)
                origin = repair(model,local,source,target,origin_dir,shared_baseline=common,origin_only=True,prepare_support=True)
            assert origin.decode()['baseline_sha256']==common.digest()
            assert common.tensor('endpoints','cpu').shape==(3,len(source),3)
            # This sparse toy target needs no new support row; do not claim
            # combined support-row MPM coverage from this common-origin check.
            assert origin.decode()['support_metadata']['status']=='no_intervention'
            assert origin.decode()['support_metadata']['count']==0
            assert len(origin.decode()['running_repeats'])==3
            g_sil = origin.tensor('gradient_silhouette','cpu')
            assert bool(torch.isfinite(g_sil).all()) and float(g_sil.norm())>0
            direction = g_sil/g_sil.norm()
            samples = []
            for sign in (-1,1):
                perturbed = model.evaluate(local['trial05_terminal'],model.coefficients[:,:3]+sign*.003*direction)
                assert perturbed['valid']
                samples.append(float(packet['reference'].terms(perturbed['x'])['silhouette']))
            assert (samples[1]-samples[0])/.006==pytest.approx(float(g_sil.norm()),rel=.04,abs=1e-6)
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
        return original(*args,on_checkpoint=checkpoint,checkpoint_iterations=(1,2),
                        checkpoint_rollout=True,checkpoint_merit=True,**kwargs)

    monkeypatch.setattr(runner,'optimize_window',with_checkpoints)
    checkpointed = runner.run_pipeline(source,target,prm,deepcopy(cfg),log=lambda *_:None)
    assert checkpoints == [(0,1),(0,2),(1,1),(1,2)]
    assert all(p['optimizer_state_after_callback_exact'] for p in callback_packets)
    for evaluate in merit_evaluators:
        with pytest.raises(RuntimeError,match='expired'):
            evaluate({})
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
