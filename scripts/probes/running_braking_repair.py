"""Fresh-window displacement repair of running motion with affine data restoration."""
import math
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.compute import to_host,cuda_execution
from physmorph.pipeline.affine_braking import affine_ball_step,projected_affine_check,geometric_running,observed_remainder
from physmorph.pipeline.frozen_body_window import project_terminal
from scripts.probes.inner_budget import summarize
from scripts.probes.reference_swap import require,sha
from scripts.probes.terminal_braking import main,assess_candidates,finite_scalar
from scripts.probes.live_braking_compensation import LiveCompensation,identity_evidence


def repair(model,packet,source,target,out,*,correction_rounds=0,quality_backtracking=False):
    """Use the caller-owned live model; never load, close or commit it here."""
    reference = packet['reference']
    require(correction_rounds in (0,2),'Unregistered correction budget')
    require(not quality_backtracking or correction_rounds==2,'Quality filtering requires the corrected model')
    rows,trials,sidecars = [],[],{}
    d0 = model.coefficients[:,:3].detach().clone()
    b0 = model.coefficients[:,3:].detach().clone()
    brake = packet['trial05_terminal'].detach().clone()
    target_x = packet['positions'][-1].detach().clone()
    dx = model.spec.prm.dx
    base_C = None
    arrived = packet['start_arrived'] & ~packet['pins']
    require(bool(arrived.any()),'No start-arrived-free cohort')

    def running(values):
        return geometric_running(values['positions'],packet['x0'],packet['dt'],arrived)

    def error(values):
        return (values['x']-target_x).square().sum(-1).mean()/(dx*dx)

    def measure(values,label,d,b,save):
        record = dict(label=label,valid=values['valid'],pins_exact=values['pins_exact'],
            min_det=finite_scalar(values['min_det']),endpoint_mse=finite_scalar(error(values)),
            running=finite_scalar(running(values)),body_energy=finite_scalar(values['body_energy']),
            coefficient_delta_rms=float((d-d0).square().sum(-1).mean().sqrt()),
            terminal_coefficients_exact=torch.equal(b,brake) if label!='repeat' else torch.equal(b,b0))
        if values['valid']:
            data = reference.terms(values['x'])
            record['data'] = {k:float(v) for k,v in data.items()}
            record['data']['weighted_render'] = packet['lambda_render']*record['data']['render']
            synthetic = dict(packet,positions=values['positions'],V=values['V'],F=values['F'])
            summary = summarize(type('One',(),{'packets':{8:synthetic}})(),source,target)
            record.update(motion=summary['rows'][0]['motion'],geometry=summary['rows'][0]['geometry'],
                source_upper_ids_sha256=summary['source_upper_ids_sha256'],
                F_change_rms=float((values['F']-packet['F'].reshape_as(values['F'])).square().mean().sqrt()),
                C_change_rms=float((values['C']-base_C).square().mean().sqrt()))
        if save:
            path = out/(label+'_'+str(len(sidecars))+'.npz')
            np.savez_compressed(path,displacement=to_host(d),terminal=to_host(b),
                positions=to_host(values['positions']),V=to_host(values['V']),F=to_host(values['F']),C=to_host(values['C']))
            sidecars[path.name] = sha(path)
        return record

    with torch.no_grad():
        for i in range(3):
            values = model.evaluate(b0,d0)
            eps = 32*torch.finfo(d0.dtype).eps
            accepted_closure = {}
            for key,expected,unit in (
                ('positions',packet['positions'],dx),('V',packet['V'],dx/(model.spec.T*packet['dt'])),
                ('F',packet['F'].reshape_as(values['F']),1.)):
                ratio = float(((values[key]-expected).abs()/(eps*(unit+expected.abs()))).max())
                accepted_closure[key] = ratio
                require(ratio<=1.,'Standalone baseline differs from accepted '+key)
            terms = reference.terms(values['x'])
            expected_render = packet['history']['d_render']+reference.pbr_weight*(packet['history']['d_pbr'] or 0.)
            for key,expected in (('volume',packet['history']['d_vol']),('render',expected_render)):
                ratio = abs(float(terms[key])-expected)/(eps*max(abs(expected),1e-12))
                accepted_closure[key] = ratio
                require(ratio<=1.,'Standalone baseline data differs: '+key)
            if base_C is None: base_C = values['C'].clone()
            record = measure(values,'repeat',d0,b0,True)
            record['accepted_closure'] = accepted_closure
            require(record['valid'],'Invalid loaded baseline')
            rows.append(record)
        brake_errors = []
        for i in range(3):
            values = model.evaluate(brake,d0)
            require(values['valid'],'Invalid selected terminal brake')
            brake_errors.append(float(running(values)))
        rows.append(measure(values,'terminal05',d0,brake,True))
    measured_noise = max(brake_errors)-min(brake_errors)
    ceilings = {k:max(r['data'][k] for r in rows[:3]) for k in ('volume','render')}
    displacement = d0.clone()
    trust = packet['history']['body_update_modes_rms'][0]
    require(trust>0,'No original displacement trust step')
    accepted = 0
    fixed_candidate_replays = []
    replay_checked = False
    for iteration in range(1 if quality_backtracking else 4):
        with torch.enable_grad():
            leaf = displacement.detach().clone().requires_grad_()
            values = model.evaluate(brake,leaf)
            require(values['valid'],'Invalid current running-repair state')
            objective = running(values)
            terms = reference.terms(values['x'])
            gradients = [torch.autograd.grad(v,leaf,retain_graph=i<2)[0].detach()
                         for i,v in enumerate((objective,terms['volume'],terms['render']))]
            require(all(bool(torch.isfinite(g).all()) for g in gradients),'Nonfinite repair gradient')
            G = torch.stack(gradients[1:])
            origin_data = torch.stack([terms[k].detach().double() for k in ('volume','render')])
            bounds = torch.tensor([ceilings[k]-float(terms[k].detach()) for k in ('volume','render')],
                                  device=leaf.device,dtype=torch.float64)
            current = float(objective.detach())
        del values,objective,terms
        path = out/f'linearization{iteration+1}.npz'
        np.savez_compressed(path,displacement=to_host(displacement),terminal=to_host(brake),
                            running_gradient=to_host(gradients[0]),data_gradients=to_host(G),
                            bounds=to_host(bounds),origin_data=to_host(origin_data))
        sidecars[path.name] = sha(path)
        threshold = 10*max(measured_noise,32*torch.finfo(displacement.dtype).eps*current)
        found = False
        for halving in range(11):
            radius = trust*(.5**halving)*math.sqrt(len(displacement))
            remainder = torch.zeros_like(bounds)
            for correction in range(correction_rounds+1):
                shifted_bounds = bounds-remainder
                step,linear = affine_ball_step(gradients[0],G,shifted_bounds,radius)
                label = f'update{iteration+1}_half{halving}'
                if correction_rounds: label += f'_correction{correction}'
                record = dict(label=label,iteration=iteration+1,correction=correction,
                              halvings=halving,linear=linear,running_update_accepted=False,
                              model_remainder_used=to_host(remainder).tolist(),
                              shifted_bounds=to_host(shifted_bounds).tolist())
                retry = False
                if step is not None:
                    candidate = project_terminal((displacement.double()+step).to(displacement.dtype),brake)
                    actual_step = candidate.double()-displacement.double()
                    check = projected_affine_check(actual_step,G,shifted_bounds,displacement.dtype)
                    norm = float(actual_step.norm())
                    radius_tolerance = 32*torch.finfo(displacement.dtype).eps*(radius+float(displacement.norm()))
                    record.update(projected_affine=check,
                                  original_projected_affine=projected_affine_check(actual_step,G,bounds,displacement.dtype),
                                  projected_step_norm=norm,trust_pass=norm<=radius+radius_tolerance,
                                  trust_roundoff=radius_tolerance)
                    if check['passed'] and record['trust_pass']:
                        with torch.no_grad():
                            values = model.evaluate(brake,candidate)
                            record.update(measure(values,record['label'],candidate,brake,False))
                            data_pass = record['valid'] and all(record['data'][k]<=ceilings[k] for k in ceilings)
                            improvement = current-record['running'] if record['running'] is not None else None
                            take = bool(data_pass and improvement>threshold)
                            record.update(data_restored=bool(data_pass),running_improvement=improvement,
                                          reduction_threshold=threshold,running_update_accepted=take)
                            if take and quality_backtracking:
                                assess_candidates(rows[:3]+[record],displacement.dtype)
                                record['data_running_passed'] = True
                                take = record['feasible']
                                record['running_update_accepted'] = take
                                record['quality_backtracking_rejected'] = not take
                            if record['valid']:
                                actual_data = torch.tensor([record['data'][k] for k in ('volume','render')],
                                                           device=G.device,dtype=torch.float64)
                                # Replace the estimate. All rounds retain the same origin and Jacobian.
                                remainder = observed_remainder(actual_data,origin_data,G,actual_step)
                                record['observed_model_remainder'] = to_host(remainder).tolist()
                                retry = not data_pass
                            if take:
                                displacement = candidate.detach().clone()
                                rows.append(measure(values,f'repair{iteration+1}',displacement,brake,True))
                                accepted+=1;found=True
                                assess_candidates(rows,displacement.dtype)
                                if correction_rounds and rows[-1]['feasible']:
                                    replay_checked = True
                                    for repeat in range(3):
                                        repeated = model.evaluate(brake,displacement)
                                        witness = measure(repeated,f'fixed_replay{repeat}',displacement,brake,True)
                                        assess_candidates(rows[:3]+[witness],displacement.dtype)
                                        witness['running_decrease_resolved'] = (
                                            witness['running'] is not None and current-witness['running']>threshold)
                                        fixed_candidate_replays.append(witness)
                                    # Stop after the first all-gate candidate even if its repeats fail.
                                    record['fixed_candidate_replay_checked'] = True
                trials.append(record)
                print(json.dumps(record,allow_nan=False),flush=True)
                if found or not retry: break
            if found: break
        if not found or replay_checked: break
    assess_candidates(rows,displacement.dtype)
    require(torch.equal(brake,packet['trial05_terminal']),'Terminal coefficient changed')
    return dict(rows=rows,trials=trials,accepted_running_updates=accepted,sidecars=sidecars,
                running_replay_noise=measured_noise,data_ceilings=ceilings,lambda_render=packet['lambda_render'],
                correction_rounds=correction_rounds,quality_backtracking=quality_backtracking,
                fixed_candidate_replays=fixed_candidate_replays,
                repeated_feasible=replay_checked and all(r['feasible'] and r['running_decrease_resolved']
                                                         for r in fixed_candidate_replays))


class RunningRepair(LiveCompensation):
    operation = staticmethod(repair)
    artifact_subdir = 'repair'
    scope = 'Fresh live callback; affine-restored running-motion diagnostic only, no archive reuse or commit'
    rendering_role = 'Prepared CIC/PBR guides the original solve, constrains each displacement trial, and screens final candidates. Reported norm shares are not causal motion shares or 4K appearance quality.'


if __name__=='__main__':
    RunningRepair.identity = identity_evidence(Path('/data/relcfd/chayo/physmorph_v2'))
    with cuda_execution('cuda:0'):
        g = torch.tensor([0.,-1.],dtype=torch.float64,device='cuda:0')
        G = torch.tensor([[1.,0.],[-1.,1e-4]],dtype=torch.float64,device='cuda:0')
        b = torch.tensor([-.6,.6-.8e-4],dtype=torch.float64,device='cuda:0')
        step,info = affine_ball_step(g,G,b,1.)
        require(step is not None and bool(torch.allclose(step,torch.tensor([-.6,-.8],device=g.device,dtype=g.dtype),
                                                        rtol=0,atol=2e-8)),'CUDA affine operator smoke failed')
        print(json.dumps(dict(cuda_affine_smoke=True,linear=info)),flush=True)
    main(RunningRepair,extra_protocol=dict(kind='Fresh live-window affine running repair',version=1,
        scope=RunningRepair.scope,identity_discrimination=RunningRepair.identity,updates=4,halvings=10,
        objective='Mean squared position-derived velocity over all steps including x0->x1, fixed start-arrived-free IDs',
        data_constraints='Affine restoration to max of three baseline prepared volume/render losses; nonlinear ceilings unchanged',
        projected_linear_check='32 input float32 eps times gradient norm * actual step norm + absolute RHS; nonlinear gates remain exact',
        terminal='Own freshly generated trial05, fixed',admission='No archive admission or production candidate commit'),
        extra_helpers=tuple(Path(__file__).resolve().with_name(name) for name in
            ('running_braking_repair.py','live_braking_compensation.py','braking_compensation.py')))
