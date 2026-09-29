"""Fresh-window displacement repair of running motion with affine data restoration."""
import math
import json
from pathlib import Path
import sys
from dataclasses import dataclass
from copy import deepcopy
from hashlib import sha256

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.compute import to_host,to_array,KDTree,cuda_execution,array_api
from physmorph.pipeline.affine_braking import affine_ball_step,projected_affine_check,geometric_running,observed_remainder
from physmorph.pipeline.frozen_body_window import project_terminal
from physmorph.pipeline.diagnostic_binding import content_digest
from physmorph.pipeline.support_braking import select_support,support_values,support_report
from scripts.probes.inner_budget import summarize
from scripts.probes.reference_swap import require,sha
from scripts.probes.terminal_braking import main,assess_candidates,finite_scalar
from scripts.probes.live_braking_compensation import LiveCompensation,identity_evidence


def repair_context(model,packet,source,target):
    evaluator = packet.get('evaluate_merit')
    return content_digest(dict(model={k:getattr(model,k,None) for k in
        ('spec','idx','weights','gate','coefficients','stress','surface_u')},
        observation={k:packet.get(k) for k in ('reference','x0','positions','V','F','pins','plan',
            'start_arrived','arrival_radius','lambda_render','dt','history','controls')},
        source=source,target=target,merit_binding=None if evaluator is None else evaluator.binding_digest()))


@dataclass(frozen=True)
class OwnedRepairEvidence:
    """Only immutable bytes/tuples; each arm receives independently decoded copies."""
    payload: bytes
    arrays: tuple
    context: str

    @classmethod
    def pack(cls,data,values,context):
        arrays = []
        for key,value in values.items():
            array = np.ascontiguousarray(to_host(value))
            arrays.append((key,str(array.dtype),tuple(array.shape),array.tobytes()))
        return cls(json.dumps(data,sort_keys=True,allow_nan=False).encode(),tuple(arrays),context)

    def decode(self):
        return json.loads(self.payload)

    def tensor(self,key,device):
        for name,dtype,shape,data in self.arrays:
            if name==key:
                return torch.tensor(np.frombuffer(data,dtype=dtype).reshape(shape).copy(),device=device)
        raise KeyError(key)

    def digest(self):
        digest = sha256(self.payload+self.context.encode())
        for key,dtype,shape,data in self.arrays:
            digest.update(json.dumps([key,dtype,shape]).encode());digest.update(data)
        return digest.hexdigest()


@dataclass(frozen=True)
class SharedRepairBaseline(OwnedRepairEvidence):
    @classmethod
    def capture(cls,rows,base_C,coefficients,sidecars,context,endpoints=None):
        data = dict(rows=rows,sidecars=sidecars,
            ceilings={k:max(r['data'][k] for r in rows) for k in ('volume','render')},
            silhouette_ceiling=max(r['data']['silhouette'] for r in rows),
            merit_ceiling=max(r['original_merit']['merit'] for r in rows))
        arrays = dict(C=base_C,coefficients=coefficients)
        if endpoints is not None: arrays['endpoints'] = endpoints
        return cls.pack(data,arrays,context)


@dataclass(frozen=True)
class SharedRepairOrigin(OwnedRepairEvidence):
    """One terminal-forward linearization and running noise reference for both arms."""


def repair(model,packet,source,target,out,*,correction_rounds=0,quality_backtracking=False,
           shared_baseline=None,baseline_only=False,selected_terminal=None,terminal_label='terminal05',
           shared_origin=None,origin_only=False,protect_silhouette=False,
           prepare_support=False,protect_support=False,save_search_endpoints=False):
    """Use the caller-owned live model; never load, close or commit it here."""
    reference = packet['reference']
    require(correction_rounds in (0,2),'Unregistered correction budget')
    require(not quality_backtracking or correction_rounds==2,'Quality filtering requires the corrected model')
    require(not origin_only or (shared_baseline is not None and shared_origin is None and not baseline_only),
            'Shared origin requires a prepared baseline')
    require(shared_origin is None or (shared_baseline is not None and quality_backtracking),
            'Shared origin requires fixed-origin quality backtracking')
    require(not protect_silhouette or shared_origin is not None,'Silhouette protection requires a shared origin')
    require(not prepare_support or baseline_only or origin_only,'Support preparation requires a common package')
    require(not protect_support or (shared_origin is not None and protect_silhouette),
            'Support protection requires shared origin and silhouette protection')
    rows,trials,sidecars = [],[],{}
    baseline_endpoints,selection = [],None
    support_target = None
    saved_first_restored = False
    d0 = model.coefficients[:,:3].detach().clone()
    b0 = model.coefficients[:,3:].detach().clone()
    selected_terminal = packet['trial05_terminal'] if selected_terminal is None else selected_terminal
    brake = selected_terminal.detach().clone()
    target_x = packet['positions'][-1].detach().clone()
    dx = model.spec.prm.dx
    base_C = None
    arrived = packet['start_arrived'] & ~packet['pins']
    require(bool(arrived.any()),'No start-arrived-free cohort')
    require(not baseline_only or (shared_baseline is None and 'evaluate_merit' in packet),
            'Shared baseline construction requires the original-merit evaluator')
    context = repair_context(model,packet,source,target) if baseline_only or shared_baseline is not None else None
    if shared_baseline is not None:
        require(context==shared_baseline.context,'Shared baseline context changed')
        require(torch.equal(model.coefficients,shared_baseline.tensor('coefficients',d0.device)),
                'Original coefficients changed between paired arms')
        shared = shared_baseline.decode()
        rows = shared['rows']
        base_C = shared_baseline.tensor('C',d0.device)
        shared_digest = shared_baseline.digest()
    if shared_origin is not None:
        require(shared_origin.context==context,'Shared origin context changed')
        origin = shared_origin.decode()
        require(origin['baseline_sha256']==shared_digest,'Origin baseline binding changed')
        require(torch.equal(brake,shared_origin.tensor('terminal',d0.device)) and
                torch.equal(d0,shared_origin.tensor('displacement',d0.device)),
                'Origin coefficients changed')
        origin_digest = shared_origin.digest()
        if 'support_metadata' in origin:
            selection = dict(origin['support_metadata'])
            selection.update({k:shared_origin.tensor('support_'+k,d0.device) for k in origin['support_array_keys']})
            require(selection['status'] in ('ready','no_intervention'),'Common support experiment was aborted')
            support_target = torch.as_tensor(to_array(target),device=d0.device).clone()
    require(not protect_support or selection is not None,'Support witnesses missing')
    require(not save_search_endpoints or selection is not None,'Endpoint witness saving requires support selection')
    data_keys = ('volume','render','silhouette') if protect_silhouette else ('volume','render')
    if protect_support: data_keys += tuple('support_'+str(i) for i in range(selection['count']))

    def scalar(record,key):
        return (record['support']['values'][int(key.split('_')[1])] if key.startswith('support_')
                else record['data'][key])

    def component_check(record):
        if shared_origin is not None:
            record['silhouette_ceiling_passed'] = bool(record['valid'] and
                record['data']['silhouette']<=shared['silhouette_ceiling'])
        if selection is not None:
            record['support_constraints_passed'] = bool(record['valid'] and record['support']['passed'])
        return ((not protect_silhouette or record.get('silhouette_ceiling_passed',True)) and
                (not protect_support or record['support_constraints_passed']))

    def running(values):
        return geometric_running(values['positions'],packet['x0'],packet['dt'],arrived)

    def error(values):
        return (values['x']-target_x).square().sum(-1).mean()/(dx*dx)

    def save_state(values,label,d,b):
        path = out/(label+'_'+str(len(sidecars))+'.npz')
        np.savez_compressed(path,displacement=to_host(d),terminal=to_host(b),
            positions=to_host(values['positions']),V=to_host(values['V']),F=to_host(values['F']),C=to_host(values['C']))
        sidecars[path.name] = sha(path)

    def measure(values,label,d,b,save,prepared=None):
        record = dict(label=label,valid=values['valid'],pins_exact=values['pins_exact'],
            min_det=finite_scalar(values['min_det']),endpoint_mse=finite_scalar(error(values)),
            running=finite_scalar(running(values)),body_energy=finite_scalar(values['body_energy']),
            coefficient_delta_rms=float((d-d0).square().sum(-1).mean().sqrt()),
            terminal_coefficients_exact=torch.equal(b,brake) if label!='repeat' else torch.equal(b,b0))
        if values['valid']:
            data = reference.terms(values['x']) if prepared is None else prepared
            record['data'] = {k:float(v) for k,v in data.items() if not k.startswith('support_')}
            record['data']['weighted_render'] = packet['lambda_render']*record['data']['render']
            if 'evaluate_merit' in packet:
                merit = packet['evaluate_merit'](values)
                require(merit['merit']==merit['physical']+merit['lambda_render']*merit['render'],
                        'Original scalar recombination changed')
                eps = 32*torch.finfo(d0.dtype).eps
                for key in ('volume','render'):
                    require(abs(merit[key]-record['data'][key])<=eps*max(abs(record['data'][key]),1e-12),
                            'Original/prepared data disagree: '+key)
                record['original_merit'] = merit
                record['accepted_history_merit_delta'] = merit['merit']-packet['history']['loss']
                if shared_baseline is not None:
                    record['original_merit_nonincrease'] = merit['merit']<=shared['merit_ceiling']
            synthetic = dict(packet,positions=values['positions'],V=values['V'],F=values['F'])
            summary = summarize(type('One',(),{'packets':{8:synthetic}})(),source,target)
            record.update(motion=summary['rows'][0]['motion'],geometry=summary['rows'][0]['geometry'],
                source_upper_ids_sha256=summary['source_upper_ids_sha256'],
                F_change_rms=float((values['F']-packet['F'].reshape_as(values['F'])).square().mean().sqrt()),
                C_change_rms=float((values['C']-base_C).square().mean().sqrt()))
            if selection is not None:
                record['support'] = support_report(values['x'],support_target,selection)
        component_check(record)
        if save: save_state(values,label,d,b)
        return record

    def invalid_arm(values,label):
        rows.append(measure(values,label,d0,brake,True))
        assess_candidates(rows,d0.dtype)
        require(shared_baseline.digest()==shared_digest and repair_context(model,packet,source,target)==context,
                'Shared baseline/context changed during invalid arm')
        return dict(rows=rows,trials=trials,accepted_running_updates=0,sidecars=sidecars,
                    invalid_selected_terminal=True,invalid_phase=label,
                    data_ceilings=shared['ceilings'],lambda_render=packet['lambda_render'],
                    running_replay_noise=None,correction_rounds=correction_rounds,
                    quality_backtracking=quality_backtracking,terminal_label=terminal_label,
                    shared_baseline_sha256=shared_digest,original_merit_ceiling=shared['merit_ceiling'],
                    fixed_candidate_replays=[],repeated_feasible=False)

    with torch.no_grad():
        for i in range(3 if shared_baseline is None else 0):
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
            if 'original_merit' in record:
                expected = packet['history']['loss']
                ratio = abs(record['original_merit']['merit']-expected)/(eps*max(abs(expected),1e-12))
                accepted_closure['original_merit'] = ratio
                require(ratio<=1.,'Original merit baseline does not close accepted scalar')
            record['accepted_closure'] = accepted_closure
            require(record['valid'],'Invalid loaded baseline')
            rows.append(record)
            if prepare_support: baseline_endpoints.append(values['x'].detach().clone())
        if baseline_only:
            require(repair_context(model,packet,source,target)==context,'Context changed while preparing baseline')
            return SharedRepairBaseline.capture(rows,base_C,model.coefficients,sidecars,context,
                torch.stack(baseline_endpoints) if prepare_support else None)
        if shared_origin is not None:
            brake_errors = origin['running_repeats']
            rows.append(origin['terminal_record'])
            component_check(rows[-1])
        else:
            brake_errors = []
            for i in range(3):
                values = model.evaluate(brake,d0)
                if not values['valid'] and shared_baseline is not None and not origin_only:
                    return invalid_arm(values,terminal_label+'_invalid_repeat'+str(i))
                require(values['valid'],'Invalid selected terminal brake')
                brake_errors.append(float(running(values)))
            rows.append(measure(values,terminal_label,d0,brake,True))
    measured_noise = max(brake_errors)-min(brake_errors)
    ceilings = (shared['ceilings'] if shared_baseline is not None else
                {k:max(r['data'][k] for r in rows[:3]) for k in ('volume','render')})
    affine_ceilings = dict(ceilings)
    if protect_silhouette or origin_only: affine_ceilings['silhouette'] = shared['silhouette_ceiling']
    if protect_support:
        affine_ceilings.update({k:0. for k in data_keys if k.startswith('support_')})
    displacement = d0.clone()
    trust = packet['history']['body_update_modes_rms'][0]
    require(trust>0,'No original displacement trust step')
    accepted = 0
    fixed_candidate_replays = []
    replay_checked = False
    for iteration in range(1 if quality_backtracking else 4):
        linear_keys = ('volume','render','silhouette') if origin_only else data_keys
        if shared_origin is not None:
            gradients = [shared_origin.tensor('gradient_'+k,d0.device) for k in ('running',*linear_keys)]
            all_origin = shared_origin.tensor('data',d0.device)
            origin_data = all_origin[:len(linear_keys)].clone()
            current = origin['running']
            require(trust==origin['trust'],'Shared origin trust changed')
        else:
            with torch.enable_grad():
                leaf = displacement.detach().clone().requires_grad_()
                values = model.evaluate(brake,leaf)
                if not values['valid'] and shared_baseline is not None and not origin_only:
                    return invalid_arm(values,terminal_label+'_invalid_linearization')
                require(values['valid'],'Invalid current running-repair state')
                objective = running(values)
                terms = reference.terms(values['x'])
                if prepare_support:
                    support_target = torch.as_tensor(to_array(target),device=d0.device).clone()
                    neighbors = KDTree(to_array(support_target)).query(to_array(support_target),k=2)[0][:,1]
                    # Match the raw metric: average the middle pair for even populations.
                    spacing = float(array_api.median(neighbors))
                    selection = select_support(shared_baseline.tensor('endpoints',d0.device),
                                               values['x'].detach(),support_target,2*spacing)
                    support_keys = tuple('support_'+str(i) for i in range(selection['count']))
                    if selection['status'] in ('ready','no_intervention'):
                        signed = support_values(values['x'],selection['targets'],selection['witnesses'],selection['radius'])
                        terms = dict(terms,**dict(zip(support_keys,signed.unbind())))
                        linear_keys += support_keys
                        affine_ceilings.update({k:0. for k in support_keys})
                    path = out/'support_selection.npz'
                    np.savez_compressed(path,radius=selection['radius'],
                        **{k:to_host(v) for k,v in selection.items() if torch.is_tensor(v)})
                    sidecars[path.name] = sha(path)
                targets = (objective,*(terms[k] for k in linear_keys))
                gradients = [torch.autograd.grad(v,leaf,retain_graph=i<len(targets)-1)[0].detach()
                             for i,v in enumerate(targets)]
                require(all(bool(torch.isfinite(g).all()) for g in gradients),'Nonfinite repair gradient')
                origin_data = torch.stack([terms[k].detach().double() for k in linear_keys])
                current = float(objective.detach())
            if origin_only:
                terminal_record = measure(values,terminal_label+'_origin',d0,brake,True,prepared=terms)
            del values,objective,terms,targets
        G = torch.stack(gradients[1:])
        bounds = torch.tensor([affine_ceilings[k]-float(origin_data[i]) for i,k in enumerate(linear_keys)],
                              device=d0.device,dtype=torch.float64)
        path = out/f'linearization{iteration+1}.npz'
        np.savez_compressed(path,displacement=to_host(displacement),terminal=to_host(brake),
                            running_gradient=to_host(gradients[0]),data_gradients=to_host(G),
                            bounds=to_host(bounds),origin_data=to_host(origin_data))
        sidecars[path.name] = sha(path)
        threshold = 10*max(measured_noise,32*torch.finfo(displacement.dtype).eps*current)
        if origin_only:
            require(shared_baseline.digest()==shared_digest and repair_context(model,packet,source,target)==context,
                    'Context/baseline changed while preparing common origin')
            metadata = dict(terminal_record=terminal_record,running_repeats=brake_errors,
                noise=measured_noise,running=current,threshold=threshold,trust=trust,keys=list(linear_keys),
                sidecars=sidecars,baseline_sha256=shared_digest)
            arrays = dict(displacement=d0,terminal=brake,data=origin_data,
                          **{'gradient_'+k:g for k,g in zip(('running',*linear_keys),gradients)})
            if selection is not None:
                metadata['support_metadata'] = {k:v for k,v in selection.items() if not torch.is_tensor(v)}
                metadata['support_array_keys'] = [k for k,v in selection.items() if torch.is_tensor(v)]
                arrays.update({'support_'+k:v for k,v in selection.items() if torch.is_tensor(v)})
            return SharedRepairOrigin.pack(metadata,arrays,context)
        if shared_origin is not None:
            require(threshold==origin['threshold'] and measured_noise==origin['noise'],'Shared running threshold changed')
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
                            if save_search_endpoints and record['valid']:
                                path = out/(record['label']+'_endpoint.npz')
                                np.savez_compressed(path,x=to_host(values['x']),displacement=to_host(candidate),
                                    terminal=to_host(brake),witness_positions=to_host(values['x'][selection['witnesses']]),
                                    support_values=np.asarray(record['support']['values'],dtype=np.float64))
                                sidecars[path.name] = sha(path)
                            data_pass = record['valid'] and all(record['data'][k]<=ceilings[k] for k in ceilings)
                            constraints_pass = data_pass and component_check(record)
                            if save_search_endpoints and constraints_pass and not saved_first_restored:
                                save_state(values,record['label']+'_first_restored',candidate,brake)
                                saved_first_restored = True
                            improvement = current-record['running'] if record['running'] is not None else None
                            take = bool(constraints_pass and improvement>threshold)
                            record.update(data_restored=bool(data_pass),running_improvement=improvement,
                                          data_constraints_restored=bool(constraints_pass),
                                          reduction_threshold=threshold,running_origin=current,running_update_accepted=take)
                            if shared_origin is not None:
                                assess_candidates(rows[:3]+[record],displacement.dtype)
                                record['running_decrease_resolved'] = improvement is not None and improvement>threshold
                                record['full_search_passed'] = (record['feasible'] and
                                    record['running_decrease_resolved'] and component_check(record))
                            if take and quality_backtracking:
                                assess_candidates(rows[:3]+[record],displacement.dtype)
                                record['data_running_passed'] = True
                                take = record['feasible']
                                record['running_update_accepted'] = take
                                record['quality_backtracking_rejected'] = not take
                            if record['valid']:
                                actual_data = torch.tensor([scalar(record,k) for k in data_keys],
                                                           device=G.device,dtype=torch.float64)
                                # Replace the estimate. All rounds retain the same origin and Jacobian.
                                remainder = observed_remainder(actual_data,origin_data,G,actual_step)
                                record['observed_model_remainder'] = to_host(remainder).tolist()
                                retry = not constraints_pass
                            if take:
                                displacement = candidate.detach().clone()
                                selected = deepcopy(record)
                                selected['label'] = f'repair{iteration+1}'
                                save_state(values,selected['label'],displacement,brake)
                                rows.append(selected)
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
                                        witness['full_search_passed'] = (witness['feasible'] and
                                            witness['running_decrease_resolved'] and component_check(witness))
                                        fixed_candidate_replays.append(witness)
                                    # Stop after the first all-gate candidate even if its repeats fail.
                                    record['fixed_candidate_replay_checked'] = True
                trials.append(record)
                print(json.dumps(record,allow_nan=False),flush=True)
                if found or not retry: break
            if found: break
        if not found or replay_checked: break
    assess_candidates(rows,displacement.dtype)
    require(torch.equal(brake,selected_terminal),'Terminal coefficient changed')
    if shared_baseline is not None:
        require(shared_baseline.digest()==shared_digest and repair_context(model,packet,source,target)==context,
                'Shared baseline/context changed during paired arm')
    if shared_origin is not None:
        require(shared_origin.digest()==origin_digest,'Shared origin changed during arm')
    return dict(rows=rows,trials=trials,accepted_running_updates=accepted,sidecars=sidecars,
                running_replay_noise=measured_noise,data_ceilings=ceilings,lambda_render=packet['lambda_render'],
                correction_rounds=correction_rounds,quality_backtracking=quality_backtracking,
                terminal_label=terminal_label,
                shared_baseline_sha256=None if shared_baseline is None else shared_digest,
                shared_origin_sha256=None if shared_origin is None else origin_digest,
                protect_silhouette=protect_silhouette,
                protect_support=protect_support,
                silhouette_ceiling=None if shared_baseline is None else shared['silhouette_ceiling'],
                original_merit_ceiling=None if shared_baseline is None else shared['merit_ceiling'],
                fixed_candidate_replays=fixed_candidate_replays,
                repeated_feasible=replay_checked and all(r['feasible'] and r['running_decrease_resolved'] and component_check(r)
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
