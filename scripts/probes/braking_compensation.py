"""Bounded displacement-mode compensation of a frozen terminal brake, no commit."""
import argparse
from datetime import datetime, timezone
import json
import math
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution,to_host
from physmorph.pipeline.frozen_body_window import FrozenBodyWindow,project_terminal
from physmorph.pipeline.prepared_reference import PreparedReference
from scripts.probes.inner_budget import summarize
from scripts.probes.reference_swap import require,sha
from scripts.probes.terminal_braking import assess_candidates,finite_scalar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    for path in (args.capture,args.snapshot,args.out):
        require(path.resolve().is_relative_to('/data/relcfd/chayo/physmorph_v2'),'Outside project data')
    args.out.mkdir(exist_ok=False)
    report_path,protocol_path = args.capture/'result.json',args.capture/'protocol.json'
    report_bytes,protocol_bytes = report_path.read_bytes(),protocol_path.read_bytes()
    report = json.loads(report_bytes);original = json.loads(protocol_bytes)
    require(sha256(protocol_bytes).hexdigest()==report['protocol_sha256'],'Capture protocol mismatch')
    json.dumps(report,allow_nan=False)
    require(original.get('extension',{}).get('version')==1 and
            original['extension'].get('witnesses')==['baseline','fresh_trial05'],'Wrong capture witness schema')
    closure = report.get('extension',{}).get('roundtrip_closure')
    require(isinstance(closure,list) and len(closure)==2,'Two capture witnesses required')
    required = ['positions','V','F','C']+['gradient_'+k for k in ('brake','volume','render')]+[
                'data_'+k for k in ('volume','silhouette','pbr','render')]
    require(all(all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) and 0<=row[k]<=1
                    for k in required) for row in closure),'Capture witness failed')
    archive = args.capture/'owned_window.npz'
    require(sha(archive)==report['extension']['archive_sha256']==report['sidecars'][archive.name], 'Archive mismatch')
    root = Path(__file__).resolve().parents[2]
    bound = {str(args.snapshot/k):v for k,v in {**original['code'],**original['helpers']}.items()}
    bound.update(original['inputs'])
    bound.update({str(report_path):sha256(report_bytes).hexdigest(),str(protocol_path):sha256(protocol_bytes).hexdigest(),str(archive):sha(archive)})
    require(all(sha(Path(k))==v for k,v in bound.items()),'Capture provenance mismatch')
    code = {str(p.relative_to(root)):sha(p) for p in sorted((root/'physmorph').rglob('*.py'))}
    require(code==original['code'],'Numerical source differs from capture')
    helpers = {str(p.relative_to(root)):sha(p) for p in (Path(__file__).resolve(),root/'scripts/probes/inner_budget.py',
               root/'scripts/probes/terminal_braking.py',root/'scripts/probes/reference_swap.py')}
    require(all(helpers[k]==original['helpers'][k] for k in helpers if k!=str(Path(__file__).resolve().relative_to(root))),
            'Imported measurement/feasibility helper differs from capture')
    protocol = dict(start_utc=datetime.now(timezone.utc).isoformat(),bound=bound,code=code,helpers=helpers,
                    updates=4,halvings=10,objective='Full same-ID endpoint squared error / dx^2',
                    scope='Displacement coefficients only; frozen trial05 terminal, stress/u and physical initial state; no commit')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows,trials,sidecars = [],[],{}
    with cuda_execution('cuda:0'):
        model,packet = FrozenBodyWindow.load(archive,'cuda:0')
        packet['reference'] = reference = PreparedReference(**packet['reference'])
        source,target = packet['source'],packet['target']
        d0 = model.coefficients[:,:3].detach().clone()
        b0 = model.coefficients[:,3:].detach().clone()
        brake = packet['trial05_terminal'].detach().clone()
        target_x = packet['positions'][-1].detach().clone()
        dx = model.spec.prm.dx
        base_C = None

        def error(values):
            return (values['x']-target_x).square().sum(-1).mean()/(dx*dx)

        def measure(values,label,d,b,save):
            record = dict(label=label,valid=values['valid'],pins_exact=values['pins_exact'],
                min_det=finite_scalar(values['min_det']),endpoint_mse=finite_scalar(error(values)),
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
                path = args.out/(label+'_'+str(len(sidecars))+'.npz')
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
                brake_errors.append(float(error(values)))
            rows.append(measure(values,'terminal05',d0,brake,True))
        measured_noise = max(brake_errors)-min(brake_errors)
        current = brake_errors[-1]
        displacement = d0.clone()
        trust = packet['history']['body_update_modes_rms'][0]
        require(trust>0,'No original displacement trust step')
        accepted = 0
        for iteration in range(4):
            with torch.enable_grad():
                leaf = displacement.detach().clone().requires_grad_()
                values = model.evaluate(brake,leaf)
                objective = error(values)
                gradient, = torch.autograd.grad(objective,leaf)
                require(bool(torch.isfinite(gradient).all()),'Nonfinite endpoint derivative')
                norm = gradient.square().sum(-1).mean().sqrt()
                if float(norm)==0: break
                direction = -gradient/norm
            del values,objective
            found = False
            threshold = 10*max(measured_noise,32*torch.finfo(displacement.dtype).eps*current)
            for halving in range(11):
                candidate = project_terminal(displacement+trust*(.5**halving)*direction,brake)
                with torch.no_grad():
                    values = model.evaluate(brake,candidate)
                    record = measure(values,f'update{iteration+1}_half{halving}',candidate,brake,False)
                    improvement = current-record['endpoint_mse'] if record['endpoint_mse'] is not None else None
                    take = bool(record['valid'] and improvement>threshold)
                    record.update(iteration=iteration+1,halvings=halving,improvement=improvement,
                                  reduction_threshold=threshold,endpoint_update_accepted=take)
                    trials.append(record)
                    print(json.dumps(record,allow_nan=False),flush=True)
                    if take:
                        displacement = candidate.detach().clone();current=record['endpoint_mse']
                        rows.append(measure(values,f'compensation{iteration+1}',displacement,brake,True))
                        accepted += 1;found = True
                        break
            if not found: break
        model.close()
        assess_candidates(rows,displacement.dtype)
        require(torch.equal(brake,packet['trial05_terminal']),'Terminal coefficient changed')
    require(all(sha(Path(k))==v for k,v in bound.items()),'Capture evidence changed')
    require(all(sha(root/k)==v for k,v in {**code,**helpers}.items()),'Source changed')
    output = dict(protocol_sha256=sha(args.out/'protocol.json'),rows=rows,trials=trials,
        accepted_endpoint_updates=accepted,sidecars=sidecars,endpoint_replay_noise=measured_noise,
        lambda_render=packet['lambda_render'],original_discretization=original['mpm'],N=len(source),
        T=model.spec.T,loss_res=original['config']['loss_res'],scope=protocol['scope'],
        rendering_role='Prepared CIC/PBR guides the original accepted solve and screens final feasibility. Compensation directions minimize endpoint error only; candidate data/weighted-render changes are reported, not causal motion shares or 4K appearance quality.')
    (args.out/'result.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(dict(out=str(args.out),feasible=[r['label'] for r in rows if r.get('feasible')])),flush=True)


if __name__=='__main__': main()
