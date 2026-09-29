"""Noncommitting terminal-body-only feasibility at W20 inner iteration eight."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as host_np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution, to_host
from physmorph.pipeline import PipelineConfig, runner
from physmorph.pipeline.braking_direction import data_tangent_direction
from physmorph.pipeline.frozen_body_window import project_terminal
from physmorph.mpm.state import MPMParams
from scripts.probes.inner_budget import summarize
from scripts.probes.reference_swap import require, sha


def finite_scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def assess_candidates(rows,dtype):
    # Observed repeat range is numerical evidence, not a statistical confidence bound.
    repeat = rows[:3]
    require(all(row['valid'] for row in repeat),'Invalid repeat')
    for row in rows[3:]:
        if not row['valid']:
            row['feasible'] = False
            continue
        checks = {}
        for key in ('volume','render'):
            checks['data_'+key] = row['data'][key] <= max(r['data'][key] for r in repeat)
        for key in ('sil_iou','upper_target_near_frac','target_near_frac','tip_n','fixed_source_upper_density'):
            checks[key] = row['geometry'][key] >= min(r['geometry'][key] for r in repeat)
        checks['chamfer'] = row['geometry']['chamfer'] <= max(r['geometry']['chamfer'] for r in repeat)
        for cohort in ('start_free','start_arrived_free'):
            for key in ('net_rms_sp','step_rms_sp','path_mean_sp'):
                checks[cohort+'_'+key] = row['motion'][cohort][key] <= max(r['motion'][cohort][key] for r in repeat)
        for key in ('physical_terminal_rms_wu_s','geometric_terminal_rms_wu_s'):
            checks[key] = row['motion']['start_arrived_free'][key] < min(r['motion']['start_arrived_free'][key] for r in repeat)
        resolved = {}
        for key in ('physical_terminal_rms_wu_s','geometric_terminal_rms_wu_s'):
            speeds = [r['motion']['start_arrived_free'][key] for r in repeat]
            threshold = 10*max(max(speeds)-min(speeds),32*torch.finfo(dtype).eps*max(speeds))
            improvement = min(speeds)-row['motion']['start_arrived_free'][key]
            resolved[key] = dict(threshold=threshold,improvement=improvement,passed=improvement>threshold)
        row.update(checks=checks,nominal_feasible=all(checks.values()),resolved_braking=resolved,
                   feasible=all(checks.values()) and all(v['passed'] for v in resolved.values()))


class Capture:
    def __init__(self, source, target, out):
        self.source,self.target,self.out = source,target,out
        self.rows = []
        self.sidecars = {}
        self.direction = None

    def observe(self,index,packet):
        require(index==19 and packet['iteration']==8,'Wrong checkpoint')
        model = packet['rollout']
        terminal = packet['controls']['body'][:,3:].detach().clone().requires_grad_()
        arrived = packet['start_arrived'] & ~packet['pins']
        require(bool(arrived.any()),'No start-arrived-free IDs')
        with torch.enable_grad():
            values = model.evaluate(terminal)
            require(values['valid'],'Invalid baseline replay')
            geometric = (values['positions'][-1]-values['positions'][-2])/packet['dt']
            objective = .5*(values['v'][arrived].square().sum(-1).mean()
                            +geometric[arrived].square().sum(-1).mean())
            terms = packet['reference'].terms(values['x'])
            closure = {}
            eps = 32*torch.finfo(terminal.dtype).eps
            for key,actual,expected,unit in (
                    ('positions',values['positions'],packet['positions'],model.spec.prm.dx),
                    ('velocities',values['V'],packet['V'],model.spec.prm.dx/(model.spec.T*packet['dt'])),
                    ('F',values['F'],packet['F'].reshape_as(values['F']),1.)):
                tolerance = eps*(unit+expected.abs())
                error = (actual.detach()-expected).abs()
                closure[key] = dict(max_abs=float(error.max()),max_tolerance_ratio=float((error/tolerance).max()))
                require(bool((error<=tolerance).all()),'Private replay differs from accepted '+key)
            for key,expected in (('volume',packet['history']['d_vol']),
                                 ('render',packet['history']['d_render']+packet['reference'].pbr_weight*(packet['history']['d_pbr'] or 0.))):
                error = abs(float(terms[key].detach())-expected)
                tolerance = eps*max(abs(expected),1e-12)
                closure[key] = dict(absolute_difference=error,tolerance=tolerance)
                require(error<=tolerance,'Private data closure failed: '+key)
            gh, = torch.autograd.grad(objective,terminal,retain_graph=True)
            gv, = torch.autograd.grad(terms['volume'],terminal,retain_graph=True)
            gr, = torch.autograd.grad(terms['render'],terminal)
        direction = data_tangent_direction(-gh,gv,gr)
        rms = direction.square().sum(-1).mean().sqrt()
        require(float(rms)>0,'No nonzero data-tangent descent direction')
        direction = direction/rms
        step = packet['history']['body_update_modes_rms'][1]
        require(step>0,'Zero original terminal update; no preregistered trust scale')
        self.direction = dict(terminal_step_rms=step,
            private_accepted_closure=closure,
            norms={key:float(value.norm()) for key,value in dict(brake=gh,volume=gv,render=gr).items()},
            directional={key:float((value*direction).sum()) for key,value in dict(brake=gh,volume=gv,render=gr).items()},
            projected_rms_before_normalization=float(rms),
            bound_saturated_fraction=float((packet['controls']['body'].norm(dim=1)>=.999).float().mean()))
        del values,terms,objective,geometric,gh,gv,gr
        base = terminal.detach()
        for label,scale in [('repeat0',0.),('repeat1',0.),('repeat2',0.),
                            ('trial4',4.),('trial2',2.),('trial1',1.),('trial05',.5),('trial025',.25)]:
            candidate = base if not scale else project_terminal(base+scale*step*direction,model.coefficients[:,:3])
            with torch.no_grad():
                values = model.evaluate(candidate)
                record = dict(label=label,scale=scale,valid=values['valid'],pins_exact=values['pins_exact'],
                              min_det=finite_scalar(values['min_det']),body_energy=finite_scalar(values['body_energy']),
                              rejection_reason=None if values['valid'] else 'Raw finite/bounds/orientation/pin gate failed',
                              coefficient_delta_rms=float((candidate-base).square().sum(-1).mean().sqrt()))
                if values['valid']:
                    if 'evaluate_merit' in packet:
                        record['original_merit'] = packet['evaluate_merit'](values)
                    terms = packet['reference'].terms(values['x'])
                    record['data'] = {key:float(v) for key,v in terms.items()}
                    record['data']['weighted_render'] = packet['lambda_render']*record['data']['render']
                    synthetic = dict(packet,positions=values['positions'],V=values['V'],F=values['F'])
                    summary = summarize(type('One',(),{'packets':{8:synthetic}})(),self.source,self.target)
                    record.update(motion=summary['rows'][0]['motion'],geometry=summary['rows'][0]['geometry'])
                    record['source_upper_ids_sha256'] = summary['source_upper_ids_sha256']
                    record['replay_endpoint_rms'] = float((values['x']-packet['positions'][-1]).square().sum(-1).mean().sqrt())
                path = self.out/(label+'.npz')
                host_np.savez_compressed(path,terminal=to_host(candidate),positions=to_host(values['positions']),
                                        V=to_host(values['V']),F=to_host(values['F']))
                self.sidecars[path.name] = sha(path)
                self.rows.append(record)
                print(json.dumps(record,allow_nan=False),flush=True)
            del values
        assess_candidates(self.rows,terminal.dtype)
        self.lambda_render = packet['lambda_render']
        self.original_history = deepcopy(packet['history'])

    def wrap(self,original):
        def wrapped(*args,**kwargs):
            if kwargs['win_index']==19:
                return original(*args,on_checkpoint=self.observe,checkpoint_iterations=(8,),checkpoint_rollout=True,
                                checkpoint_merit=getattr(self,'record_candidate_merit',False),**kwargs)
            return original(*args,**kwargs)
        return wrapped


def main(capture_type=Capture, extra_protocol=None, extra_helpers=()):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    require(args.out.resolve().is_relative_to(args.root.resolve()),'Output outside project data')
    args.out.mkdir(exist_ok=False)
    metadata_path = args.root/'work/p303/raw24a.json'
    source_path = args.root/'repro/current_pair/source_render_full_dt_iso_nn.npz'
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    cfg = PipelineConfig(**metadata['config']);cfg.stop_after_windows = 20
    require(cfg.iters==8 and not cfg.commit_pic and not cfg.shift_sub and cfg.compute_backend=='cuda','Wrong recipe')
    inputs = {str(p):sha(p) for p in (metadata_path,source_path,Path(cfg.target_reference))}
    from hashlib import sha256
    require(inputs[str(metadata_path)]==sha256(metadata_bytes).hexdigest(),'Parsed metadata bytes changed')
    root = Path(__file__).resolve().parents[2]
    code = {str(p.relative_to(root)):sha(p) for p in sorted((root/'physmorph').rglob('*.py'))}
    helpers = {str(p.relative_to(root)):sha(p) for p in (Path(__file__).resolve(),root/'scripts/probes/inner_budget.py',root/'scripts/probes/reference_swap.py',*extra_helpers)}
    protocol = dict(start_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,code=code,helpers=helpers,
        config=metadata['config'],mpm=metadata['mpm'],window=20,checkpoint=8,
        overrides=dict(stop_after_windows=20),scales=[4,2,1,.5,.25],repeats=3,
        direction='negative equal stored/geometric terminal mean square, projected into two data halfspaces',
        scope='Noncommitting positive feasibility in terminal body subspace; frozen displacement/stress/u/ref/pins; no full rest claim')
    if extra_protocol is not None:
        protocol['extension'] = extra_protocol
        protocol['scope'] = extra_protocol.get('scope',protocol['scope'])
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with host_np.load(source_path,allow_pickle=False) as archive: source,target=archive['src'],archive['tgt']
    capture = capture_type(source,target,args.out)
    with patch.object(runner,'optimize_window',capture.wrap(runner.optimize_window)):
        result = runner.run_pipeline(source,target,MPMParams(**metadata['mpm']),cfg)
    require(not any(result['guards'].values()),'Outer state guard fired')
    require(len(capture.rows)==8,'Preregistered checkpoint not reached')
    require(all(sha(Path(k))==v for k,v in inputs.items()),'Input changed')
    require(all(sha(root/k)==v for k,v in {**code,**helpers}.items()),'Source changed')
    from physmorph.pipeline.render_reporting import write_render_report
    influence = write_render_report(args.out/'run',result['history'],metadata['config'],metadata['mpm'],len(source))
    report = dict(protocol_sha256=sha(args.out/'protocol.json'),rows=capture.rows,direction=capture.direction,
        sidecars=capture.sidecars,lambda_render=capture.lambda_render,original_history=capture.original_history,
        history=result['history'],guards=result['guards'],render_influence=influence,N=len(source),T=cfg.T,
        mpm=metadata['mpm'],loss_res=cfg.loss_res,scope=protocol['scope'],extension=getattr(capture,'extra',None))
    (args.out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(out=str(args.out),feasible=[r['label'] for r in capture.rows if r.get('feasible')])),flush=True)


if __name__=='__main__': main()
