"""GPU-only numerical reference-swap audit on preregistered raw windows 19/20.

Fresh unchanged-policy realization, no replay identity claim. These are endpoint
position sensitivities at fixed F/V/controls, not feasible alternate trajectories.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as host_np
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution, to_array, to_host
from physmorph.pipeline import PipelineConfig, runner
from physmorph.mpm.state import MPMParams


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def directional(gradient, delta, masks, normal, surface):
    result = {}
    dn = (delta*normal).sum(-1, keepdim=True)*normal
    for key, mask in masks.items():
        use = mask & surface
        result[key] = dict(particles=int(mask.sum()), gradient_norm=float(gradient[mask].norm()),
            dot_displacement=float((gradient[mask]*delta[mask]).double().sum()),
            surface_particles=int(use.sum()),
            surface_normal_dot=float((gradient[use]*dn[use]).double().sum()),
            surface_tangent_dot=float((gradient[use]*(delta-dn)[use]).double().sum()),
            nonsurface_dot=float((gradient[mask & ~surface]*delta[mask & ~surface]).double().sum()))
    return result


def analyze(previous, current, common):
    require(torch.equal(previous['xT'], current['x0']), 'Observed windows do not share exact endpoint')
    old, new = previous['reference'], current['reference']
    require(old.pbr_weight == new.pbr_weight and (old.shade is None) == (new.shade is None),
            'PBR weighting or observation presence changed')
    if old.shade is not None:
        for key in ('views', 'res', 'extent', 'k', 'ambient', 'dx', 'dims', 'blur_cells'):
            require(old.shade[key] == new.shade[key], 'PBR operator changed: '+key)
        require(torch.equal(old.shade['grid_min'],new.shade['grid_min']), 'PBR grid origin changed')
    for key in ('dx', 'dims', 'm_ref', 'n_support', 'form'):
        require(old.density[key] == new.density[key], 'Density calibration changed: '+key)
    for key in ('m', 'grid_min'):
        require(torch.equal(old.density[key], new.density[key]), 'Density layout changed: '+key)
    for key in ('views', 'res', 'extent', 'k', 'w_hole', 'w_spray'):
        require(old.render[key] == new.render[key], 'Render calibration changed: '+key)
    x0, xT = current['x0'], current['xT']
    delta = xT-x0
    previous_arrived = (previous['xT']-previous['plan']).norm(dim=-1) <= previous['arrival_radius']
    current_arrived = (x0-current['plan']).norm(dim=-1) <= current['arrival_radius']
    free = ~current['pins']
    masks = dict(start_free=free, previously_and_currently_arrived_free=free & previous_arrived & current_arrived)
    normal = current['normal']
    surface = current['surface'] & (normal.norm(dim=-1) > 1e-6)
    normal = normal/normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    weight = current['lambda_render']
    require(weight > 0 and host_np.isfinite(weight), 'Positive current lambda required')
    output = dict(lambda_render=weight, previous_lambda=previous['lambda_render'],
                  old_kind=old.kind, new_kind=new.kind, samples={}, common={}, kinetic={}, motion={})
    output['pbr_grid'] = {key: None if ref.shade is None else
        {k: ref.shade[k] for k in ('dx', 'dims', 'blur_cells')} for key, ref in [('old',old), ('new',new)]}
    output['arrival'] = dict(previous_end=int(previous_arrived.sum()), current_start=int(current_arrived.sum()),
        start_predicate_roundoff_disagreements=int((current_arrived != current['start_arrived'].bool()).sum()),
        radius_previous=previous['arrival_radius'], radius_current=current['arrival_radius'],
        masks_frozen_before_current_endpoint=True)
    gradients = {}
    # The midpoint is the straight segment midpoint, NOT raw physical step10.
    for label, alpha in [('start',0.), ('segment_midpoint',.5), ('end',1.)]:
        value = x0+alpha*delta
        common_result = common(value)
        require(host_np.isfinite(common_result['value']) and bool(torch.isfinite(common_result['gradient']).all()),
                'Nonfinite common diagnostic')
        output['common'][label] = dict(value=common_result['value'],
            sensitivity=directional(common_result['gradient'], delta, masks, normal, surface))
        gradients['common_'+label] = common_result['gradient']
        samples = {}
        with torch.enable_grad():
            for name, ref in [('old',old), ('new',new)]:
                x = value.detach().clone().requires_grad_(True)
                terms = ref.terms(x)
                losses = dict(volume=terms['volume'], silhouette=terms['silhouette'],
                    weighted_pbr=ref.pbr_weight*terms['pbr'], weighted_render=weight*terms['render'],
                    data=terms['volume']+weight*terms['render'])
                samples[name] = {}
                for channel, loss in losses.items():
                    grad = torch.autograd.grad(loss, x, retain_graph=True)[0].detach()
                    require(bool(torch.isfinite(loss).all() & torch.isfinite(grad).all()),
                            'Nonfinite sampled reference value/gradient')
                    samples[name][channel] = dict(value=float(loss.detach()),
                        sensitivity=directional(grad, delta, masks, normal, surface))
                    if label == 'segment_midpoint' and channel == 'data':
                        gradients[name+'_data_midpoint'] = grad
                if name == 'new' and label == 'end':
                    for key in ('volume', 'render'):
                        expected = current['expected'][key]
                        actual = float(terms[key].detach())
                        require(abs(actual-expected) <= 2e-7+2e-5*abs(expected), 'Live/snapshot loss mismatch: '+key)
                        output.setdefault('live_parity', {})[key] = dict(expected=expected, observed=actual)
                    expected_merit = current['expected']['merit']
                    reconstructed = common_result['value']+float(terms['volume'].detach()+weight*terms['render'].detach())
                    tolerance = max(current['expected']['merit_tolerance'], 2e-7+2e-5*abs(expected_merit))
                    require(abs(expected_merit-reconstructed) <= tolerance, 'Live scalar merit does not close')
                    output['live_parity']['merit'] = dict(expected=expected_merit, observed=reconstructed,
                                                        absolute_tolerance=tolerance)
        output['samples'][label] = samples
    output['finite_changes'] = {name: {channel: output['samples']['end'][name][channel]['value']
        - output['samples']['start'][name][channel]['value'] for channel in output['samples']['end'][name]}
        for name in ('old','new')}
    output['finite_changes']['common'] = output['common']['end']['value']-output['common']['start']['value']
    output['fd'] = {}
    with torch.no_grad():
        for name, ref in [('old',old), ('new',new), ('common',None)]:
            grad = gradients['common_segment_midpoint' if name == 'common' else name+'_data_midpoint']
            ad = float((grad*delta).double().sum())
            rows = []
            for epsilon in (.1,.03,.01):
                vals = []
                for sign in (-1,1):
                    x = x0+(.5+sign*epsilon)*delta
                    if ref is None:
                        vals.append(common(x)['value'])
                    else:
                        t = ref.terms(x)
                        vals.append(float(t['volume']+weight*t['render']))
                fd = (vals[1]-vals[0])/(2*epsilon)
                require(host_np.isfinite(ad) and host_np.isfinite(fd), 'Nonfinite finite-difference diagnostic')
                rows.append(dict(epsilon=epsilon, ad=ad, fd=fd,
                                 relative_error=abs(ad-fd)/max(abs(ad),abs(fd),1e-12)))
            output['fd'][name] = rows
    V = current['V']
    pos = torch.cat((x0[None], current['positions']))
    geo = (pos[1:]-pos[:-1])/current['dt']
    for name, mask in masks.items():
        n = int(mask.sum())
        if not n:
            output['kinetic'][name] = dict(particles=0)
            continue
        physical = V[:,mask]
        geometric = geo[:,mask]
        output['kinetic'][name] = dict(particles=n,
            physical_terminal=float(physical[-1].square().sum(-1).mean()),
            physical_running=float(physical.square().sum(-1).mean()),
            physical_variance=float((physical-physical.mean(0)).square().sum(-1).mean()),
            geometric_terminal=float(geometric[-1].square().sum(-1).mean()),
            geometric_running=float(geometric.square().sum(-1).mean()),
            geometric_variance=float((geometric-geometric.mean(0)).square().sum(-1).mean()),
            net_rms_wu=float(delta[mask].square().sum(-1).mean().sqrt()))
        row = output['kinetic'][name]
        row['weighted_physical_contribution'] = {key: current['unit_weight']*current['weights'][key]
            *row['physical_'+key]*n/len(x0) for key in ('terminal','running','variance')}
        from physmorph.pipeline.motion_accounting import NAMES
        motion = current['motion']
        output['motion'][name] = dict(component_window_rms_wu={key:float(motion['sums'][i,mask].square().sum(-1).mean().sqrt())
            for i,key in enumerate(NAMES)}, actual_step_rms_wu=float(motion['squares'][:,-1,mask].mean().sqrt()))
    output['common_term_scope'] = 'Current physical/cleanup terms with F,V,controls,NN and gates fixed; constant kinetic/control/constitutive costs included in values'
    output['sensitivity_scope'] = 'Gradient dot observed endpoint displacement; negative locally rewards this direction; not energy/work, feasible motion or causal fraction'
    output['cohort_scope'] = 'Start-free and previous-end/current-start arrived intersection; no filtering using current endpoint; surface split uses current start layer normals only'
    return output, gradients


class Capture:
    def __init__(self):
        self.packets = {}
        self.committed = set()
        self.report = None
        self.expired = None
        self.gradients = {}

    def observe(self, index, packet, common):
        require(index in (18,19), 'Unexpected preregistered attempt')
        self.packets[index] = packet
        self.expired = common
        if index == 19:
            require(18 in self.committed, 'Previous preregistered window was not accepted')
            self.report, self.gradients = analyze(self.packets[18], packet, common)

    def wrap(self, original):
        def wrapped(*args, **kwargs):
            index = kwargs['win_index']
            if index in (18,19):
                require('on_reference' not in kwargs, 'Observer already installed')
                result = original(*args, on_reference=self.observe, **kwargs)
                if index in self.packets:
                    try:
                        self.expired(self.packets[index]['xT'])
                    except RuntimeError as exc:
                        require('expired' in str(exc), 'Unexpected evaluator lifetime error')
                    else:
                        raise ValueError('Reference evaluator leaked beyond callback')
                return result
            return original(*args, **kwargs)
        return wrapped

    def commit(self, index, x, F, v, row):
        if index not in (18,19):
            return
        require(row.get('frame_end') and not row.get('null_commit'), 'Audited window rejected')
        packet = self.packets[index]
        current = torch.as_tensor(to_array(x), device=packet['xT'].device)
        require(torch.equal(packet['xT'],current), 'Audit endpoint differs from outer commit')
        self.committed.add(index)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    require(args.out.resolve().is_relative_to(args.root.resolve()), 'All output must remain in project data')
    args.out.mkdir(exist_ok=False)
    metadata_path = args.root/'work/p303/raw24a.json'
    source_path = args.root/'repro/current_pair/source_render_full_dt_iso_nn.npz'
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    cfg = PipelineConfig(**metadata['config'])
    require(not cfg.commit_pic and not cfg.commit_pic_objective and not cfg.shift_sub
            and cfg.stop_after_windows == 24 and cfg.iters == 8 and cfg.motion_accounting,
            'Exact raw P303 recipe required')
    require(not cfg.geometric_rest and not cfg.geometric_variance and cfg.compute_backend == 'cuda',
            'Unexpected diagnostic treatment')
    inputs = {str(path):sha(path) for path in (metadata_path,source_path,Path(cfg.target_reference))}
    require(inputs[str(metadata_path)] == hashlib.sha256(metadata_bytes).hexdigest(), 'Metadata changed while parsing')
    root = Path(__file__).resolve().parents[2]
    code = {str(f.relative_to(root)):sha(f) for f in sorted((root/'physmorph').rglob('*.py'))}
    protocol = dict(start_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs, code=code,
        probe_sha256=sha(__file__), config=metadata['config'], mpm=metadata['mpm'],
        attempts=[19,20], label='Fresh read-only observation; old/current reference on SAME path; no policy change')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with host_np.load(source_path, allow_pickle=False) as data:
        src,tgt = data['src'],data['tgt']
    capture = Capture()
    with patch.object(runner,'optimize_window',capture.wrap(runner.optimize_window)):
        result = runner.run_pipeline(src,tgt,MPMParams(**metadata['mpm']),cfg,on_commit=capture.commit)
    require(not any(result['guards'].values()), 'Physical guards fired')
    require(capture.committed == {18,19} and capture.report is not None, 'Preregistered audit incomplete')
    require(inputs == {path:sha(path) for path in inputs}, 'Inputs changed while executing')
    require(code == {str(f.relative_to(root)):sha(f) for f in sorted((root/'physmorph').rglob('*.py'))}, 'Code changed')
    sidecars = {}
    with cuda_execution('cuda'):
        for index,packet in capture.packets.items():
            arrays = {key:to_host(value) for key,value in packet.items() if torch.is_tensor(value)}
            ref = packet['reference']
            arrays.update({f'density_{key}':to_host(value) for key,value in ref.density.items() if torch.is_tensor(value)})
            arrays['alphas'] = to_host(torch.stack(ref.render['target_alphas']))
            if ref.shade is not None:
                arrays['shades'] = to_host(torch.stack(ref.shade['shade_tgts']))
                arrays['pbr_grid_min'] = to_host(ref.shade['grid_min'])
            path = args.out/f'attempt_{index+1}.npz'
            host_np.savez_compressed(path,**arrays)
            sidecars[path.name] = sha(path)
        path = args.out/'common_and_data_gradients.npz'
        host_np.savez_compressed(path,**{key:to_host(value) for key,value in capture.gradients.items()})
        sidecars[path.name] = sha(path)
    from physmorph.pipeline.render_reporting import write_render_report
    render_report = write_render_report(args.out/'run',result['history'],metadata['config'],metadata['mpm'],len(src))
    output = dict(protocol_sha256=sha(args.out/'protocol.json'), analysis=capture.report,
        history=result['history'], guards=result['guards'], sidecars=sidecars,
        render_influence=render_report, mpm=metadata['mpm'], N=len(src), T=cfg.T, loss_res=cfg.loss_res,
        observed_outer_acceptance=True, evaluator_expiration_checked=True,
        end_utc=datetime.now(timezone.utc).isoformat())
    (args.out/'result.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(dict(out=str(args.out),finite_changes=capture.report['finite_changes'])),flush=True)


if __name__ == '__main__':
    main()
