"""Continue one prepared W20 solve through iterations 8/16/32 without restarting.

All checkpoints are inner accepted candidates on one fixed initial state and
reference. Only the final selected candidate can be outer-committed. Extra
iterations do not establish rest unless shape/supply and raw movement pass too.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as host_np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution, to_array, to_host, KDTree, array_api as np
from physmorph.pipeline import PipelineConfig, runner
from physmorph.mpm.state import MPMParams
from physmorph.metrics import sil_iou, target_extent
from scripts.probes.reference_swap import require,sha


class Capture:
    def __init__(self):
        self.packets = {}
        self.inner_history = None
        self.stats = None
        self.outer_accepted = False
        self.outer_record = None

    def observe(self,index,packet):
        require(index==19, 'Unexpected checkpoint window')
        iteration = packet['iteration']
        require(iteration in (8,16,32) and iteration not in self.packets, 'Unexpected checkpoint iteration')
        if self.packets:
            first = self.packets[8]
            for key in ('x0','pins','plan','start_arrived'):
                require(torch.equal(packet[key],first[key]), 'Prepared state/cohort changed: '+key)
            require(packet['lambda_render']==first['lambda_render'], 'Lambda changed inside one solve')
            for key in ('m','grid_min','target_grid'):
                require(torch.equal(packet['reference'].density[key],first['reference'].density[key]),
                        'Prepared density reference changed')
            for key in ('render','shade'):
                a,b = getattr(packet['reference'],key),getattr(first['reference'],key)
                targets = 'target_alphas' if key=='render' else 'shade_tgts'
                require((a is None)==(b is None), 'Prepared render presence changed')
                if a is not None:
                    require(all(torch.equal(x,y) for x,y in zip(a[targets],b[targets])), 'Prepared images changed')
        self.packets[iteration] = packet
        print(json.dumps(dict(inner_checkpoint=iteration,merit=packet['history']['loss'],
                              replay=packet['gradient_replay'])),flush=True)

    def wrap(self, original):
        def wrapped(*args,**kwargs):
            if kwargs['win_index']!=19:
                return original(*args,**kwargs)
            cfg = args[2]
            require(cfg.iters==8, 'Unexpected original budget')
            original_budget = cfg.iters
            cfg.iters = 32
            try:
                result = original(*args,on_checkpoint=self.observe,checkpoint_iterations=(8,16,32),**kwargs)
                self.inner_history = deepcopy(result[-2])
                self.stats = {key:result[-1].get(key) for key in
                              ('accepted','rejected','ls_exhausted','grad_converged','pace_bound','replay_diagnostics')}
                return result
            finally:
                cfg.iters = original_budget
        return wrapped

    def commit(self,index,x,F,v,row):
        if index==19:
            self.outer_record = deepcopy(row)
            if not row.get('frame_end') or row.get('null_commit'):
                return
            self.outer_accepted = True
            if 32 in self.packets:
                endpoint = torch.as_tensor(to_array(x),device='cuda')
                require(torch.equal(endpoint,self.packets[32]['positions'][-1]),
                        'Outer commit differs from the observed iteration32 endpoint')


def summarize(capture, source, target):
    source,target = to_array(source),to_array(target)
    st,tt = KDTree(source),KDTree(target)
    spacing = float(np.median(st.query(source,k=2)[0][:,1]))
    tspacing = float(np.median(tt.query(target,k=2)[0][:,1]))
    radius = float(np.median(tt.query(target,k=9)[0][:,8]))
    tip = target[target[:,1].argmax()]
    extent = target_extent(target)
    counts = st.query_ball_point(source,2*spacing,return_length=True)
    source_ids = np.nonzero((source[:,1] >= (source[:,1].max()+source[:,1].min())*.5)
                           & (counts < .6*np.median(counts)))[0]
    import hashlib
    source_ids_sha = hashlib.sha256(to_host(source_ids).astype('<i8').tobytes()).hexdigest()
    if not capture.packets:
        return dict(rows=[],incomplete_reason='Existing solver stopped before the first preregistered checkpoint',
                    spacing=spacing,target_spacing=tspacing,source_upper_count=len(source_ids),source_upper_ids_sha256=source_ids_sha)
    require(8 in capture.packets, 'Later checkpoint lacks iteration8; invalid observer history')
    first = capture.packets[8]
    cohorts = dict(start_free=~first['pins'], start_arrived_free=~first['pins'] & first['start_arrived'])
    rows = []
    for iteration,packet in sorted(capture.packets.items()):
        x = to_array(packet['positions'][-1]); tree = KDTree(x)
        dt = tree.query(target)[0]; ds = tt.query(x)[0]
        density = tree.query_ball_point(x[source_ids],radius,return_length=True)-1
        pos = torch.cat((packet['x0'][None],packet['positions']))
        steps = pos[1:]-pos[:-1]
        motion = {}
        for key,mask in cohorts.items():
            n = int(mask.sum())
            if not n:
                motion[key] = dict(particles=0);continue
            motion[key] = dict(particles=n,
                net_rms_sp=float((pos[-1,mask]-pos[0,mask]).square().sum(-1).mean().sqrt())/spacing,
                step_rms_sp=float(steps[:,mask].square().sum(-1).mean().sqrt())/spacing,
                path_mean_sp=float(steps[:,mask].norm(dim=-1).sum(0).mean())/spacing,
                physical_terminal_rms_wu_s=float(packet['V'][-1,mask].square().sum(-1).mean().sqrt()),
                geometric_terminal_rms_wu_s=float(steps[-1,mask].square().sum(-1).mean().sqrt())/packet['dt'])
        require(all(bool(torch.isfinite(v).all()) for v in packet['controls'].values()), 'Invalid controls')
        rows.append(dict(iteration=iteration,merit=packet['history']['loss'],
            lambda_render=packet['lambda_render'],alpha=packet['alpha'],adam_time=packet['adam_time'],
            scalar_gradient=packet['scalar_gradient'],gradient_replay=packet['gradient_replay'],
            optimizer_state_exact=packet['optimizer_state_exact'],motion=motion,
            geometry=dict(chamfer=float(ds.mean()+dt.mean()),sil_iou=sil_iou(x,target,extent),
                target_near_frac=float((dt<=2*tspacing).mean()),
                upper_target_near_frac=float((dt[target[:,1]>2.3]<=2*tspacing).mean()),
                tip_n=int((np.linalg.norm(x-tip,axis=-1)<.25).sum()),
                fixed_source_upper_density=float(density.mean()/8),
                min_endpoint_detF=float(torch.linalg.det(packet['F'].reshape(-1,3,3)).min()))))
    return dict(rows=rows,spacing=spacing,target_spacing=tspacing,source_upper_count=len(source_ids),
        source_upper_ids_sha256=source_ids_sha,
        target_tip_n=int((np.linalg.norm(target-tip,axis=-1)<.25).sum()),
        cohort_definition='Same window-start pins and frozen plan arrival; no checkpoint endpoint selection',
        gradient_scope='True current scalar/control derivative on a new rollout at the accepted controls; replay noise reported; raw norm is not constrained stationarity',
        candidate_scope='Inner accepted checkpoints in one solve; only final selected state is outer-committed; later best-frame delivery is not the checkpoint selection')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/relcfd/chayo/physmorph_v2'))
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    require(args.out.resolve().is_relative_to(args.root.resolve()), 'Output outside project data')
    args.out.mkdir(exist_ok=False)
    metadata_path = args.root/'work/p303/raw24a.json'
    source_path = args.root/'repro/current_pair/source_render_full_dt_iso_nn.npz'
    meta_bytes = metadata_path.read_bytes(); metadata = json.loads(meta_bytes)
    cfg = PipelineConfig(**metadata['config']); cfg.stop_after_windows = 20
    require(cfg.iters==8 and not cfg.commit_pic and not cfg.shift_sub and cfg.compute_backend=='cuda', 'Wrong raw recipe')
    inputs = {str(path):sha(path) for path in (metadata_path,source_path,Path(cfg.target_reference))}
    from hashlib import sha256
    require(inputs[str(metadata_path)]==sha256(meta_bytes).hexdigest(), 'Metadata changed')
    root = Path(__file__).resolve().parents[2]
    code = {str(f.relative_to(root)):sha(f) for f in sorted((root/'physmorph').rglob('*.py'))}
    protocol = dict(start_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,code=code,
        probe_sha256=sha(__file__),config=metadata['config'],mpm=metadata['mpm'],
        overrides=dict(stop_after_windows=20,attempt20_inner_budget=32),checkpoints=[8,16,32],
        no_restart=True,retain_existing_early_exit=True)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with host_np.load(source_path,allow_pickle=False) as archive: source,target = archive['src'],archive['tgt']
    capture = Capture()
    with patch.object(runner,'optimize_window',capture.wrap(runner.optimize_window)):
        result = runner.run_pipeline(source,target,MPMParams(**metadata['mpm']),cfg,on_commit=capture.commit)
    require(not any(result['guards'].values()), 'State guards fired')
    with cuda_execution('cuda'):
        report = summarize(capture,source,target)
        sidecars = {}
        for i,packet in capture.packets.items():
            require(all(bool(torch.isfinite(v).all()) for v in packet.values() if torch.is_tensor(v)),
                    'Nonfinite checkpoint state')
            arrays = {key:to_host(v) for key,v in packet.items() if torch.is_tensor(v)}
            arrays.update({'control_'+key:to_host(v) for key,v in packet['controls'].items()})
            path = args.out/f'iteration_{i}.npz'
            host_np.savez_compressed(path,**arrays);sidecars[path.name]=sha(path)
    require(all(sha(Path(key))==value for key,value in inputs.items()), 'Inputs changed')
    require(code=={str(f.relative_to(root)):sha(f) for f in sorted((root/'physmorph').rglob('*.py'))}, 'Source changed')
    from physmorph.pipeline.render_reporting import write_render_report
    influence = write_render_report(args.out/'run',result['history'],metadata['config'],metadata['mpm'],len(source))
    output = dict(protocol_sha256=sha(args.out/'protocol.json'),report=report,sidecars=sidecars,
        requested_checkpoints=[8,16,32],observed_checkpoints=sorted(capture.packets),
        final_outer_accepted=capture.outer_accepted,inner_history=capture.inner_history,
        missing_checkpoints=[i for i in (8,16,32) if i not in capture.packets],
        final_outer_record=capture.outer_record,
        final_attempt_records=[row for row in result['history'] if row.get('animation')==19],
        inner_stop=capture.stats,history=result['history'],guards=result['guards'],render_influence=influence,
        N=len(source),T=cfg.T,mpm=metadata['mpm'],loss_res=cfg.loss_res)
    (args.out/'result.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(dict(out=str(args.out),rows=report['rows'])),flush=True)


if __name__=='__main__': main()
