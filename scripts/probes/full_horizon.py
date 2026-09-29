"""Observe ordinary stopping at the configured horizon; no new solve or policy."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as host_np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import array_api as np, to_array, to_host
from scripts.probes.coverage_paths import require, sha


def array_digest(value):
    """Archive identity only; no CPU simulation or metric computation."""
    return sha256(host_np.ascontiguousarray(value).tobytes()).hexdigest()


class RestTrace:
    def __init__(self, folder):
        self.folder=Path(folder);self.folder.mkdir(exist_ok=False)
        self.attempts=[]

    def wrap(self, original):
        def observed(x0, *args, **kwargs):
            index=int(kwargs['win_index'])
            require(index==len(self.attempts), 'Unexpected attempt sequence')
            initial=to_host(x0)
            pin=kwargs.get('pin_init')
            pin_before=to_host(np.zeros(len(x0),np.bool_) if pin is None else to_array(pin)>.5)
            meta=dict(animation=index,x0_sha256=array_digest(initial),N=len(initial))
            # Own inputs before the original function; return its exact tuple.
            output=original(x0,*args,**kwargs)
            frames,F_seq,end,s,hist,stats=output
            arrays=dict(pin_before=pin_before,optimizer_raw_endpoint=to_host(frames[-1]))
            plan=stats.get('plan_img');arrived=stats.get('arrived_mask');radius=stats.get('pace_r')
            meta.update(inner_accepted=int(stats.get('accepted',0)),has_plan=plan is not None,
                        optimizer_frames=len(frames),endpoint_space=stats.get('endpoint_space'))
            if plan is not None:
                require(arrived is not None and radius is not None,'Incomplete arrival reference')
                arrays.update(plan=to_host(plan),start_arrived=to_host(arrived),radius=to_host(radius))
                require(arrays['plan'].shape==initial.shape and arrays['start_arrived'].shape==(len(initial),),
                        'Invalid arrival layout')
            else:
                require(not hist,'Accepted solve has no full plan')
            if end is not None and end.get('v') is not None:
                v=to_array(end['v']).astype(np.float64)
                arrays['optimizer_terminal_speed_squared']=to_host((v*v).sum(-1))
            path=self.folder/f'attempt_{index:03d}.npz'
            with path.open('xb') as stream: host_np.savez_compressed(stream,**arrays)
            meta.update(sidecar=path.name,sha256=sha(path))
            self.attempts.append(meta)
            return output
        return observed

    def finish(self, result):
        """Bind attempts to actual promoted archive rows, excluding rejected paths."""
        cursor=0; mapped=[]; accepted=[]; events=[]
        for row in result['history']:
            index=int(row['animation'])
            if 'c2f_render_res' in row:
                events.append(dict(row))
                continue
            if row.get('held'):
                require(array_digest(result['frames'][cursor+1])==array_digest(result['frames'][cursor]),
                        'Held archive state changed')
                cursor+=1
                continue
            require(index==len(mapped) and index<len(self.attempts),'History lacks unique observed attempt')
            meta=dict(self.attempts[index])
            require(meta['x0_sha256']==array_digest(result['frames'][cursor]), 'Attempt/archive start mismatch')
            meta['start_frame']=cursor
            committed=bool(row.get('frame_end') and not row.get('null_commit')
                           and not row.get('outer_rejected') and row.get('outer_accepted',1))
            if committed:
                cursor=int(row['frame_end'])-1
                require(cursor-meta['start_frame']==meta['optimizer_frames']-1
                        and cursor>meta['start_frame'] and meta['inner_accepted']>0,
                        'Invalid accepted frame interval')
                accepted.append(index)
            elif row.get('null_commit') and not row.get('outer_rejected'):
                require(array_digest(result['frames'][cursor+1])==meta['x0_sha256'],'Null archive state changed')
                cursor+=1
            meta.update(end_frame=cursor,committed=committed,outer_rejected=bool(row.get('outer_rejected')),
                        null_commit=bool(row.get('null_commit')),grad_converged=bool(row.get('grad_converged')))
            mapped.append(meta)
        require(len(mapped)==len(self.attempts) and cursor==len(result['frames'])-1,
                'Attempt/archive mapping incomplete')
        return dict(attempts=mapped,events=events,accepted_attempts=accepted,
            actual_last_accepted=(None if not accepted else accepted[-1]),
            actual_archive_frames=len(result['frames']),deliver_n=int(result['deliver_n']),
            truncation=result['truncation'],held_archive_rows=int(result['n_held']),
            reported_converged=bool(result['converged']),
            final_pin_sha256=(None if result['pinned'] is None else array_digest(result['pinned'])),
            stored_velocity_scope='Per-ID optimizer terminal speed squared before outer operations; accepted attempts only for committed-path claims',
            arrival_scope='Start masks and frozen full plan/radius per attempt; endpoint arrival must use promoted archive position',
            pin_scope='Before solve; next attempt and final archive pin masks close post-admission state when release policies are off',
            stop_scope='Ordinary pipeline stop and held suffix, not a physical rest certificate')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--arm',choices=('baseline','raw'),required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    require(args.out.resolve().is_relative_to(args.root.resolve()),'Output outside project data')
    from physmorph.pipeline import runner
    from scripts.probes import gpu_pipeline
    source=args.root/'repro/current_pair/source.json'
    source_bytes=source.read_bytes();metadata=json.loads(source_bytes)
    config=metadata['arms']['render_full_dt_iso_nn']['config']
    require(config['animations']==300 and config['T']==20 and config['archive_stride']==1,'Unexpected horizon')
    require(not any(config.get(k,False) for k in ('settle_pin_follow','settle_pin_yield','settle_pin_kkt',
                                                  'surface_gs_loss','render_F_geom','local_dress_iters')),
            'Uncovered release/render policy')
    code_root=Path(__file__).resolve().parents[2]
    code={str(p):sha(p) for p in sorted((code_root/'physmorph').rglob('*.py'))}
    for name in ('scripts/probes/full_horizon.py','scripts/probes/gpu_pipeline.py',
                 'scripts/probes/coverage_paths.py','scripts/ops/run_p303_probe.sh',
                 'scripts/ops/cuda_python.py','docs/full_horizon_p316.md'):
        code[str(code_root/name)]=sha(code_root/name)
    inputs={str(source):sha256(source_bytes).hexdigest()}
    for name in ('source_render_full_dt_iso_nn.npz','target_reference.npz'):
        path=source.parent/name;inputs[str(path)]=sha(path)
    with host_np.load(source.parent/'source_render_full_dt_iso_nn.npz',allow_pickle=False) as data:
        require(data['src'].shape==data['tgt'].shape==(300000,3),'Unexpected particle count')
    require(all(sha(Path(k))==v for k,v in inputs.items()),'Input changed before launch')
    cli=['gpu_pipeline.py','--root',str(args.root),'--windows','300','--iters','8',
         '--archive','--motion-accounting','--outer-render-committed','--no-shift-sub',
         '--commit-pic-objective' if args.arm=='baseline' else '--no-commit-pic','--out',str(args.out)]
    protocol=dict(start_utc=datetime.now(timezone.utc).isoformat(),arm=args.arm,inputs=inputs,code=code,
                  cli=cli,source_config=config,mpm=metadata['provenance']['mpm'],
                  scope='Ordinary configured-horizon stopping diagnostic; no repaired-state adoption',
                  storage_reservation_bytes=30000000000)
    protocol_path=args.out.with_suffix('.protocol.json')
    with protocol_path.open('x') as stream:json.dump(protocol,stream,indent=2,allow_nan=False)
    trace=RestTrace(args.out.with_name(args.out.name+'_cohorts'));summary={}
    original_run=gpu_pipeline.run_pipeline
    def traced_run(*pos,**kw):
        result=original_run(*pos,**kw)
        # A telemetry mapping failure must not discard a completed physical archive.
        try:
            summary.update(trace.finish(result))
        except Exception as error:
            summary.update(trace_error=dict(type=type(error).__name__,message=str(error)))
        return result
    with patch.object(runner,'optimize_window',trace.wrap(runner.optimize_window)), \
            patch.object(gpu_pipeline,'run_pipeline',traced_run),patch.object(sys,'argv',cli):
        gpu_pipeline.main()
    require(all(sha(Path(k))==v for k,v in {**inputs,**code}.items()),'Inputs/code changed during run')
    require(all(sha(trace.folder/a['sidecar'])==a['sha256'] for a in trace.attempts),'Cohort archive changed')
    outputs=[args.out.with_suffix('.json'),args.out.with_suffix('.npz'),
             args.out.with_name(args.out.name+'_render_full_dt_iso_nn.npz'),
             args.out.with_name(args.out.name+'.render_influence.json'),
             args.out.with_name(args.out.name+'.render_influence.md')]
    summary.update(protocol_sha256=sha(protocol_path),inputs_code_unchanged=True,
                   result_sha256=sha(args.out.with_suffix('.json')),
                   output_sha256={str(p):sha(p) for p in outputs})
    path=args.out.with_suffix('.rest_trace.json')
    with path.open('x') as stream:json.dump(summary,stream,indent=2,allow_nan=False)
    require('trace_error' not in summary,'Trace mapping failed; completed physical outputs retained')
    print(json.dumps(dict(trace=str(path),actual_last_accepted=summary['actual_last_accepted'],
                         reported_converged=summary['reported_converged'])),flush=True)


if __name__=='__main__':main()
