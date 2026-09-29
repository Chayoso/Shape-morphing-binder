"""Matched silhouette-only and fixed-material-support constrained braking."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution
from physmorph.pipeline.affine_braking import affine_ball_step
from scripts.probes.running_braking_repair import repair,repair_context
from scripts.probes.silhouette_braking_repair import write_package
from scripts.probes.live_braking_compensation import LiveCompensation,identity_evidence
from scripts.probes.terminal_braking import main
from scripts.probes.reference_swap import require,sha


def support_repair(model,packet,source,target,out):
    require('evaluate_merit' in packet,'Support repair requires original merit')
    folder=out/'baseline';folder.mkdir(exist_ok=False)
    baseline=repair(model,packet,source,target,folder,baseline_only=True,prepare_support=True)
    base_meta,files=write_package(baseline,folder)
    sidecars={'baseline/'+k:v for k,v in files.items()}
    folder=out/'origin';folder.mkdir(exist_ok=False)
    origin=repair(model,packet,source,target,folder,shared_baseline=baseline,origin_only=True,
                  prepare_support=True,correction_rounds=2,quality_backtracking=True)
    origin_meta,files=write_package(origin,folder)
    sidecars.update({'origin/'+k:v for k,v in files.items()})
    status=origin_meta['support_metadata']['status']
    arms={}
    if status=='ready':
        for label,protect in (('silhouette',False),('support',True)):
            require(baseline.digest()==base_meta['package_sha256'] and origin.digest()==origin_meta['package_sha256']
                    and repair_context(model,packet,source,target)==baseline.context==origin.context,
                    'Common support evidence/context changed before arm')
            folder=out/label;folder.mkdir(exist_ok=False)
            arm=repair(model,packet,source,target,folder,shared_baseline=baseline,shared_origin=origin,
                       protect_silhouette=True,protect_support=protect,save_search_endpoints=True,
                       correction_rounds=2,quality_backtracking=True)
            require(baseline.digest()==base_meta['package_sha256'] and origin.digest()==origin_meta['package_sha256']
                    and repair_context(model,packet,source,target)==baseline.context==origin.context,
                    'Common support evidence/context changed after arm')
            arm['shared_context_before_after_exact']=True
            arms[label]=arm
            sidecars.update({label+'/'+k:v for k,v in arm['sidecars'].items()})
            path=folder/'arm_result.json';path.write_text(json.dumps(arm,indent=2,allow_nan=False))
            sidecars[label+'/arm_result.json']=sha(path)
    require(baseline.digest()==base_meta['package_sha256'] and origin.digest()==origin_meta['package_sha256']
            and repair_context(model,packet,source,target)==baseline.context==origin.context,
            'Common support evidence/context changed')
    return dict(baseline=base_meta,origin=origin_meta,arms=arms,sidecars=sidecars,search_status=status,
                arm_order=list(arms),planned_arm_order=['silhouette','support'],
                support_role='Fixed common-baseline material witness for every stable-covered target lost at origin; exact scalar and raw checks; no reselection',
                original_merit_role='Report only; no candidate adoption or archive admission')


class SupportRepair(LiveCompensation):
    operation=staticmethod(support_repair)
    record_candidate_merit=True
    artifact_subdir='support_repair'
    scope='Matched support-constrained running repair in one fresh callback; no adoption or archive admission'
    rendering_role='Both arms share original prepared volume/render/silhouette constraints; only material support constraints differ; norm shares are not causal displacement shares or4K evidence'


if __name__=='__main__':
    SupportRepair.identity=identity_evidence(Path('/data/relcfd/chayo/physmorph_v2'))
    with cuda_execution('cuda:0'):
        b=torch.linspace(-.1,-.2,8,dtype=torch.float64,device='cuda:0')
        G=torch.eye(9,dtype=b.dtype,device=b.device)[:8]
        g=torch.zeros(9,dtype=b.dtype,device=b.device);g[-1]=1
        step,info=affine_ball_step(g,G,b,1.)
        expected=torch.cat((b,-(1-b.square().sum()).sqrt()[None]))
        require(step is not None and bool(torch.allclose(step,expected,rtol=0,atol=1e-12)),
                'CUDA eight-plane smoke failed')
        print(json.dumps(dict(cuda_eight_plane_smoke=True,linear=info)),flush=True)
    main(SupportRepair,extra_protocol=dict(kind='Matched fixed material support diagnostic',version=1,
        scope=SupportRepair.scope,identity_discrimination=SupportRepair.identity,
        arms=['silhouette','support'],shared='One original baseline triplet/endpoints and one actual terminal05 origin/linearization',
        support='All stable-original-covered targets lost at common origin; repeat0 nearest witness certified in all3 originals',
        abort='More than5 support targets or any uncertified common witness skips BOTH searches; zero targets skips as no intervention',
        constraints='Volume/render/silhouette shared; treatment additionally signed squared witness distances <=0',
        per_arm=dict(halvings=10,correction_rounds=2,accepted_updates=1,fixed_candidate_repeats=3),
        admission='No production commit, archive admission, or all-phase/full-morph/4K claim'),
        extra_helpers=tuple(Path(__file__).resolve().with_name(name) for name in
            ('support_braking_repair.py','silhouette_braking_repair.py','running_braking_repair.py',
             'live_braking_compensation.py','braking_compensation.py')))
