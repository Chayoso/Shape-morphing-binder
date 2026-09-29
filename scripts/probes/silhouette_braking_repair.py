"""Matched aggregate-render versus additional silhouette constraint on one origin."""
import json
from hashlib import sha256
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution
from physmorph.pipeline.affine_braking import affine_ball_step
from scripts.probes.running_braking_repair import repair,repair_context
from scripts.probes.live_braking_compensation import LiveCompensation,identity_evidence
from scripts.probes.terminal_braking import main
from scripts.probes.reference_swap import require,sha


def write_package(package,folder):
    metadata = dict(**package.decode(),context_sha256=package.context,package_sha256=package.digest(),
        arrays={key:dict(dtype=dtype,shape=shape,sha256=sha256(data).hexdigest())
                for key,dtype,shape,data in package.arrays})
    path = folder/'shared.json'
    path.write_text(json.dumps(metadata,indent=2,allow_nan=False))
    return metadata,dict(**metadata['sidecars'],**{'shared.json':sha(path)})


def silhouette_repair(model,packet,source,target,out):
    require('evaluate_merit' in packet,'Matched repair requires original-merit evaluation')
    folder = out/'baseline';folder.mkdir(exist_ok=False)
    baseline = repair(model,packet,source,target,folder,baseline_only=True)
    base_meta,files = write_package(baseline,folder)
    sidecars = {'baseline/'+k:v for k,v in files.items()}
    folder = out/'origin';folder.mkdir(exist_ok=False)
    origin = repair(model,packet,source,target,folder,shared_baseline=baseline,origin_only=True,
                    correction_rounds=2,quality_backtracking=True)
    origin_meta,files = write_package(origin,folder)
    sidecars.update({'origin/'+k:v for k,v in files.items()})
    arms = {}
    for label,protect in (('aggregate',False),('silhouette',True)):
        require(baseline.digest()==base_meta['package_sha256'] and origin.digest()==origin_meta['package_sha256'],
                'Common package changed before arm')
        require(repair_context(model,packet,source,target)==baseline.context==origin.context,
                'Common numerical context changed before arm')
        folder = out/label;folder.mkdir(exist_ok=False)
        arm = repair(model,packet,source,target,folder,correction_rounds=2,quality_backtracking=True,
                     shared_baseline=baseline,shared_origin=origin,protect_silhouette=protect)
        require(baseline.digest()==base_meta['package_sha256'] and origin.digest()==origin_meta['package_sha256']
                and repair_context(model,packet,source,target)==baseline.context==origin.context,
                'Common packages/context changed after arm')
        arm['shared_context_before_after_exact'] = True
        arms[label] = arm
        sidecars.update({label+'/'+k:v for k,v in arm['sidecars'].items()})
        path = folder/'arm_result.json'
        path.write_text(json.dumps(arm,indent=2,allow_nan=False))
        sidecars[label+'/arm_result.json'] = sha(path)
    return dict(baseline=base_meta,origin=origin_meta,arms=arms,sidecars=sidecars,
        arm_order=['aggregate','silhouette'],
        original_merit_role='Report only against common original maximum; never selects or stops either arm',
        component_role='Only silhouette arm requires prepared silhouette <= common baseline maximum; raw P306 gates unchanged')


class SilhouetteRepair(LiveCompensation):
    operation = staticmethod(silhouette_repair)
    record_candidate_merit = True
    artifact_subdir = 'silhouette_repair'
    scope = 'Matched terminal05 repair with one original baseline and terminal origin; no adoption or archive admission'
    rendering_role = 'Only added silhouette affine/nonlinear constraint differs across arms; prepared channels, original merit and raw quality remain distinct; not 4K appearance certification'


if __name__=='__main__':
    SilhouetteRepair.identity = identity_evidence(Path('/data/relcfd/chayo/physmorph_v2'))
    with cuda_execution('cuda:0'):
        device = 'cuda:0'
        g = torch.tensor([0.,0.,0.,1.],device=device,dtype=torch.float64)
        G = torch.tensor([[1e-5,0.,0.,0.],[0.,1e3,0.,0.],[-1e-2,-1e-2,1e-8,0.]],device=device,dtype=g.dtype)
        b = torch.tensor([-2e-6,-300.,.005-4e-9],device=device,dtype=g.dtype)
        step,info = affine_ball_step(g,G,b,1.)
        expected = torch.tensor([-.2,-.3,-.4,-(.71**.5)],device=device,dtype=g.dtype)
        require(step is not None and bool(torch.allclose(step,expected,rtol=0,atol=3e-8)),
                'CUDA three-plane smoke failed')
        print(json.dumps(dict(cuda_three_plane_smoke=True,linear=info)),flush=True)
    main(SilhouetteRepair,extra_protocol=dict(kind='Matched prepared-silhouette constraint diagnostic',version=1,
        scope=SilhouetteRepair.scope,identity_discrimination=SilhouetteRepair.identity,
        arms=['aggregate','silhouette'],terminal='Same own freshly generated terminal05 in both arms',
        shared='One immutable original baseline triplet and one terminal05 forward/4gradients/3noise observations/threshold',
        constraints=dict(aggregate=['volume','render'],silhouette=['volume','render','silhouette']),
        per_arm=dict(halvings=10,correction_rounds=2,accepted_updates=1,fixed_candidate_repeats=3),
        stopping='First P306+resolved-running candidate (+actual silhouette ceiling in treatment) gets3fixed repeats then arm stops; both arms retained regardless of repeat/merit failure',
        admission='No production commit, archive admission, or full-morph/4K promotion'),
        extra_helpers=tuple(Path(__file__).resolve().with_name(name) for name in
            ('silhouette_braking_repair.py','running_braking_repair.py','live_braking_compensation.py','braking_compensation.py')))
