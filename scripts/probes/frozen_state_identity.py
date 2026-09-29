"""One archive round trip: exact inputs, primal ownership, and all repeat pairs.

Preserves witnesses before any closure decision. Does not replace the failed
capture1 record or its unavailable original in-memory C witnesses.
"""
import argparse
from dataclasses import fields,is_dataclass
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
import warp as wp

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution,to_array,to_host
from physmorph.pipeline.frozen_body_window import FrozenBodyWindow
from physmorph.pipeline.prepared_reference import PreparedReference
from scripts.probes.reference_swap import require,sha


def exact_tree(a,b,path='root',counts=None):
    if counts is None: counts = dict(arrays=0,elements=0,scalars=0)
    if torch.is_tensor(a) or hasattr(a,'__cuda_array_interface__') or isinstance(a,np.ndarray):
        at,bt = [torch.as_tensor(to_array(v),device='cuda:0').contiguous() for v in (a,b)]
        require(at.dtype==bt.dtype and at.shape==bt.shape,'Array layout mismatch: '+path)
        require(torch.equal(at.reshape(-1).view(torch.uint8),bt.reshape(-1).view(torch.uint8)),
                'Array bit mismatch: '+path)
        counts['arrays']+=1;counts['elements']+=at.numel()
    elif is_dataclass(a):
        exact_tree({f.name:getattr(a,f.name) for f in fields(a)},
                   {f.name:getattr(b,f.name) for f in fields(b)},path,counts)
    elif isinstance(a,dict):
        require(set(a)==set(b),'Tree keys mismatch: '+path)
        for k in a: exact_tree(a[k],b[k],path+'.'+k,counts)
    elif isinstance(a,(tuple,list)):
        require(type(a)==type(b) and len(a)==len(b),'Sequence mismatch: '+path)
        for i,(x,y) in enumerate(zip(a,b)): exact_tree(x,y,path+'.'+str(i),counts)
    else:
        require(a==b and (not isinstance(a,float) or a.hex()==float(b).hex()),'Scalar mismatch: '+path)
        counts['scalars']+=1
    return counts


def initial_buffers(model):
    tr = model.adjoint.traj
    arrays = {k+'0':wp.to_torch(getattr(tr,k)[0]).clone() for k in ('x','v','C','F','Fg')}
    names = ('m','vol','lam','mu','eta','pin','Fp','body_control','bond_nbr','bond_rest','bond_frag',
             'layer_ug','layer_mask','layer_nrm','layer_nbr','layer_w','layer_u','layer_g')
    arrays.update({k:wp.to_torch(getattr(tr,k)).clone() for k in names if getattr(tr,k,None) is not None})
    arrays['stress_sequence'] = model.adjoint.dc.clone()
    return arrays


def witness(model,reference,terminal,mask):
    with torch.enable_grad():
        leaf = terminal.detach().clone().requires_grad_()
        values = model.evaluate(leaf)
        require(values['valid'],'Invalid identity witness')
        C = values['C'].clone()
        tr = model.adjoint.traj
        C_primal = wp.to_torch(tr.C[model.spec.T]).clone()
        initial = initial_buffers(model)
        geometric = (values['positions'][-1]-values['positions'][-2])/model.spec.prm.dt
        h = .5*(values['v'][mask].square().sum(-1).mean()+geometric[mask].square().sum(-1).mean())
        terms = reference.terms(values['x'])
        gradients,immutable = {},[]
        for key,value in [('brake',h),('volume',terms['volume']),('render',terms['render'])]:
            gradients[key] = torch.autograd.grad(value,leaf,retain_graph=key!='render')[0].detach()
            after = initial_buffers(model)
            immutable.append(dict(seed=key,primal_C=torch.equal(C_primal,wp.to_torch(tr.C[model.spec.T])),
                                  owned_C=torch.equal(C,values['C']),
                                  initial_buffers=all(torch.equal(v,after[k]) for k,v in initial.items())))
            del after
    return dict(arrays={k:values[k].detach().clone() for k in ('positions','V','F','C')},
                terms={k:float(v.detach()) for k,v in terms.items()},gradients=gradients,
                immutable=immutable,initial=initial,owned_C_handle=values['C'],
                raw_C_handle=wp.to_torch(tr.C[model.spec.T]),raw_C_snapshot=C_primal)


def comparison(a,b,model):
    eps = torch.finfo(torch.float32).eps
    arrays,gradients,data = {},{},{}
    for key,ref in a['arrays'].items():
        unit = dict(positions=model.spec.prm.dx,V=model.spec.prm.dx/(model.spec.T*model.spec.prm.dt),
                    F=1.,C=1/(model.spec.T*model.spec.prm.dt))[key]
        error = (b['arrays'][key]-ref).abs()
        ratio = error/(32*eps*(unit+ref.abs()))
        index = int(ratio.reshape(-1).argmax())
        arrays[key] = dict(max_abs=float(error.max()),rms=float(error.square().mean().sqrt()),
                          ratio=float(ratio.max()),failed_elements=int((ratio>1).sum()),max_ratio_flat_index=index,
                          reference_at_max_ratio=float(ref.reshape(-1)[index]),
                          candidate_at_max_ratio=float(b['arrays'][key].reshape(-1)[index]))
    for key,ref in a['gradients'].items():
        error = float((b['gradients'][key]-ref).norm()); norm=float(ref.norm())
        tolerance = 64*eps*max(norm,1e-12)
        gradients[key] = dict(error_norm=error,reference_norm=norm,tolerance=tolerance,ratio=error/tolerance)
    for key,ref in a['terms'].items():
        tolerance = 32*eps*max(abs(ref),1e-12)
        data[key] = dict(reference=ref,candidate=b['terms'][key],tolerance=tolerance,
                         ratio=abs(b['terms'][key]-ref)/tolerance)
    return dict(arrays=arrays,gradients=gradients,data=data,
                passed=all(v['ratio']<=1 for block in (arrays,gradients,data) for v in block.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    for path in (args.archive,args.out):
        require(path.resolve().is_relative_to('/data/relcfd/chayo/physmorph_v2'),'Outside project data')
    args.out.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    code = {str(f.relative_to(root)):sha(f) for f in sorted((root/'physmorph').rglob('*.py'))}
    digest = sha(args.archive)
    protocol = dict(start_utc=datetime.now(timezone.utc).isoformat(),archive_sha256=digest,code=code,
        script_sha256=sha(__file__),repeats=3,controls=['baseline','trial05'],array_eps=32,gradient_eps=64,
        scope='One new archive reuse validation; original failed C witness remains unavailable; all pairs retain existing bounds')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    signatures,sidecars,immutability,next_forward = {},{},{},[]
    with cuda_execution('cuda:0'):
        reference,packet = FrozenBodyWindow.load(args.archive,'cuda:0')
        def cupy(value):
            if torch.is_tensor(value): return to_array(value,copy=True)
            if isinstance(value,(tuple,list)): return type(value)(cupy(v) for v in value)
            return value
        for f in fields(reference.spec): setattr(reference.spec,f.name,cupy(getattr(reference.spec,f.name)))
        roundtrip = args.out/'roundtrip.npz'
        reference.save(roundtrip,packet)
        reloaded,loaded = FrozenBodyWindow.load(roundtrip,'cuda:0')
        counts = exact_tree(reference.spec,reloaded.spec)
        for name in ('idx','weights','gate','coefficients','stress','surface_u'):
            exact_tree(getattr(reference,name),getattr(reloaded,name),name,counts)
        exact_tree(packet,loaded,'observations',counts)
        sidecars[roundtrip.name] = sha(roundtrip)
        buffer_counts = {}
        for name,model,observations in [('reference',reference,packet),('reloaded',reloaded,loaded)]:
            ref = PreparedReference(**observations['reference'])
            mask = observations['start_arrived'] & ~observations['pins']
            retained = []
            static_initial = None
            for control,terminal in [('baseline',model.coefficients[:,3:]),('trial05',observations['trial05_terminal'])]:
                for repeat in range(3):
                    label = f'{name}_{control}_{repeat}'
                    result = witness(model,ref,terminal,mask)
                    path = args.out/(label+'.npz')
                    arrays = {k:to_host(v) for k,v in result['arrays'].items()}
                    arrays.update({'gradient_'+k:to_host(v) for k,v in result['gradients'].items()})
                    np.savez_compressed(path,**arrays);sidecars[path.name]=sha(path)
                    # Retain evaluate()'s actual public tensor, not our persisted clone.
                    for old in retained:
                        next_forward.append(dict(previous=old['label'],current=label,
                            owned_C_unchanged=torch.equal(old['handle'],old['snapshot']),
                            raw_C_changed=not torch.equal(old['raw'],old['raw_snapshot']),
                            changed_control=old['control']!=control))
                    retained.append(dict(label=label,control=control,handle=result['owned_C_handle'],
                        snapshot=result['owned_C_handle'].clone(),raw=result['raw_C_handle'],
                        raw_snapshot=result['raw_C_snapshot']))
                    current_static = {k:v for k,v in result['initial'].items() if k!='body_control'}
                    if static_initial is None: static_initial = current_static
                    exact_tree(static_initial,current_static,'Inputs across next forward')
                    if repeat==0 and name=='reference':
                        buffer_counts[control] = result['initial']
                    elif repeat==0:
                        exact_tree(buffer_counts[control],result['initial'],'Warp inputs')
                    signatures[label] = {k:result[k] for k in ('arrays','terms','gradients')}
                    immutability[label] = result['immutable']
                    print(json.dumps(dict(witness=label,immutable=result['immutable'],saved=path.name)),flush=True)
            model.close()
        comparisons = []
        for control in ('baseline','trial05'):
            keys = [k for k in signatures if '_'+control+'_' in k]
            for a,b in itertools.combinations(keys,2):
                row = comparison(signatures[a],signatures[b],reference)
                row.update(reference=a,candidate=b,group='within' if a.split('_')[0]==b.split('_')[0] else 'cross')
                comparisons.append(row)
    require(sha(args.archive)==digest,'Input archive changed')
    require(all(sha(root/k)==v for k,v in code.items()),'Numerical source changed')
    result = dict(protocol_sha256=sha(args.out/'protocol.json'),archive_sha256=digest,exact_input_counts=counts,
                  exact_warp_inputs=True,exact_static_inputs_across_forward=True,
                  owned_C_survives_next_forward=all(r['owned_C_unchanged'] for r in next_forward),
                  changed_control_changes_raw_C=all(r['raw_C_changed'] for r in next_forward if r['changed_control']),
                  next_forward=next_forward,immutability=immutability,
                  comparisons=comparisons,sidecars=sidecars,
                  strict_all_pairs_pass=all(row['passed'] for row in comparisons),
                  primal_ownership_pass=all(row['primal_C'] and row['owned_C'] and row['initial_buffers']
                                           for rows in immutability.values() for row in rows),scope=protocol['scope'])
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(strict_all_pairs_pass=result['strict_all_pairs_pass'],failed_pairs=sum(not r['passed'] for r in comparisons))),flush=True)


if __name__=='__main__': main()
