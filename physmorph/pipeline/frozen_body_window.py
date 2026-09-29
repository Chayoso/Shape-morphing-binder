"""Private diagnostic body-control rollout with owned initial state.

No production state is committed. The displacement mode is never rescaled to
accommodate a brake: terminal coefficients use its remaining joint radius.
An explicit displacement override supports separately preregistered compensation.
"""
from copy import deepcopy
from dataclasses import asdict, fields
import json

import numpy as np
import torch
import warp as wp

from ..compute import to_host, is_cuda_execution
from ..mpm.function import PersistentAdjoint, RolloutSpec
from ..mpm.state import MPMParams
from .endpoint_contract import endpoint_bounds, valid_endpoint


def project_terminal(terminal, displacement):
    radius = (1-displacement.square().sum(-1, keepdim=True)).clamp_min(0).sqrt()
    return terminal * (radius / terminal.norm(dim=-1, keepdim=True).clamp_min(1e-30)).clamp_max(1)


class FrozenBodyWindow:
    def __init__(self, spec, basis, gate, coefficients, stress, surface_u):
        if not spec.body_ctrl or spec.body_modes != 2:
            raise ValueError('Terminal audit requires two body modes')
        # deepcopy owns numpy/CuPy arrays and the parameter/layer structures.
        self.spec = deepcopy(spec)
        self.idx = basis.idx.detach().clone()
        self.weights = basis.weights.detach().clone()
        self.gate = gate.detach().clone()
        self.coefficients = coefficients.detach().clone()
        self.stress = stress.detach().clone().contiguous()
        self.surface_u = None if surface_u is None else surface_u.detach().clone()
        self.adjoint = None
        self.closed = False

    def save(self, path, observations):
        """Archive I/O only: numeric NPZ members and an explicit JSON tree, no pickle."""
        if self.closed:
            raise RuntimeError('Frozen body window expired')
        arrays = {}
        def encode(value):
            if torch.is_tensor(value) or isinstance(value,np.ndarray) or hasattr(value,'__cuda_array_interface__'):
                key = 'array_'+str(len(arrays))
                arrays[key] = to_host(value)
                if arrays[key].dtype.hasobject:
                    raise ValueError('Object arrays are forbidden in frozen-window archives')
                return dict(type='tensor',key=key,shape=list(arrays[key].shape),dtype=str(arrays[key].dtype))
            if isinstance(value,np.generic): return value.item()
            if isinstance(value,dict): return dict(type='dict',items={k:encode(v) for k,v in value.items()})
            if isinstance(value,(tuple,list)): return dict(type=type(value).__name__,items=[encode(v) for v in value])
            if value is None or type(value) in (bool,int,float,str): return value
            raise TypeError('Unsupported frozen-window archive value: '+str(type(value)))
        spec = {f.name:getattr(self.spec,f.name) for f in fields(self.spec)}
        spec['prm'] = asdict(self.spec.prm)
        data = dict(version=1,spec=spec,observations=observations,
                    model={k:getattr(self,k) for k in ('idx','weights','gate','coefficients','stress','surface_u')})
        manifest = json.dumps(encode(data),allow_nan=False).encode('utf-8')
        arrays['manifest'] = np.frombuffer(manifest,dtype=np.uint8)
        with open(path,'xb') as stream:
            np.savez_compressed(stream,**arrays)

    @classmethod
    def load(cls,path,device):
        if str(device).startswith('cuda') and not is_cuda_execution():
            raise ValueError('CUDA frozen-window loading requires cuda_execution context')
        with np.load(path,allow_pickle=False) as archive:
            def decode(value):
                if not isinstance(value,dict): return value
                kind = value['type']
                if kind=='tensor':
                    array = archive[value['key']]
                    if list(array.shape)!=value['shape'] or str(array.dtype)!=value['dtype']:
                        raise ValueError('Frozen-window array schema mismatch')
                    tensor = torch.tensor(array,device=device)
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError('Nonfinite frozen-window array')
                    return tensor
                if kind=='dict': return {k:decode(v) for k,v in value['items'].items()}
                if kind in ('list','tuple'):
                    items = [decode(v) for v in value['items']]
                    return tuple(items) if kind=='tuple' else items
                raise ValueError('Unknown frozen-window tree type')
            data = decode(json.loads(archive['manifest'].tobytes()))
        if data['version'] != 1:
            raise ValueError('Unsupported frozen-window version')
        spec = data['spec'];spec['device'] = device
        spec['prm'] = MPMParams(**spec['prm'])
        result = cls.__new__(cls)
        result.spec = RolloutSpec(**spec)
        for key,value in data['model'].items(): setattr(result,key,value)
        result.adjoint = None;result.closed = False
        return result,data['observations']

    def close(self):
        self.adjoint = None
        self.closed = True

    def evaluate(self, terminal, displacement=None, *, retain_full_state=False):
        """Optionally own detached post-step F snapshots for exact frame export.

        Capture happens before any subsequent rollout can overwrite the private
        trajectory. It grants neither archive admission nor candidate adoption.
        Differentiable terminal F remains the existing ``F`` output.
        """
        if self.closed:
            raise RuntimeError('Frozen body window expired')
        if str(self.spec.device).startswith('cuda') and not is_cuda_execution():
            raise ValueError('CUDA frozen-window evaluation requires cuda_execution context')
        displacement = self.coefficients[:, :3] if displacement is None else displacement
        if (displacement.shape != self.coefficients[:,:3].shape
                or displacement.device != self.coefficients.device
                or not bool(torch.isfinite(displacement).all())):
            raise ValueError('Invalid displacement coefficients')
        if (terminal.shape != displacement.shape or terminal.device != displacement.device
                or not bool(torch.isfinite(terminal).all())):
            raise ValueError('Invalid terminal coefficients')
        coeff = torch.cat((displacement, terminal), dim=1)
        if bool((coeff.detach().square().sum(-1) > 1+1e-6).any()):
            raise ValueError('Terminal coefficients exceed remaining joint radius')
        n = len(self.idx)
        field = (self.spec.prm.dx * (coeff[self.idx]*self.weights[..., None]).sum(1)*self.gate)
        body = field.reshape(n, 2, 3).permute(1, 0, 2).reshape(2*n, 3).contiguous()
        if self.adjoint is None:
            self.adjoint = PersistentAdjoint(self.spec, position_sequence=True)
        x, F, v, Fg, V, X = self.adjoint.apply_with_positions(self.stress, self.surface_u, body)
        with torch.no_grad():
            tr = self.adjoint.traj
            finite = all(bool(torch.isfinite(a).all()) for a in (x, F, v, Fg, V, X))
            finite &= all(bool(torch.isfinite(wp.to_torch(a)).all()) for a in tr.C)
            j = torch.stack([torch.linalg.det(wp.to_torch(tr.F[t]).reshape(n,3,3)).min()
                             for t in range(1,self.spec.T+1)]).min()
            je = torch.stack([torch.linalg.det(wp.to_torch(tr.F[t]).reshape(n,3,3)
                               +self.stress[t]).min() for t in range(self.spec.T)]).min()
            bounds = endpoint_bounds(self.spec.prm,x)
            valid = finite and valid_endpoint(X,bounds) and float(torch.minimum(j,je)) > 0
            pinned = wp.to_torch(tr.pin) > .5
            start = wp.to_torch(tr.x[0])
            pins_exact = torch.equal(X[:,pinned],start[pinned][None].expand(self.spec.T,-1,-1))
            C = wp.to_torch(tr.C[self.spec.T]).clone()
            full = {}
            if retain_full_state:
                F_sequence = torch.stack([wp.to_torch(tr.F[t]).reshape(n,3,3)
                                          for t in range(1,self.spec.T+1)])
                F_initial = wp.to_torch(tr.F[0]).reshape(n,3,3).clone()
                full = dict(F_sequence=F_sequence,F_initial=F_initial)
                valid &= (bool(torch.isfinite(F_sequence).all() and torch.isfinite(F_initial).all())
                          and torch.equal(F_sequence[-1].reshape_as(F),F))
        return dict(x=x,F=F,C=C,v=v,V=V,positions=X,body_energy=(field/self.spec.prm.dx).square().sum(1).mean(),
                    valid=bool(valid and pins_exact),pins_exact=pins_exact,min_det=float(torch.minimum(j,je)),**full)
