"""Private diagnostic rollout with only terminal body coefficients variable.

No production state is committed. The displacement mode is never rescaled to
accommodate a brake: terminal coefficients use its remaining joint radius.
"""
from copy import deepcopy

import torch
import warp as wp

from ..mpm.function import PersistentAdjoint
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

    def close(self):
        self.adjoint = None
        self.closed = True

    def evaluate(self, terminal):
        if self.closed:
            raise RuntimeError('Frozen body window expired')
        displacement = self.coefficients[:, :3]
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
        return dict(x=x,F=F,v=v,V=V,positions=X,body_energy=(field/self.spec.prm.dx).square().sum(1).mean(),
                    valid=bool(valid and pins_exact),pins_exact=pins_exact,min_det=float(torch.minimum(j,je)))
