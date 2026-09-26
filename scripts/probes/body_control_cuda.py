"""Run on hyde06 only: CUDA graph candidate refresh and body-force adjoint parity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import torch
from physmorph.mpm.function import RolloutSpec, PersistentAdjoint, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.pipeline.body_control import BodyControlBasis


def main():
    x = np.random.default_rng(2).uniform(-0.8, 0.8, (100, 3)).astype(np.float32)
    prm = MPMParams(dx=0.5, dt=0.005, nx=20, ny=20, nz=20,
                    grid_min=(-5., -5., -5.), drag=0., smoothing=1.)
    spec = RolloutSpec(x, 1., 400., 200., prm, 6, device='cuda:0', body_ctrl=True,
                       vol0=np.full(len(x), 0.008, np.float32))
    basis = BodyControlBasis(x, prm.grid_min, prm.dx, device='cuda:0')
    torch.manual_seed(17)
    coeff = (0.01 * torch.randn(basis.n_nodes, 3, device='cuda:0')).requires_grad_()
    dc = torch.zeros(spec.T, len(x), 3, 3, device='cuda:0')
    weight = torch.randn(len(x), 3, device='cuda:0')
    adj = PersistentAdjoint(spec)
    rows = []
    for scale in (1., -0.7, 1.):
        b = basis.expand(coeff * scale)
        xf = warp_mpm_ext(dc, spec, body_t=b)[0]
        gf, = torch.autograd.grad((xf * weight).sum(), coeff)
        xp = adj.apply(dc, body_t=basis.expand(coeff * scale))[0]
        gp, = torch.autograd.grad((xp * weight).sum(), coeff)
        row = dict(scale=scale, position_max=float((xf-xp).abs().max()),
                   grad_relative=float((gf-gp).norm()/gf.norm().clamp_min(1e-20)))
        assert row['position_max'] < 2e-5 and row['grad_relative'] < 2e-3, row
        rows.append(row)
    print(json.dumps(dict(status='PASS', T=6, dt=0.005, dx=0.5, n=100, candidates=rows)), flush=True)


if __name__ == '__main__':
    main()
