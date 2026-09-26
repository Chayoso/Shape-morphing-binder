"""Run on hyde06 only: CUDA graph candidate refresh and body-force adjoint parity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import argparse
import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree
from physmorph.mpm.function import RolloutSpec, PersistentAdjoint, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory
from physmorph.pipeline.body_control import BodyControlBasis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', type=int, choices=(1, 2), default=1)
    parser.add_argument('--reference-bonds', action='store_true')
    args = parser.parse_args()
    modes = args.modes
    x = np.random.default_rng(2).uniform(-0.8, 0.8, (100, 3)).astype(np.float32)
    bonds = {}
    if args.reference_bonds:
        nbr = cKDTree(x).query(x, k=9)[1][:, 1:].astype(np.int32)
        rest = np.linalg.norm(x[nbr] - x[:, None], axis=2).astype(np.float32)
        x[:3] = [[3., 0., 0.], [3.05, 0., 0.], [3., 0.05, 0.]]
        bonds = dict(bond_nbr=nbr, bond_rest=rest, bond_frag=np.zeros(len(x), np.float32),
                     bond_threshold=7.5)
    prm = MPMParams(dx=0.5, dt=0.005, nx=24, ny=24, nz=24,
                    grid_min=(-6., -6., -6.), drag=0., smoothing=1.)
    spec = RolloutSpec(x, 1., 400., 200., prm, 6, device='cuda:0', body_ctrl=True, body_modes=modes,
                       vol0=np.full(len(x), 0.008, np.float32), **bonds)
    basis = BodyControlBasis(x, prm.grid_min, prm.dx, device='cuda:0')
    torch.manual_seed(17)
    coeff = (0.01 * torch.randn(basis.n_nodes, 3 * modes, device='cuda:0')).requires_grad_()
    def field(scale):
        return basis.expand(coeff * scale).reshape(len(x), modes, 3).permute(1, 0, 2).reshape(-1, 3).contiguous()
    dc = torch.zeros(spec.T, len(x), 3, 3, device='cuda:0')
    weight = torch.randn(len(x), 3, device='cuda:0')
    adj = PersistentAdjoint(spec)
    body_buf = torch.zeros(modes * len(x), 3, device='cuda:0')
    candidate = Trajectory(x, spec.m, spec.lam, spec.mu, prm, spec.T, device=spec.device,
                           requires_grad=False, persistent=True, vol0=spec.vol0, bonds=spec.bonds(),
                           dFc=[wp.from_torch(dc[t], dtype=wp.mat33) for t in range(spec.T)],
                           body_control=wp.from_torch(body_buf, dtype=wp.vec3))
    candidate.capture()
    rows = []
    for scale in (1., -0.7, 1.):
        xf, _, vf, _, _ = warp_mpm_ext(dc, spec, body_t=field(scale))
        gf, = torch.autograd.grad((xf * weight).sum() + 0.01 * vf.square().sum(), coeff)
        xp, _, vp, _, _ = adj.apply(dc, body_t=field(scale))
        gp, = torch.autograd.grad((xp * weight).sum() + 0.01 * vp.square().sum(), coeff)
        body_buf.copy_(field(scale).detach())
        candidate.run()
        row = dict(scale=scale, position_max=float((xf-xp).abs().max()),
                   velocity_max=float((vf-vp).abs().max()),
                   candidate_position_max=float((xf-wp.to_torch(candidate.x[-1])).abs().max()),
                   candidate_velocity_max=float((vf-wp.to_torch(candidate.v[-1])).abs().max()),
                   grad_relative=float((gf-gp).norm()/gf.norm().clamp_min(1e-20)))
        assert row['position_max'] < 2e-5 and row['velocity_max'] < 2e-5 and row['grad_relative'] < 2e-3, row
        assert row['candidate_position_max'] < 2e-5 and row['candidate_velocity_max'] < 2e-5, row
        rows.append(row)
    print(json.dumps(dict(status='PASS', modes=modes, bond_threshold=spec.bond_threshold,
                          T=6, dt=0.005, dx=0.5, n=100, candidates=rows)), flush=True)


if __name__ == '__main__':
    main()
