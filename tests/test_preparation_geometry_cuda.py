"""Explicit hyde06 preparation gates; never execute on the local GPU.

Geometry: bonds N38/dx.5/grid24x16x16; layer N128/T20/reference N64.
Passive MPM: N27/T20/dt.002/dx.5/grid16^3, nonzero incoming v/C/F/Fp,
fixed pins and bonds, active layer relaxation, zero learned controls, layer_F off.
These are policy/forward invariants, not a full-N timing, adjoint or rest gate.
"""
from contextlib import contextmanager
from copy import deepcopy
import os

import numpy as np
import pytest
import torch
import warp as wp

import physmorph.compute as compute
from physmorph.compute import cuda_execution, cuda_module, to_array
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory
from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.preparation_geometry import prepare_bonds, prepare_layer_geometry
from scripts.probes.boundary_packet import capture_step
from test_boundary_packet import _trajectory
from test_withdrawal_cuda import NoDeviceToHostArray


pytestmark = pytest.mark.skipif(
    os.environ.get('PHYSMORPH_PREPARATION_CUDA_TEST') != '1' or not torch.cuda.is_available(),
    reason='Explicit hyde06 preparation CUDA gate')


@contextmanager
def device_only(monkeypatch):
    """Reject public numerical-array download/fallback paths; scalars are allowed."""
    from physmorph.render import knn_gpu
    def forbidden(*args, **kwargs):
        raise AssertionError('Preparation downloaded a numerical array or used CPU neighbors')
    with monkeypatch.context() as guard, NoDeviceToHostArray():
        guard.setattr(cuda_module(), 'asnumpy', forbidden)
        guard.setattr(compute, 'to_host', forbidden)
        guard.setattr(wp.array, 'numpy', forbidden)
        guard.setattr(knn_gpu, '_cpu_knn', forbidden)
        yield


def tensor(array):
    assert hasattr(array, '__cuda_array_interface__')
    result = torch.as_tensor(array, device='cuda:0')
    assert result.is_cuda
    return result


def test_cuda_bond_history_is_retained_until_actual_reconnection(monkeypatch):
    # Same known topology as the CPU policy fixture, not a copy of the algorithm.
    cloud = np.array([[i*.2, j*.2, k*.2] for i in range(4) for j in range(3)
                      for k in range(3)] + [[6., 0., 0.], [6.1, 0., 0.]], np.float32)
    neighbors = ((np.arange(len(cloud))+1) % len(cloud))[:, None].astype(np.int32)
    prm = MPMParams(dx=.5, grid_min=(-2., -2., -2.), nx=24, ny=16, nz=16)
    with cuda_execution('cuda:0'):
        cp = cuda_module()
        x, nbr = to_array(cloud), to_array(neighbors)
        history = cp.full((len(x), 1), 123., cp.float32)
        original = history.copy()
        x_before = x.copy()
        with device_only(monkeypatch):
            rest, fragmented = prepare_bonds(x, nbr, history, prm)
            assert tensor(fragmented)[-2:].all() and not tensor(fragmented)[:-2].any()
            assert torch.equal(tensor(rest)[-2:], tensor(history)[-2:])
            independent = (tensor(x)[tensor(nbr).long()]-tensor(x)[:, None]).norm(dim=2)
            torch.testing.assert_close(tensor(rest)[:-2], independent[:-2], rtol=1e-6, atol=1e-6)
            assert torch.equal(tensor(history), tensor(original))
            assert torch.equal(tensor(x), tensor(x_before))
            detached_rest = rest.copy()
            rejoined = x.copy()
            rejoined[-2:, 0] -= cp.float32(5.)
            refreshed, rejoined_mask = prepare_bonds(rejoined, nbr, rest, prm)
            initialized, initial_mask = prepare_bonds(rejoined, nbr, None, prm)
            assert not tensor(rejoined_mask).any() and not tensor(initial_mask).any()
            assert torch.equal(tensor(initialized), tensor(refreshed))
            assert torch.equal(tensor(rest), tensor(detached_rest))
            assert (tensor(refreshed)[-2:] < 123.).all()


@pytest.mark.parametrize('relax,control,fraction', [(True, False, 0.), (False, True, 0.), (True, True, .2)])
def test_cuda_layer_reference_count_and_translation(monkeypatch, relax, control, fraction):
    # Irregular fixture avoids nearest-neighbor tie ambiguity. The dense Torch
    # spacing oracle does not invoke the helper's KDTree or layer implementation.
    cloud = np.random.default_rng(42).uniform(-1., 1., (128, 3)).astype(np.float32)
    cfg = PipelineConfig(device='cuda:0', compute_backend='cuda', T=20,
        layer_relax=relax, layer_ctrl=control, layer_k=8, layer_frac=fraction,
        disc_ref=True, mass_ref_n=64)
    before_cfg = deepcopy(cfg)
    with cuda_execution('cuda:0', input_sizes=(len(cloud),)):
        x = to_array(cloud)
        original = x.copy()
        shift = to_array(np.array([1., -2., .5], np.float32))
        with device_only(monkeypatch):
            layer, spacing = prepare_layer_geometry(x, cfg)
            translated, shifted_spacing = prepare_layer_geometry(x+shift, cfg)
            assert cfg == before_cfg and torch.equal(tensor(x), tensor(original))
            dense = torch.cdist(tensor(x).double(), tensor(x).double())
            independent_spacing = torch.quantile(dense.sort(dim=1).values[:, 8], .5) * (128/64)**(1/3)
            assert spacing == pytest.approx(float(independent_spacing), rel=1e-6)
            assert shifted_spacing == pytest.approx(spacing, rel=1e-6)
            mask, normal, nbr, weight = [tensor(value) for value in layer[:4]]
            assert nbr.shape == weight.shape == (128, 16)
            assert normal.shape == (128, 3) and mask.shape == (128,)
            assert mask.dtype == normal.dtype == weight.dtype == torch.float32
            assert nbr.dtype == torch.int32 and ((nbr >= 0) & (nbr < 128)).all()
            assert torch.equal(nbr, tensor(translated[2]))
            for index in (0, 1, 3):
                torch.testing.assert_close(tensor(layer[index]), tensor(translated[index]), rtol=2e-5, atol=2e-6)
            assert mask.bool().sum() > 16 and torch.count_nonzero(weight) > 0
            assert torch.isfinite(normal).all() and torch.isfinite(weight).all() and (weight >= 0).all()
            torch.testing.assert_close(normal[mask.bool()].norm(dim=1),
                torch.ones_like(mask[mask.bool()]), rtol=2e-5, atol=2e-6)
            nonzero = weight.sum(1) > 0
            torch.testing.assert_close(weight[nonzero].sum(1), torch.ones_like(mask[nonzero]), rtol=2e-5, atol=2e-6)
            assert torch.count_nonzero(weight[~mask.bool()]) == 0
            assert layer[4] == ((fraction or 1/cfg.T) if relax else 0.)
            assert prepare_layer_geometry(None, PipelineConfig(layer_relax=False, layer_ctrl=False)) == (None, None)


def passive_trajectory(arrays, metadata, gate):
    """Reconstruct full nonzero fixture state; ug is the sole changing input."""
    prm = MPMParams(**metadata['prm'])
    prm.gate_n0 = metadata['gate_n0']
    return Trajectory(arrays['x0'], arrays['m'], arrays['lam'], arrays['mu'], prm, metadata['T'],
        Fp=arrays['Fp'], F0=arrays['F0'], v0=arrays['v0'], C0=arrays['C0'],
        Fg0=arrays['Fg0'], track_geom=True, eta=arrays['eta'], pin=arrays['pin'], pin_slip=True,
        vol0=arrays['vol'], device='cuda:0', requires_grad=False, persistent=True,
        # None creates zero dFc and u; body is absent. The 8-field tuple carries
        # ug, but None gradient data explicitly disables layer_F.
        layer=(arrays['layer_mask'], arrays['layer_nrm'], arrays['layer_nbr'], arrays['layer_w'],
               metadata['layer_frac'], None, 0., gate),
        bonds=(arrays['bond_nbr'], arrays['bond_rest'], arrays['bond_frag'], metadata['frag_thr']))


def test_cuda_zero_control_passive_path_is_exactly_independent_of_layer_u_gate(monkeypatch):
    # CPU fixture construction is input preparation only; every tested trajectory
    # and all comparisons below execute on CUDA under the shared stream context.
    packet = capture_step(_trajectory(pin_slip=True))
    assert packet['meta']['T'] == 20
    with cuda_execution('cuda:0'):
        cp = cuda_module()
        arrays = {name: to_array(value, copy=True) for name, value in packet['arrays'].items()}
        n = len(arrays['x0'])
        assert n == 27
        gates = (cp.zeros(n, cp.float32), cp.ones(n, cp.float32), cp.linspace(.1, .9, n, dtype=cp.float32))
        with device_only(monkeypatch):
            trajectories = [passive_trajectory(arrays, packet['meta'], gate) for gate in gates]
            paths = []
            assert all(torch.isfinite(tensor(gate)).all() for gate in gates)
            assert all(not torch.equal(tensor(gates[i]), tensor(gates[j])) for i in range(3) for j in range(i))
            for tr, gate in zip(trajectories, gates):
                assert torch.equal(wp.to_torch(tr.layer_ug), tensor(gate))
                assert tr.layer and not tr.layer_F and tr.body_control is None
                assert tr.T == 20 and tr.prm.dt == .002 and tr.prm.dx == .5
                assert torch.count_nonzero(wp.to_torch(tr.layer_u)) == 0
                assert torch.count_nonzero(wp.to_torch(tr._dfc(0))) == 0
                tr.run()
                paths.append({key: torch.stack([wp.to_torch(value) for value in getattr(tr, field)])
                              for key, field in (('X', 'x'), ('V', 'v'), ('F', 'F'), ('C', 'C'))})
            for actual in paths[1:]:
                for key in paths[0]:
                    assert torch.equal(actual[key], paths[0][key]), key
            # Demonstrate active dynamics and layer projection, not an all-zero
            # state that would make the ug-invariance assertion vacuous.
            original = paths[0]
            pins = tensor(arrays['pin']).bool()
            assert torch.count_nonzero(original['V'][0, ~pins]) > 0
            assert torch.count_nonzero(original['C'][0, ~pins]) > 0
            assert not torch.equal(tensor(arrays['F0']), tensor(arrays['Fp']))
            assert (original['X'][1:, ~pins]-original['X'][:1, ~pins]).abs().max() > 1e-6
            projected = torch.stack([wp.to_torch(value) for value in trajectories[0].xu[1:]])
            assert (original['X'][1:]-projected).abs().max() > 1e-6
            assert all(torch.isfinite(value).all() for value in original.values())
            assert torch.equal(original['X'][:, pins], original['X'][:1, pins].expand_as(original['X'][:, pins]))
            assert torch.count_nonzero(original['V'][1:, pins]) == torch.count_nonzero(original['C'][1:, pins]) == 0
