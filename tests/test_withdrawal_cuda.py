"""Server-only independent reconstruction and captured withdrawal forward gate."""
from copy import deepcopy

import numpy as np
import pytest
import torch
import warp as wp
from torch.utils._python_dispatch import TorchDispatchMode

from physmorph.compute import cuda_execution, to_array, warp_array
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory
from physmorph.mpm.withdrawal import OwnedWithdrawal
from scripts.probes.boundary_packet import capture_step
from test_boundary_packet import _trajectory


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='hyde06 CUDA checks')


def build_cuda_head(packet):
    """Fixture reconstruction independent of the withdrawal helper."""
    a = {name: to_array(value, copy=True) for name, value in packet['arrays'].items()}
    m = packet['meta']
    prm = MPMParams(**m['prm'])
    prm.gate_n0 = m['gate_n0']
    def w(value, dtype):
        return warp_array(value, dtype, 'cuda:0')
    return Trajectory(a['x0'], a['m'], a['lam'], a['mu'], prm, m['T'],
        Fp=a['Fp'], v0=a['v0'], C0=a['C0'], F0=a['F0'], Fg0=a['Fg0'], track_geom=True,
        eta=a['eta'], pin=a['pin'], pin_slip=True, vol0=a['vol'],
        dFc=[w(a['dFc']+np.float32(t*.0001), wp.mat33) for t in range(m['T'])],
        layer=(a['layer_mask'], a['layer_nrm'], a['layer_nbr'], a['layer_w'], m['layer_frac'],
               a['layer_g'], m['layer_depth'], a['layer_ug']),
        layer_u=w(a['layer_u'], wp.float32), body_control=w(a['body_control'], wp.vec3),
        bonds=(a['bond_nbr'], a['bond_rest'], a['bond_frag'], m['frag_thr']),
        device='cuda:0', requires_grad=False, persistent=True)


class NoDeviceToHostArray(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (func == torch.ops.aten._to_copy.default and args[0].is_cuda
                and torch.device(kwargs.get('device', args[0].device)).type == 'cpu'):
            raise AssertionError('Withdrawal state went through host memory')
        return func(*args, **kwargs)


def test_owned_withdrawal_cuda_independent_oracle_capture_and_archive_upload():
    # Input fixture/archival I/O occurs before the device-only numerical scope.
    cpu = _trajectory(pin_slip=True)
    packet = capture_step(cpu)
    with cuda_execution('cuda:0'):
        head = build_cuda_head(packet)
        head.run()
        head.layer_ug.fill_(.37)
        direct = build_cuda_head(packet)
        for name in ('x', 'v', 'C', 'F', 'Fg'):
            wp.copy(getattr(direct, name)[0], getattr(head, name)[20])
        for name in ('Fp', 'm', 'lam', 'mu', 'eta', 'pin', 'vol', 'layer_mask', 'layer_nrm',
                     'layer_nbr', 'layer_w', 'layer_ug', 'layer_g', 'bond_nbr', 'bond_rest', 'bond_frag'):
            wp.copy(getattr(direct, name), getattr(head, name))
        direct.prm = deepcopy(head.prm)
        direct.gate_n0 = head.gate_n0
        for control in direct.dFc_seq:
            control.zero_()
        direct.layer_u.zero_()
        direct.body_control = None
        with NoDeviceToHostArray():
            snapshot = OwnedWithdrawal.capture(head, 20)
            arrays = snapshot.arrays()
            assert all(hasattr(value, '__cuda_array_interface__') for value in arrays.values())
            fresh = snapshot.trajectory()
            captured = snapshot.trajectory(persistent=True)
        originals = {name: wp.to_torch(getattr(head, name)[20]).clone() for name in ('x', 'v', 'C', 'F', 'Fg')}
        direct.run()
        fresh.run()
        assert captured.capture()
        captured.run()
        # Fixed before execution: 64 float32 eps times each oracle field's scale.
        # This is forward accumulation parity, not an adjoint or natural-rest gate.
        relative = 64*torch.finfo(torch.float32).eps
        for name in ('x', 'v', 'C', 'F', 'Fg'):
            expected = torch.stack([wp.to_torch(value) for value in getattr(direct, name)])
            allowance = relative*max(1., float(expected.abs().max()))
            for replay in (fresh, captured):
                actual = torch.stack([wp.to_torch(value) for value in getattr(replay, name)])
                torch.testing.assert_close(actual, expected, rtol=relative, atol=allowance)
            assert torch.equal(wp.to_torch(getattr(head, name)[20]), originals[name])
        assert captured.T == 20 and captured.body_control is None
        assert captured.gate_n0 == head.gate_n0
        pins = wp.to_torch(captured.pin) > .5
        assert torch.equal(wp.to_torch(captured.x[20])[pins], wp.to_torch(captured.x[0])[pins])
        assert torch.count_nonzero(wp.to_torch(captured.layer_u)) == 0
        wp.to_torch(fresh.Fp).zero_()
        assert not torch.equal(wp.to_torch(captured.Fp), wp.to_torch(fresh.Fp))
    # Capture CPU arrays outside cuda_execution; mixing contexts must be rejected.
    cpu_snapshot = OwnedWithdrawal.capture(cpu, 0)
    cpu_arrays, cpu_meta = cpu_snapshot.arrays(), cpu_snapshot.metadata()
    with cuda_execution('cuda:0'):
        # Loading a CPU archival mapping is an explicit upload, not CPU physics.
        uploaded = OwnedWithdrawal.from_arrays(cpu_arrays, cpu_meta, device='cuda:0')
        restored = uploaded.trajectory()
        assert uploaded.metadata()['captured_device'] == 'cpu'
        assert uploaded.metadata()['device'] == 'cuda:0'
        torch.testing.assert_close(wp.to_torch(restored.Fp), torch.as_tensor(cpu_arrays['Fp'], device='cuda:0'), rtol=0, atol=0)
