"""Explicit hyde06 adapter CUDA gate; N27,T20,dt.002,dx.5,grid16^3."""
import os

import pytest
import torch
import warp as wp

from physmorph.compute import cuda_execution, cuda_module, to_array
from physmorph.mpm.withdrawal import OwnedWithdrawal
from physmorph.pipeline.post_assimilation_window import PostAssimilationWindow
from physmorph.plasticity.assimilation import assimilate_elastic
from test_post_assimilation_window import make_case, future_velocity
from test_frozen_withdrawal_window_cuda import upload_owner
from test_withdrawal_adjoint_fd import direction_like
from test_withdrawal_cuda import NoDeviceToHostArray

pytestmark = pytest.mark.skipif(
    os.environ.get('PHYSMORPH_POST_WINDOW_CUDA_TEST') != '1' or not torch.cuda.is_available(),
    reason='Explicit hyde06 post-assimilation adapter gate')


def cpu_fixture():
    owner, saved, cfg = make_case()
    return owner, saved.arrays(), saved.metadata(), cfg


def gpu_case(fixture, *, slip=False):
    owner, saved_arrays, metadata, cfg = fixture
    owner.adjoint = None
    owner.spec.pin_slip = slip
    owner = upload_owner(owner)
    cfg.device, cfg.compute_backend = 'cuda:0', 'cuda'
    cfg.settle_pin_slip = slip
    arrays = {key: to_array(value, copy=True) for key, value in saved_arrays.items()}
    # Recompute the ordinary handoff from the actual GPU head, with production
    # subset assimilation. Source CPU simulation supplies only the fixture policies.
    with torch.no_grad():
        head = owner.evaluate(owner.coefficients[:, 3:])
        old = to_array(owner.spec.pin).astype(bool)
        pins = arrays['pin'].astype(bool)
        F = to_array(head['F'].reshape(-1, 3, 3))
        P = to_array(owner.spec.Fp)
        first = assimilate_elastic(F, P, eta=cfg.assim, isochoric=cfg.assim_iso)
        first[old] = P[old]
        first[pins & ~old] = assimilate_elastic(F[pins & ~old], first[pins & ~old], eta=1., isochoric=False)
        arrays.update(x0=to_array(head['x'], copy=True), F0=F.copy(),
            v0=to_array(head['v'], copy=True), C0=to_array(head['C'], copy=True), Fp=first)
        arrays['v0'][pins], arrays['C0'][pins] = 0., 0.
    metadata['device'] = metadata['captured_device'] = 'cuda:0'
    metadata['pin_mode'] = int(slip)
    # Match the native production recipe: physical F is captured; Fg is auxiliary.
    metadata['track_geom'] = False
    arrays.pop('Fg0')
    successor = OwnedWithdrawal.from_arrays(arrays, metadata, device='cuda:0')
    owner.adjoint = None
    return owner, successor, cfg


@pytest.mark.parametrize('mode', [0, 1], ids=['displacement', 'terminal'])
@pytest.mark.parametrize('slip', [False, True], ids=['no_slip', 'slip'])
def test_captured_reduced_modes_match_plain_and_fd(mode, slip, monkeypatch):
    fixture = cpu_fixture()
    with cuda_execution('cuda:0'):
        owner, successor, cfg = gpu_case(fixture, slip=slip)
        plain = PostAssimilationWindow(owner, successor, cfg, capture=False)
        model = PostAssimilationWindow(owner, successor, cfg)
        leaves = [owner.coefficients[:, :3].clone().requires_grad_(),
                  owner.coefficients[:, 3:].clone().requires_grad_()]
        def no_host(*_args, **_kwargs):
            pytest.fail('Post-assimilation adapter downloaded a numerical array')
        with monkeypatch.context() as patch, NoDeviceToHostArray():
            patch.setattr(cuda_module(), 'asnumpy', no_host)
            patch.setattr(wp.array, 'numpy', no_host)
            expected = plain.evaluate(leaves[1], leaves[0])
            reference = torch.autograd.grad(future_velocity(expected), leaves)
            actual = model.evaluate(leaves[1], leaves[0])
            gradients = torch.autograd.grad(future_velocity(actual), leaves)
        assert expected['valid'] and actual['valid']
        assert model.adjoint.boundary_forward_graph is not None
        assert model.adjoint.boundary_backward_graph is not None
        for key in ('positions', 'V', 'F', 'C', 'coast_X', 'coast_V', 'coast_F', 'coast_C', 'coast_Fp'):
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-5)
        assert torch.equal(actual['body_energy'], expected['body_energy'])
        for value, wanted in zip(gradients, reference):
            torch.testing.assert_close(value, wanted, rtol=2e-4, atol=2e-5)
            assert torch.isfinite(value).all() and value.norm() > 1e-5
        direction = direction_like(leaves[mode], mode)
        ad = float((gradients[mode].double()*direction.double()).sum())
        assert abs(ad) > 5e-5
        for epsilon in (1e-3, 5e-4):
            losses = []
            for sign in (-1, 1):
                trial = [value.detach().clone() for value in leaves]
                trial[mode].add_(direction, alpha=sign*epsilon)
                values = model.evaluate(trial[1], trial[0])
                assert values['valid']
                losses.append(float(future_velocity(values)))
            fd = (losses[1]-losses[0])/(2*epsilon)
            print(dict(mode=mode, slip=slip, epsilon=epsilon, analytical=ad, finite_difference=fd))
            assert ad == pytest.approx(fd, rel=.02, abs=5e-6)


def test_actual_gpu_handoff_and_independent_successor_coast():
    fixture = cpu_fixture()
    with cuda_execution('cuda:0'):
        owner, successor, cfg = gpu_case(fixture)
        model = PostAssimilationWindow(owner, successor, cfg)
        values = model.evaluate(owner.coefficients[:, 3:])
        assert values['valid'] and values['pins_exact']
        arrays = successor.arrays()
        torch.testing.assert_close(values['coast_Fp'], torch.as_tensor(arrays['Fp'], device=cfg.device),
                                   rtol=4e-6, atol=4e-6)
        pins = torch.as_tensor(arrays['pin'], device=cfg.device).bool()
        assert torch.equal(values['coast_pins'], pins)
        assert torch.equal(values['coast_X'][:, pins], values['x'][pins][None].expand(cfg.T+1, -1, -1))
        assert torch.count_nonzero(values['coast_V'][:, pins]) == 0
        assert torch.count_nonzero(values['coast_C'][:, pins]) == 0
        independent = successor.trajectory(persistent=True)
        independent.rollout()
        for name, key in (('x', 'coast_X'), ('v', 'coast_V'), ('C', 'coast_C'), ('F', 'coast_F')):
            expected = torch.stack([wp.to_torch(value).clone() for value in getattr(independent, name)])
            torch.testing.assert_close(values[key], expected.reshape_as(values[key]), rtol=2e-5, atol=3e-6)


def test_side_stream_and_expired_owner_cannot_backpropagate():
    fixture = cpu_fixture()
    with cuda_execution('cuda:0'):
        primary = torch.cuda.current_stream()
        warp_stream = wp.Stream('cuda:0')
        stream = wp.stream_to_torch(warp_stream)
        stream.wait_stream(primary)
        with torch.cuda.stream(stream), cuda_module().cuda.ExternalStream(stream.cuda_stream), wp.ScopedStream(warp_stream):
            owner, successor, cfg = gpu_case(fixture)
            model = PostAssimilationWindow(owner, successor, cfg)
            terminal = owner.coefficients[:, 3:].clone().requires_grad_()
            values = model.evaluate(terminal)
            gradient, = torch.autograd.grad(future_velocity(values), (terminal,), retain_graph=True)
            assert torch.isfinite(gradient).all() and gradient.norm() > 1e-5
            owner.close()
            with pytest.raises(RuntimeError, match='expired'):
                torch.autograd.grad(future_velocity(values), (terminal,))
        primary.wait_stream(stream)
