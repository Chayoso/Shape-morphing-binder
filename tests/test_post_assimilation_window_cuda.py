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
from pin_collider_reference import first_step_pin_velocity

pytestmark = pytest.mark.skipif(
    os.environ.get('PHYSMORPH_POST_WINDOW_CUDA_TEST') != '1' or not torch.cuda.is_available(),
    reason='Explicit hyde06 post-assimilation adapter gate')


def cpu_fixture(fp64=False):
    owner, saved, cfg = make_case(fp64=fp64)
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
        first = assimilate_elastic(F, P, eta=cfg.assim, isochoric=cfg.assim_iso, fp64=cfg.assim_fp64)
        first[old] = P[old]
        first[pins & ~old] = assimilate_elastic(F[pins & ~old], first[pins & ~old], eta=1., isochoric=False,
                                               fp64=cfg.assim_fp64)
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
@pytest.mark.parametrize('fp64', [False, True])
def test_captured_reduced_modes_match_plain_and_fd(mode, slip, fp64, monkeypatch):
    fixture = cpu_fixture(fp64)
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
        # Slip brackets refined after the preserved p335_cuda1/CPU radius audit;
        # this is a local FD gate, not a claim that the large stencil is smooth.
        radii = (1e-4, 5e-5) if slip else (1e-3, 5e-4)
        for epsilon in radii:
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


@pytest.mark.parametrize('fp64', [False, True])
def test_actual_gpu_handoff_and_independent_successor_coast(fp64):
    fixture = cpu_fixture(fp64)
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


@pytest.mark.parametrize('fp64', [False, True])
def test_side_stream_and_expired_owner_cannot_backpropagate(fp64):
    fixture = cpu_fixture(fp64)
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


def test_captured_new_pin_anchor_pullback_and_detached_mass_negative():
    fixture = cpu_fixture()
    with cuda_execution('cuda:0'):
        owner, successor, cfg = gpu_case(fixture, slip=True)
        model = PostAssimilationWindow(owner, successor, cfg)
        broken = PostAssimilationWindow(owner, successor, cfg, capture=False)
        broken.adjoint = broken._make_adjoint()
        broken.adjoint.coast.gmpin = wp.clone(broken.adjoint.coast.gmpin, requires_grad=False)
        broken.adjoint._configure_seeds()
        broken.adjoint.capture_enabled = True
        with broken.adjoint._scope():
            broken.adjoint._capture()
        leaves = [owner.coefficients[:, :3].clone().requires_grad_(),
                  owner.coefficients[:, 3:].clone().requires_grad_()]
        actual = model.evaluate(leaves[1], leaves[0])
        incomplete = broken.evaluate(leaves[1], leaves[0])
        assert actual['valid'] and incomplete['valid']
        for key in ('positions', 'V', 'coast_X', 'coast_V', 'coast_F', 'coast_C', 'coast_Fp'):
            torch.testing.assert_close(actual[key], incomplete[key], rtol=1e-5, atol=1e-5)
        weights = (torch.arange(len(owner.idx)*3, dtype=torch.float64, device=cfg.device).reshape(-1, 3)*.43+.2).sin()
        def acceleration(value):
            return ((value['coast_V'][1].double()-value['coast_V'][0].double())*weights).mean()/owner.spec.prm.dt
        torch.autograd.grad(acceleration(actual), leaves, retain_graph=True)
        full = wp.to_torch(model.adjoint.coast.x[0].grad).clone()
        mass_covector = wp.to_torch(model.adjoint.coast.gmpin.grad).clone()
        assert mass_covector.norm() > 1e-7
        torch.autograd.grad(acceleration(actual), leaves, retain_graph=True)
        torch.testing.assert_close(wp.to_torch(model.adjoint.coast.gmpin.grad), mass_covector,
                                   rtol=2e-4, atol=2e-5)
        torch.autograd.grad(actual['coast_V'].sum()*0, leaves)
        assert torch.count_nonzero(wp.to_torch(model.adjoint.coast.gmpin.grad)) == 0
        torch.autograd.grad(acceleration(incomplete), leaves)
        missing = wp.to_torch(broken.adjoint.coast.x[0].grad).clone()
        direction = torch.zeros_like(actual['x'])
        direction[3] = direction.new_tensor([.3, -.7, .5])
        direction /= direction.norm()
        ad = float((full.double()*direction.double()).sum())
        bad = float((missing.double()*direction.double()).sum())
        assert abs(ad) > 5e-5 and bad == 0.
        independent = successor.trajectory(persistent=True)
        x0 = wp.to_torch(independent.x[0]).clone()
        v0 = wp.to_torch(independent.v[0]).clone()
        for name, key in (('x', 'coast_X'), ('v', 'coast_V'), ('C', 'coast_C'), ('F', 'coast_F')):
            expected = wp.to_torch(getattr(independent, name)[0]).reshape_as(actual[key][0])
            torch.testing.assert_close(actual[key][0], expected, rtol=2e-5, atol=3e-6)
        torch.testing.assert_close(actual['coast_Fp'], wp.to_torch(independent.Fp),
                                   rtol=4e-6, atol=4e-6)
        independent.step(0)
        torch.testing.assert_close(actual['coast_V'][1], wp.to_torch(independent.v[1]),
                                   rtol=2e-5, atol=3e-6)
        # The raw FP32 witness failed in p335_cuda2 and remains archived. This
        # separate reference computes only the collider path in FP64 on GPU.
        mass = wp.to_torch(independent.m).clone()
        grid_mass = wp.to_torch(independent.gm[0]).clone()
        grid_momentum = wp.to_torch(independent.gmom[0]).clone()
        def reference(x):
            return first_step_pin_velocity(x, actual['coast_pins'], mass,
                                           grid_mass, grid_momentum, owner.spec.prm)
        def reference_loss(value):
            return ((value['velocity']-v0.double())*weights).mean()/owner.spec.prm.dt
        base_x = x0.double().requires_grad_()
        base = reference(base_x)
        for key, source in (('pin_mass', independent.gmpin), ('grid_velocity', independent.gvel[0]),
                            ('velocity', independent.v[1])):
            torch.testing.assert_close(base[key], wp.to_torch(source).double(), rtol=2e-5, atol=3e-6)
        reference_gradient, = torch.autograd.grad(reference_loss(base), base_x)
        reference_ad = float((reference_gradient*direction.double()).sum())
        assert ad == pytest.approx(reference_ad, rel=.02, abs=5e-6)
        radii, differences = (1e-4, 5e-5), []
        for epsilon in radii:
            losses = []
            for sign in (-1, 1):
                x = x0.double()+sign*epsilon*direction.double()
                value = reference(x)
                for key, branch in base['branches'].items():
                    assert torch.equal(value['branches'][key], branch), (epsilon, sign, key)
                losses.append(float(reference_loss(value)))
                wp.to_torch(independent.x[0]).copy_(x.float())
                independent.step(0)
                # GPU atomic summation need not reproduce bit-identical grids;
                # anchor changes themselves must not change free P2G inputs.
                torch.testing.assert_close(wp.to_torch(independent.gm[0]), grid_mass, rtol=2e-5, atol=3e-6)
                torch.testing.assert_close(wp.to_torch(independent.gmom[0]), grid_momentum, rtol=2e-5, atol=3e-6)
            fd = (losses[1]-losses[0])/(2*epsilon)
            allowance = max(.02*abs(fd), 5e-6)
            print(dict(scope='captured_fp64_fixed_successor_anchor', epsilon=epsilon,
                       full=ad, reference_ad=reference_ad, detached_gmpin=bad, fd=fd))
            assert abs(ad-fd) <= allowance and abs(bad-fd) > allowance
            differences.append(fd)
        rounding_floor = 64*torch.finfo(torch.float64).eps*max(1., abs(float(reference_loss(base).detach())))/min(radii)
        assert abs(differences[-1]-differences[-2]) <= max(1e-6*abs(differences[-1]), rounding_floor)
