"""Server-only reduced-body withdrawal stream, derivative and ownership gates."""
from dataclasses import fields

import pytest
import torch
import warp as wp

from physmorph.compute import cuda_execution, cuda_module, to_array
from physmorph.pipeline.frozen_withdrawal_window import FrozenWithdrawalWindow
from test_frozen_withdrawal_window import make_owner, coast_observable
from test_withdrawal_adjoint_fd import direction_like
from test_withdrawal_cuda import NoDeviceToHostArray


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='hyde06 CUDA checks')


def upload_owner(owner):
    def upload(value):
        if isinstance(value, tuple):
            return tuple(upload(item) for item in value)
        return to_array(value, copy=True) if hasattr(value, 'shape') else value
    for item in fields(owner.spec):
        setattr(owner.spec, item.name, upload(getattr(owner.spec, item.name)))
    owner.spec.device = 'cuda:0'
    for name in ('idx', 'weights', 'gate', 'coefficients', 'stress', 'surface_u'):
        value = getattr(owner, name)
        if value is not None:
            setattr(owner, name, value.detach().to('cuda:0'))
    return owner


@pytest.mark.parametrize('mode', [0, 1], ids=['displacement', 'terminal'])
def test_reduced_coast_captured_parity_and_directional_derivative(mode, monkeypatch):
    owner = make_owner()
    # Same smooth pinned branch as the preregistered CPU FD test.
    owner.spec.pin_slip = False
    with cuda_execution('cuda:0'):
        owner = upload_owner(owner)
        leaves = [owner.coefficients[:, :3].clone().requires_grad_(),
                  owner.coefficients[:, 3:].clone().requires_grad_()]
        def no_host(*_args, **_kwargs):
            pytest.fail('Prepared withdrawal numerical array downloaded to host')
        with monkeypatch.context() as patch, NoDeviceToHostArray():
            patch.setattr(cuda_module(), 'asnumpy', no_host)
            patch.setattr(wp.array, 'numpy', no_host)
            plain = FrozenWithdrawalWindow(owner, capture=False)
            graph = FrozenWithdrawalWindow(owner, capture=True)
            expected = plain.evaluate(leaves[1], leaves[0])
            reference = torch.autograd.grad(coast_observable(expected), leaves)
            actual = graph.evaluate(leaves[1], leaves[0])
            gradients = torch.autograd.grad(coast_observable(actual), leaves)
        assert graph.adjoint.g_fwd is not None and graph.adjoint.g_bwd is not None
        assert expected['valid'] and actual['valid'] and actual['pins_exact']
        for name in ('x', 'F', 'v', 'C', 'V', 'positions', 'F_sequence', 'Fg_sequence',
                     'C_sequence', 'coast_X', 'coast_V', 'coast_F', 'coast_Fg', 'coast_C'):
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual['body_energy'], expected['body_energy'], rtol=0, atol=0)
        for got, want in zip(gradients, reference):
            torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-5)
            assert torch.isfinite(got).all() and got.norm() > 1e-5
        direction = direction_like(leaves[mode], mode)
        analytical = (gradients[mode].double()*direction.double()).sum().item()
        assert abs(analytical) > 5e-5
        for epsilon in (1e-3, 5e-4):
            losses = []
            for sign in (-1., 1.):
                trial = [value.detach().clone() for value in leaves]
                trial[mode].add_(direction, alpha=sign*epsilon)
                candidate = graph.evaluate(trial[1], trial[0])
                assert candidate['valid']
                losses.append(coast_observable(candidate).item())
            observed = (losses[1]-losses[0])/(2*epsilon)
            print(dict(mode=mode, epsilon=epsilon, analytical=analytical, finite_difference=observed))
            assert analytical == pytest.approx(observed, rel=.02, abs=5e-6)
        # Archival state remains owned after the perturbed forwards.
        torch.testing.assert_close(actual['C_sequence'], expected['C_sequence'], rtol=1e-5, atol=1e-5)


def test_side_stream_and_callback_expiration_on_autograd_worker():
    owner = make_owner()
    with cuda_execution('cuda:0'):
        primary = torch.cuda.current_stream()
        warp_stream = wp.Stream('cuda:0')
        stream = wp.stream_to_torch(warp_stream)
        stream.wait_stream(primary)
        with torch.cuda.stream(stream), cuda_module().cuda.ExternalStream(stream.cuda_stream), wp.ScopedStream(warp_stream):
            owner = upload_owner(owner)
            model = FrozenWithdrawalWindow(owner, capture=True)
            terminal = owner.coefficients[:, 3:].clone().requires_grad_()
            values = model.evaluate(terminal)
            loss = coast_observable(values)+values['body_energy']
            first, = torch.autograd.grad(loss, terminal, retain_graph=True)
            repeated, = torch.autograd.grad(loss, terminal, retain_graph=True)
            torch.testing.assert_close(first, repeated, rtol=2e-4, atol=2e-5)
            assert torch.isfinite(first).all() and first.norm() > 1e-5
            original_values = values
            saved = values['coast_X'].detach().clone()
        primary.wait_stream(stream)
        with pytest.raises(RuntimeError, match='stream'):
            model.evaluate(terminal)
        with torch.cuda.stream(stream), cuda_module().cuda.ExternalStream(stream.cuda_stream), wp.ScopedStream(warp_stream):
            values = model.evaluate(terminal)
            owner.close()
            with pytest.raises(RuntimeError, match='expired'):
                torch.autograd.grad(coast_observable(values), terminal)
            with pytest.raises(RuntimeError, match='expired'):
                torch.autograd.grad(values['body_energy'], terminal)
            with pytest.raises(RuntimeError, match='expired'):
                model.evaluate(terminal)
            assert torch.equal(original_values['coast_X'], saved)
        primary.wait_stream(stream)


def test_constructor_and_evaluation_reject_unaligned_stream_before_owned_data_work(monkeypatch):
    import physmorph.pipeline.frozen_withdrawal_window as module
    owner = make_owner()
    with cuda_execution('cuda:0'):
        owner = upload_owner(owner)
        model = FrozenWithdrawalWindow(owner, capture=True)
        terminal = owner.coefficients[:, 3:].clone()
        assert model.evaluate(terminal)['valid']
        primary = torch.cuda.current_stream()
        other = torch.cuda.Stream()
        other.wait_stream(primary)
        def premature(*_args, **_kwargs):
            pytest.fail('Owned-data operation ran before stream validation')
        with torch.cuda.stream(other), monkeypatch.context() as patch:
            patch.setattr(module, 'deepcopy', premature)
            patch.setattr(torch, 'isfinite', premature)
            with pytest.raises(RuntimeError, match='stream'):
                FrozenWithdrawalWindow(owner, capture=True)
            with pytest.raises(RuntimeError, match='stream'):
                model.evaluate(terminal)
        primary.wait_stream(other)
