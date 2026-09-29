"""Server-only joint head/coast derivative and CUDA graph/stream gates."""
from dataclasses import fields

import pytest
import torch
import warp as wp

from physmorph.compute import cuda_execution, cuda_module, to_array
from physmorph.mpm.withdrawal_adjoint import WithdrawalAdjoint
from test_withdrawal_adjoint_fd import coupled_case, mixed_coast_loss, direction_like
from test_withdrawal_cuda import NoDeviceToHostArray


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='hyde06 CUDA checks')


def upload(spec, controls):
    def convert(value):
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        return to_array(value, copy=True) if hasattr(value, 'shape') else value
    for field in fields(spec):
        setattr(spec, field.name, convert(getattr(spec, field.name)))
    spec.device = 'cuda:0'
    return spec, tuple(value.detach().to('cuda:0').requires_grad_() for value in controls)


@pytest.mark.parametrize('channel', [0, 1, 2], ids=['stress', 'surface_u', 'body'])
def test_captured_joint_derivative_matches_uncaptured_and_finite_difference(channel, monkeypatch):
    spec, controls = coupled_case()
    with cuda_execution('cuda:0'):
        spec, controls = upload(spec, controls)
        plain = WithdrawalAdjoint(spec, capture=False)
        graph = WithdrawalAdjoint(spec, capture=True)
        assert graph.g_fwd is not None and graph.g_bwd is not None
        def no_host(*_args, **_kwargs):
            pytest.fail('Numerical withdrawal path copied through host memory')
        with monkeypatch.context() as patch, NoDeviceToHostArray():
            patch.setattr(cuda_module(), 'asnumpy', no_host)
            patch.setattr(wp.array, 'numpy', no_host)
            expected = plain.apply(*controls)
            reference = torch.autograd.grad(mixed_coast_loss(expected), controls)
            actual = graph.apply(*controls)
            gradients = torch.autograd.grad(mixed_coast_loss(actual), controls)
        # FP32 atomic accumulation; fixed before the first server run.
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        for got, want in zip(gradients, reference):
            torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-5)
            assert torch.isfinite(got).all() and got.norm() > 1e-7
        direction = direction_like(controls[channel], channel)
        analytical = (gradients[channel].double() * direction.double()).sum().item()
        assert abs(analytical) > 5e-5
        radii = (1e-4, 5e-5) if channel == 2 else (1e-3, 5e-4)
        for epsilon in radii:
            losses = []
            for sign in (-1., 1.):
                shifted = [value.detach().clone() for value in controls]
                shifted[channel].add_(direction, alpha=sign*epsilon)
                losses.append(mixed_coast_loss(graph.apply(*shifted)).item())
            observed = (losses[1]-losses[0])/(2*epsilon)
            print(dict(channel=channel, epsilon=epsilon, analytical=analytical, finite_difference=observed))
            assert analytical == pytest.approx(observed, rel=.02, abs=5e-6)


def test_captured_seeds_reuse_and_torch_side_stream_binding():
    spec, controls = coupled_case()
    with cuda_execution('cuda:0'):
        primary = torch.cuda.current_stream()
        # Align all libraries BEFORE any asynchronous fixed-input upload.
        warp_stream = wp.Stream('cuda:0')
        stream = wp.stream_to_torch(warp_stream)
        stream.wait_stream(primary)
        with torch.cuda.stream(stream), cuda_module().cuda.ExternalStream(stream.cuda_stream), wp.ScopedStream(warp_stream):
            spec, controls = upload(spec, controls)
            model = WithdrawalAdjoint(spec, capture=True)
            out = model.apply(*controls)
            multi = (.25*out.x.sum() + .25*out.X[-1].sum() + .5*out.coast_X[0].sum()
                     + .25*out.v.sum() + .5*out.V[-1].sum() + out.coast_V[0].sum()
                     + .75*out.F.sum() + out.coast_F[0].sum()
                     + out.Fg.sum() + 1.25*out.coast_Fg[0].sum())
            a = torch.autograd.grad(multi, controls, retain_graph=True)
            single = out.x.sum() + 1.75*out.v.sum() + 1.75*out.F.sum() + 2.25*out.Fg.sum()
            b = torch.autograd.grad(single, controls, retain_graph=True)
            c = torch.autograd.grad(single, controls, retain_graph=True)
            for first, second, third in zip(a, b, c):
                torch.testing.assert_close(first, second, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(second, third, rtol=2e-4, atol=2e-5)
            zero, = torch.autograd.grad(out.coast_F.sum()*0, controls[:1])
            assert torch.count_nonzero(zero) == 0
            owned = out.coast_X.detach().clone()
            # Failed validation invalidates old contexts before touching buffers.
            previous = model.apply(*controls)
            with pytest.raises(ValueError, match='float32'):
                model.apply(controls[0].double(), *controls[1:])
            with pytest.raises(RuntimeError, match='stale'):
                torch.autograd.grad(previous.coast_X.sum(), controls[0])
            model.apply(*(v.detach()*0 for v in controls))
            assert torch.equal(out.coast_X.detach(), owned)
        primary.wait_stream(stream)
        with pytest.raises(RuntimeError, match='constructor CUDA device/stream'):
            model.apply(*controls)
        # Direct backward obeys the same stream binding; autograd itself may
        # schedule a backward on the original forward stream.
        with pytest.raises(RuntimeError, match='constructor CUDA device/stream'):
            model.backward()


def test_constructor_rejects_unaligned_library_streams():
    spec, controls = coupled_case()
    with cuda_execution('cuda:0'):
        spec, _ = upload(spec, controls)
        primary = torch.cuda.current_stream()
        other = torch.cuda.Stream()
        other.wait_stream(primary)
        with torch.cuda.stream(other):
            with pytest.raises(RuntimeError, match='stream'):
                WithdrawalAdjoint(spec, capture=True)
        primary.wait_stream(other)


def test_changing_fragment_activity_in_production_captured_adjoint():
    from physmorph.mpm.function import PersistentAdjoint
    from test_fragment_activity_adjoint import changing_activity_case, activity, observation, body_direction
    spec, dc, body = changing_activity_case()
    with cuda_execution('cuda:0'):
        spec, (dc, body) = upload(spec, (dc, body))
        adj = PersistentAdjoint(spec, position_sequence=True)
        assert adj.g_fwd is not None and adj.g_bwd is not None
        out = adj.apply_with_positions(dc, body_t=body)
        masks = activity(adj.traj)
        assert torch.equal(masks[:, 0], torch.tensor([1., 1., 1., 0., 0., 0., 0., 0.], device='cuda:0'))
        assert len({a.ptr for a in adj.traj.frag_steps}) == spec.T
        gradient, = torch.autograd.grad(observation(out), body)
        direction = body_direction(body)
        analytical = (gradient.double()*direction.double()).sum().item()
        assert analytical < -1e-3
        for epsilon in (1e-3, 5e-4):
            losses = []
            for sign in (-1., 1.):
                value = adj.apply_with_positions(dc, body_t=body.detach()+sign*epsilon*direction)
                assert torch.equal(activity(adj.traj), masks)
                losses.append(observation(value).item())
            observed = (losses[1]-losses[0])/(2*epsilon)
            print(dict(kind='changing_fragment', epsilon=epsilon, analytical=analytical, finite_difference=observed))
            assert abs(analytical-observed) <= max(.02*abs(observed), 2e-5)
