"""Opt-in CUDA gates for the fixed-policy head/assimilation/coast derivative.

N=27, T=20, dt=.002, dx=.5, grid=16^3. These tests do not differentiate
admission or successor preparation and do not certify physical quality.
"""
from copy import deepcopy
import os

import pytest
import torch
import warp as wp

from physmorph.compute import cuda_execution, cuda_module, to_array
from physmorph.mpm.post_assimilation_adjoint import PostAssimilationAdjoint
from physmorph.mpm.traj import Trajectory
from physmorph.plasticity.assimilation import assimilate_elastic
from test_post_assimilation_adjoint import case
from test_withdrawal_adjoint_cuda import upload
from test_withdrawal_adjoint_fd import direction_like, mixed_coast_loss
from test_withdrawal_cuda import NoDeviceToHostArray


pytestmark = pytest.mark.skipif(
    os.environ.get('PHYSMORPH_POST_ASSIM_CUDA_TEST') != '1' or not torch.cuda.is_available(),
    reason='Requires explicit PHYSMORPH_POST_ASSIM_CUDA_TEST=1 on hyde06')


def upload_case(spec, controls, kwargs):
    """Upload all fixed policies only after all three libraries share a stream."""
    stream = torch.cuda.current_stream().cuda_stream
    assert stream == cuda_module().cuda.get_current_stream().ptr
    assert stream == wp.get_stream('cuda:0').cuda_stream
    spec, controls = upload(spec, controls)
    def convert(value):
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        return to_array(value, copy=True) if hasattr(value, 'shape') else value
    kwargs = dict(kwargs)
    kwargs['next_pins'] = kwargs['next_pins'].to('cuda:0').clone()
    for key in ('successor_layer', 'successor_bonds', 'successor_eta'):
        kwargs[key] = convert(kwargs[key])
    return spec, controls, kwargs


def model_for(spec, kwargs, *, capture=True):
    model = PostAssimilationAdjoint(spec, **dict(kwargs, capture=capture))
    if capture:
        for name in ('head_forward_graph', 'coast_forward_graph', 'head_backward_graph',
                     'coast_backward_graph', 'boundary_forward_graph', 'boundary_backward_graph'):
            assert getattr(model, name) is not None, name
    return model


def no_host(*_args, **_kwargs):
    pytest.fail('Numerical post-assimilation path downloaded an array')


def tensor(array):
    return wp.to_torch(array).detach().clone()


@pytest.mark.parametrize('channel', [0, 1, 2], ids=['stress', 'surface_u', 'body'])
def test_captured_forward_vjp_and_all_control_finite_differences(channel, monkeypatch):
    fixture = case()
    with cuda_execution('cuda:0'):
        spec, controls, kwargs = upload_case(*fixture)
        plain, graph = model_for(spec, kwargs, capture=False), model_for(spec, kwargs)
        with monkeypatch.context() as patch, NoDeviceToHostArray():
            patch.setattr(cuda_module(), 'asnumpy', no_host)
            patch.setattr(wp.array, 'numpy', no_host)
            expected = plain.apply(*controls)
            reference = torch.autograd.grad(mixed_coast_loss(expected), controls)
            actual = graph.apply(*controls)
            gradients = torch.autograd.grad(mixed_coast_loss(actual), controls)
        # Existing FP32 Warp atomic parity bounds; FD remains independently strict.
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        for got, want in zip(gradients, reference):
            torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-5)
            assert torch.isfinite(got).all() and float(got.norm()) > 1e-7
        # Future-only covectors actually reach both constitutive and APIC paths.
        assert float(tensor(graph.coast.Fp.grad).norm()) > 1e-7
        assert float(tensor(graph.coast.C[0].grad).norm()) > 1e-7
        direction = direction_like(controls[channel], channel)
        analytical = float((gradients[channel].double()*direction.double()).sum())
        assert abs(analytical) > 5e-5
        # The no-slip fixture supports these same CPU preregistered brackets.
        for epsilon in (1e-3, 5e-4):
            losses = []
            for sign in (-1., 1.):
                shifted = [value.detach().clone() for value in controls]
                shifted[channel].add_(direction, alpha=sign*epsilon)
                losses.append(float(mixed_coast_loss(graph.apply(*shifted))))
            observed = (losses[1]-losses[0])/(2*epsilon)
            print(dict(channel=channel, epsilon=epsilon, analytical=analytical, finite_difference=observed))
            assert analytical == pytest.approx(observed, rel=.02, abs=5e-6)


def test_forward_matches_actual_assimilation_subset_and_independent_coast():
    fixture = case()
    with cuda_execution('cuda:0'):
        spec, controls, kwargs = upload_case(*fixture)
        model = model_for(spec, kwargs)
        out = model.apply(*controls)
        head, coast = model.head, model.coast
        F, P = to_array(tensor(head.F[spec.T])), to_array(tensor(head.Fp))
        old, pins = spec.pin.astype(bool), to_array(kwargs['next_pins'])
        new = pins & ~old
        first = assimilate_elastic(F, P, eta=.5, isochoric=True)
        first[old] = P[old]
        first[new] = assimilate_elastic(F[new], first[new], eta=1., isochoric=False)
        torch.testing.assert_close(tensor(coast.Fp), torch.as_tensor(first, device='cuda:0'),
                                   rtol=4e-6, atol=4e-6)
        v, C = tensor(head.v[spec.T]), tensor(head.C[spec.T])
        pins_t = kwargs['next_pins']
        v[pins_t], C[pins_t] = 0, 0
        independent = Trajectory(x0=to_array(tensor(head.x[spec.T])), v0=to_array(v),
            C0=to_array(C), F0=F, Fg0=to_array(tensor(head.Fg[spec.T])), Fp=first,
            m=spec.m, lam=spec.lam, mu=spec.mu, vol0=spec.vol0,
            eta=kwargs['successor_eta'], pin=pins.astype('float32'),
            layer=kwargs['successor_layer'], bonds=kwargs['successor_bonds'],
            prm=deepcopy(coast.prm), T=spec.T, device='cuda:0', requires_grad=False,
            track_geom=True, persistent=True)
        independent.rollout()
        for name, actual in (('x', out.coast_X), ('v', out.coast_V),
                             ('F', out.coast_F), ('Fg', out.coast_Fg)):
            expected = torch.stack([tensor(value).reshape(model.N, -1)
                                    for value in getattr(independent, name)])
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        for name in ('x', 'v', 'C', 'F', 'Fg'):
            assert getattr(coast, name)[0].ptr != getattr(head, name)[spec.T].ptr
        assert coast.Fp.ptr != head.Fp.ptr
        assert torch.count_nonzero(tensor(coast.v[0])[pins_t]) == 0
        assert torch.count_nonzero(tensor(coast.C[0])[pins_t]) == 0


def test_captured_head_coast_seed_addition_repeat_and_missing_seed_clear():
    fixture = case()
    with cuda_execution('cuda:0'):
        spec, controls, kwargs = upload_case(*fixture)
        model = model_for(spec, kwargs)
        out = model.apply(*controls)
        head = .01*sum(value.double().square().mean() for value in out[:6])
        coast = mixed_coast_loss(out)+.013*out.coast_X[0].sum()+.007*out.coast_F[0].sum()
        a = torch.autograd.grad(head, controls, retain_graph=True)
        b = torch.autograd.grad(coast, controls, retain_graph=True)
        combined = torch.autograd.grad(head+coast, controls, retain_graph=True)
        again = torch.autograd.grad(head+coast, controls, retain_graph=True)
        for x, y, z, repeated in zip(a, b, combined, again):
            torch.testing.assert_close(x+y, z, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(z, repeated, rtol=2e-4, atol=2e-5)
        zero = torch.autograd.grad(out.coast_F.sum()*0, controls)
        assert all(torch.count_nonzero(value) == 0 for value in zero)


@pytest.mark.parametrize('drop', ['Fp', 'C'])
def test_captured_omitted_boundary_paths_fail_independent_control_fd(drop, monkeypatch):
    fixture = case()
    with cuda_execution('cuda:0'):
        spec, controls, kwargs = upload_case(*fixture)
        model = model_for(spec, kwargs)
        def observation(out):
            if drop == 'C':
                return mixed_coast_loss(out)
            # First-step acceleration exposes Fp sensitivity that the mixed
            # terminal observable can hide within the fixed FD allowance.
            ids = torch.arange(model.N*3, dtype=torch.float64, device='cuda:0').reshape(-1, 3)
            weights = (ids*.43+.2).sin()
            return ((out.coast_V[1].double()-out.coast_V[0].double())*weights).mean()/spec.prm.dt
        def no_eager(*_args, **_kwargs):
            pytest.fail('Captured omission gate entered an eager trajectory or boundary')
        with monkeypatch.context() as capture_guard:
            for method in ('_record_head', '_record_coast', '_boundary_forward', '_boundary_vjp'):
                capture_guard.setattr(model, method, no_eager)
            out = model.apply(*controls)
            snapshot = tuple(value.detach().clone() for value in out)
            full = torch.autograd.grad(observation(out), controls, retain_graph=True)
            assert float(tensor(model.coast.Fp.grad)[model.new_pins].norm()) > 1e-7
            combine = model._head_covectors
            calls = []
            def omit_path(fp_to_F):
                # This value must come from the replayed Torch boundary VJP.
                assert fp_to_F.data_ptr() == model.captured_fp_to_F.data_ptr()
                calls.append(True)
                if drop == 'Fp':
                    assert float(fp_to_F.norm()) > 1e-7
                    fp_to_F = torch.zeros_like(fp_to_F)
                else:
                    assert float(tensor(model.coast.C[0].grad).norm()) > 1e-7
                    model.coast.C[0].grad.zero_()
                combine(fp_to_F)
            with monkeypatch.context() as patch:
                patch.setattr(model, '_head_covectors', omit_path)
                # Same primal/output object; only one reverse boundary edge is removed.
                broken = torch.autograd.grad(observation(out), controls)
            assert calls == [True]
            assert all(torch.equal(value, owned) for value, owned in zip(out, snapshot))
            witnesses = []
            for channel, (actual, incomplete) in enumerate(zip(full, broken)):
                direction = direction_like(controls[channel], channel)
                ad = float((actual.double()*direction.double()).sum())
                bad = float((incomplete.double()*direction.double()).sum())
                losses, epsilon = [], 5e-4
                for sign in (-1., 1.):
                    shifted = [value.detach().clone() for value in controls]
                    shifted[channel].add_(direction, alpha=sign*epsilon)
                    losses.append(float(observation(model.apply(*shifted))))
                fd = (losses[1]-losses[0])/(2*epsilon)
                allowance = max(.02*abs(fd), 5e-6)
                assert abs(ad-fd) <= allowance, (drop, channel, ad, fd)
                print(dict(drop=drop, channel=channel, epsilon=epsilon,
                           full=ad, omitted=bad, fd=fd, allowance=allowance))
                witnesses.append(abs(bad-fd) > allowance)
            assert any(witnesses), f'{drop} omission did not fail an independent captured FD direction'


def test_changed_controls_owned_policies_stale_generation_and_context_stream():
    fixture = case()
    with cuda_execution('cuda:0'):
        primary = torch.cuda.current_stream()
        warp_stream = wp.Stream('cuda:0')
        stream = wp.stream_to_torch(warp_stream)
        stream.wait_stream(primary)
        with torch.cuda.stream(stream), cuda_module().cuda.ExternalStream(stream.cuda_stream), wp.ScopedStream(warp_stream):
            spec, controls, kwargs = upload_case(*fixture)
            plain, graph = model_for(spec, kwargs, capture=False), model_for(spec, kwargs)
            old_out = graph.apply(*controls)
            snapshot = tuple(value.detach().clone() for value in old_out)
            kwargs['next_pins'].fill_(True)
            kwargs['successor_eta'].fill(100.)
            kwargs['successor_layer'][0].fill(0.)
            assert int(graph.next_pins.sum()) == 3
            assert float(tensor(graph.coast.eta).max()) < 100.
            assert float(tensor(graph.coast.layer_mask).max()) > 0.
            changed = tuple((value.detach()+.0005*direction_like(value, i)).requires_grad_()
                            for i, value in enumerate(controls))
            expected = plain.apply(*changed)
            ref = torch.autograd.grad(mixed_coast_loss(expected), changed)
            actual = graph.apply(*changed)
            got = torch.autograd.grad(mixed_coast_loss(actual), changed)
            assert float((actual.coast_X-snapshot[6]).abs().max()) > 1e-6
            for observed, wanted in zip(actual, expected):
                torch.testing.assert_close(observed, wanted, rtol=1e-5, atol=1e-5)
            for observed, wanted in zip(got, ref):
                torch.testing.assert_close(observed, wanted, rtol=2e-4, atol=2e-5)
            assert all(torch.equal(value, owned) for value, owned in zip(old_out, snapshot))
            with pytest.raises(RuntimeError, match='stale'):
                torch.autograd.grad(old_out.coast_X.sum(), controls[0])
            previous = graph.apply(*controls)
            with pytest.raises(ValueError, match='float32'):
                graph.apply(controls[0].double(), *controls[1:])
            with pytest.raises(RuntimeError, match='stale'):
                torch.autograd.grad(previous.coast_X.sum(), controls[0])
        primary.wait_stream(stream)
        with pytest.raises(RuntimeError, match='constructor CUDA device/stream'):
            graph.apply(*controls)
        with pytest.raises(RuntimeError, match='constructor CUDA device/stream'):
            graph.backward()
    with pytest.raises(RuntimeError, match='cuda_execution'):
        graph.forward()


def test_constructor_rejects_unaligned_torch_cupy_warp_streams():
    fixture = case()
    with cuda_execution('cuda:0'):
        spec, _, kwargs = upload_case(*fixture)
        primary = torch.cuda.current_stream()
        other = torch.cuda.Stream()
        other.wait_stream(primary)
        with torch.cuda.stream(other):
            with pytest.raises(RuntimeError, match='stream'):
                model_for(spec, kwargs)
        primary.wait_stream(other)
