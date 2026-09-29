"""Explicit hyde06 assimilation gates; never enable ordinary local collection.

Production-size FP32 forwards must remain exact. FP64 differences test the
fixed smooth branches, and graph capture is a real gate, never an xfail/skip
when inverse/SVD/indexing proves uncapturable on the server's installed stack.
"""
import math
import os

import pytest
import torch
import warp as wp

from physmorph.compute import cuda_execution, cuda_module, to_array
from physmorph.plasticity import assimilation as production
from physmorph.plasticity.assimilation_adjoint import (
    assimilate_elastic_differentiable as assimilate, assimilate_handoff)
from test_withdrawal_cuda import NoDeviceToHostArray


pytestmark = pytest.mark.skipif(
    os.environ.get('PHYSMORPH_ASSIMILATION_CUDA_TEST') != '1' or not torch.cuda.is_available(),
    reason='Requires explicit PHYSMORPH_ASSIMILATION_CUDA_TEST=1 on hyde06')


def rotation(angle, dtype):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=dtype, device='cuda:0')


def inputs(dtype=torch.float64):
    r, q = rotation(.43, dtype), rotation(-.72, dtype)
    spectra = torch.tensor([[1., 1., 1.], [1.7, 1.7, 1.7], [2., 2., .7],
                            [1.3, .82, 1.05], [12., 4., 1/48]], dtype=dtype, device='cuda:0')
    F = r @ torch.diag_embed(spectra) @ q
    P = torch.eye(3, dtype=dtype, device='cuda:0').repeat(len(spectra), 1, 1)
    P[3:] = torch.tensor([[1.1, .15, .03], [0., .91, .08], [.05, 0., 1.04]],
                         dtype=dtype, device='cuda:0')
    # Nonidentity, noncommuting Fp also participates in the active-clamp case.
    F[4] = F[4] @ P[4]
    return F, P


def aligned():
    stream = torch.cuda.current_stream().cuda_stream
    assert stream == cuda_module().cuda.get_current_stream().ptr
    assert stream == wp.get_stream('cuda:0').cuda_stream


@pytest.mark.parametrize('iso', [False, True])
@pytest.mark.parametrize('eta', [.35, 1.])
def test_actual_production_n20000_float32_forward_is_exact(monkeypatch, iso, eta):
    with cuda_execution('cuda:0'):
        aligned()
        F, P = (value.repeat(4000, 1, 1) for value in inputs(torch.float32))
        assert len(F) == 20000
        F_before, P_before = F.clone(), P.clone()
        calls = []
        original = production._assimilate_torch
        def record(*args, **kwargs):
            calls.append(len(args[0]))
            return original(*args, **kwargs)
        def no_host(*_args, **_kwargs):
            pytest.fail('Assimilation numerical array downloaded to host')
        with monkeypatch.context() as patch, NoDeviceToHostArray():
            patch.setattr(production, '_assimilate_torch', record)
            patch.setattr(cuda_module(), 'asnumpy', no_host)
            expected = torch.as_tensor(production.assimilate_elastic(
                to_array(F), to_array(P), eta=eta, isochoric=iso), device='cuda:0')
            actual = assimilate(F, P, eta=eta, isochoric=iso)
        assert calls == [20000], 'Must exercise the real production Torch dispatch'
        assert torch.isfinite(actual).all() and torch.equal(actual, expected)
        assert torch.equal(F, F_before) and torch.equal(P, P_before)


@pytest.mark.parametrize('iso', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_repeated_spectra_finite_repeated_vjp_and_identity_linear_oracle(iso, dtype):
    with cuda_execution('cuda:0'):
        aligned()
        F, P = inputs(dtype)
        F = F[:3].clone().requires_grad_()
        P = P[:3].clone().requires_grad_()
        weight = torch.tensor([[[.2, .7, -.5], [-.3, 1.2, .4], [.8, -.6, .1]]],
                              dtype=dtype, device='cuda:0').expand_as(F)
        result = assimilate(F, P, eta=.37, isochoric=iso)
        first = torch.autograd.grad((result*weight).sum(), (F, P), retain_graph=True)
        again = torch.autograd.grad((result*weight).sum(), (F, P))
        for actual, repeated in zip(first, again):
            assert torch.isfinite(actual).all() and bool((actual.norm(dim=(1, 2)) > 1e-6).all())
            assert torch.equal(actual, repeated)
        identity_F = torch.eye(3, dtype=dtype, device='cuda:0')[None].requires_grad_()
        identity_P = torch.eye(3, dtype=dtype, device='cuda:0')[None].requires_grad_()
        w = weight[:1]
        grads = torch.autograd.grad((assimilate(identity_F, identity_P, eta=.37, isochoric=iso)*w).sum(),
                                    (identity_F, identity_P))
        symmetric, skew = (w+w.transpose(1, 2))/2, (w-w.transpose(1, 2))/2
        if iso:
            symmetric = symmetric-torch.diag_embed(symmetric.diagonal(dim1=1, dim2=2).sum(1)[:, None].expand(-1, 3)/3)
        oracle = (.37*symmetric, .63*symmetric+skew)
        tolerance = (2e-6, 2e-7) if dtype == torch.float32 else (1e-11, 1e-12)
        for actual, expected in zip(grads, oracle):
            torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])


@pytest.mark.parametrize('iso', [False, True])
@pytest.mark.parametrize('case', ['noncommuting', 'cumulative_clamps', 'singular_value_floor'])
def test_float64_joint_F_Fp_directional_finite_differences(iso, case):
    with cuda_execution('cuda:0'):
        aligned()
        F, P = inputs()
        if case == 'noncommuting':
            F, P, eta, steps, atol = F[3:4], P[3:4], .37, (2e-6, 1e-6), 2e-7
        elif case == 'cumulative_clamps':
            F, P, eta, steps, atol = F[4:5], P[4:5], 1., (1e-7, 5e-8), 3e-6
        else:
            F = torch.diag(torch.tensor([4., 4., 5e-7], dtype=F.dtype, device=F.device))[None]
            P, eta, steps, atol = torch.eye(3, dtype=F.dtype, device=F.device)[None], .4, (1e-9, 5e-10), 5e-5
        if case != 'singular_value_floor':
            assert float((F@P-P@F).norm()) > .01
        if case == 'cumulative_clamps':
            with torch.no_grad():
                _, s, vh = torch.linalg.svd(F@torch.linalg.inv(P))
                powered = s.clamp_min(1e-3)**eta
                if iso: powered = powered/powered.prod(1, keepdim=True)**(1/3)
                raw = vh.transpose(1, 2)@torch.diag_embed(powered)@vh@P
                singular = torch.linalg.svdvals(raw)
                assert bool((singular < .2).any()) and bool((singular > 5.).any())
        generator = torch.Generator(device='cuda:0').manual_seed(333)
        dF, dP, w = [torch.randn(F.shape, generator=generator, dtype=F.dtype, device=F.device) for _ in range(3)]
        F, P = F.clone().requires_grad_(), P.clone().requires_grad_()
        value = assimilate(F, P, eta=eta, isochoric=iso)
        gradients = torch.autograd.grad((value*w).sum(), (F, P))
        assert all(torch.isfinite(g).all() and float(g.norm()) > 1e-7 for g in gradients)
        analytical = sum((gradient*direction).sum() for gradient, direction in zip(gradients, (dF, dP)))
        assert abs(float(analytical)) > 50*atol
        for step in steps:
            with torch.no_grad():
                high = assimilate(F+step*dF, P+step*dP, eta=eta, isochoric=iso)
                low = assimilate(F-step*dF, P-step*dP, eta=eta, isochoric=iso)
                observed = ((high-low)*w).sum()/(2*step)
            print(dict(case=case, isochoric=iso, epsilon=step, ad=float(analytical), fd=float(observed)))
            torch.testing.assert_close(analytical, observed, rtol=2e-6, atol=atol)


@pytest.mark.parametrize('new_count', [128, 20000])
@pytest.mark.parametrize('settle', [False, True])
def test_actual_runner_subset_handoff_dispatch_numerical_parity(monkeypatch, new_count, settle):
    with cuda_execution('cuda:0'):
        aligned()
        F, P = (value.repeat(4800, 1, 1) for value in inputs(torch.float32))
        old = torch.zeros(len(F), dtype=torch.bool, device=F.device)
        new = torch.zeros_like(old)
        old[:1024], new[1024:1024+new_count] = True, True
        ordinary = production.assimilate_elastic
        torch_calls = []
        original = production._assimilate_torch
        def record(*args, **kwargs):
            torch_calls.append(len(args[0]))
            return original(*args, **kwargs)
        with monkeypatch.context() as patch, NoDeviceToHostArray():
            patch.setattr(production, '_assimilate_torch', record)
            first = ordinary(to_array(F), to_array(P), eta=.4, isochoric=True)
            if settle:
                first[to_array(old)] = to_array(P)[to_array(old)]
                # Actual runner dispatches ONLY the newly admitted subset.
                first[to_array(new)] = ordinary(to_array(F)[to_array(new)], first[to_array(new)],
                                                eta=1., isochoric=False)
            expected = torch.as_tensor(first, device='cuda:0')
            actual = assimilate_handoff(F, P, old, new, eta=.4, isochoric=True, settle_pin_assim=settle)
        assert torch_calls == ([24000, 20000] if settle and new_count >= 20000 else [24000])
        assert torch.isfinite(actual).all()
        if not settle:
            assert torch.equal(actual, expected)
        else:
            assert torch.equal(actual[old], P[old])
            # Torch/CuPy subset SVD dispatch differs below20000; preserve that
            # production oracle and preregister numerical, not bitwise, parity.
            allowance = 64*torch.finfo(torch.float32).eps
            error = (actual-expected).abs()
            ratio = error/(allowance*(1+expected.abs()))
            print(dict(new_count=new_count, max_absolute_error=float(error.max()), max_tolerance_ratio=float(ratio.max())))
            torch.testing.assert_close(actual, expected, rtol=allowance, atol=allowance)


@pytest.mark.parametrize('iso', [False, True])
@pytest.mark.parametrize('count,dtype', [(1, torch.float64), (20000, torch.float32)],
                         ids=['n1_float64', 'n20000_float32'])
def test_torch_cuda_graph_forward_backward_replays_changed_inputs_and_seed(iso, count, dtype):
    _assert_graph_replays(count, dtype, iso)


def test_torch_cuda_graph_frozen_pin_handoff_forward_backward():
    _assert_graph_replays(20000, torch.float32, True, handoff=True)


def _assert_graph_replays(count, dtype, iso, *, handoff=False):
    with cuda_execution('cuda:0'):
        aligned()
        F0, P0 = (value[3:4].repeat(count, 1, 1) for value in inputs(dtype))
        F, P = F0.clone().requires_grad_(), P0.clone().requires_grad_()
        seed = (torch.arange(9, dtype=F.dtype, device=F.device).reshape(1, 3, 3)/7-.4).repeat(count, 1, 1)
        old = torch.zeros(count, dtype=torch.bool, device=F.device)
        new = torch.zeros_like(old)
        if handoff:
            old[:1024], new[1024:2048] = True, True
        old_before, new_before = old.clone(), new.clone()
        forward_tolerance = (2e-6, 2e-7) if dtype == torch.float32 else (1e-11, 1e-12)
        gradient_tolerance = (2e-6, 2e-7) if dtype == torch.float32 else (1e-10, 1e-11)
        def evaluate():
            out = (assimilate_handoff(F, P, old, new, eta=.37, isochoric=iso)
                   if handoff else assimilate(F, P, eta=.37, isochoric=iso))
            gradients = torch.autograd.grad((out*seed).sum(), (F, P))
            return out, gradients
        # Initialize the installed inverse/SVD/backward libraries on this same
        # aligned nondefault stream before attempting a real capture.
        for _ in range(3): evaluate()
        torch.cuda.current_stream().synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            captured, captured_gradients = evaluate()
        for replay in range(2):
            if replay:
                with torch.no_grad():
                    F.copy_(F0+torch.eye(3, dtype=F.dtype, device=F.device)*.002)
                    P.copy_(P0+torch.eye(3, dtype=P.dtype, device=P.device)*.001)
                    seed.mul_(-.7).add_(.1)
            graph.replay()
            expected, expected_gradients = evaluate()
            torch.testing.assert_close(captured, expected, rtol=forward_tolerance[0], atol=forward_tolerance[1])
            for actual, wanted in zip(captured_gradients, expected_gradients):
                assert torch.isfinite(actual).all() and float(actual.norm()) > 1e-7
                torch.testing.assert_close(actual, wanted, rtol=gradient_tolerance[0], atol=gradient_tolerance[1])
            if handoff:
                assert torch.equal(captured[old], P[old])
                assert torch.count_nonzero(captured_gradients[0][old]) == 0
                assert torch.equal(captured_gradients[1][old], seed[old])
                assert torch.equal(old, old_before) and torch.equal(new, new_before)
                first_only = assimilate(F, P, eta=.37, isochoric=iso)
                assert float((captured[new]-first_only[new]).abs().max()) > 1e-3
            if replay == 0:
                previous = (captured.clone(), *(value.clone() for value in captured_gradients))
            else:
                for actual, before in zip((captured, *captured_gradients), previous):
                    assert float((actual-before).abs().max()) > 1e-5
