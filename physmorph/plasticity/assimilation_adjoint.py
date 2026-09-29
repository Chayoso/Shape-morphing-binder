"""Opt-in derivative of the ordinary elastic assimilation, with fixed branches.

The forward uses the production Torch operations, including both cumulative
clamps. Spectral pullbacks avoid differentiating SVD vectors at repeated values.
This is not the derivative of pin admission, growth, consensus or preparation.
"""
from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable

from .assimilation import _project_logsv_torch


def _sym(a):
    return (a + a.transpose(-1, -2)) * 0.5


def _quotient(s, values, repeated, branch):
    """Spectral divided difference with the smooth repeated-spectrum limit.

    A close pair uses that limit only inside the same clamp branch. This does
    not perturb a singular value or change the forward. Kinks remain kinks.
    """
    delta = s[:, :, None] - s[:, None, :]
    scale = torch.maximum(s[:, :, None], s[:, None, :])
    near = delta.abs() <= math.sqrt(torch.finfo(s.dtype).eps) * scale
    same = branch[:, :, None] == branch[:, None, :]
    use_limit = near & same
    denominator = torch.where(use_limit, torch.ones_like(delta), delta)
    value = (values[:, :, None] - values[:, None, :]) / denominator
    limit = (repeated[:, :, None] + repeated[:, None, :]) * 0.5
    return torch.where(use_limit, limit, value)


class _ElasticIncrement(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Fe, eta, isochoric):
        U, s, Vh = torch.linalg.svd(Fe)
        powered = s.clamp_min(1e-3) ** eta
        normalizer = powered.prod(1, keepdim=True) ** (1.0 / 3.0) if isochoric else torch.ones_like(s[:, :1])
        values = powered / normalizer if isochoric else powered
        out = Vh.transpose(1, 2) @ torch.diag_embed(values) @ Vh
        ok = torch.linalg.det(Fe) > 1e-6
        out[~ok] = torch.eye(3, dtype=Fe.dtype, device=Fe.device)
        ctx.save_for_backward(U, s, Vh, powered, normalizer, values, ok)
        ctx.eta, ctx.isochoric = eta, isochoric
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        U, s, Vh, powered, normalizer, values, ok = ctx.saved_tensors
        E = _sym(Vh @ grad @ Vh.transpose(1, 2))
        active = s > 1e-3
        slope = ctx.eta * s.clamp_min(1e-3) ** (ctx.eta - 1) * active
        L = _quotient(s, powered, slope, active) / normalizer[:, :, None]
        denom = s[:, :, None] + s[:, None, :]
        # Fe may be singular on a skipped row. Its complete pullback is zero.
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        result = (2 * s[:, :, None] / denom) * L * E
        diagonal = E.diagonal(dim1=-2, dim2=-1)
        if ctx.isochoric:
            weighted = diagonal * values
            ds = (weighted - weighted.sum(1, keepdim=True) / 3) * ctx.eta / s.clamp_min(1e-3) * active
        else:
            ds = diagonal * slope
        result = result - torch.diag_embed(result.diagonal(dim1=-2, dim2=-1)) + torch.diag_embed(ds)
        result = U @ result @ Vh
        return torch.where(ok[:, None, None], result, torch.zeros_like(result)), None, None


class _CumulativeBand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, smin, smax, isochoric):
        U, s, Vh = torch.linalg.svd(value)
        clipped = s.clamp(smin, smax)
        if isochoric:
            q = _project_logsv_torch(clipped.log(), math.log(smin), math.log(smax),
                                     torch.zeros_like(s[:, 0]))
        else:
            q = clipped
        ctx.save_for_backward(U, s, Vh, clipped, q)
        ctx.smin, ctx.smax, ctx.isochoric = smin, smax, isochoric
        return U @ torch.diag_embed(q) @ Vh

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        U, s, Vh, clipped, q = ctx.saved_tensors
        E = U.transpose(1, 2) @ grad @ Vh.transpose(1, 2)
        symmetric = _sym(E)
        skew = E - symmetric
        pre_active = (s > ctx.smin) & (s < ctx.smax)
        diagonal = E.diagonal(dim1=-2, dim2=-1)
        if ctx.isochoric:
            # Determine the final projection's active set from its exact output
            # bounds, not from the first clamp or a guessed bisection derivative.
            lo = torch.exp(torch.full_like(q, math.log(ctx.smin)))
            hi = torch.exp(torch.full_like(q, math.log(ctx.smax)))
            free = (q > lo) & (q < hi)
            count = free.sum(1, keepdim=True).clamp_min(1)
            weighted = diagonal * q * free
            ds = (weighted - weighted.sum(1, keepdim=True) / count) * free * pre_active / clipped
            repeated = q * free * pre_active / clipped
            pre_region = pre_active.to(torch.int8) + 2 * (s >= ctx.smax).to(torch.int8)
            final_region = free.to(torch.int8) + 2 * (q >= hi).to(torch.int8)
            branch = pre_region + 3 * final_region
        else:
            ds = diagonal * pre_active
            repeated = pre_active.to(s.dtype)
            branch = pre_active.to(torch.int8) + 2 * (s >= ctx.smax).to(torch.int8)
        plus = _quotient(s, q, repeated, branch)
        denominator = s[:, :, None] + s[:, None, :]
        minus = (q[:, :, None] + q[:, None, :]) / denominator
        result = plus * symmetric + minus * skew
        result = result - torch.diag_embed(result.diagonal(dim1=-2, dim2=-1)) + torch.diag_embed(ds)
        return U @ result @ Vh, None, None, None


def _validate(F, Fp, eta, smin, smax):
    if not isinstance(F, torch.Tensor) or not isinstance(Fp, torch.Tensor):
        raise TypeError('Assimilation adjoint requires Torch tensors')
    if F.ndim != 3 or F.shape[1:] != (3, 3) or F.shape != Fp.shape or F.shape[0] == 0:
        raise ValueError('Assimilation adjoint requires matching nonempty (N,3,3) tensors')
    if F.dtype not in (torch.float32, torch.float64) or Fp.dtype != F.dtype or Fp.device != F.device:
        raise ValueError('Assimilation adjoint requires matching float32/float64 dtype and device')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in (eta, smin, smax)) or not 0 < smin < smax:
        raise ValueError('Assimilation constants must be finite, with 0 < smin < smax')


def assimilate_elastic_differentiable(F, Fp, eta=0.5, smin=0.2, smax=5.0, isochoric=False):
    """First derivative of ordinary assimilation on finite, nonsingular inputs.

    Float32 matches the production CUDA operation order; float64 is for gradient
    verification. The det gate and clamp branches are held fixed. No derivative
    at their boundaries or higher-order derivative is claimed. No host copies.
    """
    _validate(F, Fp, eta, smin, smax)
    if eta <= 0:
        return Fp.clone()
    Fe = F @ torch.linalg.inv(Fp)
    increment = _ElasticIncrement.apply(Fe, eta, bool(isochoric))
    return _CumulativeBand.apply(increment @ Fp, smin, smax, bool(isochoric))


def assimilate_handoff(F, Fp, old_pins, new_pins, *, eta=0.5, isochoric=False,
                       smin=0.2, smax=5.0, settle_pin_assim=True):
    """Two ordinary calls with frozen, disjoint old/new pin masks.

    Caller owns correct masks and policy scope. New pins consume the first call's
    result. This maps only Fp; v/C projection and the trajectory bridge are separate.
    """
    _validate(F, Fp, eta, smin, smax)
    for pins in (old_pins, new_pins):
        if not isinstance(pins, torch.Tensor) or pins.shape != F.shape[:1] or pins.dtype != torch.bool or pins.device != F.device:
            raise ValueError('Pin masks must be boolean (N,) tensors on the state device')
    torch._assert_async(~(old_pins & new_pins).any(), 'Old and new pin masks must be disjoint')
    first = assimilate_elastic_differentiable(F, Fp, eta, smin, smax, isochoric)
    if not settle_pin_assim:
        return first
    first = torch.where(old_pins[:, None, None], Fp, first)
    admitted = assimilate_elastic_differentiable(F, first, 1.0, smin, smax, False)
    return torch.where(new_pins[:, None, None], admitted, first)
