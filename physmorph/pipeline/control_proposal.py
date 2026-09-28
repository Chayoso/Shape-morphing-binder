"""Shared in-place Adam proposal and scalar line-search decrease rule.

Callers own rollback/acceptance. Diagnostic callers must supply their own
parameter and moment copies; this helper does not clone live solver state.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def apply_proposal(leaves, gradients, mom, vel, *, step, alpha, lr_scale,
                   beta1, beta2, eps, stress_index=0, material_index=None,
                   body_index=None, surface_index=None, ctrl_scale=None,
                   body_scale=None, surface_project=None, dfc_clip=0.,
                   material_clip=1., surface_bound=None):
    """Mutate supplied leaves/moments using the production update and bound order.

    ``step`` is the proposed Adam time (previous accepted time + 1). Stress and
    body scales act on the preconditioned update; surface projection acts before
    the learning rate. Bounds act on the resulting parameters, not the gradient.
    """
    count = len(leaves)
    if not (count == len(gradients) == len(mom) == len(vel) == len(lr_scale)):
        raise ValueError('Proposal requires matching parameter/gradient/moment/scale lists')
    indices = [i for i in (stress_index, material_index, body_index, surface_index) if i is not None]
    if len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices):
        raise ValueError('Proposal leaf roles must be distinct valid indices')
    if surface_index is not None and surface_bound is None:
        raise ValueError('Surface proposal requires the prepared displacement bound')
    for index, (parameter, gradient, first, second, scale) in enumerate(
            zip(leaves, gradients, mom, vel, lr_scale)):
        first.mul_(beta1).add_(gradient, alpha=1-beta1)
        second.mul_(beta2).addcmul_(gradient, gradient, value=1-beta2)
        corrected_first = first / (1-beta1**step)
        corrected_second = second / (1-beta2**step)
        direction = corrected_first / (corrected_second.sqrt()+eps)
        if ctrl_scale is not None and index == stress_index:
            direction = direction*ctrl_scale
        if body_scale is not None and index == body_index:
            direction = direction*body_scale
        if surface_project is not None and index == surface_index:
            direction = surface_project(direction)
        parameter -= (alpha*scale)*direction
    if dfc_clip > 0:
        stress = leaves[stress_index]
        magnitude = stress.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1)
        stress *= (dfc_clip/magnitude.clamp_min(1e-8)).clamp(max=1.0)
    if material_index is not None:
        leaves[material_index].clamp_(-material_clip, material_clip)
    if surface_index is not None:
        surface = leaves[surface_index]
        surface.copy_(torch.maximum(torch.minimum(surface, surface_bound), -surface_bound))
    if body_index is not None:
        body = leaves[body_index]
        body.div_(body.norm(dim=1, keepdim=True).clamp_min(1.0))


def required_decrease(current, predicted, unit_ratio, noise_rel, armijo_c1):
    """Armijo only for a descent slope; stale-moment directions use noise floor."""
    noise_floor = noise_rel*max(abs(current), 1.0/unit_ratio)
    return max(armijo_c1*predicted, noise_floor) if predicted > 0.0 else noise_floor
