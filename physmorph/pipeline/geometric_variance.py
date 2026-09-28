"""Temporal rates of the saved positional path, distinct from physical momentum."""
from __future__ import annotations

import math
import torch


def path_rates(start, positions, promoted, dt):
    """X contains post-layer x1..xT; the delivered path ends at promoted, not raw xT.

    The start is fixed state. Intermediate X and the promoted endpoint keep their
    autograd paths. This is an observation only; it does not edit x/v/C/F.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('path rates require finite positive dt')
    if start.ndim != 2 or start.shape[1] != 3 or len(start) == 0:
        raise ValueError('path rates require nonempty Nx3 start')
    if (positions.ndim != 3 or positions.shape[0] < 1 or positions.shape[1:] != start.shape
            or promoted.shape != start.shape):
        raise ValueError('path rates require X=(T,N,3), T>=1 and promoted=(N,3)')
    if any(x.device != start.device or x.dtype != start.dtype for x in (positions, promoted)):
        raise ValueError('path rate dtype/device mismatch')
    before = torch.cat((start.detach()[None], positions[:-1]), dim=0)
    saved = torch.cat((positions[:-1], promoted[None]), dim=0)
    return (saved-before)/dt


def temporal_variance(rates):
    """Mean particle population variance; uniform drift has zero variance."""
    return (rates-rates.mean(dim=0, keepdim=True)).square().sum(-1).mean()


@torch.no_grad()
def path_telemetry(start, positions, promoted, physical_v, dt):
    """Report both observables and displacements; never interpret variance as rest."""
    rates = path_rates(start, positions, promoted, dt)
    raw_before = torch.cat((start[None], positions[:-1]), dim=0)
    raw_rates = (positions-raw_before)/dt
    jump = (promoted-positions[-1])/dt
    return dict(
        geometric_variance=float(temporal_variance(rates)),
        physical_variance=float(temporal_variance(physical_v)),
        raw_geometric_variance=float(temporal_variance(raw_rates)),
        saved_phase_rms_wu=[float(v) for v in (rates.square().sum(-1).mean(-1).sqrt()*dt)],
        raw_phase_rms_wu=[float(v) for v in (raw_rates.square().sum(-1).mean(-1).sqrt()*dt)],
        remap_rms_wu=float(jump.square().sum(-1).mean().sqrt()*dt),
        net_rms_wu=float((promoted-start).square().sum(-1).mean().sqrt()),
        saved_path_mean_wu=float(rates.norm(dim=-1).sum(0).mean()*dt),
        raw_path_mean_wu=float(raw_rates.norm(dim=-1).sum(0).mean()*dt),
        normalization='all particles; temporal population variance',
        limitation='constant drift and redistributed motion may lower variance; not a rest certificate')
