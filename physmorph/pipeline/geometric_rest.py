"""Terminal geometric motion, distinct from stored momentum and pin admission."""
from __future__ import annotations

import math
import torch


def terminal_motion(raw, previous, promoted, eligible, dt):
    """Whole-cloud mean squared speeds on a fixed eligible material cohort.

    The raw final step and endpoint remap are separate nonnegative terms. A
    small saved transition alone may conceal cancellation between them. This
    is terminal geometry only, not proof of rest over the complete trajectory.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('geometric rest requires a finite positive dt')
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0:
        raise ValueError('geometric rest expects nonempty Nx3 positions')
    if any(x.shape != raw.shape or x.device != raw.device or x.dtype != raw.dtype
           for x in (previous, promoted)):
        raise ValueError('geometric rest position shape/device/dtype mismatch')
    if eligible.shape != (len(raw),) or eligible.dtype != torch.bool or eligible.device != raw.device:
        raise ValueError('geometric rest requires a same-device boolean cohort')
    r = (raw - previous) / dt
    j = (promoted - raw) / dt
    weight = eligible.to(raw.dtype)

    def average(value):
        return (weight * value).mean()

    raw_sq = average(r.square().sum(1))
    remap_sq = average(j.square().sum(1))
    return dict(raw_sq=raw_sq, remap_sq=remap_sq, total=raw_sq + remap_sq,
                delivered_sq=average((r + j).square().sum(1)),
                cross=average(2 * (r * j).sum(1)))
