"""Ownership and scope of the opt-in shared XPIC endpoint objective."""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch


def validate_endpoint_config(cfg):
    if not cfg.commit_pic_objective:
        return
    if not cfg.commit_pic:
        raise ValueError('commit_pic_objective requires commit_pic')
    unsupported = [name for name in ('shift_sub', 'reattach', 'settle_commit',
                                    'grad_dump', 'use_gauss_loss') if getattr(cfg, name)]
    if cfg.lg_sweeps > 0:
        unsupported.append('lg_sweeps')
    if cfg.assim_consensus or cfg.w_grow > 0:
        unsupported.append('geometry-dependent assimilation')
    if cfg.w_corr > 0 or (cfg.h1_outside and cfg.w_h1 > 0):
        unsupported.append('correspondence/outside-H1 gradient path')
    if os.environ.get('PHYSMORPH_REPLAY_LOAD') or os.environ.get('PHYSMORPH_REPLAY_SAVE'):
        unsupported.append('legacy control replay hooks')
    if unsupported:
        raise ValueError('commit_pic_objective does not support ' + ', '.join(unsupported))


def endpoint_bounds(prm, reference):
    origin = torch.as_tensor(prm.grid_min, device=reference.device, dtype=reference.dtype)
    dims = torch.tensor([prm.nx, prm.ny, prm.nz], device=reference.device, dtype=reference.dtype)
    return origin + 2 * prm.dx, origin + prm.dx * dims - 2 * prm.dx


def valid_endpoint(x, bounds):
    lo, hi = bounds
    return bool((torch.isfinite(x) & (x >= lo) & (x <= hi)).all())


@dataclass(frozen=True)
class OwnedEndpoint:
    """The exact evaluated endpoint, owned independently of rollout buffers.

    Retaining the per-window map prevents accidental reuse across a new stencil.
    The consumer checks its raw input, start positions and pins before promotion;
    it must not run another scatter/gather whose atomic rounding could differ.
    """
    operator: object
    start: torch.Tensor
    pin: torch.Tensor
    raw: torch.Tensor
    promoted: torch.Tensor
    source: str

    @classmethod
    def capture(cls, operator, start, pin, raw, promoted, source):
        return cls(operator, *(v.detach().clone() for v in (start, pin, raw, promoted)), source)

    def promote(self, raw, start, pin, bounds):
        for label, actual, expected in (('raw', raw, self.raw), ('start', start, self.start),
                                        ('pins', pin, self.pin)):
            if actual.device != expected.device or actual.shape != expected.shape or not torch.equal(actual, expected):
                raise ValueError(f'shared PIC endpoint {label} no longer matches its evaluated state')
        if not valid_endpoint(self.promoted, bounds):
            raise ValueError('shared PIC endpoint requires finite in-bounds promoted positions; repair is forbidden')
        if not torch.equal(self.promoted[self.pin], self.start[self.pin]):
            raise ValueError('shared PIC endpoint changed a window-start pin')
        return self.promoted.clone()
