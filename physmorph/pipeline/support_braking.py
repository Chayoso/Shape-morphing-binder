"""Fixed material support witnesses for the opt-in braking diagnostic."""
import math

import torch

from ..compute import KDTree, to_array, to_host, is_cuda_execution


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def support_values(x, targets, witnesses, radius):
    """One FP64 signed squared-distance scalar per fixed target/material pair."""
    _require(math.isfinite(radius) and radius > 0 and 0 < radius*radius < math.inf, 'Invalid support radius')
    _require(x.ndim == 2 and x.shape[1] == 3 and targets.shape == (len(witnesses), 3)
             and witnesses.ndim == 1 and witnesses.dtype == torch.int64
             and x.device == targets.device == witnesses.device, 'Invalid support layout')
    _require(bool(((witnesses >= 0) & (witnesses < len(x))).all()), 'Invalid support witness ID')
    delta = x[witnesses].double() - targets.double()
    _require(bool(torch.isfinite(delta).all()), 'Nonfinite support position')
    return (delta.square().sum(-1) - radius*radius) / (radius*radius)


@torch.no_grad()
def target_neighbors(x, target):
    _require(x.is_cuda == is_cuda_execution(), 'Target-neighbor backend/device mismatch')
    _require(x.ndim == target.ndim == 2 and x.shape[1] == target.shape[1] == 3
             and x.device == target.device and len(x) > 0 and len(target) > 0
             and x.is_floating_point() and target.is_floating_point()
             and bool(torch.isfinite(x).all() and torch.isfinite(target).all()), 'Invalid target-neighbor inputs')
    d, ids = KDTree(to_array(x)).query(to_array(target))
    d = torch.as_tensor(d, device=x.device).clone()
    ids = torch.as_tensor(ids, device=x.device, dtype=torch.int64).clone()
    _require(d.shape == ids.shape == (len(target),) and bool(torch.isfinite(d).all()
             and (d >= 0).all() and ((ids >= 0) & (ids < len(x))).all()), 'Invalid target-neighbor output')
    return d, ids


@torch.no_grad()
def select_support(baseline, origin, target, radius, limit=5):
    """All stable-covered targets lost by the shared origin; never truncate."""
    _require(baseline.shape == (3, *origin.shape) and baseline.device == origin.device == target.device,
             'Invalid support baseline layout')
    _require(limit == 5, 'Unregistered support budget')
    distances, nearest = zip(*(target_neighbors(x, target) for x in (*baseline, origin)))
    distances, nearest = torch.stack(distances), torch.stack(nearest)
    covered = distances <= radius
    stable = covered[:3].all(0)
    ids = torch.nonzero(stable & ~covered[3]).flatten()
    witnesses, q = nearest[0, ids].clone(), target[ids].clone()
    values = torch.stack([support_values(x, q, witnesses, radius) for x in (*baseline, origin)])
    # The common original witness must pass both equivalent arithmetic forms.
    direct = torch.stack([(x[witnesses].double()-q.double()).norm(dim=-1) <= radius for x in baseline])
    certified = bool((values[:3] <= 0).all() and direct.all())
    status = ('over_budget' if len(ids) > limit else
              'uncertified_witness' if not certified else
              'no_intervention' if not len(ids) else 'ready')
    return dict(status=status, count=len(ids), radius=radius,
                target_ids=ids, witnesses=witnesses, targets=q,
                baseline_covered=covered[:3].clone(), origin_covered=covered[3].clone(),
                endpoint_distances=distances, endpoint_nearest=nearest, witness_values=values)


@torch.no_grad()
def support_report(x, target, selection):
    """Actual per-ID support and all-target loss/gain sets; never reselect pairs."""
    ids, witnesses, q = (selection[k] for k in ('target_ids', 'witnesses', 'targets'))
    radius = selection['radius']
    _require(ids.dtype == torch.int64 and ids.ndim == 1 and ids.device == target.device
             and selection['count'] == len(ids) and bool(((ids >= 0) & (ids < len(target))).all())
             and torch.equal(q, target[ids]), 'Support target identity mismatch')
    _require(selection['baseline_covered'].shape == (3, len(target))
             and selection['baseline_covered'].dtype == torch.bool
             and selection['baseline_covered'].device == target.device, 'Support coverage layout mismatch')
    values = support_values(x, q, witnesses, radius)
    distance, nearest = target_neighbors(x, target)
    covered = distance <= radius
    direct = (x[witnesses].double()-q.double()).norm(dim=-1) <= radius
    scalar_bits = values <= 0
    agreement = bool(torch.equal(direct, scalar_bits))
    protected = covered[ids]
    original = selection['baseline_covered']
    stable_covered = original.all(0)
    stable_uncovered = (~original).all(0)
    dump = lambda value: to_host(value).tolist()
    return dict(values=dump(values), witness_within_radius=dump(direct),
                protected_raw_covered=dump(protected), scalar_radius_agreement=agreement,
                passed=agreement and bool(scalar_bits.all() and direct.all() and protected.all()),
                lost_ids=[dump(torch.nonzero(row & ~covered).flatten()) for row in original],
                gained_ids=[dump(torch.nonzero(~row & covered).flatten()) for row in original],
                stable_lost_ids=dump(torch.nonzero(stable_covered & ~covered).flatten()),
                stable_gained_ids=dump(torch.nonzero(stable_uncovered & covered).flatten()),
                baseline_ambiguous_ids=dump(torch.nonzero(~(stable_covered | stable_uncovered)).flatten()),
                protected_nearest_ids=dump(nearest[ids]))
