"""Fixed-reference raw geometry diagnostics for P296 endpoint candidates.

Use inside the caller's physmorph.compute context: CUDA stays on that backend;
normal CPU context is the test/reference path. The region and thresholds below
are the existing bunny geometry_curve diagnostics, not new physics constants.
No renderer or loss operator is evaluated. This module sets no execution/cache
environment and does not import the quality_compare CLI.
"""
from __future__ import annotations

import math

from physmorph.compute import array_api as np, KDTree, to_array, is_cuda_execution
from physmorph.metrics import target_extent, sil_iou, hole_frac


def _points(value, name, minimum, *, copy=False):
    value = to_array(value, copy=copy)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < minimum:
        raise ValueError(f'{name} must have shape (N,3) with N >= {minimum}')
    if value.dtype.kind != 'f' or not bool(np.isfinite(value).all()):
        raise ValueError(f'{name} must contain finite floating-point positions')
    return value


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


def _stats(values):
    return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=float(np.max(values))) if values.size else None


def prepare_reference(source, target):
    """Own a fixed target and its tree; reuse this dict for every candidate.

    Median NN spacing uses k=2 including self. Density radius is exactly the
    existing audit's median target k=9 distance[:,8], NOT 2*source spacing.
    Callers must not mutate the returned target/tree/reference dictionary.
    """
    source = _points(source, 'source', 2)
    target = _points(target, 'target', 9, copy=True)
    source_tree, target_tree = KDTree(source), KDTree(target)
    source_spacing = _positive(np.median(source_tree.query(source, k=2)[0][:, 1]), 'source spacing')
    target_spacing = _positive(np.median(target_tree.query(target, k=2)[0][:, 1]), 'target spacing')
    radius = _positive(np.median(target_tree.query(target, k=9)[0][:, 8]), 'density radius')
    extent = _positive(target_extent(target), 'target extent')
    return dict(source_spacing=source_spacing, target_spacing=target_spacing,
                targettree=target_tree, target=target, extent=extent,
                tip=target[target[:, 1].argmax()].copy(), radius=radius,
                source_n=len(source), target_n=len(target), cuda=is_cuda_execution())


def _finite_report(value):
    if isinstance(value, dict):
        return all(_finite_report(item) for item in value.values())
    return not isinstance(value, float) or math.isfinite(value)


def endpoint_quality(x, reference):
    """JSON scalars with geometry_curve definitions and independent hole_frac.

    source_gap_sp is candidate -> target distance; target_gap_sp is target ->
    candidate distance. Both use the FIXED target's median NN spacing. Empty top
    regions are None, not zero-error evidence. Invalid full clouds fail closed.
    """
    if reference['cuda'] != is_cuda_execution():
        raise ValueError('Reference and endpoint require the same compute backend context')
    x = _points(x, 'endpoint', 1)
    target, target_tree = reference['target'], reference['targettree']
    spacing, radius, extent = (reference[key] for key in ('target_spacing', 'radius', 'extent'))
    tree = KDTree(x)
    source_distance = target_tree.query(x)[0]
    target_distance = tree.query(target)[0]
    top, top_target = x[:, 1] > 2.3, target[:, 1] > 2.3
    counts = tree.query_ball_point(x[top], radius, return_length=True)-1 if bool(top.any()) else None
    result = dict(
        finite=True, target_fixed=True, source_n=reference['source_n'], target_n=reference['target_n'],
        endpoint_n=len(x), source_spacing=reference['source_spacing'], target_spacing=spacing,
        extent=extent, density_radius=radius,
        chamfer=float(source_distance.mean()+target_distance.mean()),
        sil_iou=sil_iou(x, target, extent), hole_frac=hole_frac(x, extent),
        target_near_frac=float((target_distance <= 2*spacing).mean()),
        target_gap_sp=_stats(target_distance/spacing),
        source_gap_sp=_stats(source_distance/spacing),
        top_target_near_frac=float((target_distance[top_target] <= 2*spacing).mean()) if bool(top_target.any()) else None,
        top_target_gap_sp=_stats(target_distance[top_target]/spacing),
        source_out_far_frac=float((source_distance > 4.5*spacing).mean()),
        tip_n=int((np.linalg.norm(x-reference['tip'], axis=1) < .25).sum()),
        top_n=int(top.sum()), top_target_n=int(top_target.sum()),
        top_density=float(counts.mean()/8) if counts is not None else None,
        top_under_half=float((counts < 4).mean()) if counts is not None else None,
        definitions=dict(
            scope='same existing bunny diagnostic region; these are diagnostic thresholds, not new physics constants',
            target_fixed='owned target, target tree, extent, spacings, density radius and argmax-y tip shared across candidates',
            spacing='source and target median self-query k=2 distance[:,1]; source spacing is metadata only here',
            chamfer='mean(candidate->target NN) + mean(target->candidate NN), in world units',
            projection='physmorph.metrics binary 3x3 footprint; sil_iou: res128, 8 azimuths x elevations(0,.5,-.5); hole_frac: res160, views((.6,.18),(2.2,.18))',
            extent='1.15 * max target Euclidean radius about origin; never fitted to endpoint',
            gaps='source_gap_sp: candidate->target; target_gap_sp/top_target_gap_sp: target->candidate; all divided by target median NN spacing',
            target_near='target->candidate NN <= 2*target_spacing',
            source_out_far='candidate->target NN > 4.5*target_spacing',
            top='candidate y>2.3 for density; fixed target y>2.3 for top coverage/gaps',
            density='inclusive ball radius = median target self-query k=9 distance[:,8]; subtract one self; mean count/8; under-half count<4',
            tip='first target argmax-y particle; candidate Euclidean distance <0.25 world units',
            empty='empty top regions yield null; empty or nonfinite full inputs and nonpositive native spacings are rejected',
            limitation='endpoint geometry only; no trajectory/rest/equilibrium or physical-hole-removal claim'))
    if not _finite_report(result):
        raise ValueError('Raw geometry diagnostics produced nonfinite results')
    return result
