"""P331 fixed-reference, all-phase raw quality gates; no renderer or loss calls.

Construct and use inside the caller's compute context. The three original
forwards set per-phase envelopes; candidates never modify those references.
Only masks and scalar envelopes survive observe(), never trajectory graphs.
"""
from copy import deepcopy
from hashlib import sha256
import math

from physmorph.compute import (array_api as np, KDTree, ndimage, to_array,
                              to_host, is_cuda_execution)
from physmorph.metrics import _splat_body, target_extent


RESOLUTIONS = (128, 256)
# Exact P329 raw observation rings, without elevation-dependent azimuth offsets.
VIEWS = tuple((j*2*math.pi/8, elevation)
              for elevation in (0., .5, -.5) for j in range(8))
LOWER = ('target_covered', 'upper_target_covered', 'tip_count',
         'fixed_source_density', 'iou')
UPPER = ('under_half_count', 'hole_pixels', 'clipped_centers')


def _native(value):
    if not is_cuda_execution() and (hasattr(value, '__cuda_array_interface__')
                                   or bool(getattr(value, 'is_cuda', False))):
        raise ValueError('CUDA raw quality requires the active CUDA compute context')
    return to_array(value)


def _points(value, name, minimum=1):
    value = _native(value)
    if (value.ndim != 2 or value.shape[1] != 3 or len(value) < minimum
            or value.dtype != np.float32 or not bool(np.isfinite(value).all())):
        raise ValueError(name+' requires finite native float32 (N,3) positions')
    return value


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid '+name)
    return value


def _clipped_centers(x, res, azimuth, elevation, extent):
    # Preserve _splat_body's FP32 basis, floor, and half-open center bounds.
    right = np.asarray([math.cos(azimuth), 0., -math.sin(azimuth)], np.float32)
    up = np.asarray([-math.sin(elevation)*math.sin(azimuth), math.cos(elevation),
                     -math.sin(elevation)*math.cos(azimuth)], np.float32)
    projection = np.stack((x @ right, x @ up), axis=1)
    pixels = np.floor((projection+extent)/(2*extent)*res).astype(np.int64)
    return (~((pixels >= 0).all(1) & (pixels < res).all(1))).sum(dtype=np.int64)


class RawWithdrawalQuality:
    """Exactly three baseline forwards followed by immutable candidate checks."""

    def __init__(self, source, target, x0, pins):
        self.cuda = is_cuda_execution()
        source = _points(source, 'source', 2)
        self.target = _points(target, 'target', 9).copy()
        self.x0 = _points(x0, 'x0').copy()
        self.pins = _native(pins).copy()
        if len(source) != len(self.x0) or self.pins.shape != (len(source),) or self.pins.dtype != np.bool_:
            raise ValueError('Source/x0 particle identities and boolean pins must match')
        self.device = self.target.device.id if self.cuda else None
        self._context()
        source_tree, target_tree = KDTree(source), KDTree(self.target)
        self.source_spacing = _positive(np.median(source_tree.query(source, k=2)[0][:, 1]), 'source spacing')
        self.target_spacing = _positive(np.median(target_tree.query(self.target, k=2)[0][:, 1]), 'target spacing')
        self.radius = _positive(np.median(target_tree.query(self.target, k=9)[0][:, 8]), 'density radius')
        self.extent = _positive(target_extent(self.target), 'target extent')
        self.tip = self.target[self.target[:, 1].argmax()].copy()
        self.upper_target = self.target[:, 1] > 2.3
        counts = source_tree.query_ball_point(source, 2*self.source_spacing, return_length=True)
        upper = source[:, 1] >= (source[:, 1].min()+source[:, 1].max())*.5
        self.source_ids = np.flatnonzero(upper & (counts < .6*np.median(counts))).astype(np.int64)
        if not len(self.source_ids) or not bool(self.upper_target.any()):
            raise ValueError('Raw quality requires nonempty fixed source and upper-target cohorts')
        self.cohort = dict(count=len(self.source_ids),
            ids_sha256=sha256(to_host(self.source_ids.astype('<i8')).tobytes()).hexdigest(),
            pinned_count=int(self.pins[self.source_ids].sum()), includes_pinned_ids=True,
            definition='Original source y>=bbox midpoint and inclusive r=2*source-spacing count<0.6*source median count; all selected IDs, no resampling',
            density_definition='Inclusive neighbors at fixed target median k9[:,8] radius, subtract self; FP64 mean count/8; under-half count<4')
        self.target_masks = {}
        self.target_holes = {}
        for res in RESOLUTIONS:
            masks = np.stack([_splat_body(self.target, res, az, el, self.extent) for az, el in VIEWS])
            self.target_masks[res] = masks
            self.target_holes[res] = np.stack([ndimage.binary_fill_holes(mask) & ~mask for mask in masks])
        self.baseline_labels = []
        self.minimum, self.maximum = {}, {}
        self.stable_covered = None
        self.steps = None

    def _context(self):
        if self.cuda != is_cuda_execution():
            raise ValueError('Raw quality requires its original compute backend context')
        if self.cuda and self.target.device.id != np.cuda.runtime.getDevice():
            raise ValueError('Raw quality requires its original CUDA device')

    def _phase(self, x):
        """Native arrays only; radius counts use the bounded compute.KDTree wrapper."""
        tree = KDTree(x)
        covered = tree.query(self.target)[0] <= 2*self.target_spacing
        neighbors = tree.query_ball_point(x[self.source_ids], self.radius, return_length=True)-1
        result = dict(covered=covered, target_covered=covered.sum(dtype=np.int64),
            upper_target_covered=covered[self.upper_target].sum(dtype=np.int64),
            tip_count=(np.linalg.norm(x-self.tip, axis=1) < .25).sum(dtype=np.int64),
            fixed_source_density=neighbors.mean(dtype=np.float64)/8.,
            under_half_count=(neighbors < 4).sum(dtype=np.int64))
        result.update(target_near_fraction=covered.mean(dtype=np.float64),
            upper_target_near_fraction=covered[self.upper_target].mean(dtype=np.float64),
            under_half_fraction=(neighbors < 4).mean(dtype=np.float64))
        image = {key: [] for key in ('iou', 'hole_pixels', 'extra_hole_pixels', 'clipped_centers')}
        for res in RESOLUTIONS:
            rows = {key: [] for key in image}
            for view, (azimuth, elevation) in enumerate(VIEWS):
                body = _splat_body(x, res, azimuth, elevation, self.extent)
                holes = ndimage.binary_fill_holes(body) & ~body
                target = self.target_masks[res][view]
                intersection = (body & target).sum(dtype=np.int64)
                union = (body | target).sum(dtype=np.int64)
                rows['iou'].append(intersection.astype(np.float64)/np.maximum(union, 1))
                rows['hole_pixels'].append(holes.sum(dtype=np.int64))
                rows['extra_hole_pixels'].append((holes & ~self.target_holes[res][view]).sum(dtype=np.int64))
                rows['clipped_centers'].append(_clipped_centers(x, res, azimuth, elevation, self.extent))
            for key in image:
                image[key].append(np.stack(rows[key]))
        result.update({key: np.stack(value) for key, value in image.items()})
        return result

    def _trajectory(self, values):
        head, coast = _native(values['positions']), _native(values['coast_X'])
        if (head.ndim != 3 or head.shape[1:] != self.x0.shape or len(head) < 1
                or coast.shape != (len(head)+1, *self.x0.shape)
                or head.dtype != np.float32 or coast.dtype != np.float32
                or not bool(np.isfinite(head).all()) or not bool(np.isfinite(coast).all())):
            raise ValueError('Invalid head/coast phase layout or finite state')
        if not bool(np.array_equal(head[-1], coast[0])):
            raise ValueError('Head endpoint and coast boundary differ')
        if self.steps is not None and len(head) != self.steps:
            raise ValueError('Head/coast phase count changed')
        return head, coast

    def observe(self, label, values, baseline=False):
        self._context()
        if baseline and (len(self.baseline_labels) >= 3 or label in self.baseline_labels):
            raise ValueError('Exactly three distinctly labelled baselines are immutable')
        head, coast = self._trajectory(values)
        # Phase arrays are consumed in order; none is retained by the observer.
        observations = [self._phase(self.x0)]
        observations.extend(self._phase(x) for x in head)
        observations.extend(self._phase(x) for x in coast[1:])
        arrays = {key: np.stack([row[key] for row in observations]) for key in observations[0]}
        for key, value in arrays.items():
            if not bool(np.isfinite(value).all()):
                raise ValueError('Nonfinite raw measurement: '+key)
        valid = bool(values.get('valid', True)) and bool(values.get('pins_exact', True))
        if baseline and not valid:
            raise ValueError('Invalid original forward cannot establish baseline quality')
        if baseline:
            if self.stable_covered is None:
                self.stable_covered = arrays['covered'].copy()
                self.minimum = {key: arrays[key].copy() for key in LOWER}
                self.maximum = {key: arrays[key].copy() for key in UPPER}
                self.steps = len(head)
            else:
                self.stable_covered &= arrays['covered']
                for key in LOWER:
                    self.minimum[key] = np.minimum(self.minimum[key], arrays[key])
                for key in UPPER:
                    self.maximum[key] = np.maximum(self.maximum[key], arrays[key])
            self.baseline_labels.append(str(label))
        ready = len(self.baseline_labels) == 3
        failures = []
        lost = np.zeros(len(observations), dtype=np.int64)
        if not baseline:
            if not ready:
                failures.append(dict(gate='three_baselines_required'))
            elif self.stable_covered is not None:
                lost = (self.stable_covered & ~arrays['covered']).sum(axis=1, dtype=np.int64)
                for key, bounds, lower in ((key, self.minimum, True) for key in LOWER):
                    self._compare(key, arrays[key], bounds[key], lower, failures)
                for key in UPPER:
                    self._compare(key, arrays[key], self.maximum[key], False, failures)
                for phase in to_host(np.flatnonzero(lost)).tolist():
                    failures.append(dict(gate='lost_stable_target_ids', phase=phase, count=int(lost[phase])))
            if not valid:
                failures.append(dict(gate='invalid_forward'))
        rows = []
        # This is the JSON reporting boundary, after all numerical work on device.
        host = {key: to_host(value).tolist() for key, value in arrays.items() if key != 'covered'}
        host_lost = to_host(lost).tolist()
        for phase in range(len(observations)):
            kind, step = ('initial', 0) if phase == 0 else (('head', phase) if phase <= len(head)
                                                          else ('coast', phase-len(head)))
            row = dict(phase=phase, kind=kind, step=step, lost_stable_target_ids=host_lost[phase])
            row.update({key: value[phase] for key, value in host.items()})
            row['failing_gates'] = [entry for entry in failures if entry.get('phase') == phase]
            rows.append(row)
        return dict(label=str(label), baseline=bool(baseline), passed=not failures,
            baseline_count=len(self.baseline_labels), baseline_labels=list(self.baseline_labels),
            baseline_ready=ready, phases=rows,
            failing_gates=failures, cohort=deepcopy(self.cohort),
            source_spacing=self.source_spacing, target_spacing=self.target_spacing,
            density_radius=self.radius, extent=self.extent, target_count=len(self.target),
            upper_target_count=int(self.upper_target.sum()), resolutions=list(RESOLUTIONS),
            views=[list(view) for view in VIEWS],
            scope='Independent raw 2T+1 phases; fixed reference; candidate selection, not post-selection validation or rest',
            comparison='Integer gates exact; floating gates allow only 64*float64_eps*(1+abs(bound)); baseline extrema from three originals',
            extra_hole_scope='Report-only bitmap count of body holes outside target enclosed-hole pixels; not max(body-hole count minus target-hole count,0). Absolute enclosed-hole counts gate each view')

    @staticmethod
    def _compare(key, value, bound, lower, failures):
        tolerance = 64*np.finfo(np.float64).eps*(1+np.abs(bound)) if value.dtype.kind == 'f' else 0
        bad = value < bound-tolerance if lower else value > bound+tolerance
        for index in to_host(np.argwhere(bad)).tolist():
            where = tuple(index)
            item = dict(gate=key, phase=index[0], actual=float(value[where]) if value.dtype.kind == 'f'
                        else int(value[where]), bound=float(bound[where]) if value.dtype.kind == 'f'
                        else int(bound[where]), relation='>=' if lower else '<=')
            if len(index) == 3:
                item.update(resolution=RESOLUTIONS[index[1]], view=index[2])
            failures.append(item)

    def archive_state(self):
        """Owned numeric arrays on the active backend for the caller's NPZ writer."""
        self._context()
        result = dict(baseline_count=np.asarray(len(self.baseline_labels), dtype=np.int64),
            source_ids=self.source_ids.copy(), source_cohort_pins=self.pins[self.source_ids].copy(),
            upper_target_ids=np.flatnonzero(self.upper_target).astype(np.int64),
            views=np.asarray(VIEWS, dtype=np.float64), resolutions=np.asarray(RESOLUTIONS, dtype=np.int64))
        result.update({key: np.asarray(value, dtype=np.float64) for key, value in dict(
            source_spacing=self.source_spacing, target_spacing=self.target_spacing,
            density_radius=self.radius, extent=self.extent).items()})
        result.update(target_tip=self.tip.copy(),
            target_count=np.asarray(len(self.target), dtype=np.int64),
            source_count=np.asarray(len(self.x0), dtype=np.int64),
            head_steps=np.asarray(0 if self.steps is None else self.steps, dtype=np.int64),
            target_hole_pixels=np.stack([self.target_holes[res].sum(axis=(1, 2), dtype=np.int64)
                                        for res in RESOLUTIONS]))
        if self.stable_covered is not None:
            result['stable_covered'] = self.stable_covered.copy()
            result.update({'minimum_'+key: value.copy() for key, value in self.minimum.items()})
            result.update({'maximum_'+key: value.copy() for key, value in self.maximum.items()})
        return result
