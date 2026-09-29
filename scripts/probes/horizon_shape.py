"""Accepted archive phases plus raw terminal states; shape observations without rendering."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as host_np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import array_api as np, cuda_execution, to_array, to_host, KDTree, ndimage
from physmorph.metrics import _splat_body, target_extent
from physmorph.pipeline.render_loss import make_views
from scripts.probes.coverage_paths import require, sha
from scripts.probes.horizon_motion import FrameReader, bound_json
from scripts.probes.silhouette_pixels import projection


def physical_frame_indices(windows):
    """One initial frame plus all accepted steps, never the runner-held padding."""
    result = [0]
    for row in windows:
        start, end = int(row['start_frame']), int(row['end_frame'])
        require(start >= result[-1] and end > start, 'Invalid/overlapping accepted intervals')
        result.extend(range(start+1, end+1))
    return result


def sample_schedule(windows):
    physical_frame_indices(windows)  # Validate the disjoint accepted intervals.
    result = [dict(kind='archive', archive_frame=0, paired_archive_frame=0, animation=None)]
    for row in windows:
        start, end = int(row['start_frame']), int(row['end_frame'])
        for index in range(start+1, end+1):
            if index == end:
                result.append(dict(kind='optimizer_raw_endpoint', archive_frame=None,
                                   paired_archive_frame=index, animation=int(row['animation'])))
            result.append(dict(kind='promoted_endpoint' if index == end else 'archive',
                               archive_frame=index, paired_archive_frame=index, animation=int(row['animation'])))
    return result


def pack_mask_rows(masks):
    """CuPy packbits has no axis option; zero-pad rows then pack the flat buffer."""
    rows = masks.reshape(len(masks), -1)
    pad = (-rows.shape[1]) % 8
    if pad:
        rows = np.pad(rows, ((0, 0), (0, pad)), mode='constant', constant_values=False)
    return np.packbits(rows).reshape(len(rows), rows.shape[1]//8)


class ShapeObserver:
    def __init__(self, target, resolutions=(128, 256), views=None):
        self.target = target
        self.extent = target_extent(target)
        self.resolutions = tuple(resolutions)
        self.views = list(make_views(8, (0., .5, -.5)) if views is None else views)
        self.target_tree = KDTree(target)
        self.spacing = float(np.median(self.target_tree.query(target, k=2)[0][:, 1]))
        require(self.spacing > 0 and self.extent > 0, 'Invalid target sampling')
        self.targets = {res: np.stack([_splat_body(target, res, az, el, self.extent)
                                      for az, el in self.views]) for res in self.resolutions}
        self.target_holes = {res: np.stack([ndimage.binary_fill_holes(mask) & ~mask for mask in masks])
                             for res, masks in self.targets.items()}
        self.previous_holes = None

    def frame(self, x):
        require(x.ndim == 2 and x.shape[1] == 3 and bool(np.isfinite(x).all()), 'Invalid raw phase')
        x = np.ascontiguousarray(x, np.float32)
        distance = KDTree(x).query(self.target)[0]
        covered = distance <= 2*self.spacing
        outside_target = self.target_tree.query(x)[0] > 2*self.spacing
        row = dict(target_covered=int(covered.sum()), target_particles=len(self.target),
                   source_outside_target_support=int(outside_target.sum()),
                   source_outside_extent_box=int((np.abs(x) > self.extent).any(1).sum()), views={})
        packed = dict(covered_target_bits=np.packbits(covered))
        holes_now = {}
        for res in self.resolutions:
            bodies = np.stack([_splat_body(x, res, az, el, self.extent) for az, el in self.views])
            holes = np.stack([ndimage.binary_fill_holes(mask) & ~mask for mask in bodies])
            target = self.targets[res]
            body_counts = bodies.sum((1, 2))
            hole_counts = holes.sum((1, 2))
            union = (bodies | target).sum((1, 2))
            intersection = (bodies & target).sum((1, 2))
            values = dict(body_pixels=body_counts, internal_hole_pixels=hole_counts,
                projected_outside_centers=np.stack([(~projection(x, res, az, el, self.extent)[1]).sum()
                                                    for az, el in self.views]),
                hole_fraction=hole_counts/np.maximum(body_counts+hole_counts, 1),
                target_iou=np.where(union > 0, intersection/np.maximum(union, 1), 1.),
                hole_pixels_outside_target_holes=(holes & ~self.target_holes[res]).sum((1, 2)))
            if self.previous_holes is not None:
                values.update(new_hole_pixels=(holes & ~self.previous_holes[res]).sum((1, 2)),
                              removed_hole_pixels=(~holes & self.previous_holes[res]).sum((1, 2)))
            row['views'][str(res)] = {k: to_host(v).tolist() for k, v in values.items()}
            packed[f'body_{res}_bits'] = pack_mask_rows(bodies)
            holes_now[res] = holes
        self.previous_holes = holes_now
        return row, packed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/data/relcfd/chayo/physmorph_v2').resolve()
    require(args.motion.resolve().is_relative_to(root) and args.out.resolve().is_relative_to(root),
            'Inputs/output outside project data')
    args.out.mkdir(exist_ok=False)
    motion_path = args.motion/'result.json'
    motion, digest = bound_json(motion_path)
    bindings = {str(motion_path): digest, **motion['bindings']}
    require(sha(args.motion/'particles.npz') == motion['particles_sha256'], 'Unbound motion analysis')
    bindings[str(args.motion/'particles.npz')] = motion['particles_sha256']
    require(all(sha(Path(k)) == v for k, v in bindings.items()), 'Motion inputs changed')
    raw_paths = [Path(p) for p in bindings if p.endswith('_render_full_dt_iso_nn.npz')]
    require(len(raw_paths) == 1, 'Expected one complete raw archive')
    raw = raw_paths[0]
    code_root = Path(__file__).resolve().parents[2]
    code = {str(p): sha(p) for p in sorted((code_root/'physmorph').rglob('*.py'))}
    for name in ('horizon_shape.py', 'horizon_motion.py', 'coverage_paths.py', 'full_horizon.py', 'silhouette_pixels.py'):
        path = Path(__file__).with_name(name)
        code[str(path)] = sha(path)
    frames = physical_frame_indices(motion['windows'])
    samples = sample_schedule(motion['windows'])
    trace_paths = [Path(p) for p in bindings if p.endswith('.rest_trace.json')]
    require(len(trace_paths) == 1, 'Expected one traced producer')
    trace, trace_sha = bound_json(trace_paths[0])
    require(trace_sha == bindings[str(trace_paths[0])], 'Trace binding changed')
    attempts = {int(row['animation']): row for row in trace['attempts']}
    prefix = raw.with_name(raw.name.removesuffix('_render_full_dt_iso_nn.npz'))
    reader = FrameReader(raw)
    require(frames[-1] == motion['windows'][-1]['end_frame'] and
            reader.shape[0] == motion['actual_frames'], 'Physical frame scope mismatch')
    require(reader.shape[1] == 300000 and motion['discretization']['T'] == 20 and len(samples) <= 6301,
            'P316 particle/horizon size exceeds the reserved scope')
    rows, chunks = [], []
    with cuda_execution('cuda:0'):
        with host_np.load(raw, allow_pickle=False) as z:
            target = to_array(z['tgt'])
            require(target.shape == (300000, 3), 'P316 target size exceeds the reserved scope')
            observer = ShapeObserver(target)
        target_bits = {f'target_{res}_bits': to_host(pack_mask_rows(mask))
                       for res, mask in observer.targets.items()}
        with (args.out/'target_masks.npz').open('xb') as stream:
            host_np.savez_compressed(stream, **target_bits)
        pending, pending_idx = {}, []
        try:
            for ordinal, sample in enumerate(samples):
                if sample['kind'] == 'optimizer_raw_endpoint':
                    attempt = attempts[sample['animation']]
                    path = prefix.with_name(prefix.name+'_cohorts')/attempt['sidecar']
                    require(str(path) in bindings and bindings[str(path)] == attempt['sha256'], 'Unbound raw endpoint')
                    with host_np.load(path, allow_pickle=False) as z:
                        x = to_array(z['optimizer_raw_endpoint'])
                else:
                    index = sample['archive_frame']
                    x = to_array(reader.read(index, index)[0])
                row, bits = observer.frame(x)
                row.update(sample, sample_index=ordinal)
                rows.append(row); pending_idx.append(ordinal)
                for name, value in bits.items():
                    pending.setdefault(name, []).append(value)
                if len(pending_idx) == 20 or ordinal == len(samples)-1:
                    path = args.out/f'samples_{pending_idx[0]:05d}_{pending_idx[-1]:05d}.npz'
                    arrays = {name: to_host(np.stack(values)) for name, values in pending.items()}
                    with path.open('xb') as stream:
                        host_np.savez_compressed(stream, sample_indices=host_np.asarray(pending_idx, host_np.int64), **arrays)
                    chunks.append(dict(path=path.name, sha256=sha(path), sample_indices=pending_idx))
                    print(json.dumps(dict(last_sample=sample, observed_samples=len(rows),
                                          target_covered=row['target_covered'],
                                          holes128=row['views']['128']['internal_hole_pixels'])), flush=True)
                    pending, pending_idx = {}, []
        finally:
            reader.close()
        target_holes = {str(res): to_host(value.sum((1, 2))).tolist() for res, value in observer.target_holes.items()}
    require(all(sha(Path(k)) == v for k, v in {**bindings, **code}.items()), 'Inputs/code changed during phase audit')
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), bindings=bindings, code=code,
        discretization=motion['discretization'], arm=motion['arm'], resolutions=list(observer.resolutions),
        views=observer.views, extent_wu=observer.extent, target_spacing_wu=observer.spacing,
        target_hole_pixels=target_holes, samples=rows, accepted_archive_frames=frames, chunks=chunks,
        target_masks_sha256=sha(args.out/'target_masks.npz'), delivery_frame=motion['delivery_frame'],
        render_influence=motion['render_influence'],
        scope='All accepted archived frames plus initial state and separately labelled raw optimizer endpoints; '
              'phase T in the archive is the promoted/PIC endpoint. The additional raw xT shares that time, not an extra dt. '
              'Held/null/rejected duplicates excluded; new/removed holes follow the labelled observation sequence. '
              'Coverage is target IDs within two target spacings; incomplete coverage during transport is expected. '
              'Binary 3x3 masks at fixed target extent,24 views and128/256 pixels; no Gaussian renderer or loss operator. '
              'Projected interior holes may be genuine shape openings; target-hole differences are diagnostic only. '
              'Axis-aligned extent-box exclusion and per-view projected center exclusion are separate observations. '
              'Zero counts do not prove3D watertightness or4K appearance. No physical state is changed.')
    with (args.out/'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
