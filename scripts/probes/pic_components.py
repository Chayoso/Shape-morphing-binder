"""P296: partition an accepted PIC jump; inspect candidate endpoints without committing them."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.probes.boundary_replay import sha, write_json

SELECTED = (1, 24, 30)
QUALITY_DIRECTIONS = dict(chamfer=-1, sil_iou=1, hole_frac=-1, target_near_frac=1,
                          top_target_near_frac=1, tip_n=1, source_out_far_frac=-1,
                          top_density=1, top_under_half=-1)


def utc():
    return datetime.now(timezone.utc).isoformat()


def quality_contrast(current, candidate):
    """An exact directional screen, with no newly fitted metric tolerances."""
    deltas, no_worse = {}, {}
    for name, direction in QUALITY_DIRECTIONS.items():
        a, b = current[name], candidate[name]
        deltas[name] = None if a is None or b is None else b-a
        no_worse[name] = None if deltas[name] is None else direction*deltas[name] >= 0
    return dict(candidate_minus_current=deltas, no_worse=no_worse,
                all_observed_no_worse=all(value is True for value in no_worse.values()),
                scope='fixed endpoint screen; no epsilon and no trajectory/repair inference')


def endpoint_admissibility(x, start, pin, bounds):
    from physmorph.pipeline.endpoint_contract import valid_endpoint
    shape = x.shape == start.shape and x.device == start.device and x.dtype == start.dtype
    finite_in_bounds = shape and valid_endpoint(x, bounds)
    pins_exact = shape and torch.equal(x[pin], start[pin])
    return dict(same_shape_device_dtype=shape, finite_in_bounds=finite_in_bounds,
                pins_exact=pins_exact, passed=shape and finite_in_bounds and pins_exact)


class Capture:
    def __init__(self, out, reference, selected=SELECTED):
        self.out, self.reference, self.selected = Path(out), reference, tuple(selected)
        self.accepted, self.pending, self.rows = 0, None, []

    def observe(self, tr, promoted, win_index):
        import warp as wp
        from physmorph.pipeline.motion_accounting import collect_rollout
        selected = self.accepted + 1 in self.selected
        self.pending = dict(attempt=int(win_index)+1, selected=selected)
        if not selected:
            return
        accounting = collect_rollout(tr, tr.prm.dt)
        self.pending.update(sums=accounting['sums'], step_closure_wu=accounting['closure_wu'],
                            layer_mask=(wp.to_torch(tr.layer_mask).detach().clone() >= .5 if tr.layer
                                        else torch.zeros(tr.N, dtype=torch.bool, device=wp.to_torch(tr.x[0]).device)),
                            pin=wp.to_torch(tr.pin).detach().clone() > .5, prm=tr.prm, T=tr.T,
                            raw=wp.to_torch(tr.x[tr.T]).detach().clone(),
                            start=wp.to_torch(tr.x[0]).detach().clone())

    def wrap(self, original):
        def wrapped(*args, **kwargs):
            self.pending = None
            if 'on_rollout' in kwargs:
                raise ValueError('Diagnostic observer already installed')
            result = original(*args, on_rollout=self.observe, **kwargs)
            if self.pending is None or not self.pending['selected']:
                return result
            from physmorph.pipeline.endpoint_contract import endpoint_bounds
            from scripts.probes.pic_component_metrics import decompose_pic
            from scripts.probes.pic_endpoint_quality import endpoint_quality
            owned = result[-1]['owned_endpoint']
            pending = self.pending
            if owned is None or not all(torch.equal(getattr(owned, key), pending[key])
                                        for key in ('start', 'raw', 'pin')):
                raise RuntimeError('Validated accounting and owned PIC endpoint disagree')
            part = decompose_pic(owned.operator, owned.start, owned.raw, owned.promoted,
                                 owned.pin, pending['sums'], pending['layer_mask'],
                                 self.reference['source_spacing'])
            bounds = endpoint_bounds(pending['prm'], owned.start)
            admissible = {name: endpoint_admissibility(x, owned.start, owned.pin, bounds)
                          for name, x in part['endpoints'].items()}
            geometry = {name: endpoint_quality(x, self.reference) if admissible[name]['passed'] else None
                        for name, x in part['endpoints'].items()}
            if geometry['current'] is None:
                raise RuntimeError('Actual accepted endpoint is not admissible')
            comparisons = {name: quality_contrast(geometry['current'], value)
                           if part['report']['valid'] and value is not None else None
                           for name, value in geometry.items() if name != 'current'}
            pending.update(endpoints=part['endpoints'], report=part['report'], admissible=admissible,
                           geometry=geometry, comparisons=comparisons)
            return result
        return wrapped

    def commit(self, attempt, x, F, v, record):
        if not record.get('frame_end'):
            self.pending = None
            return
        self.accepted += 1
        pending = self.pending
        if pending is None or pending['attempt'] != int(attempt)+1:
            raise RuntimeError('Accepted commit lacks its validated observer')
        if not pending['selected']:
            self.pending = None
            return
        from physmorph.compute import to_host
        current = pending['endpoints']['current']
        if not torch.equal(current, torch.as_tensor(x, dtype=current.dtype, device=current.device)):
            raise RuntimeError('Runner did not commit the owned endpoint that was analyzed')
        path = self.out / f'endpoint_{self.accepted:03d}.npz'
        arrays = {name: to_host(value) for name, value in pending['endpoints'].items()}
        arrays.update(start=to_host(pending['start']), pin=to_host(pending['pin']),
                      layer_mask=to_host(pending['layer_mask']))
        with path.open('xb') as stream:
            np.savez(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        row = dict(commit=self.accepted, attempt=int(attempt)+1, T=pending['T'],
                   step_component_closure_wu=pending['step_closure_wu'],
                   decomposition=pending['report'], admissible=pending['admissible'],
                   geometry=pending['geometry'], comparisons=pending['comparisons'],
                   comparison_eligible=pending['report']['valid'],
                   comparison_ineligible_reason=None if pending['report']['valid'] else 'decomposition contract failed',
                   endpoints_path=str(path), endpoints_sha256=sha(path), endpoints_bytes=path.stat().st_size)
        write_json(self.out / f'endpoint_{self.accepted:03d}.json', row)
        self.rows.append(row)
        print(json.dumps(dict(commit=self.accepted, attempt=int(attempt)+1,
                              decomposition_valid=row['decomposition']['valid'],
                              endpoint_bytes=row['endpoints_bytes'])), flush=True)
        self.pending = None
        if not row['decomposition']['valid']:
            raise RuntimeError('Accepted decomposition failed numerical/pin contract; see endpoint JSON')


def code_hashes():
    paths = sorted((ROOT / 'physmorph').rglob('*.py'))
    paths += [ROOT / 'scripts/probes' / name for name in (
        'pic_components.py', 'pic_component_metrics.py', 'pic_endpoint_quality.py', 'boundary_replay.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def run(root, out, selected, windows):
    from physmorph.compute import cuda_execution
    from physmorph.pipeline import PipelineConfig, runner
    from physmorph.mpm.state import MPMParams
    from scripts.probes.pic_endpoint_quality import prepare_reference
    baseline_path = root / 'work/p295/full60_v2/protocol.json'
    baseline_bytes = baseline_path.read_bytes()
    baseline = json.loads(baseline_bytes)
    hashes = code_hashes()
    for name, expected in baseline['code_sha256'].items():
        if name.startswith('physmorph/') and hashes.get(name) != expected:
            raise RuntimeError(f'P296 simulation core must equal P295: {name}')
    source = Path(baseline['source'])
    before = source.stat()
    with np.load(source, allow_pickle=False) as archive:
        src, tgt = archive['src'], archive['tgt']
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError('Input archive changed while reading')
    for name, value in (('source', src), ('target', tgt)):
        if hashlib.sha256(value.tobytes()).hexdigest() != baseline[name + '_array_sha256']:
            raise RuntimeError(f'{name} differs from immutable P295 inputs')
    config = dict(baseline['config'], stop_after_windows=windows)
    if sha(config['target_reference']) != baseline['target_reference_sha256']:
        raise RuntimeError('Target shading reference changed')
    prm = MPMParams(**baseline['mpm'])
    with cuda_execution('cuda'):
        reference = prepare_reference(src, tgt)
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(start_utc=utc(), selected_accepted_windows=list(selected), attempt_cap=windows,
                    baseline_protocol=str(baseline_path), baseline_protocol_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
                    source=str(source), source_array_sha256=baseline['source_array_sha256'],
                    target_array_sha256=baseline['target_array_sha256'],
                    config=config, mpm=baseline['mpm'], code_sha256=hashes,
                    quality_directions=QUALITY_DIRECTIONS,
                    quality_policy='exact same-endpoint directional screen; missing values unresolved; no fitted epsilon',
                    numerical_policy='helper records dtype roundoff/owned-map/repeat/linearity errors; fail closed',
                    cohorts='current window-start free and layer-free IDs; not P295 next-window cohorts',
                    candidates=['raw', 'current', 'advection_only', 'preserve_relaxation'],
                    scope='linear partition of one fixed optimized rollout, not causal channel-off experiments',
                    limits=['candidate endpoints are never committed', 'F/v/C and controls belong to the original rollout',
                            'endpoint quality does not establish intermediate continuity, later rest or watertightness'],
                    reference={k: reference[k] for k in ('source_spacing', 'target_spacing', 'radius', 'extent')},
                    visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write_json(out / 'protocol.json', protocol)
    capture = Capture(out, reference, selected)
    started = time.monotonic()
    with patch.object(runner, 'optimize_window', capture.wrap(runner.optimize_window)):
        result = runner.run_pipeline(src, tgt, prm, PipelineConfig(**config), on_commit=capture.commit)
    missing = sorted(set(selected) - {row['commit'] for row in capture.rows})
    report = dict(end_utc=utc(), seconds=time.monotonic()-started, accepted=capture.accepted,
                  history=result['history'], guards=result['guards'], missing_windows=missing,
                  selected=capture.rows, protocol_sha256=sha(out / 'protocol.json'),
                  torch_peak_bytes=torch.cuda.max_memory_allocated())
    write_json(out / 'result.json', report)
    print(json.dumps(dict(seconds=report['seconds'], accepted=report['accepted'],
                          guards=report['guards'], missing=missing)), flush=True)
    if missing:
        raise RuntimeError('Pre-registered window missing; no substitution allowed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selected', type=int, nargs='+', default=SELECTED)
    parser.add_argument('--windows', type=int, default=60)
    args = parser.parse_args()
    args.root, args.out = args.root.resolve(), args.out.resolve()
    if any(not path.is_relative_to(Path('/data')) for path in (args.root, args.out)):
        parser.error('All probe paths must be under /data')
    if not args.selected or min(args.selected) < 1 or len(set(args.selected)) != len(args.selected):
        parser.error('Selected accepted windows must be distinct positive integers')
    if args.windows < 1 or max(args.selected) > args.windows:
        parser.error('Positive attempt cap must cover every selected accepted window')
    for name in ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH'):
        if not os.environ.get(name) or not Path(os.environ[name]).resolve().is_relative_to(Path('/data')):
            parser.error(f'{name} must be explicit under /data')
    if not torch.cuda.is_available():
        parser.error('CUDA required; no CPU numerical fallback')
    import warp as wp
    wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
    wp.init()
    run(args.root, args.out, tuple(args.selected), args.windows)


if __name__ == '__main__':
    main()
