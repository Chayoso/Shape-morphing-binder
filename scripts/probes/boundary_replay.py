"""Read-only conditional PIC handoff experiment; GPU numerics, archive-only CPU I/O.

Capture an actual accepted next window, then replay only its first step at its
original T. The intervention removes the prior PIC position correction on next-
window free IDs. All other post-commit state, controls and pin anchors stay fixed.
This is not a historical no-PIC optimization or a state-consistent repair.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
BOUNDARIES = (1, 24, 30)  # One-based accepted commits, fixed before results.
ATOL, RTOL = 3e-6, 3e-5


def check_original_step(got, reference, field, dt, dx):
    """Compare dimensionless state; C has units 1/s, unlike x or F."""
    factors = dict(x1=1 / dx, pre_layer=1 / dx, v1=dt / dx, F1=1., C1=dt)
    names = dict(x1='x/dx', pre_layer='x/dx', v1='dt*v/dx', F1='F', C1='dt*C')
    scale = factors[field]
    difference = got.double() - reference.double()
    error = difference.abs() * scale
    allowed = ATOL + RTOL * (reference.double() * scale).abs()
    return dict(passed=bool((error <= allowed).all()), normalization=names[field],
                max_abs_native_units=float(difference.abs().max()),
                component_rms_native_units=float(difference.square().mean().sqrt()),
                max_abs_dimensionless=float(error.max()),
                max_tolerance_ratio=float((error / allowed).max()))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def save_packet(path, packet):
    # Owned arrays cross the host boundary solely for checkpoint I/O.
    with Path(path).open('xb') as stream:
        np.savez(stream, __meta__=json.dumps(packet['meta'], allow_nan=False), **packet['arrays'])
        stream.flush()
        os.fsync(stream.fileno())


def load_packet(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(meta=json.loads(str(archive['__meta__'])),
                    arrays={key: archive[key] for key in archive.files if key != '__meta__'})


class Capture:
    """Inner observers are tentative; only outer accepted callbacks admit evidence."""

    def __init__(self, out, boundaries=BOUNDARIES):
        self.out = Path(out)
        self.boundaries = tuple(boundaries)
        self.accepted = 0
        self.pending = None
        self.candidate = None
        self.saved = []

    def observe(self, tr, promoted, win_index):
        from physmorph.compute import to_host
        import warp as wp
        from scripts.probes.boundary_packet import capture_step
        candidate = dict(attempt=int(win_index) + 1, packet=None, boundary=None)
        if self.pending is not None:
            candidate['packet'] = capture_step(tr)
        if self.accepted + 1 in self.boundaries:
            candidate['boundary'] = dict(
                previous=to_host(wp.to_torch(tr.x[tr.T - 1])),
                raw=to_host(wp.to_torch(tr.x[tr.T])), promoted=to_host(promoted))
        self.candidate = candidate

    def commit(self, attempt, x, F, v, record):
        if not record.get('frame_end'):
            self.candidate = None
            return
        self.accepted += 1
        candidate = self.candidate
        if candidate is None or candidate['attempt'] != int(attempt) + 1:
            raise RuntimeError('Accepted outer commit lacks its validated inner observer')
        if self.pending is not None:
            packet = candidate['packet']
            if packet is None:
                raise RuntimeError('Missing next-window checkpoint')
            packet['arrays'].update(self.pending['arrays'])
            packet['meta'].update(boundary_commit=self.pending['commit'],
                                  boundary_attempt=self.pending['attempt'],
                                  next_commit=self.accepted, next_attempt=int(attempt) + 1,
                                  intervention='free x0 only; actual post-commit F/Fp/v/C, pins and controls fixed',
                                  shift_sub=False, first_step_only=True)
            path = self.out / f'boundary_{self.pending["commit"]:03d}.npz'
            save_packet(path, packet)
            self.saved.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                                   boundary_commit=self.pending['commit'], next_commit=self.accepted,
                                   next_attempt=int(attempt) + 1))
            print(json.dumps(dict(checkpoint=self.saved[-1])), flush=True)
            self.pending = None
        if candidate['boundary'] is not None:
            self.pending = dict(commit=self.accepted, attempt=int(attempt) + 1,
                                arrays=candidate['boundary'])
        self.candidate = None

    def wrap(self, original):
        def observed(*args, **kwargs):
            self.candidate = None
            if 'on_rollout' in kwargs:
                raise ValueError('Diagnostic observer already installed')
            return original(*args, on_rollout=self.observe, **kwargs)
        return observed


def code_hashes():
    paths = sorted((ROOT / 'physmorph').rglob('*.py'))
    paths += [ROOT / 'scripts/probes' / name for name in (
        'boundary_replay.py', 'boundary_packet.py', 'boundary_metrics.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def capture(root, out, boundaries, windows):
    from physmorph.compute import cuda_execution, KDTree, to_array
    from physmorph.pipeline import PipelineConfig
    from physmorph.pipeline import runner
    from physmorph.mpm.state import MPMParams
    from scripts.probes.pic_endpoint import SOURCE_SHA

    source = root / 'output/c291/c291_bunny_mixed60_render_full_dt_iso_nn.npz'
    metadata = root / 'output/c291/c291_bunny_mixed60.json'
    metadata_bytes = metadata.read_bytes()
    original = json.loads(metadata_bytes)
    before = source.stat()
    with np.load(source, allow_pickle=False) as archive:
        src, tgt = archive['src'], archive['tgt']
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError('Source changed during read')
    source_sha = hashlib.sha256(src.tobytes()).hexdigest()
    if src.shape != (300000, 3) or src.dtype != np.float32 or source_sha != SOURCE_SHA:
        raise ValueError('Expected immutable original mixed60 source')
    config = {k: v for k, v in original['arms']['render_full_dt_iso_nn']['config'].items()
              if k in PipelineConfig.__dataclass_fields__}
    config.update(compute_backend='cuda', stop_after_windows=windows, iters=8,
                  target_reference=str(root / 'work/gpu_refactor/target_reference.npz'),
                  commit_pic=True, commit_pic_objective=True, shift_sub=False,
                  outer_render_committed=True, geometric_rest=False, grad_dump='',
                  motion_accounting=False)
    prm = MPMParams(**original['provenance']['mpm'])
    with cuda_execution('cuda'):
        distances, _ = KDTree(to_array(src)).query(to_array(src), k=2)
        spacing = float(torch.quantile(torch.as_tensor(distances[:, 1], device='cuda'), .5))
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(start_utc=utc(), boundaries=list(boundaries), attempt_cap=windows,
                    source=str(source), source_array_sha256=source_sha,
                    target_array_sha256=hashlib.sha256(tgt.tobytes()).hexdigest(),
                    metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                    target_reference_sha256=sha(config['target_reference']),
                    config=config, mpm=asdict(prm), native_spacing_wu=spacing,
                    code_sha256=code_hashes(), closure_tolerance=dict(atol=ATOL, rtol=RTOL,
                        protocol_version=2, units='dimensionless',
                        normalization=dict(x1='x/dx', pre_layer='x/dx', v1='dt*v/dx', F1='F', C1='dt*C')),
                    closure_fields=['x1', 'pre_layer', 'v1', 'F1', 'C1'],
                    comparison='same actual accepted next rollout; only free x0 is intervened',
                    limitations=['post-PIC assimilation, pins and optimized controls are conditioned on',
                                 'density-dependent bond activation and P2G weights may change',
                                 'one-step causal response does not prove a quality repair'],
                    visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write_json(out / 'protocol.json', protocol)
    coordinator = Capture(out, boundaries)
    started = time.monotonic()
    with patch.object(runner, 'optimize_window', coordinator.wrap(runner.optimize_window)):
        result = runner.run_pipeline(src, tgt, prm, PipelineConfig(**config), on_commit=coordinator.commit)
    record = dict(end_utc=utc(), seconds=time.monotonic() - started, accepted=coordinator.accepted,
                  checkpoints=coordinator.saved, history=result['history'], guards=result['guards'],
                  missing_boundaries=sorted(set(boundaries) - {p['boundary_commit'] for p in coordinator.saved}),
                  torch_peak_bytes=torch.cuda.max_memory_allocated(),
                  protocol_sha256=sha(out / 'protocol.json'))
    write_json(out / 'capture.json', record)
    print(json.dumps({k: record[k] for k in ('seconds', 'accepted', 'guards', 'missing_boundaries')}), flush=True)
    if record['missing_boundaries']:
        raise RuntimeError('Pre-registered boundary not reached; do not substitute another')


def replay(out):
    from physmorph.compute import cuda_execution
    from scripts.probes.boundary_packet import replay_step
    from scripts.probes.boundary_metrics import summarize_boundary

    protocol = json.loads((out / 'protocol.json').read_text())
    capture_record = json.loads((out / 'capture.json').read_text())
    if sha(out / 'protocol.json') != capture_record['protocol_sha256']:
        raise RuntimeError('Capture protocol changed')
    if code_hashes() != protocol['code_sha256']:
        raise RuntimeError('Replay code differs from frozen capture snapshot')
    report = dict(start_utc=utc(), protocol_sha256=sha(out / 'protocol.json'),
                  capture_sha256=sha(out / 'capture.json'), comparisons=[],
                  mpm=protocol['mpm'], N=300000, T=protocol['config']['T'],
                  native_spacing_wu=protocol['native_spacing_wu'],
                  limitations=protocol['limitations'])
    all_valid = not capture_record['missing_boundaries']
    for checkpoint in capture_record['checkpoints']:
        path = Path(checkpoint['path'])
        if sha(path) != checkpoint['sha256']:
            raise RuntimeError(f'Checkpoint changed: {path}')
        packet = load_packet(path)
        with cuda_execution('cuda'):
            arrays = packet['arrays']
            data = {key: torch.as_tensor(arrays[key], device='cuda').clone()
                    for key in ('previous', 'raw', 'promoted', 'pin', 'layer_mask')}
            data['pin'] = data['pin'] > .5
            data['layer_mask'] = data['layer_mask'] > .5
            original_x0 = torch.as_tensor(arrays['x0'], device='cuda')
            if not torch.equal(original_x0, data['promoted']):
                raise RuntimeError('Actual next x0 differs from previous promoted endpoint')
            alt_x0 = torch.where(data['pin'][:, None], original_x0, data['raw'])
            branches = dict(pic=replay_step(packet, device='cuda'),
                            pic_repeat=replay_step(packet, device='cuda'),
                            free_raw=replay_step(packet, x0override=alt_x0, device='cuda'))
            closure = {}
            for key in protocol['closure_fields']:
                actual = torch.as_tensor(arrays['original_' + key], device='cuda')
                actual = actual.reshape_as(branches['pic'][key])
                closure[key] = check_original_step(branches['pic'][key], actual, key,
                                                    protocol['mpm']['dt'], protocol['mpm']['dx'])
            free = ~data['pin']
            response = ((branches['pic']['x1'] - branches['pic']['x0']) -
                        (branches['free_raw']['x1'] - branches['free_raw']['x0']))[free]
            observed = torch.as_tensor(arrays['original_x1'], device='cuda')
            closure_delta = (branches['pic']['x1'] - observed)[free]
            repeat_delta = (branches['pic']['x1'] - branches['pic_repeat']['x1'])[free]
            rms = lambda t: float(t.double().square().sum(-1).mean().sqrt()) if t.numel() else 0.
            signal, noise, capture_error = rms(response), rms(repeat_delta), rms(closure_delta)
            metrics = summarize_boundary(data, branches, protocol['mpm']['dt'],
                                         protocol['native_spacing_wu'])
            pin_fields = ('position_exact', 'pre_layer_position_exact', 'v1_zero', 'C1_zero', 'F1_same_as_pic')
            pins_valid = all(metrics['pin_checks'][arm][key]
                             for arm in branches for key in pin_fields)
            valid = all(c['passed'] for c in closure.values()) and pins_valid
            cohort_resolution = {}
            for name, mask in (('all_free', free), ('layer_free', free & data['layer_mask'])):
                delta = ((branches['pic']['x1'] - branches['pic']['x0']) -
                         (branches['free_raw']['x1'] - branches['free_raw']['x0']))[mask]
                original_noise = (branches['pic']['x1'] - observed)[mask]
                repeat_noise = (branches['pic']['x1'] - branches['pic_repeat']['x1'])[mask]
                s, c, r = rms(delta), rms(original_noise), rms(repeat_noise)
                resolved = bool(mask.any()) and s > 10 * max(c, r)
                cohort_resolution[name] = dict(count=int(mask.sum()), response_rms_wu=s,
                    original_closure_rms_wu=c, repeat_rms_wu=r, exceeds_10x_noise=resolved,
                    interpretation=('closure_failed' if not valid else 'resolved_response' if resolved else
                                    'inconclusive_at_measured_noise' if bool(mask.any()) else 'empty_cohort'))
            entry = dict(boundary=packet['meta']['boundary_commit'],
                         next_attempt=packet['meta']['next_attempt'], checkpoint_sha256=checkpoint['sha256'],
                         closure=closure, pin_contract_passed=pins_valid, closure_passed=valid,
                         free_response_rms_wu=signal, free_repeat_rms_wu=noise,
                         free_original_closure_rms_wu=capture_error,
                         response_exceeds_10x_noise=signal > 10 * max(noise, capture_error),
                         cohort_resolution=cohort_resolution, metrics=metrics)
            if 'bond_active' in branches['pic']:
                entry['bond_activation_changed_count'] = int(torch.count_nonzero(
                    branches['pic']['bond_active'] != branches['free_raw']['bond_active']))
            all_valid = all_valid and valid
            report['comparisons'].append(entry)
        print(json.dumps({k: entry[k] for k in ('boundary', 'closure_passed', 'free_response_rms_wu',
                                               'free_repeat_rms_wu', 'response_exceeds_10x_noise')}), flush=True)
    report.update(end_utc=utc(), closure_passed=all_valid)
    write_json(out / 'replay.json', report)
    if not all_valid:
        raise RuntimeError('Original first-step closure failed: causal interpretation blocked')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'replay'))
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--boundaries', type=int, nargs='+', default=BOUNDARIES)
    parser.add_argument('--windows', type=int, default=60)
    args = parser.parse_args()
    args.root, args.out = args.root.resolve(), args.out.resolve()
    for path in (args.root, args.out):
        if not path.is_relative_to(Path('/data')):
            parser.error('Server probe paths must be under /data')
    if not args.boundaries or min(args.boundaries) < 1 or len(set(args.boundaries)) != len(args.boundaries):
        parser.error('Boundaries must be distinct positive accepted-commit indices')
    for name in ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH'):
        if not os.environ.get(name) or not Path(os.environ[name]).resolve().is_relative_to(Path('/data')):
            parser.error(f'{name} must be explicit under /data')
    if not torch.cuda.is_available():
        parser.error('CUDA required; no CPU numerical fallback')
    import warp as wp
    wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
    wp.init()
    if args.mode == 'capture':
        capture(args.root, args.out, tuple(args.boundaries), args.windows)
    else:
        replay(args.out)


if __name__ == '__main__':
    main()
