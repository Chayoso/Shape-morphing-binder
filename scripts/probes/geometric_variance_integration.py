"""Cap1 integration evidence; does not convert the failed primitive gates to a pass.

Production CLI runs only CUDA on the original 300k recipe. CPU tests exercise the
same coordinator with a small actual pipeline. The scalar audit never rerolls
physics: only the variance argument to its prepared objective is substituted.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
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
ROUNDING_ULPS = 64
MERIT_ROUNDING_ULPS = 4
SOURCE_SHA = '71eb14d38c2efb41379a12b0ce017430e093883cbce51188265ed9e2948d0e34'
TARGET_SHA = '8de5081712e5fb55f7c1c281e0bbe787ba8c2e948bc6c3635d2d01ac45dd8ed7'


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def independent_variances(start, positions, promoted, velocities, dt):
    """Population variance, vector norm squared, averaged over all particles."""
    require(positions.ndim == 3 and positions.shape == velocities.shape, 'Invalid path shape')
    saved = torch.cat((start[None], positions[:-1], promoted[None]), 0).double()
    rates = (saved[1:]-saved[:-1])/dt
    physical = velocities.double()
    def summary(values):
        variance = ((values-values.mean(0)).square().sum(-1)).mean()
        second = values.square().sum(-1).mean()
        return variance, second
    geometric, geometric_second = summary(rates)
    physical_var, physical_second = summary(physical)
    return dict(geometric=geometric, physical=physical_var,
                geometric_second=geometric_second, physical_second=physical_second,
                saved=saved)


def objective_check(audit, independent, second_moment):
    """Same prepared scalar objective, independent observable, no fitted tolerance."""
    observed = float(audit['observed_variance'])
    weight = float(audit['effective_weight'])
    eps = torch.finfo(torch.float32).eps
    variance_allowance = ROUNDING_ULPS*eps*max(float(second_moment), abs(observed), float(independent), 1e-30)
    require(abs(observed-float(independent)) <= variance_allowance,
            'Recorded variance differs from independent path arithmetic')
    evaluate = audit['evaluate']
    zero = float(evaluate(0.))
    with_independent = float(evaluate(float(independent)))
    with_recorded = float(evaluate(observed))
    contribution = weight*float(independent)
    delta = with_independent-zero
    arithmetic_allowance = MERIT_ROUNDING_ULPS*eps*max(
        abs(zero), abs(with_independent), abs(contribution), 1e-30)
    require(abs(delta-contribution) <= arithmetic_allowance,
            'Prepared scalar objective does not apply the recorded variance weight')
    replay_allowance = float(audit['replay_tolerance'])
    require(abs(with_recorded-float(audit['final_merit'])) <= replay_allowance+arithmetic_allowance,
            'Prepared objective differs from the validated final merit')
    require(with_recorded <= float(audit['accepted_merit'])+replay_allowance+arithmetic_allowance,
            'Prepared objective violates the existing accepted/replay merit allowance')
    return dict(observed_variance=observed, independent_variance=float(independent),
        variance_allowance=variance_allowance, effective_weight=weight,
        E_without_variance=zero, E_with_independent=with_independent, E_with_recorded=with_recorded,
        E_final=float(audit['final_merit']), E_accepted=float(audit['accepted_merit']),
        replay_tolerance=replay_allowance, expected_contribution=contribution,
        measured_contribution=delta, contribution_error=abs(delta-contribution),
        arithmetic_allowance=arithmetic_allowance,
        nonzero_participation_resolved=contribution > arithmetic_allowance)


class IntegrationAudit:
    def __init__(self, cfg, prm):
        self.cfg, self.prm = cfg, prm
        self.gradient_calls, self.gradient_seeds, self.windows, self.commits = [], [], [], []
        self.previous_output = None
        self.owned = None

    def check_prior_output(self):
        if self.previous_output is not None:
            output, saved = self.previous_output
            require(torch.equal(output, saved), 'Returned X changed after a later forward/backward')

    def observe_positions(self, outputs):
        self.check_prior_output()
        positions = outputs[-1]
        require(positions.shape == (self.cfg.T, len(outputs[0]), 3), 'X does not contain the full path')
        require(torch.isfinite(positions).all(), 'Nonfinite differentiable X')
        require(torch.equal(outputs[0], positions[-1]), 'Differentiable X terminal differs from xT')
        self.previous_output = (positions.detach(), positions.detach().clone())
        self.gradient_calls.append(dict(T=len(positions), N=len(positions[0]),
                                        terminal_exact=True, finite=True))
        if positions.requires_grad:
            def observe_seed(gradient):
                if gradient is None:
                    self.gradient_seeds.append(dict(present=False, early_max=0., terminal_max=0., rms=0.))
                    return
                require(torch.isfinite(gradient).all(), 'Nonfinite X seed')
                self.gradient_seeds.append(dict(present=True, early_max=float(gradient[:-1].abs().max()),
                    terminal_max=float(gradient[-1].abs().max()),
                    rms=float(gradient.double().square().mean().sqrt())))
            positions.register_hook(observe_seed)
        return outputs

    def objective(self, tr, promoted, win_index, audit):
        import warp as wp
        require(self.owned is None, 'Cap1 observer unexpectedly saw another validated rollout')
        self.check_prior_output()
        view = lambda value: wp.to_torch(value)
        positions = torch.stack([view(tr.x[t]) for t in range(1, tr.T+1)])
        velocities = torch.stack([view(tr.v[t]) for t in range(1, tr.T+1)])
        tensors = dict(start=view(tr.x[0]), raw=view(tr.x[tr.T]), promoted=promoted,
            positions=positions, velocities=velocities, F=view(tr.F[tr.T]), v=view(tr.v[tr.T]),
            C=view(tr.C[tr.T]), Fp=view(tr.Fp), pin=view(tr.pin),
            first_dFc=view(tr._dfc(0)), last_dFc=view(tr._dfc(tr.T-1)))
        if tr.body_control is not None:
            tensors['body'] = view(tr.body_control)
        if tr.layer:
            tensors['u'] = view(tr.layer_u)
        self.owned = {name: value.detach().clone() for name, value in tensors.items()}
        independent = independent_variances(tensors['start'], positions, promoted, velocities, self.prm.dt)
        selected = 'geometric' if self.cfg.geometric_variance else 'physical'
        contribution = objective_check(audit, independent[selected], independent[selected+'_second'])
        for name, value in tensors.items():
            require(torch.equal(value, self.owned[name]), f'Objective audit mutated {name}')
        # Stacked X/V are owned temporaries; re-read actual trajectory arrays too.
        for t in range(tr.T):
            require(torch.equal(view(tr.x[t+1]), self.owned['positions'][t]), 'Objective audit mutated trajectory X')
            require(torch.equal(view(tr.v[t+1]), self.owned['velocities'][t]), 'Objective audit mutated trajectory V')
        pins = self.owned['pin'] > .5
        require(torch.equal(promoted[pins], self.owned['start'][pins]), 'Promotion moved a window-start pin')
        self.windows.append(dict(attempt=int(win_index)+1, commit_source=audit['commit_source'],
            active_pin_count=int(pins.sum()), active_pin_check_vacuous=not bool(pins.any()),
            selected_observable=selected, geometric_variance=float(independent['geometric']),
            physical_variance=float(independent['physical']), objective=contribution,
            observer_state_unchanged=True))

    def wrap_optimizer(self, original):
        def wrapped(*args, **kwargs):
            require('on_objective' not in kwargs, 'Objective observer already installed')
            result = original(*args, on_objective=self.objective, **kwargs)
            frames, _, end, _, history, stats = result
            require(bool(history) and self.owned is not None, 'No accepted inner trajectory to audit')
            expected = self.owned
            device = expected['raw'].device
            tensor = lambda value: torch.as_tensor(value, device=device)
            require(torch.equal(tensor(frames[0]), expected['start']), 'Optimizer start mismatch')
            require(torch.equal(tensor(frames[-1]), expected['raw']), 'Optimizer raw endpoint mismatch')
            for name in ('F', 'v', 'C'):
                require(torch.equal(tensor(end[name]).reshape_as(expected[name]), expected[name]),
                        'Optimizer returned a different '+name)
            package = stats.get('owned_endpoint')
            require(package is not None and torch.equal(package.promoted, expected['promoted'])
                    and torch.equal(package.raw, expected['raw']), 'Owned endpoint mismatch')
            self.windows[-1]['inner_accepted_iterations'] = len(history)
            self.windows[-1]['optimizer_package_exact'] = True
            return result
        return wrapped

    def commit(self, attempt, x, F, v, record):
        require(record.get('frame_end') is not None, 'Cap1 outer commit was rejected')
        require(self.owned is not None, 'Commit lacks validated objective observer')
        device = self.owned['raw'].device
        for name, value, expected in (('x', x, self.owned['promoted']),
                                      ('F', F, self.owned['F']), ('v', v, self.owned['v'])):
            require(torch.equal(torch.as_tensor(value, device=device).reshape_as(expected), expected),
                    'Committed '+name+' differs from the owned evaluated state')
        self.commits.append(dict(attempt=int(attempt)+1, frame_end=int(record['frame_end']),
                                 owned_x_F_v_exact=True))

    def finish(self, result):
        self.check_prior_output()
        require(len(self.commits) == len(self.windows) == 1, 'Expected exactly one accepted window')
        require(not any(result['guards'].values()), 'Pipeline state guard fired')
        device = self.owned['raw'].device
        endpoint = self.commits[0]['frame_end']-1
        archive = torch.stack([torch.as_tensor(frame, device=device) for frame in result['frames'][:endpoint+1]])
        require(endpoint == self.cfg.T, 'Unexpected archive stride or missing raw frames')
        expected = torch.cat((self.owned['start'][None], self.owned['positions'][:-1],
                              self.owned['promoted'][None]), 0)
        require(torch.equal(archive, expected), 'Archived full path differs from the evaluated saved path')
        require(self.windows[0]['objective']['nonzero_participation_resolved'],
                'Variance contribution is unresolved; integration participation is not established')
        if self.cfg.geometric_variance:
            require(self.gradient_calls and any(row['early_max'] > 0 for row in self.gradient_seeds),
                    'No observed earlier-X gradient participation')
        else:
            require(not self.gradient_calls, 'Control unexpectedly used position-sequence bridge')
        return dict(windows=self.windows, commits=self.commits, gradient_forwards=self.gradient_calls,
                    gradient_seeds=self.gradient_seeds, full_archive_exact=True, guards=result['guards'],
                    state_check_scope=dict(optimizer_return=['raw x', 'F', 'v', 'C'],
                                           outer_commit=['promoted x', 'F', 'v'],
                                           committed_C_independently_checked=False),
                    bounded_integration_passed=True, primitive_gate_passed=False,
                    no_quality_or_rest_conclusion=True)


def run_audited(source, target, prm, cfg, log=print, coordinator=None):
    from physmorph.pipeline import runner, optimizer
    from physmorph.mpm.function import PersistentAdjoint
    coordinator = coordinator or IntegrationAudit(cfg, prm)
    original_apply = PersistentAdjoint.apply_with_positions
    original_ordinary = optimizer.warp_mpm_ext_with_positions
    def persistent(adjoint, *args, **kwargs):
        coordinator.check_prior_output()
        return coordinator.observe_positions(original_apply(adjoint, *args, **kwargs))
    def ordinary(*args, **kwargs):
        coordinator.check_prior_output()
        return coordinator.observe_positions(original_ordinary(*args, **kwargs))
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner, 'optimize_window', coordinator.wrap_optimizer(runner.optimize_window)))
        stack.enter_context(patch.object(PersistentAdjoint, 'apply_with_positions', persistent))
        stack.enter_context(patch.object(optimizer, 'warp_mpm_ext_with_positions', ordinary))
        result = runner.run_pipeline(source, target, prm, cfg, log=log, on_commit=coordinator.commit)
    report = coordinator.finish(result)
    return result, report, coordinator


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--mode', choices=('control', 'geometry'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-protocol', type=Path)
    parser.add_argument('--primitive-v1', type=Path)
    parser.add_argument('--primitive-v2', type=Path)
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    baseline_path = (args.baseline_protocol or root/'work/p295/full60_v2/protocol.json').resolve()
    primitive_paths = [(args.primitive_v1 or root/'work/p297/position_sequence.json').resolve(),
                       (args.primitive_v2 or root/'work/p297/position_sequence_v2.json').resolve()]
    for path in (root, out, baseline_path, *primitive_paths):
        require(path.is_relative_to('/data'), 'All paths must resolve under /data')
    require(not out.exists(), 'Output directory already exists')
    caches = {name: os.environ.get(name) for name in ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH')}
    require(all(value and Path(value).resolve().is_relative_to('/data') for value in caches.values()),
            'Explicit /data caches are required')
    require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
    primitives = []
    for path in primitive_paths:
        primitive_bytes = path.read_bytes()
        value = json.loads(primitive_bytes)
        require(value.get('passed') is False, 'Expected preserved failed primitive evidence')
        primitives.append(dict(path=str(path), sha256=hashlib.sha256(primitive_bytes).hexdigest(), strict_passed=False))
    import warp as wp
    wp.config.kernel_cache_dir = caches['WARP_CACHE_PATH']
    from physmorph.pipeline import PipelineConfig
    from physmorph.mpm.state import MPMParams
    from physmorph.compute import to_host
    baseline_bytes = baseline_path.read_bytes()
    baseline = json.loads(baseline_bytes)
    source_path = Path(baseline['source']).resolve()
    require(source_path.is_relative_to('/data'), 'Source outside /data')
    before = source_path.stat()
    with np.load(source_path, allow_pickle=False) as archive:
        source, target = archive['src'], archive['tgt']
    after = source_path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns),
            'Source changed during read')
    hashes = [hashlib.sha256(value.tobytes()).hexdigest() for value in (source, target)]
    require(source.shape == target.shape == (300000, 3) and source.dtype == target.dtype == np.float32,
            'Expected original float32 N300k inputs')
    require(hashes == [SOURCE_SHA, TARGET_SHA] == [baseline['source_array_sha256'], baseline['target_array_sha256']],
            'Original input hashes mismatch')
    config = dict(baseline['config'])
    config.update(stop_after_windows=1, geometric_variance=args.mode == 'geometry')
    cfg = PipelineConfig(**config)
    prm = MPMParams(**baseline['mpm'])
    require(cfg.T == 20 and cfg.iters == 8 and prm.dt == 1/240 and cfg.compute_backend == 'cuda'
            and cfg.device == 'cuda' and cfg.commit_pic and cfg.commit_pic_objective
            and cfg.outer_render_committed and not cfg.shift_sub and not cfg.geometric_rest
            and cfg.archive_stride == 1 and not cfg.render_F_geom and not cfg.rest_commit,
            'This probe supports only the original shared-PIC T20 cap1 contract')
    require(sha(cfg.target_reference) == baseline['target_reference_sha256'], 'Prepared render reference changed')
    code = sorted((ROOT/'physmorph').rglob('*.py')) + [Path(__file__).resolve()]
    protocol = dict(start_utc=datetime.now(timezone.utc).isoformat(), mode=args.mode,
        source=str(source_path), source_array_sha256=hashes[0], target_array_sha256=hashes[1],
        baseline_protocol=str(baseline_path), baseline_protocol_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
        source_metadata_sha256=baseline['metadata_sha256'], target_reference_sha256=sha(cfg.target_reference),
        config=asdict(cfg), mpm=asdict(prm), code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in code},
        cache_directories=caches, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        preserved_primitive_evidence=primitives,
        rounding_allowance='variance:64*float32_eps*second-moment magnitude; scalar subtraction:4*float32_eps*max absolute merits/contribution; not gradient tolerance',
        primitive_status='v1 and v2 strict CUDA adjoint gates remain failed; this is separate cap1 integration evidence',
        scope='first transport window only; active pin count may be zero; no terminal rest or quality conclusion')
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'protocol.json', protocol)
    start = time.monotonic()
    result, report, coordinator = run_audited(source, target, prm, cfg)
    torch.cuda.synchronize()
    report.update(seconds=time.monotonic()-start, end_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(out/'protocol.json'), history=result['history'],
        torch_peak_bytes=torch.cuda.max_memory_allocated())
    if args.archive:
        with (out/'accepted_path.npz').open('xb') as stream:
            np.savez(stream, frames=np.stack(result['frames']),
                raw=to_host(coordinator.owned['raw']), F=to_host(coordinator.owned['F']),
                v=to_host(coordinator.owned['v']), C=to_host(coordinator.owned['C']),
                active_pin=to_host(coordinator.owned['pin']))
            stream.flush(); os.fsync(stream.fileno())
        report['archive'] = dict(name='accepted_path.npz', sha256=sha(out/'accepted_path.npz'),
                                 bytes=(out/'accepted_path.npz').stat().st_size)
    save_json(out/'integration.json', report)
    print(json.dumps(dict(out=str(out), bounded_integration_passed=True,
                         primitive_gate_passed=False, seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
