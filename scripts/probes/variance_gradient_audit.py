"""Same prepared first-gradient audit; no alternate solve, update or quality claim.

Production CLI is CUDA-only and observes attempt 24 of the physical-variance
recipe. Counterfactuals stop before Adam, line search and physical advancement.
The two previously failed strict CUDA primitive gates remain failed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def norm(values):
    # Match optimizer._norm exactly for the balancer, including working dtype.
    return float(torch.sqrt(sum(value.pow(2).sum() for value in values)))


def dot64(a, b):
    return sum((x.double()*y.double()).sum() for x, y in zip(a, b))


def difference(a, b):
    return [x-y for x, y in zip(a, b)]


def vector_comparison(a, b):
    """Double reductions on the original device; zero directions are undefined."""
    aa, bb, ab = float(dot64(a, a)), float(dot64(b, b)), float(dot64(a, b))
    delta = difference(b, a)
    dd = float(dot64(delta, delta))
    count = sum(x.numel() for x in a)
    cosine = max(-1., min(1., ab/math.sqrt(aa*bb))) if aa > 0 and bb > 0 else None
    return dict(a_l2=math.sqrt(aa), b_l2=math.sqrt(bb), dot=ab,
        cosine=cosine, angle_degrees=None if cosine is None else math.degrees(math.acos(cosine)),
        difference_l2=math.sqrt(dd), difference_component_rms=math.sqrt(dd/count),
        difference_max_abs=max(float(x.abs().max()) for x in delta),
        relative_l2_to_a=math.sqrt(dd/aa) if aa > 0 else None,
        elements=count)


def grouped_comparison(a, b, names):
    return dict(joint=vector_comparison(a, b), leaves=[
        dict(index=i, name=name, shape=list(x.shape), **vector_comparison([x], [y]))
        for i, (name, x, y) in enumerate(zip(names, a, b))])


def updated_balancer(state, physical, render):
    from physmorph.pipeline.render_loss import LambdaBalancer
    balancer = LambdaBalancer.__new__(LambdaBalancer)
    balancer.__dict__.update(deepcopy(state))
    require(balancer.active, 'Audit requires an active render balancer')
    value = balancer.update(norm(physical), norm(render))
    return float(value), deepcopy(vars(balancer))


def composite(physical, render, weight, transport, names, u_render_only):
    values = [a+weight*b for a, b in zip(physical, render)]
    if transport is not None:
        values = [a+b for a, b in zip(values, transport)]
    if u_render_only:
        for i, name in enumerate(names):
            if name == 'surface_u':
                values[i] = weight*render[i]
    return values


def analyze_gradient_audit(audit):
    """Algebraic intervention at one graph; no tolerance inferred from repeats."""
    from physmorph.pipeline.grad_combine import pcgrad
    require(audit['same_graph'] and audit['alternate_forward_count'] == 0,
            'Only the same prepared forward graph is supported')
    require(audit['mode'] in ('render', 'off') and not audit['grad_h1'],
            'Unsupported post-gradient conditioning')
    names, values = audit['leaf_names'], audit['gradients']
    physical, geometric, render = (values[key] for key in ('physical', 'geometric', 'render_raw'))
    require(names and len(names) == len(physical), 'Missing leaf correspondence')
    device = physical[0].device
    for key, gradients in values.items():
        if key == 'transport' and gradients is None:
            continue
        require(len(gradients) == len(physical), 'Leaf count changed')
        for ref, value in zip(physical, gradients):
            require(value.device == device and value.shape == ref.shape and value.dtype == ref.dtype,
                    'Gradient layout/device mismatch')
            require(not value.requires_grad and torch.isfinite(value).all(), 'Nonfinite or graph-owned gradient')
    projected_a, conflict_a = pcgrad(physical, render) if audit['mode'] == 'render' else (list(render), False)
    projected_b, conflict_b = pcgrad(geometric, render) if audit['mode'] == 'render' else (list(render), False)
    lambda_a, state_a = updated_balancer(audit['balancer_state'], physical, projected_a)
    lambda_b, state_b = updated_balancer(audit['balancer_state'], geometric, projected_b)
    def combine(p, r, weight):
        return composite(p, r, weight, values['transport'], names, audit['layer_u_render_only'])
    directions = dict(baseline=combine(physical, projected_a, lambda_a),
        physical_only=combine(geometric, projected_a, lambda_a),
        projected_fixed_lambda=combine(geometric, projected_b, lambda_a),
        adaptive=combine(geometric, projected_b, lambda_b))
    stage_names = list(directions)
    stages = [directions[key] for key in stage_names]
    components = [difference(stages[i+1], stages[i]) for i in range(3)]
    total_delta = difference(stages[-1], stages[0])
    summed = [sum(parts) for parts in zip(*components)]
    gram = [[float(dot64(a, b)) for b in components] for a in components]
    total_energy = float(dot64(total_delta, total_delta))
    repeat_a = grouped_comparison(physical, values['physical_repeat'], names)
    repeat_b = grouped_comparison(geometric, values['geometric_repeat'], names)
    change = grouped_comparison(physical, geometric, names)
    repeat_floor = max(repeat_a['joint']['difference_l2'], repeat_b['joint']['difference_l2'])
    expected_core_delta = audit['effective_weight']*(audit['geometric_variance']-audit['physical_variance'])
    measured_core_delta = audit['geometric_physics_core']-audit['physics_core']
    return dict(scope=audit['scope'], same_graph=True, alternate_forward_count=0,
        N=audit['N'], T=audit['T'], dt=audit['dt'], mode=audit['mode'],
        leaf_names=names, layer_u_render_only=audit['layer_u_render_only'], device=str(device),
        physical_variance=audit['physical_variance'], geometric_variance=audit['geometric_variance'],
        effective_weight=audit['effective_weight'], expected_physics_core_delta=expected_core_delta,
        measured_physics_core_delta=measured_core_delta,
        scalar_delta_roundoff=measured_core_delta-expected_core_delta,
        balancer_before=deepcopy(audit['balancer_state']), balancer_baseline_after=state_a,
        balancer_geometric_after=state_b, lambda_baseline=lambda_a, lambda_geometric=lambda_b,
        lambda_ratio=lambda_b/lambda_a if lambda_a != 0 else None,
        pcgrad_conflict_baseline=conflict_a, pcgrad_conflict_geometric=conflict_b,
        physical_gradient_change=change, physical_repeat=repeat_a, geometric_repeat=repeat_b,
        physical_change_to_max_repeat_l2=change['joint']['difference_l2']/repeat_floor if repeat_floor > 0 else None,
        repeat_scope='One repeated adjoint per observable, on the same graph; descriptive context, not calibration, confidence bound or pass criterion',
        adjoint_order=['physical production', 'geometric', 'geometric repeat', 'physical repeat'],
        raw_render_vs_physical=grouped_comparison(physical, render, names),
        raw_render_vs_geometric=grouped_comparison(geometric, render, names),
        projected_render_change=grouped_comparison(projected_a, projected_b, names),
        directions={key:grouped_comparison(directions['baseline'], direction, names)
                    for key, direction in directions.items()},
        stage_changes=[dict(stage=stage_names[i]+' -> '+stage_names[i+1],
                            **grouped_comparison(stages[i], stages[i+1], names)) for i in range(3)],
        delta_partition=dict(order=['physical substitution', 'PCGrad at fixed baseline lambda', 'lambda adaptation'],
            gram=gram, total_delta_squared=total_energy, gram_sum=sum(map(sum, gram)),
            signed_projection_on_total=[float(dot64(x, total_delta))/total_energy if total_energy > 0 else None
                                        for x in components],
            closure=vector_comparison(total_delta, summed),
            caveat='Path-dependent telescoping gradient partition; components can cancel and norm shares are not movement shares'),
        primitive_gate_passed=False, no_quality_or_motion_conclusion=True)


class GradientAudit:
    def __init__(self, selected_index=23):
        self.selected_index = selected_index
        self.audit = None
        self.accepted = 0
        self.commit_record = None

    def observe(self, win_index, audit):
        require(int(win_index) == self.selected_index and self.audit is None, 'Unexpected/repeated selected gradient')
        require(int(audit['win_index']) == self.selected_index, 'Observer window identity mismatch')
        self.audit = analyze_gradient_audit(audit)

    def wrap(self, original):
        def wrapped(*args, **kwargs):
            require('on_gradient_audit' not in kwargs, 'An audit hook is already installed')
            if int(kwargs['win_index']) == self.selected_index:
                kwargs['on_gradient_audit'] = self.observe
            return original(*args, **kwargs)
        return wrapped

    def commit(self, attempt, x, F, v, record):
        outer_accepted = record.get('frame_end') is not None
        if outer_accepted:
            self.accepted += 1
        if int(attempt) == self.selected_index:
            self.commit_record = dict(attempt=int(attempt)+1, outer_accepted=outer_accepted,
                accepted_ordinal=self.accepted if outer_accepted else None, record=deepcopy(record))

    def finish(self, result):
        observed = self.audit is not None
        accepted = self.commit_record is not None and self.commit_record['outer_accepted']
        lambda_exact = bool(observed and accepted and
            self.commit_record['record'].get('lambda') == self.audit['lambda_baseline'])
        measured_lambda = self.commit_record['record'].get('lambda') if accepted else None
        expected_lambda = self.audit['lambda_baseline'] if observed else None
        # Rejected trials are preserved explicitly, never relabeled as accepted late state.
        return dict(selected_attempt=self.selected_index+1, observed=observed,
            selected_outer_accepted=bool(accepted), audit=self.audit, selected_commit=self.commit_record,
            guards=result['guards'], history=result['history'], outer_accepted_commits=self.accepted,
            baseline_lambda_matches_production_exactly=lambda_exact,
            baseline_lambda_verification=dict(measured_outer_record=measured_lambda,
                reconstructed=expected_lambda,
                difference=measured_lambda-expected_lambda
                    if measured_lambda is not None and expected_lambda is not None else None,
                criterion='exact equality; actual lambda is held after first inner iteration'),
            observation_valid=bool(observed and accepted and lambda_exact and not any(result['guards'].values())),
            selection='Fixed attempted window; accepted ordinal reported only after actual outer admission; later delivery truncation is not an audit admission criterion',
            primitive_gate_passed=False, no_quality_or_motion_conclusion=True)


def run_audited(source, target, prm, cfg, selected_index=23, log=print):
    from physmorph.pipeline import runner
    coordinator = GradientAudit(selected_index)
    with patch.object(runner, 'optimize_window', coordinator.wrap(runner.optimize_window)):
        result = runner.run_pipeline(source, target, prm, cfg, on_commit=coordinator.commit, log=log)
    return result, coordinator.finish(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--baseline-protocol', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    baseline_path = (args.baseline_protocol or root/'work/p297/integration_control/protocol.json').resolve()
    for path in (root, out, baseline_path):
        require(path.is_relative_to('/data'), 'All paths must resolve under /data')
    require(not out.exists(), 'Output directory already exists')
    caches = {name:os.environ.get(name) for name in ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH')}
    require(all(value and Path(value).resolve().is_relative_to('/data') for value in caches.values()),
            'Explicit /data caches required')
    require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
    import warp as wp
    wp.config.kernel_cache_dir = caches['WARP_CACHE_PATH']
    from physmorph.pipeline import PipelineConfig
    from physmorph.mpm.state import MPMParams
    from scripts.probes.geometric_variance_integration import SOURCE_SHA, TARGET_SHA, save_json, sha
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
    config['stop_after_windows'] = 24  # The sole recipe override; no alternate rollout/config.
    cfg, prm = PipelineConfig(**config), MPMParams(**baseline['mpm'])
    require(cfg.T == 20 and cfg.iters == 8 and prm.dt == 1/240 and cfg.device == 'cuda'
        and cfg.compute_backend == 'cuda' and cfg.commit_pic and cfg.commit_pic_objective
        and cfg.outer_render_committed and not cfg.shift_sub and not cfg.geometric_rest
        and not cfg.geometric_variance and cfg.lambda_auto > 0 and cfg.w_kin_var > 0
        and not cfg.grad_h1 and cfg.grad_project and cfg.grad_project_mode == 'render',
        'Only the original physical-variance shared-PIC T20 recipe is supported')
    reference = Path(cfg.target_reference).resolve()
    require(reference.is_relative_to('/data') and sha(reference) == baseline['target_reference_sha256'],
            'Prepared render reference changed')
    primitives = []
    for entry in baseline['preserved_primitive_evidence']:
        path = Path(entry['path']).resolve()
        require(path.is_relative_to('/data'), 'Primitive evidence outside /data')
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == entry['sha256'] and json.loads(data).get('passed') is False,
                'Preserved failed strict evidence changed')
        primitives.append(deepcopy(entry))
    require(len(primitives) == 2, 'Both failed strict primitive results must be preserved')
    code = sorted((ROOT/'physmorph').rglob('*.py')) + [Path(__file__).resolve(),
        ROOT/'scripts/probes/geometric_variance_integration.py']
    protocol = dict(start_utc=datetime.now(timezone.utc).isoformat(),
        baseline_protocol=str(baseline_path), baseline_protocol_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
        source=str(source_path), source_array_sha256=hashes[0], target_array_sha256=hashes[1],
        source_metadata_sha256=baseline['source_metadata_sha256'], target_reference_sha256=sha(reference),
        config=asdict(cfg), mpm=asdict(prm), selected_attempt=24,
        config_override=dict(stop_after_windows=dict(before=baseline['config']['stop_after_windows'], after=24)),
        code_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in code},
        cache_directories=caches, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        preserved_primitive_evidence=primitives, primitive_gate_passed=False,
        scope='Same graph first-gradient substitution in a physical baseline rerun; no alternate preparation, solver step, raw archive or quality gate')
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'protocol.json', protocol)
    start = time.monotonic()
    result, report = run_audited(source, target, prm, cfg)
    torch.cuda.synchronize()
    report.update(seconds=time.monotonic()-start, end_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(out/'protocol.json'), torch_peak_bytes=torch.cuda.max_memory_allocated())
    save_json(out/'gradient_audit.json', report)
    print(json.dumps(dict(out=str(out), observation_valid=report['observation_valid'], primitive_gate_passed=False)), flush=True)
    if not report['observation_valid']:
        raise SystemExit('Selected attempt lacked a guard-free outer-accepted observation; see preserved report')


if __name__ == '__main__':
    main()
