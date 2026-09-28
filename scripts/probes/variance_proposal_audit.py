"""P299: two noncommitting first-trial proposals at the same baseline lambda.

The CUDA CLI reruns the original physical-variance recipe through attempt24.
Only A/B/A proposals are evaluated; no alternate solve or raw archive is saved.
Repeat differences are descriptive and never set a cross-arm tolerance.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
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

import numpy as host_np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.probes.variance_gradient_audit import (
    GradientAudit, composite, grouped_comparison, norm, require, updated_balancer,
)


def finite_json(value):
    """Unavailable/unsafe scalar diagnostics are null, never JSON NaN/Infinity."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def native_alpha(direction, settings):
    value = float(settings['unscaled_alpha'])
    if settings['adaptive_alpha']:
        value *= max(settings['min_alpha_scale'], min(1., settings['target_norm']/max(norm(direction), 1e-30)))
    return value


def distribution(values):
    values = values.double().reshape(-1)
    if not len(values):
        return dict(count=0, mean=None, median=None, p95=None, max=None, rms=None)
    return dict(count=len(values), mean=float(values.mean()),
                median=float(torch.quantile(values, .5)), p95=float(torch.quantile(values, .95)),
                max=float(values.max()), rms=float(values.square().mean().sqrt()))


def displacement(values, spacing):
    lengths = values.double().reshape(-1, 3).norm(dim=-1)
    return dict(wu=distribution(lengths), sp=distribution(lengths/spacing))


def reversal(steps, spacing):
    lengths = steps.double().norm(dim=-1)
    valid = (lengths[:-1] > 1e-4*spacing) & (lengths[1:] > 1e-4*spacing)
    reverse = (steps[:-1].double()*steps[1:].double()).sum(-1) < 0
    count = int(valid.sum())
    return dict(eligible=count, reversed=int((valid & reverse).sum()),
                fraction=float(reverse[valid].double().mean()) if count else None)


def path_summary(steps, spacing):
    lengths = steps.double().norm(dim=-1)
    path = lengths.sum(0)
    net = steps.double().sum(0).norm(dim=-1)
    moving = path > 1e-4*spacing
    return dict(path_wu=distribution(path), path_sp=distribution(path/spacing),
                net_wu=distribution(net), net_sp=distribution(net/spacing),
                net_over_path=distribution(net[moving]/path[moving]),
                reversal=reversal(steps, spacing))


def trajectory_summary(reference, trial, mask, dt, spacing):
    """Fixed IDs; raw MPM+layer steps, PIC jump and saved final phase stay distinct."""
    x0, X, end, V = reference['x0'], trial['positions'], trial['promoted'], trial['physical_v']
    require(X.ndim == 3 and X.shape == V.shape and X.shape[1:] == x0.shape,
            'Trial path/velocity shape mismatch')
    require(mask.dtype == torch.bool and mask.shape == x0.shape[:1], 'Invalid fixed cohort')
    require(dt > 0 and spacing > 0 and math.isfinite(dt+spacing), 'Invalid units')
    raw = (X-torch.cat((x0[None], X[:-1]), 0))[:, mask]
    jump = (end-X[-1])[mask]
    saved = raw.clone()
    saved[-1] += jump
    return dict(particles=int(mask.sum()), T=len(X), dt=dt, spacing=spacing,
        raw_all=displacement(raw, spacing), raw_interior=displacement(raw[:-1], spacing),
        raw_final=displacement(raw[-1], spacing), pic_jump=displacement(jump, spacing),
        saved_final=displacement(saved[-1], spacing), saved_all=displacement(saved, spacing),
        raw_path=path_summary(raw, spacing), saved_path=path_summary(saved, spacing),
        physical_velocity_wu_per_s=distribution(V[:, mask].double().norm(dim=-1)),
        raw_minus_dt_physical_v=displacement(raw-dt*V[:, mask], spacing),
        endpoint_closure_max_wu=float((saved.sum(0)-(end-x0)[mask]).abs().max()) if bool(mask.any()) else None,
        definition='positions contain post-layer physical steps1..T; saved final phase is raw final step plus PIC, not PIC alone; physical_v is stored momentum velocity',
        limitation='window-start fixed free IDs, not an after-arrival/rest cohort; lower motion can mean slower transport')


def prepare_metrics(source, target, device):
    from physmorph.compute import array_api as xp, KDTree, to_array, to_host
    from scripts.probes.pic_endpoint_quality import prepare_reference
    reference = prepare_reference(source, target)
    src = to_array(source)
    counts = KDTree(src).query_ball_point(src, 2*reference['source_spacing'], return_length=True)
    mask = (counts < .6*xp.median(counts)) & (src[:, 1] >= (src[:, 1].min()+src[:, 1].max())/2)
    ids = xp.flatnonzero(mask)
    eligible = len(ids)
    if len(ids) > 20000:
        ids = ids[xp.linspace(0, len(ids)-1, 20000, dtype=xp.int64)]
    selected = torch.zeros(len(src), dtype=torch.bool, device=device)
    if len(ids):
        selected[torch.as_tensor(ids, device=device)] = True
    host_ids = to_host(ids)  # Explicit provenance output, not host numerical work.
    return reference, selected, dict(eligible_count=eligible, sampled_count=len(ids),
        ids_sha256=hashlib.sha256(host_ids.tobytes()).hexdigest(),
        definition='pre-treatment source y>=bbox midpoint and radius2sp count<0.6median; deterministic at-most20000 IDs; original source shape, not future ear membership')


def fixed_supply(x, mask, reference):
    from physmorph.compute import KDTree, to_array
    values, selected = to_array(x), to_array(mask)
    if not bool(selected.any()):
        return dict(particles=0, density=None, under_half=None)
    counts = KDTree(values).query_ball_point(values[selected], reference['radius'], return_length=True)-1
    return dict(particles=int(selected.sum()), density=float(counts.mean()/8),
                under_half=float((counts < 4).mean()),
                current_y_gt_2_3_frac=float((values[selected, 1] > 2.3).mean()))


def pin_report(reference, trial):
    pin, x0 = reference['pins'], reference['x0']
    expected = x0[pin]
    raw = trial['positions'][:, pin]
    promoted = trial['promoted'][pin]
    return dict(particles=int(pin.sum()), raw_exact=bool(torch.equal(raw, expected[None].expand_as(raw))),
                promoted_exact=bool(torch.equal(promoted, expected)),
                raw_max_wu=float((raw-expected).abs().max()) if bool(pin.any()) else None,
                promoted_max_wu=float((promoted-expected).abs().max()) if bool(pin.any()) else None)


def trial_comparison(a, b, names):
    controls = grouped_comparison(a['controls_delta'], b['controls_delta'], names)
    state = {key: (grouped_comparison([a[key]], [b[key]], [key])['joint']
                   if bool(torch.isfinite(a[key]).all() & torch.isfinite(b[key]).all()) else None)
             for key in ('positions', 'promoted', 'physical_v', 'F', 'v', 'C')}
    return dict(controls=controls, controls_exact=all(torch.equal(x, y) for x, y in zip(a['controls_delta'], b['controls_delta'])),
                state=state, merit_difference={key: b['merits'][key]-a['merits'][key] for key in ('physical', 'geometric')},
                state_units=dict(positions='wu', promoted='wu', physical_v='wu/s', F='dimensionless', v='wu/s', C='1/s'),
                definition='B minus A within matched quantities; replay differences are descriptive, not a tolerance or confidence interval')


def analyze_proposals(audit, geometry_reference, source_mask, source_metadata):
    from physmorph.compute import to_host
    from physmorph.pipeline.grad_combine import pcgrad
    from scripts.probes.pic_endpoint_quality import endpoint_quality
    require(audit['same_graph'] and audit['alternate_forward_count'] == 0, 'Gradients need the same prepared graph')
    require(audit['mode'] in ('render', 'off') and not audit['grad_h1'], 'Unsupported gradient conditioning')
    gradients, names = audit['gradients'], audit['leaf_names']
    p, q, r = [gradients[key] for key in ('physical', 'geometric', 'render_raw')]
    require(len(names) == len(p) == len(q) == len(r) > 0, 'Gradient leaf mismatch')
    for values in (p, q, r, gradients['transport']):
        if values is None:
            continue
        require(len(values) == len(p), 'Gradient leaf count changed')
        for a, b in zip(p, values):
            require(a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
                    and not b.requires_grad and torch.isfinite(b).all(), 'Invalid gradient ownership/layout')
    rp, conflict_a = pcgrad(p, r) if audit['mode'] == 'render' else (list(r), False)
    rq, conflict_b = pcgrad(q, r) if audit['mode'] == 'render' else (list(r), False)
    settings, reference = audit['trial_settings'], audit['trial_reference']
    weight = float(settings['lambda'])
    reconstructed, _ = updated_balancer(audit['balancer_state'], p, rp)
    require(math.isfinite(weight) and weight > 0 and weight == reconstructed, 'Baseline lambda mismatch')
    require(settings['alpha'] > 0 and math.isfinite(settings['alpha']), 'Invalid fixed first-trial alpha')
    directions = [composite(a, b, weight, gradients['transport'], names, audit['layer_u_render_only'])
                  for a, b in ((p, rp), (q, rq))]
    native_a, native_b = [native_alpha(direction, settings) for direction in directions]
    require(native_a == settings['alpha'], 'Baseline adaptive alpha mismatch')
    masks = dict(all_window_start_free=~reference['pins'],
                 source_upper_free=source_mask & ~reference['pins'])
    cohort_metadata = {name: dict(particles=int(mask.sum()),
        ids_sha256=hashlib.sha256(to_host(torch.nonzero(mask).flatten()).tobytes()).hexdigest(),
        definition='same material IDs in reference/A/B/A-repeat, selected once before trial evaluation')
        for name, mask in masks.items()}
    evaluate = audit['trial_evaluate']  # Callback-lifetime only; never retained.
    a = evaluate(directions[0], observable='physical')
    b = evaluate(directions[1], observable='geometric')
    repeat = evaluate(directions[0], observable='physical')
    trials = dict(A=a, B=b, A_repeat=repeat)
    spacing = geometry_reference['source_spacing']
    results = {}
    for label, trial in trials.items():
        require(trial['lambda'] == weight and trial['alpha'] == settings['alpha'], 'Trial gain changed')
        require(len(trial['controls_delta']) == len(names), 'Trial control layout changed')
        finite = all(bool(torch.isfinite(trial[key]).all()) for key in
                     ('positions', 'promoted', 'physical_v', 'F', 'v', 'C'))
        values = {key: deepcopy(trial[key]) for key in ('state_ok', 'Jmin', 'merits', 'predicted_decrease',
                  'required_decrease', 'first_trial_merit_ok', 'alpha', 'lambda', 'observable', 'stats_restore_exact')}
        values.update(finite=finite, pins=pin_report(reference, trial))
        values.update({key: trial.get(key) for key in ('physical_variance', 'geometric_variance',
                      'prepared_inputs_exact', 'prepared_input_names')})
        values['motion'] = {name: trajectory_summary(reference, trial, mask, audit['dt'], spacing)
                            for name, mask in masks.items()} if finite else None
        # A-repeat gets full state/merit differences; repeating expensive geometry is unnecessary.
        values['geometry'] = endpoint_quality(trial['promoted'], geometry_reference) if finite and label != 'A_repeat' else None
        values['fixed_supply'] = {name: fixed_supply(trial['promoted'], mask, geometry_reference)
                                  for name, mask in dict(source_upper_surface=source_mask, **masks).items()} if finite else None
        results[label] = values
    comparisons = dict(A_to_B=trial_comparison(a, b, names), A_to_A_repeat=trial_comparison(a, repeat, names))
    pin_exact = all(value['pins']['raw_exact'] and value['pins']['promoted_exact'] for value in results.values())
    restored = all(value['stats_restore_exact'] for value in results.values())
    return finite_json(dict(lambda_baseline=weight, trial_settings=deepcopy(settings),
        alpha_comparison=dict(fixed_for_all_trials=settings['alpha'], native_baseline=native_a,
            native_geometric_at_baseline_lambda=native_b,
            composite_norms=[norm(direction) for direction in directions],
            limitation='Native B alpha is metadata only; no trial at that alpha was evaluated. Failing the fixed-alpha proposal does not refute a full geometric-policy solve.'),
        selected_win_index=audit['win_index'], N=audit['N'], T=audit['T'], dt=audit['dt'],
        leaf_names=names, mode=audit['mode'], layer_u_render_only=audit['layer_u_render_only'],
        pcgrad_conflict_A=conflict_a, pcgrad_conflict_B=conflict_b,
        gradient_comparison=grouped_comparison(*directions, names),
        source_cohort=source_metadata, cohorts=cohort_metadata, source_spacing=spacing,
        target_spacing=geometry_reference['target_spacing'], density_radius=geometry_reference['radius'],
        reference_motion={name: trajectory_summary(reference, reference, mask, audit['dt'], spacing)
                          for name, mask in masks.items()},
        reference_geometry=endpoint_quality(reference['promoted'], geometry_reference),
        trials=results, comparisons=comparisons,
        contract_valid=bool(comparisons['A_to_A_repeat']['controls_exact'] and pin_exact and restored),
        evaluation_order=['A physical variance', 'B geometric variance', 'A repeated physical variance'],
        definition='Only physical-gradient substitution plus its PCGrad reference changes; baseline lambda, initial Adam state, first-trial alpha and preparation are shared',
        limits='Noncommitting first proposals; first_trial_merit_ok is not full line-search or outer acceptance. No weight is calibrated, no alternative lambda update is used, no rest/repair/policy promotion follows.',
        primitive_gate_passed=False))


class ProposalAudit(GradientAudit):
    def __init__(self, source, target, device, selected_index=23):
        super().__init__(selected_index)
        self.geometry_reference, self.source_mask, self.source_metadata = prepare_metrics(source, target, device)

    def observe(self, win_index, audit):
        require(int(win_index) == self.selected_index and self.audit is None, 'Unexpected/repeated proposal audit')
        require(int(audit['win_index']) == self.selected_index, 'Audit window mismatch')
        self.audit = analyze_proposals(audit, self.geometry_reference, self.source_mask, self.source_metadata)

    def wrap(self, original):
        def wrapped(*args, **kwargs):
            require('on_gradient_audit' not in kwargs and 'audit_proposals' not in kwargs, 'Audit already installed')
            if int(kwargs['win_index']) == self.selected_index:
                kwargs.update(on_gradient_audit=self.observe, audit_proposals=True)
            return original(*args, **kwargs)
        return wrapped

    def finish(self, result):
        report = super().finish(result)
        report.pop('no_quality_or_motion_conclusion')
        report['observation_valid'] &= bool(self.audit and self.audit['contract_valid'])
        report['no_policy_promotion'] = True
        return finite_json(report)


def run_audited(source, target, prm, cfg, selected_index=23, log=print):
    from physmorph.compute import cuda_execution
    from physmorph.pipeline import runner
    context = cuda_execution(cfg.device) if cfg.compute_backend == 'cuda' else nullcontext()
    with context:
        coordinator = ProposalAudit(source, target, cfg.device, selected_index)
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
    caches = {name: os.environ.get(name) for name in ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH')}
    require(all(value and Path(value).resolve().is_relative_to('/data') for value in caches.values()), 'Explicit /data caches required')
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
    with host_np.load(source_path, allow_pickle=False) as archive:
        source, target = archive['src'], archive['tgt']
    after = source_path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed during read')
    hashes = [hashlib.sha256(value.tobytes()).hexdigest() for value in (source, target)]
    require(source.shape == target.shape == (300000, 3) and source.dtype == target.dtype == host_np.float32, 'Expected original float32 N300k inputs')
    require(hashes == [SOURCE_SHA, TARGET_SHA] == [baseline['source_array_sha256'], baseline['target_array_sha256']], 'Original input hashes mismatch')
    config = dict(baseline['config'])
    config['stop_after_windows'] = 24
    cfg, prm = PipelineConfig(**config), MPMParams(**baseline['mpm'])
    require(cfg.T == 20 and cfg.iters == 8 and prm.dt == 1/240 and cfg.device == 'cuda'
        and cfg.compute_backend == 'cuda' and cfg.commit_pic and cfg.commit_pic_objective
        and cfg.outer_render_committed and not cfg.shift_sub and not cfg.geometric_rest
        and not cfg.geometric_variance and cfg.lambda_auto > 0 and cfg.w_kin_var > 0
        and not cfg.grad_h1 and cfg.grad_project and cfg.grad_project_mode == 'render',
        'Only original physical-variance shared-PIC T20 recipe is supported')
    reference = Path(cfg.target_reference).resolve()
    require(reference.is_relative_to('/data') and sha(reference) == baseline['target_reference_sha256'], 'Render reference changed')
    primitives = []
    for entry in baseline['preserved_primitive_evidence']:
        path = Path(entry['path']).resolve()
        require(path.is_relative_to('/data'), 'Primitive evidence outside /data')
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == entry['sha256'] and json.loads(data).get('passed') is False, 'Failed strict evidence changed')
        primitives.append(deepcopy(entry))
    require(len(primitives) == 2, 'Both failed strict primitive results must remain')
    dependencies = [Path(__file__), ROOT/'scripts/probes/variance_gradient_audit.py',
                    ROOT/'scripts/probes/pic_endpoint_quality.py', ROOT/'scripts/probes/geometric_variance_integration.py']
    code = sorted((ROOT/'physmorph').rglob('*.py')) + dependencies
    protocol = dict(start_utc=datetime.now(timezone.utc).isoformat(),
        baseline_protocol=str(baseline_path), baseline_protocol_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
        source=str(source_path), source_array_sha256=hashes[0], target_array_sha256=hashes[1],
        source_metadata_sha256=baseline['source_metadata_sha256'], target_reference_sha256=sha(reference),
        config=asdict(cfg), mpm=asdict(prm), selected_attempt=24,
        config_override=dict(stop_after_windows=dict(before=baseline['config']['stop_after_windows'], after=24)),
        code_sha256={path.resolve().relative_to(ROOT).as_posix(): sha(path) for path in code},
        cache_directories=caches, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        preserved_primitive_evidence=primitives, primitive_gate_passed=False,
        scope='Same prepared noncommitting A/B/A first proposals at baseline lambda; baseline continuation only, no alternate solve or raw archive')
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'protocol.json', protocol)
    start = time.monotonic()
    result, report = run_audited(source, target, prm, cfg)
    torch.cuda.synchronize()
    report.update(seconds=time.monotonic()-start, end_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=sha(out/'protocol.json'), torch_peak_bytes=torch.cuda.max_memory_allocated())
    save_json(out/'proposal_audit.json', report)
    print(json.dumps(dict(out=str(out), observation_valid=report['observation_valid'], primitive_gate_passed=False)), flush=True)
    if not report['observation_valid']:
        raise SystemExit('Proposal observation contract or selected baseline outer admission failed; preserved report')


if __name__ == '__main__':
    main()
