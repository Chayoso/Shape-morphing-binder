"""P300 archive-only transient supply audit, on the reviewed cap24 evidence.

Production geometry is CUDA-only. Hash/NPZ/JSON I/O uses the host. This samples
saved states, including W1; it cannot recover the raw x[T] replaced by PIC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
QUALITY_PROBE_SHA = '0b593c6f445c2f602498e7ce41dc3d3761752420c069cffb524531d7ca651fa2'
SOURCE_IDS_SHA = '5388295dbac4ef4d869e4afc6585c1273d740d9e9ebe3eee730aca18e7a41da1'
IDENTITY_KEYS = ('device', 'inode', 'bytes', 'mtime_ns')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_quality(quality, probe_sha):
    """Report hashes are supplied independently, not inferred from this content."""
    require(probe_sha == quality.get('probe_sha256') == QUALITY_PROBE_SHA,
            'Exact reviewed cap24 quality probe required')
    require(quality.get('requested_windows') == 24 and quality.get('n') == 300000
            and quality.get('T') == 20 and quality.get('loss_res') == 36,
            'Only the original N300k cap24/T20/loss36 report is supported')
    cfg, mpm = quality['config'], quality['mpm']
    require(cfg.get('archive_stride') == 1 and cfg.get('stop_after_windows') == 24
            and cfg.get('commit_pic') is True and cfg.get('commit_pic_objective') is True
            and cfg.get('compute_backend') == 'cuda' and not cfg.get('shift_sub'),
            'Unsupported saved-state contract')
    require(mpm['dt'] == 1/240 and mpm['dx'] == .3062907543956724,
            'Unexpected physical discretization')
    require(quality.get('config_changes') == {}, 'Expected equal old/new policy configurations')
    cohort = quality['cohorts']['fixed_source_upper_surface']
    require(cohort['sampled_count'] == cohort['eligible_count'] == 6712
            and cohort['ids_sha256'] == SOURCE_IDS_SHA, 'Fixed source cohort differs')
    for name in ('native_spacing', 'target_spacing', 'density_radius'):
        require(math.isfinite(quality[name]) and quality[name] > 0, 'Invalid fixed '+name)


def verify_artifacts(arm, file_record):
    """Full hashes and exact identity precede any run JSON or NPZ parsing."""
    prefix = arm['prefix']
    paths = [prefix+suffix for suffix in ('.json', '.log', '_render_full_dt_iso_nn.npz')]
    entries = arm['artifact_provenance']
    require(len(entries) == 3 and [item['path'] for item in entries] == paths,
            'Run artifact membership or path differs')
    require(entries[0]['sha256'] == arm['run_json_sha256'], 'Run JSON hash bindings differ')
    for item in entries:
        path = Path(item['path'])
        require(not path.is_symlink() and path.is_file(), 'Evidence must be a regular non-symlink file')
        actual = file_record(path)
        require(all(actual[key] == item[key] for key in ('sha256', *IDENTITY_KEYS)),
                'Evidence hash or identity differs: '+str(path))
    return entries


def window_frames(records, physical_indices, steps=20):
    """Previous accepted anchor plus each saved rollout, excluding null/held rows."""
    require(type(steps) is int and steps > 0, 'Positive integer T required')
    expected, windows, anchor = [0], [], 0
    for ordinal, record in enumerate(records, 1):
        require(not record.get('held') and not record.get('null_commit'), 'Nonphysical record passed to window traversal')
        end = int(record['frame_end'])
        start = end-steps
        require(start > anchor, 'Overlapping or invalid physical window')
        indices = [anchor, *range(start, end)]
        windows.append(dict(commit=ordinal, attempt=int(record['animation'])+1,
                            frame_indices=indices))
        expected.extend(indices[1:])
        anchor = end-1
    require(windows and expected == physical_indices, 'Physical frame map differs from reviewed scope')
    return windows


def frame_sample(x, source_ids, upper_target, radius, target_spacing):
    """One current-cloud tree; source counts exclude self, target query is fixed."""
    from physmorph.compute import array_api as np, KDTree, to_array
    x = to_array(x)
    require(x.ndim == 2 and x.shape[1] == 3 and bool(np.isfinite(x).all()), 'Malformed/nonfinite saved cloud')
    tree = KDTree(x)
    counts = tree.query_ball_point(x[source_ids], radius, return_length=True)-1
    gaps = tree.query(upper_target)[0] if len(upper_target) else np.empty(0, dtype=np.float64)
    require(bool((counts >= 0).all()), 'Source self-neighbor was not counted')
    return counts, gaps/target_spacing


def _rate(count, total):
    return count/total if total else None


def _summary(counts, gaps):
    from physmorph.compute import array_api as np
    return dict(source_particles=len(counts), source_density=float(counts.mean()/8) if len(counts) else None,
                source_under_half=_rate(int((counts < 4).sum()), len(counts)),
                source_min_count=int(counts.min()) if len(counts) else None,
                target_particles=len(gaps), target_covered=int((gaps <= 2).sum()),
                target_coverage=_rate(int((gaps <= 2).sum()), len(gaps)),
                target_gap_p95_sp=float(np.percentile(gaps, 95)) if len(gaps) else None,
                target_gap_max_sp=float(gaps.max()) if len(gaps) else None)


def window_summary(counts, gaps, window):
    """Device reductions over one window; thresholds are the historical metrics.

    Returns JSON plus device masks for union accounting. No event is eligible
    merely because an unfinished target region is uncovered at an endpoint.
    """
    from physmorph.compute import array_api as np, to_array
    counts, gaps = to_array(counts), to_array(gaps)
    indices = window['frame_indices']
    require(counts.ndim == gaps.ndim == 2 and len(counts) == len(gaps) == len(indices)
            and len(indices) >= 3, 'Window requires anchor, interior and endpoint')
    require(bool(np.isfinite(counts).all()) and bool(np.isfinite(gaps).all())
            and bool((counts >= 0).all()) and bool((gaps >= 0).all()), 'Invalid supply samples')
    source_eligible = (counts[0] >= 4) & (counts[-1] >= 4)
    target_eligible = (gaps[0] <= 2) & (gaps[-1] <= 2)
    source_loss = (counts[1:-1] < 4) & source_eligible[None, :]
    target_loss = (gaps[1:-1] > 2) & target_eligible[None, :]
    source_union, target_union = source_loss.any(0), target_loss.any(0)
    frames = [dict(phase=phase, frame=frame, **_summary(counts[phase], gaps[phase]))
              for phase, frame in enumerate(indices)]

    def losses(mask, eligible):
        eligible_n = int(eligible.sum())
        each = mask.sum(1)
        union_n = int(mask.any(0).sum())
        worst_index = int(each.argmax()) if len(each) else None
        worst_count = int(each[worst_index]) if worst_index is not None else 0
        return dict(eligible_ids=eligible_n, interior_observations=eligible_n*len(each),
                    lost_observations=int(each.sum()), union_lost_ids=union_n,
                    union_fraction=_rate(union_n, eligible_n),
                    worst=dict(phase=worst_index+1, frame=indices[worst_index+1],
                               count=worst_count, fraction=_rate(worst_count, eligible_n))
                          if eligible_n and worst_count else None,
                    per_interior_frame_counts=[int(value) for value in each],
                    status='available_descriptive' if eligible_n else 'inconclusive_no_endpoint_supported_ids')

    def dip(key):
        a, b = frames[0][key], frames[-1][key]
        if a is None or b is None:
            return None
        interior = min(frames[1:-1], key=lambda row: row[key])
        return dict(endpoint_floor=min(a, b), interior_min=interior[key],
                    below_endpoint_floor=max(0., min(a, b)-interior[key]),
                    phase=interior['phase'], frame=interior['frame'])

    source_report, target_report = losses(source_loss, source_eligible), losses(target_loss, target_eligible)
    lost_counts, lost_gaps = counts[1:-1][source_loss], gaps[1:-1][target_loss]
    zero_source = source_loss & (counts[1:-1] == 0)
    source_report['severity'] = dict(
        minimum_count=int(lost_counts.min()) if lost_counts.size else None,
        zero_count_observations=int(zero_source.sum()),
        zero_count_union_ids=int(zero_source.any(0).sum()))
    target_report['severity'] = dict(
        maximum_gap_sp=float(lost_gaps.max()) if lost_gaps.size else None,
        p95_gap_sp=float(np.percentile(lost_gaps, 95)) if lost_gaps.size else None)
    result = dict(**window, frames=frames,
                  source_endpoint_supported_loss=source_report,
                  target_endpoint_covered_loss=target_report,
                  source_density_dip=dip('source_density'), target_coverage_dip=dip('target_coverage'))
    return result, source_union, target_union, source_eligible, target_eligible


def summarize_arm(run, windows, source_ids, target_ids, target, radius, target_spacing):
    from physmorph.compute import array_api as np, to_array, to_host
    start = time.perf_counter()
    upper_target = target[target_ids]
    masks = [np.zeros(len(ids), dtype=bool) for ids in (source_ids, target_ids, source_ids, target_ids)]
    reports, previous, sampled = [], None, 0
    for window in windows:
        values = [previous] if previous is not None else [frame_sample(
            run['frames'][window['frame_indices'][0]], source_ids, upper_target, radius, target_spacing)]
        sampled += int(previous is None)
        for frame in window['frame_indices'][1:]:
            require(run['frames'][frame].shape == (300000, 3), 'Particle identity/count changed')
            values.append(frame_sample(run['frames'][frame], source_ids, upper_target, radius, target_spacing))
            sampled += 1
        previous = values[-1]
        report, *current = window_summary(np.stack([value[0] for value in values]),
                                          np.stack([value[1] for value in values]), window)
        for union, value in zip(masks, current):
            union |= value
        reports.append(report)
        print(json.dumps(dict(event='accepted_window_audited', prefix=run['prefix'],
                              commit=window['commit'], saved_frames=sampled,
                              elapsed_seconds=time.perf_counter()-start)), flush=True)

    def union_report(ids, lost, eligible):
        lost_ids, eligible_ids = ids[lost], ids[eligible]
        return dict(eligible_union_count=len(eligible_ids), lost_union_count=len(lost_ids),
                    lost_union_fraction=_rate(len(lost_ids), len(eligible_ids)),
                    lost_ids=to_host(lost_ids).tolist(),
                    lost_ids_sha256=digest(to_host(lost_ids).tobytes()),
                    eligible_ids_sha256=digest(to_host(eligible_ids).tobytes()),
                    denominator='Union of IDs eligible in at least one window; an event requires eligibility in its own window')

    return dict(windows=reports, saved_frames_sampled=sampled, seconds=time.perf_counter()-start,
                source_event_union=union_report(source_ids, masks[0], masks[2]),
                target_event_union=union_report(target_ids, masks[1], masks[3]),
                endpoint_scope=run['endpoint_scope'], physical_indices=run['physical_indices'])


def audit(quality, quality_sha256, out):
    require(not out.exists(), 'Output exists; preserve earlier evidence')
    started = time.perf_counter()
    quality_bytes = quality.read_bytes()
    require(digest(quality_bytes) == quality_sha256, 'Quality JSON does not match supplied SHA256')
    report = json.loads(quality_bytes)
    from scripts.probes import constitutive_quality as cq
    validate_quality(report, digest(Path(cq.__file__).read_bytes()))
    helpers = {name: digest((ROOT/name).read_bytes()) for name in cq.HELPERS}
    treatment = report['source_treatment']
    cq.validate_snapshots(treatment['old'], treatment['new'],
                          [treatment['helper_sha256'], treatment['helper_sha256'], helpers])
    loaded = cq.snapshot(ROOT)
    require(loaded['digest'] == treatment['new']['digest']
            and loaded['files'] == treatment['new']['files'], 'Loaded numerical source differs from quality audit')
    require(report['input_hashes'] == dict(source=cq.SOURCE_SHA, target=cq.TARGET_SHA,
            metadata=cq.METADATA_SHA, target_reference=cq.REFERENCE_SHA), 'Original input hash bindings differ')
    for name in ('baseline', 'candidate'):
        require(all(Path(item['path']).resolve().is_relative_to('/data')
                    for item in report[name]['artifact_provenance']), 'Evidence paths must resolve below /data')
    evidence = {name: verify_artifacts(report[name], cq.file_record) for name in ('baseline', 'candidate')}

    # All artifact bytes have been verified before loaders parse archive members.
    from scripts.probes.render_influence import load_run
    from scripts.probes.quality_compare import bounded_ids
    from physmorph.compute import array_api as np, KDTree, cuda_execution, to_array, to_host
    require(Path(sys.modules['physmorph'].__file__).resolve().parent.parent == ROOT,
            'Imported numerical source is outside inspected root')
    runs = {}
    for name, expected_digest in zip(('baseline', 'candidate'), cq.SOURCE_DIGESTS):
        run = load_run(Path(report[name]['prefix']))
        cq.validate_run_binding(run['meta'], expected_digest)
        cq.validate_inputs(run['source'], run['target'])
        require(run['arm']['config'] == report['config'] and run['meta']['mpm'] == report['mpm'], 'Run/report config mismatch')
        run.update(cq.scope_history(run['arm']['history'], run['delivered'], len(run['frames']), windows=24))
        require(run['endpoint_scope'] == report[name]['endpoint_scope'], 'Delivered endpoint scope differs')
        require(run['physical_indices'] == report[name]['raw_checks']['frame_indices'], 'Accepted frame map differs')
        require([(i+1, int(row['frame_end'])-1) for i, row in enumerate(run['records'])]
                == [(row['commit'], row['frame']) for row in report[name]['curve']], 'Geometry endpoint map differs')
        runs[name] = run

    result = dict(quality_path=str(quality), quality_sha256=quality_sha256,
                  probe_sha256=digest(Path(__file__).read_bytes()), quality_probe_sha256=QUALITY_PROBE_SHA,
                  numerical_source_sha256=loaded['digest'], helper_sha256=helpers,
                  artifact_provenance=evidence, input_hashes=report['input_hashes'],
                  mpm=report['mpm'], T=20, loss_res=36, n=300000,
                  source_spacing=report['native_spacing'], target_spacing=report['target_spacing'],
                  density_radius=report['density_radius'], no_policy_promotion=True)
    with cuda_execution('cuda'):
        source, target = to_array(runs['baseline']['source']), to_array(runs['baseline']['target'])
        spacing, radius = report['native_spacing'], report['density_radius']
        source_counts = KDTree(source).query_ball_point(source, 2*spacing, return_length=True)
        source_mask = ((source_counts < .6*np.median(source_counts))
                       & (source[:, 1] >= (source[:, 1].min()+source[:, 1].max())/2))
        source_ids, source_meta = bounded_ids(source_mask)
        require({key: source_meta[key] for key in ('eligible_count', 'sampled_count', 'ids_sha256')}
                == {key: report['cohorts']['fixed_source_upper_surface'][key]
                    for key in ('eligible_count', 'sampled_count', 'ids_sha256')}, 'Recomputed source ID cohort differs')
        target_ids = np.flatnonzero(target[:, 1] > 2.3)
        require(len(target_ids) == 15312, 'Original fixed upper target cohort differs')
        result['cohorts'] = dict(source_upper=source_meta,
            target_upper=dict(count=len(target_ids), ids_sha256=digest(to_host(target_ids).tobytes())))
        for name, run in runs.items():
            windows = window_frames(run['records'], run['physical_indices'])
            result[name] = summarize_arm(run, windows, source_ids, target_ids, target, radius, report['target_spacing'])
    for entries in evidence.values():
        for item in entries:
            require(cq.file_identity(Path(item['path'])) == {key: item[key] for key in IDENTITY_KEYS},
                    'Evidence changed during traversal')
    require(quality.read_bytes() == quality_bytes, 'Quality evidence changed during traversal')
    result.update(seconds=time.perf_counter()-started, definitions=dict(
        scope='Every retained accepted window including W1, with previous accepted anchor; null/held frames excluded; each arm keeps its own delivered duration',
        density='Fixed original 6712 source IDs; current neighbor ball at recorded target median k9[:,8] radius; exclude self; density=mean(count)/8; under-half=count<4',
        coverage='Fixed original target y>2.3 IDs; current-cloud nearest distance divided by recorded target spacing; covered <=2',
        events='Source count>=4 or target gap<=2 at BOTH window endpoints, but respectively count<4 or gap>2 in a saved interior phase1..19; all eligible denominators retained',
        severity='Per-window severity uses actual endpoint-eligible loss observations only; zero-count union IDs are unique within that window, not summed across windows; empty minimum/gap populations are null',
        extrema='Aggregate dips below the lesser endpoint aggregate and same-ID transient losses are different diagnostics; absolute early low coverage includes unfinished transport',
        saved_phase='Phase0=previous accepted anchor; phases1..19 are raw saved physical/layer positions; phase20 is promoted endpoint, raw final physical/layer step PLUS PIC',
        blind_spot='Raw x[T] before PIC is absent. No claim about that state, between-substep continuity, unseen surface material, whole-object watertightness or after-arrival rest',
        threshold_scope='Existing bunny diagnostic regions/radii/count thresholds only; no new physical acceptance criterion',
        causality='Whole optimizer/code comparison, not isolated fixed-lambda attribution; saved-state density/coverage is not a hole-repair pass'))
    data = json.dumps(result, indent=2, allow_nan=False).encode('utf-8')
    require(len(data) < 10_000_000, 'JSON exceeds the bounded output budget')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--quality-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.quality, args.out):
        require(path.resolve().is_relative_to('/data'), 'All production paths must be below /data')
    for name in ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH'):
        value = os.environ.get(name)
        require(value and Path(value).resolve().is_relative_to('/data'), 'Explicit /data cache required: '+name)
    sys.path.insert(0, str(ROOT))
    import torch
    require(torch.cuda.is_available(), 'CUDA required; no CPU numerical fallback')
    result = audit(args.quality.resolve(), args.quality_sha256, args.out.resolve())
    print(json.dumps(dict(out=str(args.out), seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
