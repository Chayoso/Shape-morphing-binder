"""Matched render-channel ablation; raw geometry only, no render/loss operators.

Run on hyde06 through cuda_python.py. The treatment includes the render-dependent
acceptance/stopping policy, so endpoint differences are not a gradient-only effect.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as host_np
import warp as wp

wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import array_api as np, cuda_execution, KDTree, to_array
from physmorph.metrics import sil_iou, target_extent
from scripts.probes.morph_raw_qa import ARM, audit, frames_array


def load_run(prefix):
    meta = json.loads(Path(str(prefix) + '.json').read_text())
    arm = meta['arms'][ARM]
    archive = str(prefix) + '_' + ARM + '.npz'
    frames = frames_array(archive)
    with host_np.load(archive) as data:
        source, target = data['src'], data['tgt']
        delivered = min(int(data['deliver_n']), len(frames))
        pins, pin_at = data['pinned'], data['pinned_at']
    records = [r for r in arm['history'] if r.get('frame_end')
               and not r.get('null_commit') and int(r['frame_end']) <= delivered]
    return dict(prefix=str(prefix), meta=meta, arm=arm, source=source, target=target,
                frames=frames, delivered=delivered, records=records, pins=pins, pin_at=pin_at)


def count_optimizer_attempts(history, through_animation=None):
    """Count optimizer-call rows, optionally through a zero-based outer index.

    Accepted, rejected, null and gradient-converged calls count. Copied-frame
    and resolution-change metadata do not; this is not len(history).
    """
    return sum(1 for row in history
               if 'animation' in row and not row.get('held') and 'c2f_render_res' not in row
               and (through_animation is None or int(row['animation']) <= through_animation))


def channel_summary(records):
    keys = ('g_share', 'g_raw_cos', 'lambda')
    result = {k: float(host_np.median([r[k] for r in records if r.get(k) is not None]))
              for k in keys if any(r.get(k) is not None for r in records)}
    channels = sorted({name for r in records for name in (r.get('render_channels') or {})})
    result['channels'] = {
        name: {key: float(host_np.median([r['render_channels'][name][key] for r in records
                                        if name in (r.get('render_channels') or {})]))
               for key in ('nominal_share', 'raw_cos', 'projected_cos')}
        for name in channels}
    return result


def compare(on_prefix, off_prefix, out, provenance_review=None, repeat_prefix=None):
    runs = [load_run(on_prefix), load_run(off_prefix)]
    a, b = runs
    if not (host_np.array_equal(a['source'], b['source']) and host_np.array_equal(a['target'], b['target'])):
        raise ValueError('Inputs differ; not a matched render-channel ablation')
    cfg_a, cfg_b = [run['arm']['config'] for run in runs]
    changes = {k: [cfg_a.get(k), cfg_b.get(k)] for k in sorted(set(cfg_a) | set(cfg_b))
               if cfg_a.get(k) != cfg_b.get(k)}
    if set(changes) != {'lambda_auto'} or cfg_b['lambda_auto'] != 0:
        raise ValueError(f'Unexpected ablation changes: {changes}')
    if a['meta']['provenance']['mpm'] != b['meta']['provenance']['mpm']:
        raise ValueError('MPM discretisation mismatch')
    on_hash, off_hash = [run['meta']['provenance']['code_hash'] for run in runs]
    source_review = None
    if on_hash != off_hash:
        if provenance_review is None:
            raise ValueError('Simulation source hashes differ; an explicit source-equivalence review is required')
        source_review = json.loads(Path(provenance_review).read_text())
        if (source_review.get('on_code_hash'), source_review.get('off_code_hash')) != (on_hash, off_hash):
            raise ValueError('Source-equivalence review does not match the executed source hashes')
        if not source_review.get('active_numerical_code_equivalent') or not source_review.get('evidence'):
            raise ValueError('Source-equivalence review lacks a conclusion and evidence')
    repeat = None
    if repeat_prefix is not None:
        repeat_meta = json.loads(Path(str(repeat_prefix)+'.json').read_text())
        repeat_changes = {k: [cfg_a.get(k), repeat_meta['config'].get(k)] for k in cfg_a
                          if cfg_a.get(k) != repeat_meta['config'].get(k)}
        if set(repeat_changes) - {'stop_after_windows'} or repeat_meta['mpm'] != a['meta']['provenance']['mpm']:
            raise ValueError('Cross-snapshot prefix control changed numerical settings')
        with host_np.load(str(repeat_prefix)+'.npz') as data:
            repeat_frames = data['commits']
        repeat = dict(prefix=str(repeat_prefix), code_hash=repeat_meta.get('code_sha256'),
                      config_changes=repeat_changes, rows=[],
                      limitation='cross-snapshot repeat context only; old compact archive has no input hashes, stream/cache ownership changed, not a controlled repeat distribution')
    with cuda_execution('cuda'):
        source, target = to_array(a['source']), to_array(a['target'])
        source_tree, target_tree = KDTree(source), KDTree(target)
        spacing = float(np.median(source_tree.query(source, k=2)[0][:, 1]))
        target_spacing = float(np.median(target_tree.query(target, k=2)[0][:, 1]))
        extent = target_extent(target)
        tip = target[target[:, 1].argmax()]
        radius = float(np.median(target_tree.query(target, k=9)[0][:, 8]))
        target_tip_n = int((np.linalg.norm(target-tip, axis=1) < .25).sum())
        initial_chamfer = float(target_tree.query(source)[0].mean() + source_tree.query(target)[0].mean())
        for run in runs:
            curve = []
            for ordinal, record in enumerate(run['records'], 1):
                frame = int(record['frame_end'])-1
                x = to_array(run['frames'][frame])
                tree = KDTree(x)
                dx = target_tree.query(x)[0]
                dt = tree.query(target)[0]
                top = x[:, 1] > 2.3
                counts = tree.query_ball_point(x[top], radius, return_length=True)-1 if top.any() else None
                curve.append(dict(commit=ordinal, attempt=int(record['animation'])+1, frame=frame,
                                  chamfer=float(dx.mean()+dt.mean()), sil_iou=sil_iou(x, target, extent),
                                  target_near_frac=float((dt <= 2*target_spacing).mean()),
                                  source_out_far_frac=float((dx > 4.5*target_spacing).mean()),
                                  tip_n=int((np.linalg.norm(x-tip, axis=1) < .25).sum()),
                                  top_n=int(top.sum()), top_density=(float(counts.mean()/8) if counts is not None else None),
                                  top_under_half=(float((counts < 4).mean()) if counts is not None else None),
                                  Jmin_traj=record.get('Jmin_traj'), pinned_frac=record.get('pinned_frac'),
                                  arrived_end_frac=record.get('arrived_end_frac')))
            run['curve'] = curve
            print(json.dumps({'prefix': run['prefix'], 'commits': len(curve), 'endpoint': curve[-1]}), flush=True)
        common = min(len(run['curve']) for run in runs)
        pairs = []
        for k in sorted({i for i in (1, 3, 6, 10, 20, 30, common) if i <= common}):
            xa, xb = [to_array(run['frames'][run['curve'][k-1]['frame']]) for run in runs]
            distance = np.linalg.norm(xa-xb, axis=1)/spacing
            pairs.append(dict(commit=k, on=a['curve'][k-1], off=b['curve'][k-1],
                              position_difference_sp=dict(median=float(np.median(distance)),
                                                          p95=float(np.percentile(distance, 95)))))
        if repeat is not None:
            for k in range(min(len(repeat_frames), len(a['curve']))):
                x = to_array(a['frames'][a['curve'][k]['frame']])
                distance = np.linalg.norm(x-to_array(repeat_frames[k]), axis=1)/spacing
                repeat['rows'].append(dict(commit=k+1, median_sp=float(np.median(distance)),
                                           p95_sp=float(np.percentile(distance, 95))))
        progress = []
        for fraction in (.75, .5, .25, .15, .1, .075, .05):
            threshold = initial_chamfer*fraction
            selected = [next((row for row in run['curve'] if row['chamfer'] <= threshold), None) for run in runs]
            progress.append(dict(chamfer_fraction=fraction, threshold_wu=threshold, on=selected[0], off=selected[1],
                                 actual_chamfer_gap_wu=(selected[0]['chamfer']-selected[1]['chamfer']
                                                       if all(selected) else None)))
        # The individual audits select a different final surface cohort per arm.
        # Also measure exactly the same material IDs at common accepted commits.
        cohort = np.zeros(len(source), dtype=bool)
        for run in runs:
            if any(run['arm']['config'].get(k) for k in ('settle_pin_yield', 'settle_pin_follow', 'settle_pin_kkt')):
                raise ValueError('Common endpoint-free cohort requires monotone pin admission')
            final = to_array(run['frames'][run['delivered']-1])
            counts = KDTree(final).query_ball_point(final, 2*spacing, return_length=True)
            admitted = to_array(run['pins']) & (to_array(run['pin_at']) <= int(run['records'][-1]['animation'])+1)
            cohort |= (counts < .6*np.median(counts)) & ~admitted
        ids = np.flatnonzero(cohort)
        all_cohort_count = int(len(ids))
        if len(ids) > 20000:
            ids = ids[np.linspace(0, len(ids)-1, 20000, dtype=np.int64)]
        common_tail = dict(definition='union of both endpoint low-density surfaces minus each endpoint admitted pins; fixed material IDs',
                           surface_rule='raw neighbor count at radius 2 source native spacings < 0.6 median count',
                           eligible_count=all_cohort_count, sampled_count=int(len(ids)),
                           selection_limitation='symmetric endpoint-selected cohort; descriptive, not pre-treatment sampling',
                           commit_range=[max(1, common-10), common], arms={})
        if len(ids) and common >= 3:
            for name, run in zip(('on', 'off'), runs):
                positions = np.stack([to_array(run['frames'][row['frame']])[ids]
                                      for row in run['curve'][max(0, common-11):common]])
                moves = positions[1:]-positions[:-1]
                lengths = np.linalg.norm(moves, axis=2)
                active = (lengths[:-1] > 1e-4*spacing) & (lengths[1:] > 1e-4*spacing)
                reversal = (moves[:-1]*moves[1:]).sum(2) < 0
                path = lengths.sum(0)
                moving = path > 1e-4*spacing
                efficiency = np.linalg.norm(positions[-1]-positions[0], axis=1)[moving]/path[moving]
                common_tail['arms'][name] = dict(step_sp_median=float(np.median(lengths)/spacing),
                                                 step_sp_p95=float(np.percentile(lengths, 95)/spacing),
                                                 reversal_fraction=float(reversal[active].mean()) if active.any() else None,
                                                 moving_pair_count=int(active.sum()),
                                                 net_over_path_median=float(np.median(efficiency)) if len(efficiency) else None)
    result = dict(config_changes=changes, mpm=a['meta']['provenance']['mpm'], n=len(a['source']),
                  T=cfg_a['T'], loss_res=cfg_a['loss_res'], native_spacing=spacing, target_spacing=target_spacing,
                  coverage_radius=radius, target_tip_n=target_tip_n, initial_chamfer=initial_chamfer,
                  source_equivalence_review=source_review,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  definitions=dict(top_region='y > 2.3 wu; fixed spatial bunny region',
                                   top_density='mean raw neighbors within target median r8, excluding self, divided by 8',
                                   top_under_half='fraction of top particles with fewer than 4 raw neighbors within r8',
                                   tip_n='raw count within 0.25 wu of target maximum-y particle',
                                   target_near_frac='target particles within 2 target native spacings of source cloud',
                                   progress='first crossing of fixed fraction of initial symmetric mean NN distance'),
                  equal_accepted_commits=pairs, first_chamfer_threshold_crossings=progress)
    result['common_material_cohort_tail'] = common_tail
    result['cross_snapshot_repeat_context'] = repeat
    for name, run in zip(('on', 'off'), runs):
        result[name] = dict(prefix=run['prefix'], code_hash=run['meta']['provenance']['code_hash'],
                           seconds=run['meta'].get('seconds'), commits=len(run['records']),
                           attempts=count_optimizer_attempts(run['arm']['history']),
                           attempts_scope='optimizer calls in complete original run, including after delivery',
                           history_records=len(run['arm']['history']), guards=run['arm']['guards'],
                           curve=run['curve'], gradient_summary=channel_summary(run['records']),
                           raw_audit=audit(Path(run['prefix']), compute_backend='cuda'))
    Path(out).write_text(json.dumps(result, indent=2))
    print(json.dumps({'out': str(out), 'matched_config': changes}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--on', required=True, type=Path)
    parser.add_argument('--off', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--provenance-review', type=Path,
                        help='Explicit source-equivalence record, required when simulation code hashes differ')
    parser.add_argument('--repeat-prefix', type=Path, help='Optional saved CUDA prefix for cross-snapshot repeat context')
    args = parser.parse_args()
    compare(args.on, args.off, args.out, args.provenance_review, args.repeat_prefix)
