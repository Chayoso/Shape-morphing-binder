"""P292 matched raw-state quality audit; execute on hyde06 with CUDA only.

The baseline/candidate must share inputs, MPM parameters and exact simulation code.
Default: only body_rprop may change. Explicit stress_taper and commit_pic_off
interventions permit their one flag plus cap 60 -> 8 and compare a common prefix.
Numerical geometry, cohorts and motion use CUDA;
archive/hash/JSON I/O use the host. No renderer or optimization-loss operator is read.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as host_np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import array_api as np, cuda_execution, KDTree, to_array, to_host
from physmorph.metrics import sil_iou, target_extent
from scripts.probes.morph_raw_qa import audit
from scripts.probes.render_influence import load_run, channel_summary

wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
FULL_INTERVENTIONS = ('body_rprop', 'commit_pic_off_full')


def stats(values):
    return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=float(np.max(values))) if values.size else None


def accepted_raw_indices(records, steps):
    """Exact accepted rollout states at archive_stride=1; null holds are omitted."""
    indices = [0]
    for row in records:
        end = int(row['frame_end'])
        start = end-int(steps)
        if start <= indices[-1]:
            raise ValueError('Accepted frame spans overlap or have inconsistent T')
        indices.extend(range(start, end))
    return indices


def checked_config_changes(ca, cb, intervention):
    changes = {k: [ca.get(k), cb.get(k)] for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)}
    if intervention == 'body_rprop':
        valid = set(changes) == {'body_rprop'} and not ca.get('body_rprop', False) and cb.get('body_rprop') is True
    elif intervention == 'stress_taper':
        valid = (set(changes) == {'ctrl_taper_sp', 'stop_after_windows'}
                 and ca.get('ctrl_taper_sp', 0.) == 0. and cb.get('ctrl_taper_sp') == 2.
                 and ca.get('stop_after_windows') == 60 and cb.get('stop_after_windows') == 8
                 and not ca.get('body_rprop', False) and not cb.get('body_rprop', False))
    elif intervention == 'commit_pic_off':
        valid = (set(changes) == {'commit_pic', 'stop_after_windows'}
                 and ca.get('commit_pic') is True and cb.get('commit_pic') is False
                 and ca.get('stop_after_windows') == 60 and cb.get('stop_after_windows') == 8
                 and not ca.get('body_rprop', False) and not cb.get('body_rprop', False))
    elif intervention == 'commit_pic_off_full':
        valid = (set(changes) == {'commit_pic'}
                 and ca.get('commit_pic') is True and cb.get('commit_pic') is False
                 and ca.get('stop_after_windows') == 60 and cb.get('stop_after_windows') == 60
                 and not ca.get('body_rprop', False) and not cb.get('body_rprop', False))
    else:
        raise ValueError(f'Unknown intervention: {intervention}')
    if not valid:
        raise ValueError(f'Unexpected {intervention} configuration changes: {changes}')
    return changes


def checked_runs(baseline, candidate, intervention):
    runs = [load_run(baseline), load_run(candidate)]
    a, b = runs
    if not (host_np.array_equal(a['source'], b['source']) and host_np.array_equal(a['target'], b['target'])):
        raise ValueError('Source or target clouds differ')
    configs = [run['arm']['config'] for run in runs]
    changes = checked_config_changes(*configs, intervention)
    hashes = [run['meta']['provenance']['code_hash'] for run in runs]
    if not hashes[0] or hashes[0] != hashes[1]:
        raise ValueError('Exact simulation source hash match required')
    code_root = Path(sys.modules['physmorph'].__file__).resolve().parent.parent
    source_files = sorted((code_root/'physmorph').rglob('*.py'))
    audit_hash = hashlib.sha256(b''.join(p.relative_to(code_root).as_posix().encode()+b'\0'+p.read_bytes()
                                       for p in source_files)).hexdigest()
    if audit_hash != hashes[0]:
        raise ValueError('Audit numerical source must match the simulation snapshot')
    if a['meta']['provenance']['mpm'] != b['meta']['provenance']['mpm']:
        raise ValueError('MPM discretisation mismatch')
    for run, config in zip(runs, configs):
        if config.get('compute_backend') != 'cuda':
            raise ValueError('Both simulations must use the strict CUDA backend')
        if int(config.get('archive_stride', 1)) != 1:
            raise ValueError('P292 exact raw-step audit requires archive_stride=1')
        if any(config.get(k) for k in ('settle_pin_yield', 'settle_pin_follow', 'settle_pin_kkt')):
            raise ValueError('This audit requires monotone pin admission')
        if not run['records']:
            raise ValueError('At least one accepted commit is required')
        if len(run['source']) != len(run['frames'][0]):
            raise ValueError('Material particle IDs must be preserved')
        run['physical_indices'] = accepted_raw_indices(run['records'], config['T'])
        run['interior_hold_frames'] = run['physical_indices'][-1]+1-len(run['physical_indices'])
    return runs, changes


def scoped_runs(runs, intervention):
    if intervention in FULL_INTERVENTIONS:
        return runs, dict(kind='full runs', endpoint_claim='final accepted states at each arm stopping point')
    common = min(8, *(len(run['records']) for run in runs))
    scoped = []
    for original in runs:
        run = dict(original)
        run['records'] = original['records'][:common]
        run['delivered'] = int(run['records'][-1]['frame_end'])
        run['physical_indices'] = accepted_raw_indices(run['records'], run['arm']['config']['T'])
        run['interior_hold_frames'] = run['physical_indices'][-1]+1-len(run['physical_indices'])
        scoped.append(run)
    return scoped, dict(kind='common accepted prefix', requested_commits=8, analyzed_commits=common,
                        original_commits=[len(run['records']) for run in runs],
                        original_delivered_frames=[run['delivered'] for run in runs],
                        endpoint_claim='prefix endpoint only; no final quality or convergence inference',
                        metadata='Original archives, run metadata and configurations remain unchanged')


def geometry_curve(run, target, target_tree, extent, target_spacing, radius, tip):
    top_target = target[:, 1] > 2.3
    curve = []
    for ordinal, record in enumerate(run['records'], 1):
        frame = int(record['frame_end'])-1
        x = to_array(run['frames'][frame])
        tree = KDTree(x)
        source_distance = target_tree.query(x)[0]
        target_distance = tree.query(target)[0]
        top = x[:, 1] > 2.3
        counts = tree.query_ball_point(x[top], radius, return_length=True)-1 if top.any() else None
        row = dict(commit=ordinal, attempt=int(record['animation'])+1, frame=frame,
                   chamfer=float(source_distance.mean()+target_distance.mean()),
                   sil_iou=sil_iou(x, target, extent),
                   target_near_frac=float((target_distance <= 2*target_spacing).mean()),
                   target_gap_sp=stats(target_distance/target_spacing),
                   top_target_near_frac=float((target_distance[top_target] <= 2*target_spacing).mean()) if top_target.any() else None,
                   top_target_gap_sp=stats(target_distance[top_target]/target_spacing),
                   source_out_far_frac=float((source_distance > 4.5*target_spacing).mean()),
                   tip_n=int((np.linalg.norm(x-tip, axis=1) < .25).sum()), top_n=int(top.sum()),
                   top_density=float(counts.mean()/8) if counts is not None else None,
                   top_under_half=float((counts < 4).mean()) if counts is not None else None)
        for key in ('Jmin_traj', 'pinned_frac', 'arrived_end_frac', 'v_mean', 'v_absmax',
                    'body_step_node_mean', 'body_step_transit_nodes_frac',
                    'body_step_transit_min', 'body_step_arrived_median', 'body_update_modes_rms',
                    'body_rms_wu', 'body_terminal_rms_wu', 'body_step_scale'):
            row[key] = record.get(key)
        curve.append(row)
    return curve


def tip_history(run, tip):
    """All simulated raw frames, including intermediate states; no held padding."""
    counts = []
    peak_count, peak_frame, peak_ids = -1, None, None
    for frame in run['physical_indices']:
        x = to_array(run['frames'][frame])
        inside = np.linalg.norm(x-tip, axis=1) < .25
        count = int(inside.sum())
        counts.append(count)
        if count > peak_count:
            peak_count, peak_frame, peak_ids = count, frame, inside.copy()
    retained = int((inside & peak_ids).sum())
    return dict(simulated_frames=len(run['physical_indices']), frame_indices=run['physical_indices'],
                counts=counts, peak_n=peak_count, first_peak_frame=peak_frame,
                final_n=counts[-1], retained_peak_material_n=retained,
                lost_peak_material_n=peak_count-retained, new_since_peak_n=counts[-1]-retained,
                final_over_peak=float(counts[-1]/peak_count) if peak_count else None,
                note='A transient overshoot can raise the peak; attrition alone is not a shape-quality verdict')


def admitted_mask(run, through_attempt=None):
    admitted = np.zeros(len(run['source']), dtype=bool)
    pins, when = to_array(run['pins']), to_array(run['pin_at'])
    for row in run['records']:
        attempt = int(row['animation'])+1
        if through_attempt is None or attempt <= through_attempt:
            admitted |= pins & (when == attempt)
    return admitted


def scoped_pin_motion(run, spacing):
    """All archive rows inside the explicitly scoped prefix, including null holds."""
    pins, when = to_array(run['pins']), to_array(run['pin_at'])
    admission = np.full(len(pins), -1, dtype=np.int64)
    for row in run['records']:
        admission[pins & (when == int(row['animation'])+1)] = int(row['frame_end'])-1
    valid = (admission >= 0) & (admission < run['delivered']-1)
    anchor = np.zeros((len(pins), 3), dtype=np.float32)
    maxima = np.zeros(len(pins), dtype=np.float32)
    for frame in range(run['delivered']):
        x = to_array(run['frames'][frame])
        new = valid & (admission == frame)
        anchor[new] = x[new]
        active = valid & (admission < frame)
        maxima[active] = np.maximum(maxima[active], np.linalg.norm(x[active]-anchor[active], axis=1))
    values = maxima[valid]
    return dict(delivered_frames_checked=run['delivered'], admitted_points=int((admission >= 0).sum()),
                checked_particles=int(valid.sum()), moved_particles_exact=int((values > 0).sum()),
                max_wu=float(values.max()) if len(values) else 0.,
                per_particle_max_drift_sp=stats(values/spacing))


def bounded_ids(mask, limit=20000):
    ids = np.flatnonzero(mask)
    count = int(len(ids))
    if count > limit:
        ids = ids[np.linspace(0, count-1, limit, dtype=np.int64)]
    # Provenance output is an explicit host I/O boundary, not a numerical operation.
    host_ids = to_host(ids)
    return ids, dict(eligible_count=count, sampled_count=int(len(ids)),
                     ids_sha256=hashlib.sha256(host_ids.tobytes()).hexdigest())


def cohort_motion(run, ids, common, spacing):
    if not len(ids) or common < 3:
        return None
    rows = run['curve'][max(0, common-11):common]
    positions = np.stack([to_array(run['frames'][row['frame']])[ids] for row in rows])
    moves = positions[1:]-positions[:-1]
    lengths = np.linalg.norm(moves, axis=2)
    active = (lengths[:-1] > 1e-4*spacing) & (lengths[1:] > 1e-4*spacing)
    reversal = (moves[:-1]*moves[1:]).sum(2) < 0
    path = lengths.sum(0)
    moving = path > 1e-4*spacing
    efficiency = np.linalg.norm(positions[-1]-positions[0], axis=1)[moving]/path[moving]
    # Same material IDs in both arms; the normal estimator is arm-specific and frozen
    # at the common interval endpoint, not a pre-treatment or shared direction basis.
    endpoint = to_array(run['frames'][rows[-1]['frame']])
    neighbors = KDTree(endpoint).query(endpoint[ids], k=33)[1]
    normal = endpoint[ids]-endpoint[neighbors].mean(1)
    magnitude = np.linalg.norm(normal, axis=1)
    valid = magnitude > 1e-9
    normal /= np.maximum(magnitude[:, None], 1e-9)
    signed = (moves*normal[None]).sum(2)
    tangent = np.linalg.norm(moves-signed[:, :, None]*normal[None], axis=2)
    raw_indices = [frame for frame in run['physical_indices']
                   if rows[0]['frame'] <= frame <= rows[-1]['frame']]
    raw_positions = np.stack([to_array(run['frames'][frame])[ids] for frame in raw_indices])
    raw_moves = raw_positions[1:]-raw_positions[:-1]
    raw_lengths = np.linalg.norm(raw_moves, axis=2)
    raw_signed = (raw_moves*normal[None]).sum(2)
    raw_tangent = np.linalg.norm(raw_moves-raw_signed[:, :, None]*normal[None], axis=2)
    raw_active = (raw_lengths[:-1] > 1e-4*spacing) & (raw_lengths[1:] > 1e-4*spacing)
    raw_reversal = (raw_moves[:-1]*raw_moves[1:]).sum(2) < 0
    return dict(commit_range=[rows[0]['commit'], rows[-1]['commit']],
                frame_range=[rows[0]['frame'], rows[-1]['frame']],
                raw_physical_states=len(raw_indices),
                excluded_interior_hold_frames=rows[-1]['frame']-rows[0]['frame']+1-len(raw_indices),
                pinned_frac_start=float(admitted_mask(run, rows[0]['attempt'])[ids].mean()),
                pinned_frac_end=float(admitted_mask(run, rows[-1]['attempt'])[ids].mean()),
                step_sp=stats(lengths.ravel()/spacing),
                normal_sp_per_commit=stats(np.abs(signed[:, valid]).ravel()/spacing),
                tangent_sp_per_commit=stats(tangent[:, valid].ravel()/spacing),
                reversal_fraction=float(reversal[active].mean()) if active.any() else None,
                reversed_pairs=int(reversal[active].sum()), moving_pairs=int(active.sum()),
                moving_material_points=int(moving.sum()),
                net_over_path=stats(efficiency),
                normal_basis='arm-specific frozen 33NN centroid-offset normal at common interval endpoint',
                normal_valid_points=int(valid.sum()),
                raw_step_sp=stats(raw_lengths.ravel()/spacing),
                normal_sp_per_raw_step=stats(np.abs(raw_signed[:, valid]).ravel()/spacing),
                tangent_sp_per_raw_step=stats(raw_tangent[:, valid].ravel()/spacing),
                raw_reversal_fraction=float(raw_reversal[raw_active].mean()) if raw_active.any() else None,
                raw_reversed_pairs=int(raw_reversal[raw_active].sum()), raw_moving_pairs=int(raw_active.sum()))


def compare(baseline, candidate, out, intervention='body_rprop'):
    if out.exists():
        raise FileExistsError('Use a new output path to preserve prior evidence')
    loaded, changes = checked_runs(baseline, candidate, intervention)
    runs, analysis_scope = scoped_runs(loaded, intervention)
    names = ('baseline', 'candidate')
    a = runs[0]
    with cuda_execution('cuda'):
        source, target = to_array(a['source']), to_array(a['target'])
        source_tree, target_tree = KDTree(source), KDTree(target)
        spacing = float(np.median(source_tree.query(source, k=2)[0][:, 1]))
        target_spacing = float(np.median(target_tree.query(target, k=2)[0][:, 1]))
        radius = float(np.median(target_tree.query(target, k=9)[0][:, 8]))
        tip = target[target[:, 1].argmax()]
        initial_chamfer = float(target_tree.query(source)[0].mean()+source_tree.query(target)[0].mean())
        target_tip_n = int((np.linalg.norm(target-tip, axis=1) < .25).sum())
        source_counts = source_tree.query_ball_point(source, 2*spacing, return_length=True)
        source_y_mid = float((source[:, 1].min()+source[:, 1].max())/2)
        source_mask = (source_counts < .6*np.median(source_counts)) & (source[:, 1] >= source_y_mid)
        source_ids, source_meta = bounded_ids(source_mask)
        if not len(source_ids):
            raise ValueError('Source-only upper-surface cohort is empty')
        endpoint_union = np.zeros(len(source), dtype=bool)
        for name, run in zip(names, runs):
            run['curve'] = geometry_curve(run, target, target_tree, target_extent(target), target_spacing, radius, tip)
            run['tip_history'] = tip_history(run, tip)
            final = to_array(run['frames'][run['delivered']-1])
            counts = KDTree(final).query_ball_point(final, 2*spacing, return_length=True)
            endpoint_union |= (counts < .6*np.median(counts)) & ~admitted_mask(run)
            print(json.dumps({'arm': name, 'endpoint': run['curve'][-1],
                              'tip_peak_n': run['tip_history']['peak_n']}), flush=True)
        endpoint_ids, endpoint_meta = bounded_ids(endpoint_union)
        common = min(len(run['curve']) for run in runs)
        common_boundary = np.zeros(len(source), dtype=bool)
        common_free = np.ones(len(source), dtype=bool)
        for run in runs:
            row = run['curve'][common-1]
            x = to_array(run['frames'][row['frame']])
            counts = KDTree(x).query_ball_point(x, 2*spacing, return_length=True)
            common_boundary |= counts < .6*np.median(counts)
            common_free &= ~admitted_mask(run, row['attempt'])
        free_ids, free_meta = bounded_ids(common_boundary & common_free)
        pairs = []
        for k in sorted({i for i in (1, 3, 6, 10, 20, 30, 40, 50, common) if i <= common}):
            xa, xb = [to_array(run['frames'][run['curve'][k-1]['frame']]) for run in runs]
            pairs.append(dict(commit=k, baseline=runs[0]['curve'][k-1], candidate=runs[1]['curve'][k-1],
                              position_difference_sp=stats(np.linalg.norm(xa-xb, axis=1)/spacing)))
        progress = []
        for fraction in (.75, .5, .25, .225, .22, .215, .2, .15, .1):
            threshold = initial_chamfer*fraction
            selected = [next((row for row in run['curve'] if row['chamfer'] <= threshold), None) for run in runs]
            progress.append(dict(chamfer_fraction=fraction, threshold_wu=threshold,
                                 baseline=selected[0], candidate=selected[1],
                                 actual_chamfer_gap_wu=selected[1]['chamfer']-selected[0]['chamfer'] if all(selected) else None))
        cohorts = {}
        for name, ids, metadata in (('source_upper_surface', source_ids, source_meta),
                                    ('endpoint_free_union', endpoint_ids, endpoint_meta),
                                    ('common_endpoint_free_both', free_ids, free_meta)):
            cohorts[name] = dict(**metadata, arms={arm: cohort_motion(run, ids, common, spacing)
                                                  for arm, run in zip(names, runs)})
    result = dict(probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  intervention=intervention, analysis_scope=analysis_scope,
                  config_changes=changes, code_hash=a['meta']['provenance']['code_hash'],
                  mpm=a['meta']['provenance']['mpm'], n=len(a['source']), T=a['arm']['config']['T'],
                  loss_res=a['arm']['config']['loss_res'], native_spacing=spacing,
                  target_spacing=target_spacing, coverage_radius=radius, target_tip_n=target_tip_n,
                  initial_chamfer=initial_chamfer, source_y_mid=source_y_mid,
                  definitions=dict(source_upper_surface='pre-treatment source geometry only: y >= source bbox midpoint and radius2sp neighbor count <0.6median',
                                   endpoint_free_union='union of each endpoint sparse-surface/unpinned cohort; outcome-selected and descriptive',
                                   common_endpoint_free_both='union of sparse boundaries at same common accepted endpoint, intersected with IDs unpinned in BOTH arms there; outcome-selected',
                                   sampling='sorted material IDs, all up to20000 then fixed evenly spaced index selection; same IDs in both arms',
                                   top_region='y>2.3wu, fixed spatial bunny region; target coverage uses target points in same region',
                                   density='neighbor count within target median r8 excluding self, divided by8; under-half means count<4',
                                   target_coverage='fraction within2 target native NN spacings of current source cloud',
                                   tip='radius0.25wu around the maximum-y target particle',
                                   motion='source-native spacings per accepted commit or per archived raw step as labeled; held tail excluded',
                                   velocity='history v_mean: mean speed; v_absmax: maximum absolute component, wu/s',
                                   reversal='negative dot of successive displacement vectors with each magnitude>1e-4 source native spacings',
                                   caveats='No renderer consumed; coverage counts do not prove watertightness, direction changes do not prove periodic oscillation'),
                  equal_accepted_commits=pairs, first_chamfer_threshold_crossings=progress, cohorts=cohorts)
    result['dependencies'] = {
        name: hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()
        for name in ('scripts.probes.render_influence', 'scripts.probes.morph_raw_qa')}
    for name, run in zip(names, runs):
        with cuda_execution('cuda'):
            prefix_pin = scoped_pin_motion(run, spacing) if intervention not in FULL_INTERVENTIONS else None
        result[name] = dict(prefix=run['prefix'], seconds=run['meta'].get('seconds'),
                           runtime_scope='complete original run, not the analyzed prefix',
                           commits=len(run['records']),
                           attempts=(int(run['records'][-1]['animation'])+1 if intervention not in FULL_INTERVENTIONS
                                     else len(run['arm']['history'])),
                           original_attempts=len(run['arm']['history']),
                           guards=run['arm']['guards'], guards_scope='complete original run',
                           curve=run['curve'], tip_history=run['tip_history'],
                           min_accepted_detF=min(row['Jmin_traj'] for row in run['curve'] if row['Jmin_traj'] is not None),
                           interior_hold_frames=run['interior_hold_frames'],
                           gradient_summary=channel_summary(run['records']),
                           scoped_pin_motion=prefix_pin,
                           raw_audit=audit(Path(run['prefix']), compute_backend='cuda') if intervention in FULL_INTERVENTIONS else None)
        if run['interior_hold_frames'] and result[name]['raw_audit'] and result[name]['raw_audit']['tail_unpinned_surface']:
            result[name]['raw_audit']['tail_unpinned_surface']['time_basis_warning'] = (
                'Legacy audit includes interior null holds; use corrected common-cohort raw-step motion for this comparison')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({'out': str(out), 'config_changes': changes, 'source_cohort_n': source_meta['sampled_count']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--intervention', choices=(*FULL_INTERVENTIONS, 'stress_taper', 'commit_pic_off'), default='body_rprop')
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.out, args.intervention)
