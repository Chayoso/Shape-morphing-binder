"""P292/P293 archive-only CUDA phase audit; no replay, optimization or rendering.

Reconstruct the exact common-free cohort from the reviewed quality comparison.
Phase T contains the last rollout step AND all commit position corrections. Their
individual contributions are not identifiable from the saved archive alone.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import array_api as np, cuda_execution, KDTree, to_array
from scripts.probes.quality_compare import (checked_runs, admitted_mask, bounded_ids,
                                            VARIANCE_INTERVENTIONS)


def phase_frame_indices(records, first_commit, last_commit, steps):
    """One prior accepted endpoint plus T real states; omit null-held rows."""
    if not 1 <= first_commit < last_commit <= len(records):
        raise ValueError('Invalid accepted-commit interval')
    rows = []
    previous = int(records[first_commit-1]['frame_end'])-1
    for ordinal in range(first_commit+1, last_commit+1):
        end = int(records[ordinal-1]['frame_end'])
        start = end-steps
        if start <= previous:
            raise ValueError('Accepted phase spans overlap')
        rows.append([previous, *range(start, end)])
        previous = end-1
    return rows


def distribution(x, include_rms=False):
    if not x.size:
        return None
    result = dict(mean=float(x.mean()), p05=float(np.percentile(x, 5)),
                  median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(x.max()))
    if include_rms:
        result['rms'] = float(np.sqrt(np.mean(np.asarray(x, np.float64)**2)))
    return result


def motion_summary(moves, normals, tangent1, tangent2, spacing, include_rms=False):
    lengths = np.linalg.norm(moves, axis=-1)/spacing
    signed = (moves*normals).sum(-1)/spacing
    tangent = np.linalg.norm(moves-signed[..., None]*spacing*normals, axis=-1)/spacing
    summarize = lambda values: distribution(values, include_rms=include_rms)
    return dict(displacement_sp=summarize(lengths.ravel()),
                signed_normal_sp=summarize(signed.ravel()),
                absolute_normal_sp=summarize(np.abs(signed).ravel()),
                tangent_length_sp=summarize(tangent.ravel()),
                signed_tangent1_sp=summarize(((moves*tangent1).sum(-1)/spacing).ravel()),
                signed_tangent2_sp=summarize(((moves*tangent2).sum(-1)/spacing).ravel()))


def phase_audit(baseline, candidate, reference, out):
    if out.exists():
        raise FileExistsError('Use a new output path to preserve evidence')
    report = json.loads(reference.read_text())
    intervention = report['intervention']
    if intervention not in ('body_rprop', 'commit_pic_off', 'commit_pic_off_full',
                           'render_arrival_handoff', 'commit_pic_objective_prefix',
                           'geometric_rest_prefix', 'geometric_rest_full', *VARIANCE_INTERVENTIONS):
        raise ValueError('Phase diagnostic permits only explicitly reviewed interventions')
    runs, changes = checked_runs(baseline, candidate, intervention)
    quality_file = Path(sys.modules['scripts.probes.quality_compare'].__file__)
    if hashlib.sha256(quality_file.read_bytes()).hexdigest() != report['probe_sha256']:
        raise ValueError('Loaded quality probe differs from executed reference')
    if report['code_hash'] != runs[0]['meta']['provenance']['code_hash']:
        raise ValueError('Reference numerical provenance differs')
    for name, run in zip(('baseline', 'candidate'), runs):
        if Path(report[name]['prefix']) != Path(run['prefix']):
            raise ValueError('Reference names different simulation artifacts')
    previous_cohort = report['cohorts']['common_endpoint_free_both']
    if intervention in VARIANCE_INTERVENTIONS and any(previous_cohort['arms'][name] is None
                                              for name in ('baseline', 'candidate')):
        reason = ('empty_common_free_cohort' if not previous_cohort['sampled_count'] else
                  'fewer_than_three_common_commits')
        result = dict(status='inconclusive', reason=reason, intervention=intervention,
                      probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      quality_json_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                      quality_probe_sha256=report['probe_sha256'], code_hash=report['code_hash'],
                      analysis_scope=report['analysis_scope'], cohort=previous_cohort,
                      mpm=report['mpm'], n=report['n'], T=report['T'],
                      native_spacing_wu=report['native_spacing'],
                      baseline=None, candidate=None,
                      scope='No matched phase motion measured; W1 excluded; no rest or repair conclusion')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps({'out': str(out), 'status': 'inconclusive', 'reason': reason}), flush=True)
        return
    ranges = [previous_cohort['arms'][name]['commit_range'] for name in ('baseline', 'candidate')]
    if ranges[0] != ranges[1]:
        raise ValueError('Reference cohort intervals differ')
    first_commit, last_commit = ranges[0]
    steps = int(report['T'])
    if steps != 20:
        raise ValueError('This P292 phase probe is scoped to the matched T=20 runs')
    spacing = float(report['native_spacing'])
    dt = float(report['mpm']['dt'])
    result = dict(probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  quality_json_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  quality_probe_sha256=report['probe_sha256'], code_hash=report['code_hash'],
                  mpm=report['mpm'], n=report['n'], T=steps, native_spacing_wu=spacing,
                  config_changes=changes, intervention=intervention,
                  analysis_scope=report['analysis_scope'], accepted_interval=ranges[0],
                  definitions=dict(
                      phase='1..T within accepted rollouts; phase T includes final rollout step and commit corrections',
                      interval='starts at first listed accepted endpoint; ends at last listed accepted endpoint',
                      normals='arm-specific frozen 33NN centroid-offset normal at common endpoint',
                      tangent_basis='t1=normalize(cross(global_y,n)); use global_x where abs(n_y)>0.9; t2=cross(n,t1)',
                      path_share='sum of phase-T displacement lengths / sum of all raw displacement lengths',
                      reversal='negative consecutive displacement dot; both lengths exceed 1e-4 source-native spacing',
                      speed='geometric endpoint displacement/dt; distinct from recorded terminal state mean speed',
                      attribution='Archive cannot separate last physical step, layer changes, commit PIC and subgrid shift'))
    if intervention in VARIANCE_INTERVENTIONS:
        result.update(status='available_descriptive', audited_windows=[first_commit+1, last_commit],
                      window1_excluded=True)
        result['definitions'].update(
            interval='starts at accepted commit1; windows2..common (at most8); W1 excluded and cap1 is separate evidence',
            attribution='phase20 contains the raw final physical/layer step PLUS shared PIC; motion_accounting separates them only on per-arm cohorts, not matched-ID causal evidence',
            policy='whole adaptive-lambda comparison; lower variance/movement may reflect slower progress or changed pins, not rest',
            rms='float64 root mean squared vector lengths/components in source-native spacing units')
        if intervention == 'geometric_variance_full':
            result['definitions']['interval'] = (
                f'starts at accepted commit{first_commit}; windows{first_commit+1}..{last_commit}; '
                'at most the last10 common windows; W1 excluded; each arm full endpoint is separate')
    with cuda_execution('cuda'):
        n = len(runs[0]['source'])
        boundary, free = np.zeros(n, bool), np.ones(n, bool)
        for run in runs:
            row = run['records'][last_commit-1]
            endpoint = to_array(run['frames'][int(row['frame_end'])-1])
            counts = KDTree(endpoint).query_ball_point(endpoint, 2*spacing, return_length=True)
            boundary |= counts < .6*np.median(counts)
            free &= ~admitted_mask(run, int(row['animation'])+1)
        ids, cohort = bounded_ids(boundary & free)
        if any(cohort[key] != previous_cohort[key] for key in ('eligible_count', 'sampled_count', 'ids_sha256')):
            raise ValueError('Reconstructed common material cohort differs')
        result['cohort'] = cohort
        for name, run in zip(('baseline', 'candidate'), runs):
            indices = phase_frame_indices(run['records'], first_commit, last_commit, steps)
            positions = np.stack([np.stack([to_array(run['frames'][f])[ids] for f in window])
                                  for window in indices])
            moves = positions[:, 1:]-positions[:, :-1]
            lengths = np.linalg.norm(moves, axis=3)
            endpoint = to_array(run['frames'][indices[-1][-1]])
            neighbors = KDTree(endpoint).query(endpoint[ids], k=33)[1]
            normal = endpoint[ids]-endpoint[neighbors].mean(1)
            norm = np.linalg.norm(normal, axis=1)
            if (norm <= 1e-9).any():
                raise ValueError('Invalid normal in exact reference cohort')
            normal /= norm[:, None]
            axis = np.zeros_like(normal)
            axis[:, 1] = 1
            polar = np.abs(normal[:, 1]) > .9
            axis[polar, 1], axis[polar, 0] = 0, 1
            tangent1 = np.cross(axis, normal)
            tangent1 /= np.linalg.norm(tangent1, axis=1)[:, None]
            tangent2 = np.cross(normal, tangent1)
            summarize_motion = lambda values: motion_summary(values, normal, tangent1, tangent2, spacing,
                                                              include_rms=intervention in VARIANCE_INTERVENTIONS)
            phases = [dict(phase=p+1, **summarize_motion(moves[:, p]))
                      for p in range(steps)]
            total_path = lengths.sum((0, 1))
            last_path = lengths[:, -1].sum(0)
            moving = total_path > 1e-4*spacing
            flat = moves.reshape(-1, len(ids), 3)
            flat_length = lengths.reshape(-1, len(ids))
            active = (flat_length[1:] > 1e-4*spacing) & (flat_length[:-1] > 1e-4*spacing)
            reversed_pair = (flat[1:]*flat[:-1]).sum(2) < 0
            reversal_groups = {}
            destination = np.arange(1, len(flat)) % steps + 1
            for label, select in (
                    ('into_final_phase', destination == steps),
                    ('from_final_to_next_phase1', destination == 1),
                    ('interior_phases', (destination != steps) & (destination != 1))):
                eligible = active[select]
                reversed_count = int((reversed_pair[select] & eligible).sum())
                count = int(eligible.sum())
                reversal_groups[label] = dict(reversed_pairs=reversed_count, eligible_pairs=count,
                                              fraction=reversed_count/count if count else None)
            per_window = []
            for j, window in enumerate(indices):
                ordinal = first_commit+j+1
                record = run['records'][ordinal-1]
                xa = to_array(run['frames'][window[-2]])
                xb = to_array(run['frames'][window[-1]])
                geometric_speed = np.linalg.norm(xb-xa, axis=1)/dt
                per_window.append(dict(commit=ordinal, attempt=int(record['animation'])+1,
                                       frame_indices=window,
                                       final_path_fraction=float(lengths[j, -1].sum()/lengths[j].sum()),
                                       final_geometric_mean_speed_all_particles_wu_s=float(geometric_speed.mean()),
                                       recorded_terminal_mean_speed_wu_s=record.get('v_mean'),
                                       final_phase=summarize_motion(moves[j, -1])))
            result[name] = dict(phases=phases, per_window=per_window,
                                first_19_phases=summarize_motion(moves[:, :-1]),
                                final_phase=summarize_motion(moves[:, -1]),
                                final_phase_total_path_share=float(last_path.sum()/total_path.sum()),
                                per_particle_final_path_share=distribution(last_path[moving]/total_path[moving]),
                                per_particle_raw_net_over_path=distribution(
                                    np.linalg.norm(positions[-1, -1]-positions[0, 0], axis=1)[moving]/total_path[moving]),
                                reversal_groups=reversal_groups)
            print(json.dumps({'arm': name, 'final_path_share': result[name]['final_phase_total_path_share'],
                              'reversal_groups': reversal_groups}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({'out': str(out), 'cohort': cohort}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    phase_audit(args.baseline, args.candidate, args.reference, args.out)
