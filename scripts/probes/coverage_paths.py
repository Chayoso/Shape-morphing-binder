"""Read saved P312 trajectories on CUDA; never construct or replay a model."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as host_np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import array_api as np, to_array, to_host, KDTree, cuda_execution


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def observations(path):
    """Decode only selected numeric observations; archive I/O, no model object."""
    with host_np.load(path, allow_pickle=False) as archive:
        root = json.loads(archive['manifest'].tobytes())['items']
        require(root['version'] == 1, 'Unknown frozen archive version')
        fields = root['observations']['items']
        out = {}
        for key in ('x0', 'pins', 'start_arrived', 'source', 'target', 'plan'):
            node = fields[key]
            require(node['type'] == 'tensor', 'Expected numeric observation: ' + key)
            value = archive[node['key']]
            require(list(value.shape) == node['shape'] and str(value.dtype) == node['dtype']
                    and not value.dtype.hasobject, 'Observation schema mismatch: ' + key)
            out[key] = to_array(value)
        out['dt'] = fields['dt']
    return out


def coverage_sets(distance, cutoff):
    """Rows: three original repeats and one terminal origin; columns: target IDs."""
    require(distance.ndim == 2 and distance.shape[0] == 4, 'Expected four endpoint rows')
    require(bool(np.isfinite(distance).all()) and cutoff > 0, 'Invalid distance or cutoff')
    covered = distance <= cutoff
    stable_covered = covered[:3].all(0)
    stable_uncovered = (~covered[:3]).all(0)
    ambiguous = ~(stable_covered | stable_uncovered)
    lost = covered[:3] & ~covered[3]
    gained = ~covered[:3] & covered[3]
    selected = np.flatnonzero(lost.any(0) | gained.any(0) | ambiguous)
    return dict(covered=covered, stable_covered=stable_covered,
                stable_uncovered=stable_uncovered, ambiguous=ambiguous,
                lost=lost, gained=gained, selected=selected)


def follow_paths(x0, X, V, targets, endpoint_ids, pins, arrived, cutoff, dt):
    """Freeze endpoint suppliers, separately query the actual per-phase nearest."""
    require(X.shape == V.shape and X.ndim == 4 and X.shape[0] == 4, 'Invalid trajectory shape')
    require(endpoint_ids.shape == (4, len(targets), 4), 'Expected four suppliers per endpoint')
    fixed_ids = endpoint_ids.transpose(1, 0, 2).reshape(len(targets), 16)
    material_ids = np.unique(fixed_ids)
    require(bool(((fixed_ids >= 0) & (fixed_ids < len(x0))).all()), 'Invalid supplying ID')
    cohort = np.where(pins[material_ids], 0, np.where(arrived[material_ids], 1, 2))
    initial = np.broadcast_to(x0[None, None], (4, 1) + x0.shape)
    phases = np.concatenate((initial, X), axis=1)
    selected_X = phases[:, :, material_ids]
    selected_V = V[:, :, material_ids]
    # Float64 geometry matches the endpoint KDTree distance convention.
    fixed_distance = np.linalg.norm(
        phases[:, :, fixed_ids].astype(np.float64) - targets[None, None, :, None, :], axis=-1)
    nearest_d, nearest_i, counts = [], [], []
    for run in range(4):
        run_d, run_i, run_n = [], [], []
        for phase in phases[run]:
            tree = KDTree(phase)
            d, i = tree.query(targets)
            run_d.append(d); run_i.append(i)
            run_n.append(tree.query_ball_point(targets, cutoff, return_length=True))
        nearest_d.append(np.stack(run_d)); nearest_i.append(np.stack(run_i)); counts.append(np.stack(run_n))
    nearest_ids = np.stack(nearest_i)
    outside = ~np.isin(nearest_ids, material_ids)
    outside_target = (nearest_ids[..., None] != fixed_ids[None, None]).all(-1)
    return dict(fixed_ids=fixed_ids, material_ids=material_ids, cohort=cohort,
                positions=selected_X, V=selected_V,
                geometric_V=np.diff(selected_X.astype(np.float64), axis=1) / dt,
                fixed_distance=fixed_distance, nearest_distance=np.stack(nearest_d),
                nearest_ids=nearest_ids, nearest_outside_global_endpoint_union=outside,
                nearest_outside_target_endpoint_set=outside_target, occupancy=np.stack(counts))


def analyze(obs, states, expected_rows):
    x0, source, target = (obs[k] for k in ('x0', 'source', 'target'))
    pins, arrived = obs['pins'], obs['start_arrived']
    N, T = len(x0), states[0]['positions'].shape[0]
    require(x0.shape == source.shape == (N, 3) and target.ndim == 2 and target.shape[1] == 3,
            'Invalid cloud shapes')
    require(pins.shape == arrived.shape == (N,) and pins.dtype == arrived.dtype == np.bool_,
            'Invalid material-ID cohorts')
    require(obs['plan'].shape == (N, 3) and obs['dt'] > 0, 'Invalid plan or dt')
    for label, value in obs.items():
        if label != 'dt': require(bool(np.isfinite(value).all()), 'Nonfinite observation: ' + label)
    for state in states:
        for key in ('positions', 'V'):
            require(state[key].shape == (T, N, 3) and bool(np.isfinite(state[key]).all()),
                    'Invalid saved trajectory: ' + key)
        require(bool((state['positions'][:, pins] == x0[pins]).all()), 'Pinned path changed')
    spacing = float(np.median(KDTree(source).query(source, k=2)[0][:, 1]))
    tspacing = float(np.median(KDTree(target).query(target, k=2)[0][:, 1]))
    cutoff = 2 * tspacing
    require(spacing > 0 and tspacing > 0, 'Invalid native spacing')
    endpoint_d, endpoint_i = zip(*(KDTree(s['positions'][-1]).query(target, k=4) for s in states))
    endpoint_d, endpoint_i = np.stack(endpoint_d), np.stack(endpoint_i)
    distance = endpoint_d[:, :, 0]
    sets = coverage_sets(distance, cutoff)
    upper = target[:, 1] > 2.3
    require(bool(upper.any()), 'Empty upper target cohort')
    for index, row in enumerate(expected_rows):
        for name, mask in (('target_near_frac', np.ones(len(target), dtype=np.bool_)),
                           ('upper_target_near_frac', upper)):
            count, n = int(sets['covered'][index, mask].sum()), int(mask.sum())
            require(count == round(row['geometry'][name] * n)
                    and abs(count / n - row['geometry'][name]) <= 2 * host_np.finfo(float).eps,
                    'Saved endpoint coverage failed closure: ' + name)
    selected = sets['selected']
    paths = follow_paths(x0, np.stack([s['positions'] for s in states]),
                         np.stack([s['V'] for s in states]), target[selected],
                         endpoint_i[:, selected], pins, arrived, cutoff, obs['dt'])
    require(bool((paths['nearest_distance'][:, -1] == distance[:, selected]).all()),
            'Path endpoint query differs')
    require(bool(((paths['nearest_distance'] <= cutoff) == (paths['occupancy'] > 0)).all()),
            'Nearest/count coverage mismatch')
    phase_covered = paths['nearest_distance'] <= cutoff
    paths.update(target_ids=selected, targets=target[selected], phase_covered=phase_covered,
                 plan=obs['plan'][paths['material_ids']], source=source[paths['material_ids']])
    endpoint = dict(target=target, distance=endpoint_d, nearest_ids=endpoint_i,
                    margins=cutoff-distance, margins_sp=(cutoff-distance)/spacing,
                    repeat_spread=np.ptp(distance[:3], axis=0), upper=upper, **sets)
    host = lambda v: to_host(v).tolist()
    records = []
    for j in range(len(selected)):
        first_loss, first_gain = [], []
        for r in range(3):
            loss = np.flatnonzero(phase_covered[r, :, j] & ~phase_covered[3, :, j])
            gain = np.flatnonzero(~phase_covered[r, :, j] & phase_covered[3, :, j])
            first_loss.append(int(loss[0]) if len(loss) else None)
            first_gain.append(int(gain[0]) if len(gain) else None)
        records.append(dict(target_id=int(selected[j]), position=host(target[selected[j]]),
                            endpoint_margins_wu=host(endpoint['margins'][:, selected[j]]),
                            endpoint_margins_sp=host(endpoint['margins_sp'][:, selected[j]]),
                            baseline_repeat_spread_wu=float(endpoint['repeat_spread'][selected[j]]),
                            endpoint_nearest_ids=host(endpoint_i[:, selected[j], 0]),
                            endpoint_covered=host(sets['covered'][:, selected[j]]),
                            first_lost_phase_vs_baselines=first_loss, first_gained_phase_vs_baselines=first_gain))
    summary = dict(N=N, T=T, target_N=len(target), dt=obs['dt'], spacing=spacing,
                   target_spacing=tspacing, cutoff=cutoff, upper_N=int(upper.sum()),
                   covered=host(sets['covered'].sum(1)), upper_covered=host(sets['covered'][:, upper].sum(1)),
                   lost_ids=[host(np.flatnonzero(row)) for row in sets['lost']],
                   gained_ids=[host(np.flatnonzero(row)) for row in sets['gained']],
                   stable_covered=int(sets['stable_covered'].sum()),
                   stable_uncovered=int(sets['stable_uncovered'].sum()),
                   ambiguous_ids=host(np.flatnonzero(sets['ambiguous'])),
                   stable_lost_ids=host(np.flatnonzero(sets['stable_covered'] & ~sets['covered'][3])),
                   stable_gained_ids=host(np.flatnonzero(sets['stable_uncovered'] & sets['covered'][3])),
                   material_ids=host(paths['material_ids']), material_cohorts=host(paths['cohort']),
                   phase_nearest_outside_global_endpoint_union=int(paths['nearest_outside_global_endpoint_union'].sum()),
                   phase_nearest_outside_target_endpoint_set=int(paths['nearest_outside_target_endpoint_set'].sum()),
                   outside_global_material_ids=host(np.unique(paths['nearest_ids'][paths['nearest_outside_global_endpoint_union']])),
                   outside_target_material_ids=host(np.unique(paths['nearest_ids'][paths['nearest_outside_target_endpoint_set']])),
                   cohort_labels=['start_pinned', 'start_arrived_free', 'remaining_free'], targets=records,
                   endpoint_closure=True, pins_exact=True)
    return summary, endpoint, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(args.out.resolve().is_relative_to(args.root.resolve()), 'Output outside project data')
    args.out.mkdir(exist_ok=False)
    run = args.root / 'work/p303/silhouette_repair1'
    result_path, protocol_path = run/'result.json', run/'protocol.json'
    result_bytes, protocol_bytes = result_path.read_bytes(), protocol_path.read_bytes()
    result_sha, protocol_sha = sha256(result_bytes).hexdigest(), sha256(protocol_bytes).hexdigest()
    result, protocol = json.loads(result_bytes), json.loads(protocol_bytes)
    require(result_sha == '66bf10d244604da395206a4891f7ef6471b48ec5469999ff24e53d3e8b1880a3', 'Wrong P312 result')
    require(protocol_sha == result['protocol_sha256'], 'Wrong P312 protocol')
    extension = result['extension']; folder = run/'silhouette_repair'
    names = [f'baseline/repeat_{i}.npz' for i in range(3)] + ['origin/terminal05_origin_1.npz']
    inputs = {str(folder/name):extension['sidecars'][name] for name in names + ['live_window.npz']}
    inputs.update(protocol['inputs'])
    inputs.update({str(result_path):result_sha, str(protocol_path):protocol_sha})
    code_root = Path(__file__).resolve().parents[2]
    code = {str(code_root/name):sha(code_root/name) for name in
            ('scripts/probes/coverage_paths.py', 'physmorph/compute.py', 'scripts/probes/inner_budget.py',
             'docs/coverage_paths_p313.md', 'scripts/ops/cuda_python.py', 'scripts/ops/run_p303_probe.sh')}
    require(all(sha(Path(k)) == v for k,v in inputs.items()), 'Input checksum mismatch')
    require(code[str(code_root/'physmorph/compute.py')] == protocol['code']['physmorph/compute.py'],
            'KDTree source changed since P312')
    require(code[str(code_root/'scripts/probes/inner_budget.py')] == protocol['helpers']['scripts/probes/inner_budget.py'],
            'Metric source changed since P312')
    spec = dict(start_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs, code=code,
                trajectory_order=names, mpm=protocol['mpm'], config=protocol['config'],
                no_forward=True, no_adoption=True, scope='Saved original triplet versus saved shared terminal05 origin only')
    own_protocol = args.out/'protocol.json'
    own_protocol.write_text(json.dumps(spec, indent=2, allow_nan=False))
    with cuda_execution('cuda:0'):
        obs = observations(folder/'live_window.npz')
        source_path = args.root/'repro/current_pair/source_render_full_dt_iso_nn.npz'
        with host_np.load(source_path, allow_pickle=False) as archive:
            require(bool((obs['source'] == to_array(archive['src'])).all())
                    and bool((obs['target'] == to_array(archive['tgt'])).all()), 'Captured input cloud changed')
        states = []
        for name in names:
            with host_np.load(folder/name, allow_pickle=False) as archive:
                states.append({k:to_array(archive[k]) for k in ('positions', 'V')})
        rows = extension['baseline']['rows'] + [extension['origin']['terminal_record']]
        summary, endpoint, paths = analyze(obs, states, rows)
        require(summary['N'] == 300000 and summary['T'] == 20 and summary['dt'] == protocol['mpm']['dt'],
                'Unexpected discretisation')
        sidecars = {}
        for name, data in (('endpoints', endpoint), ('paths', paths)):
            path = args.out/(name+'.npz')
            host_np.savez_compressed(path, **to_host(data))
            sidecars[path.name] = sha(path)
    require(all(sha(Path(k)) == v for k,v in {**inputs, **code}.items()), 'Inputs/code changed during analysis')
    summary.update(protocol_sha256=sha(own_protocol), sidecars=sidecars, inputs_code_unchanged=True,
                   rendering=dict(role='Inherited P312 only; no fresh rendering or rendering intervention',
                                  lambda_render=result['lambda_render'], evidence=result['render_influence']),
                   scope=spec['scope'])
    (args.out/'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(dict(done=True, selected_targets=len(summary['targets']),
                          lost=[len(x) for x in summary['lost_ids']], gained=[len(x) for x in summary['gained_ids']]),
                     allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
