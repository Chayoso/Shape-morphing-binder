"""Stream completed P316 archives; GPU geometry, no replay or stopping policy."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as host_np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import array_api as np, cuda_execution, to_array, to_host, KDTree
from physmorph.pipeline.settlement import accepted_arrivals
from scripts.probes.coverage_paths import require, sha
from scripts.probes.full_horizon import array_digest


class FrameReader:
    """Bounded archive I/O; decode at most one accepted window, including x0."""
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)
        self.stream = self.archive.open('frames.npy')
        version = host_np.lib.format.read_magic(self.stream)
        self.shape, fortran, self.dtype = host_np.lib.format._read_array_header(self.stream, version)
        require(not fortran and self.dtype == host_np.dtype('float32') and
                len(self.shape) == 3 and self.shape[2] == 3, 'Unsupported frame storage')
        self.offset = self.stream.tell()
        self.row_bytes = math.prod(self.shape[1:]) * self.dtype.itemsize

    def read(self, start, end):
        require(0 <= start <= end < self.shape[0], 'Frame interval outside archive')
        self.stream.seek(self.offset + start * self.row_bytes)
        size = (end-start+1) * self.row_bytes
        data = self.stream.read(size)
        require(len(data) == size, 'Incomplete archive interval')
        return host_np.frombuffer(data, self.dtype).reshape(end-start+1, *self.shape[1:])

    def close(self):
        self.stream.close()
        self.archive.close()


def reversal_pairs(first, second, spacing):
    """Existing raw-phase eligibility rule; not a rest/admission threshold."""
    require(spacing > 0, 'Invalid source spacing')
    eligible = ((np.linalg.norm(first, axis=-1) > 1e-4*spacing) &
                (np.linalg.norm(second, axis=-1) > 1e-4*spacing))
    return eligible, eligible & ((first*second).sum(-1) < 0)


def window_motion(positions, raw_endpoint, plan, radius, start_arrived, pins, speed2, dt, spacing):
    """Same IDs and same frozen plan; geometric speed is distinct from stored v."""
    require(positions.ndim == 3 and positions.shape[0] >= 2 and positions.shape[2] == 3,
            'Invalid accepted path')
    n = positions.shape[1]
    require(raw_endpoint.shape == plan.shape == (n, 3) and
            pins.shape == start_arrived.shape == speed2.shape == (n,), 'Invalid window layout')
    require(dt > 0 and radius > 0 and bool(np.isfinite(positions).all()) and
            bool(np.isfinite(raw_endpoint).all()) and bool(np.isfinite(plan).all()) and
            bool(np.isfinite(speed2).all()) and bool((speed2 >= 0).all()), 'Invalid window values')
    require(bool((positions[:, pins] == positions[0, pins]).all()) and
            bool((raw_endpoint[pins] == positions[0, pins]).all()), 'Pinned material moved')
    x = positions.astype(np.float64)
    steps = np.diff(x, axis=0)
    raw_last = raw_endpoint.astype(np.float64)-x[-2]
    correction = x[-1]-raw_endpoint.astype(np.float64)
    step2 = (steps*steps).sum(-1)
    start_geo = accepted_arrivals(positions[0], plan, radius)
    end = accepted_arrivals(positions[-1], plan, radius)
    raw_end = accepted_arrivals(raw_endpoint, plan, radius)
    eligible, reversed_pairs = reversal_pairs(steps[:-1], steps[1:], spacing)
    return dict(start_geometric_arrived=start_geo, end_arrived=end, raw_end_arrived=raw_end,
        start_mask_roundtrip_disagreement=start_geo != start_arrived,
        entered=(~start_arrived) & end, departed=start_arrived & ~end,
        path_wu=np.sqrt(step2).sum(0), step_rms_wu=np.sqrt(step2.mean(0)),
        net_wu=np.linalg.norm(x[-1]-x[0], axis=1), net_vector=x[-1]-x[0],
        adjacent_step_reversed_count=reversed_pairs.sum(0), adjacent_step_eligible_count=eligible.sum(0),
        raw_terminal_geometric_speed_wu_s=np.linalg.norm(raw_last, axis=1)/dt,
        promoted_terminal_geometric_speed_wu_s=np.sqrt(step2[-1])/dt,
        optimizer_terminal_stored_speed_wu_s=np.sqrt(speed2),
        endpoint_correction_wu=np.linalg.norm(correction, axis=1),
        correction_opposes_raw_step=(correction*raw_last).sum(-1) < 0,
        correction_changes_arrival=raw_end != end, step_squared=step2,
        first_step=steps[0], last_step=steps[-1])


def distributions(values, mask):
    n = int(mask.sum())
    if not n:
        return dict(particles=0)
    out = dict(particles=n)
    for key, value in values.items():
        if value.ndim != 1 or value.dtype == np.bool_:
            continue
        v = value[mask].astype(np.float64)
        out[key] = dict(mean=float(v.mean()), rms=float(np.sqrt((v*v).mean())),
                        p95=float(np.quantile(v, .95)), maximum=float(v.max()))
    return out


def arrival_relabel(previous, current):
    """Call only after confirming identical archive positions across references."""
    require(previous.shape == current.shape, 'Reference cohort shape changed')
    return dict(newly_labelled=current & ~previous, no_longer_labelled=previous & ~current)


def followup_statistics(fields):
    windows = fields['observed_windows']
    steps = fields['observed_steps']
    observed = windows > 0
    nw, ns = float(windows.sum()), float(steps.sum())
    per_id = dict(
        step_rms_wu=np.sqrt(fields['step_squared_wu2_sum']/np.maximum(steps, 1)),
        optimizer_terminal_stored_speed_rms_wu_s=np.sqrt(
            fields['terminal_stored_speed_squared_sum']/np.maximum(windows, 1)),
        mean_step_length_wu=fields['path_wu_sum']/np.maximum(steps, 1))
    return dict(observed_particle_windows=int(nw), observed_particle_steps=int(ns),
        pooled_step_rms_wu=float(np.sqrt(fields['step_squared_wu2_sum'].sum()/ns)) if ns else None,
        pooled_optimizer_terminal_stored_speed_rms_wu_s=float(np.sqrt(
            fields['terminal_stored_speed_squared_sum'].sum()/nw)) if nw else None,
        per_id_normalized=distributions(per_id, observed),
        per_id_accumulated_sums_and_denominators=distributions(fields, observed))


def bound_json(path):
    data = Path(path).read_bytes()
    return json.loads(data), sha256(data).hexdigest()


def analyze(prefix):
    prefix = Path(prefix)
    paths = {key: prefix.with_suffix(suffix) for key, suffix in
             (('trace', '.rest_trace.json'), ('record', '.json'), ('protocol', '.protocol.json'))}
    loaded = {k: bound_json(p) for k, p in paths.items()}
    bindings = {str(paths[k]): v[1] for k, v in loaded.items()}
    trace, record, protocol = (loaded[k][0] for k in ('trace', 'record', 'protocol'))
    require('trace_error' not in trace and trace['inputs_code_unchanged'], 'Incomplete P316 trace')
    require(trace['protocol_sha256'] == bindings[str(paths['protocol'])] and
            trace['result_sha256'] == bindings[str(paths['record'])], 'P316 JSON binding mismatch')
    raw_path = prefix.with_name(prefix.name+'_render_full_dt_iso_nn.npz')
    render_path = prefix.with_name(prefix.name+'.render_influence.json')
    render_report, render_hash = bound_json(render_path)
    for p, digest in ((raw_path, sha(raw_path)), (render_path, render_hash)):
        require(digest == trace['output_sha256'][str(p)], 'P316 output hash mismatch: '+str(p))
        bindings[str(p)] = digest
    # Reuse exactly the producer's arrival arithmetic/backend, not a silently
    # updated module from another checkout. The new analyzer is bound separately.
    for suffix in ('physmorph/compute.py', 'physmorph/pipeline/settlement.py'):
        frozen = [v for p, v in protocol['code'].items() if p.replace('\\', '/').endswith('/'+suffix)]
        current = Path(__file__).resolve().parents[2]/suffix
        require(len(frozen) == 1 and sha(current) == frozen[0], 'Producer dependency mismatch: '+suffix)
        bindings[str(current)] = frozen[0]
    cfg = record['config']
    require(cfg['archive_stride'] == 1 and not any(cfg.get(k, False) for k in
        ('settle_pin_follow', 'settle_pin_yield', 'settle_pin_kkt', 'local_dress_iters')),
        'Unsupported archive or pin-release policy')
    with host_np.load(raw_path, allow_pickle=False) as z:
        source = to_array(z['src'])
        pins = to_array(z['pinned'])
        pin_at = to_array(z['pinned_at'])
        require(int(z['deliver_n']) == trace['deliver_n'] == record['arms']['render_full_dt_iso_nn']['deliver_n'],
                'Delivery metadata mismatch')
        require(bool(np.array_equal(pins, pin_at >= 0)), 'Invalid final admission state')
    reader = FrameReader(raw_path)
    n = len(pins)
    spacing = float(np.median(KDTree(source).query(source, k=2)[0][:, 1]))
    require(spacing > 0, 'Invalid source spacing')
    require(reader.shape == (trace['actual_archive_frames'], n, 3), 'Archive/trace size mismatch')
    cohort_masks = dict(final_free=~pins, eventually_pinned=pins)
    totals = {k: np.zeros(n, np.float64) for k in
              ('path_wu', 'endpoint_correction_wu', 'adjacent_step_reversed_count',
               'adjacent_step_eligible_count', 'boundary_step_reversed_count', 'boundary_step_eligible_count',
               'net_reversed_count', 'net_eligible_count', 'entered_count', 'departed_count',
               'reference_newly_labelled_count', 'reference_no_longer_labelled_count')}
    first_arrived = np.full(n, -1, np.int32)
    first_endpoint_arrived = np.full(n, -1, np.int32)
    after_arrival = {scope: {name: np.zeros(n, np.float64) for name in
        ('observed_windows', 'observed_steps', 'path_wu_sum', 'step_squared_wu2_sum', 'terminal_stored_speed_squared_sum')}
        for scope in ('free', 'pinned')}
    first_pinned = None
    windows, references = [], []
    previous_arrival = previous_position_hash = previous_net = previous_last_step = None
    last = delivery = None
    delivery_frame = int(trace['deliver_n'])-1
    history_rows = [r for r in record['history'] if not r.get('held') and 'c2f_render_res' not in r]
    require([int(r['animation']) for r in history_rows] == list(range(len(trace['attempts']))),
            'Missing/duplicate attempt history')
    require([int(r['animation']) for r in trace['attempts']] == list(range(len(trace['attempts']))),
            'Missing/duplicate trace attempt')
    accepted_history = [int(r['animation']) for r in history_rows if r.get('frame_end') and
                        not r.get('null_commit') and not r.get('outer_rejected') and r.get('outer_accepted', 1)]
    require(accepted_history == trace['accepted_attempts'] and
            (accepted_history[-1] if accepted_history else None) == trace['actual_last_accepted'],
            'Accepted history/trace scope mismatch')
    history = {int(r['animation']): r for r in history_rows}
    try:
        for attempt in trace['attempts']:
            a = int(attempt['animation'])
            row = history[a]
            committed = bool(row.get('frame_end') and not row.get('null_commit') and
                             not row.get('outer_rejected') and row.get('outer_accepted', 1))
            require(committed == attempt['committed'], 'Trace/history commit mismatch')
            if committed:
                require(int(row['frame_end'])-1 == attempt['end_frame'] and row.get('accepted', 0) > 0,
                        'Trace/history accepted interval mismatch')
            path = prefix.with_name(prefix.name+'_cohorts')/attempt['sidecar']
            require(sha(path) == attempt['sha256'], 'Attempt archive hash mismatch')
            bindings[str(path)] = attempt['sha256']
            with host_np.load(path, allow_pickle=False) as z:
                obs = {k: to_array(z[k]) for k in z.files}
            pin_before = obs['pin_before']
            require(bool(np.array_equal(pin_before, pins & (pin_at <= a))), 'Admission chronology mismatch')
            if first_pinned is None:
                first_pinned = pin_before.copy()
                cohort_masks['initial_free'] = ~first_pinned
            start, end = int(attempt['start_frame']), int(attempt['end_frame'])
            positions_host = reader.read(start, end)
            x0hash = array_digest(positions_host[0])
            require(x0hash == attempt['x0_sha256'], 'Window archive start mismatch')
            x = to_array(positions_host)
            if not attempt['committed']:
                require(start == end or bool(np.array_equal(x[0], x[-1])), 'Uncommitted position changed')
                continue
            require(attempt['has_plan'] and len(x)-1 == cfg['T'], 'Missing accepted path/reference')
            if attempt['has_plan']:
                radius = float(obs['radius'])
                start_geo = accepted_arrivals(x[0], obs['plan'], radius)
                first_arrived = np.where((first_arrived < 0) & obs['start_arrived'], a, first_arrived)
                if previous_arrival is None:
                    cohort_masks['first_accepted_start_arrived'] = obs['start_arrived'].copy()
                if previous_arrival is not None:
                    require(x0hash == previous_position_hash, 'Reference changed alongside unaccounted position')
                    relabel = arrival_relabel(previous_arrival, start_geo)
                    references.append(dict(animation=a, frame=start,
                        newly_labelled=int(relabel['newly_labelled'].sum()),
                        no_longer_labelled=int(relabel['no_longer_labelled'].sum()),
                        stored_start_roundtrip_disagreement=int((start_geo != obs['start_arrived']).sum())))
                    for name, value in relabel.items():
                        totals['reference_'+name+'_count'] += value
                previous_arrival = accepted_arrivals(x[-1], obs['plan'], radius)
                previous_position_hash = array_digest(positions_host[-1])
            values = window_motion(x, obs['optimizer_raw_endpoint'], obs['plan'], radius,
                obs['start_arrived'], pin_before, obs['optimizer_terminal_speed_squared'], record['mpm']['dt'], spacing)
            if protocol['arm'] == 'raw':
                require(bool((values['endpoint_correction_wu'] == 0).all()), 'Raw path externally corrected')
            for name in ('path_wu', 'endpoint_correction_wu', 'adjacent_step_reversed_count', 'adjacent_step_eligible_count'):
                totals[name] += values[name]
            for name in ('entered', 'departed'):
                totals[name+'_count'] += values[name]
            if previous_net is not None:
                eligible, reversals = reversal_pairs(previous_net, values['net_vector'], spacing)
                totals['net_eligible_count'] += eligible
                totals['net_reversed_count'] += reversals
                eligible, reversals = reversal_pairs(previous_last_step, values['first_step'], spacing)
                totals['boundary_step_eligible_count'] += eligible
                totals['boundary_step_reversed_count'] += reversals
            previous_net = values['net_vector'].copy()
            previous_last_step = values['last_step'].copy()
            # Qualification begins AFTER the first accepted endpoint arrival.
            # Escapees/reentries remain in this fixed-ID chronology; never drop them.
            qualified = first_endpoint_arrived >= 0
            for scope, mask in (('free', qualified & ~pin_before), ('pinned', qualified & pin_before)):
                target = after_arrival[scope]
                target['observed_windows'] += mask
                target['observed_steps'] += mask*(len(x)-1)
                target['path_wu_sum'] += mask*values['path_wu']
                target['step_squared_wu2_sum'] += mask*values['step_squared'].sum(0)
                target['terminal_stored_speed_squared_sum'] += mask*obs['optimizer_terminal_speed_squared']
            first_endpoint_arrived = np.where((first_endpoint_arrived < 0) & values['end_arrived'], a, first_endpoint_arrived)
            masks = dict(cohort_masks, free_at_start=~pin_before,
                arrived_free_both_ends=(~pin_before) & obs['start_arrived'] & values['end_arrived'],
                previously_arrived_still_free=qualified & ~pin_before)
            row = dict(animation=a, start_frame=start, end_frame=end,
                end_arrived=int(values['end_arrived'].sum()),
                start_mask_roundtrip_disagreement=int(values['start_mask_roundtrip_disagreement'].sum()),
                correction_changes_arrival=int(values['correction_changes_arrival'].sum()),
                correction_opposes_raw_step=int(values['correction_opposes_raw_step'].sum()),
                entered=int(values['entered'].sum()), departed=int(values['departed'].sum()),
                cohorts={k: distributions(values, mask) for k, mask in masks.items()},
                step_rms_wu={k: to_host(np.sqrt(values['step_squared'][:, mask].mean(1))).tolist()
                             if bool(mask.any()) else None for k, mask in cohort_masks.items()},
                render_observations={k: history[a].get(k) for k in
                    ('d_vol', 'd_dt', 'd_sil', 'd_render', 'lambda', 'g_share', 'loss', 'render_res')})
            windows.append(row)
            last = {k: v.copy() for k, v in values.items() if v.ndim == 1}
            if end <= delivery_frame:
                delivery_pin = pins & (pin_at <= a+1)
                delivery = dict(animation=a, end_frame=end,
                    pinned_after_commit=int(delivery_pin.sum()),
                    free_after_commit=int((~delivery_pin).sum()),
                    free_at_delivery_commit=distributions(totals, ~delivery_pin),
                    retrospective_full_horizon_cohorts={k: distributions(totals, m) for k, m in cohort_masks.items()})
            print(json.dumps(dict(animation=a, final_free=row['cohorts']['final_free'],
                                  end_frame=end)), flush=True)
    finally:
        reader.close()
    require(windows and last is not None, 'No accepted physical windows')
    require(windows[-1]['animation'] == trace['actual_last_accepted'], 'Last accepted scope mismatch')
    require(delivery is not None and 0 <= delivery_frame < trace['actual_archive_frames'],
            'Invalid delivery scope')
    delivery_reader = FrameReader(raw_path)
    try:
        require(array_digest(delivery_reader.read(delivery_frame, delivery_frame)[0]) ==
                array_digest(delivery_reader.read(delivery['end_frame'], delivery['end_frame'])[0]),
                'Delivery is not a complete accepted endpoint or identical held/null row')
    finally:
        delivery_reader.close()
    require(all(sha(Path(k)) == v for k, v in bindings.items()), 'Analysis inputs changed')
    data = dict(totals, final_pinned=pins, pinned_at=pin_at, first_accepted_start_arrival=first_arrived,
                first_accepted_endpoint_arrival=first_endpoint_arrived,
                **{f'after_endpoint_arrival_{scope}_{k}': v for scope, fields in after_arrival.items() for k, v in fields.items()},
                **{'last_'+k: v for k, v in last.items()})
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), arm=protocol['arm'], bindings=bindings,
        discretization=dict(N=n, T=cfg['T'], dt=record['mpm']['dt'], dx_wu=record['mpm']['dx'],
                            loss_res=cfg['loss_res'], iters=cfg['iters'], source_spacing_wu=spacing),
        actual_last_accepted=trace['actual_last_accepted'], reported_converged=trace['reported_converged'],
        actual_frames=trace['actual_archive_frames'], delivery_frame=delivery_frame,
        held_archive_rows=trace['held_archive_rows'], truncation=trace['truncation'],
        physical_accepted_windows=len(windows), windows=windows, reference_relabels=references,
        full_path_cohorts={k: distributions(totals, m) for k, m in cohort_masks.items()},
        after_endpoint_arrival={scope: followup_statistics(fields) for scope, fields in after_arrival.items()},
        delivery_scope=delivery, guards=record['guards'],
        render_influence=render_report,
        scope='Accepted promoted paths only; stored terminal speeds precede outer operations. '
              'Final-free and eventually-pinned are fixed retrospective cohorts, not matched across arms. '
              'First endpoint arrival qualifies only subsequent accepted windows; escapees remain tracked. '
              'Accumulated reversal counts describe the entire accepted path, not just post-arrival motion. '
              'Reversal pairs require both lengths >1e-4 source spacing; zero eligible counts stay explicit. '
              'Negative dot products measure reversals, not certified oscillations; coarse arrival is not rest. '
              'No coverage, hole, passive-release, all-frame render or 4K quality certificate.'), data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/data/relcfd/chayo/physmorph_v2').resolve()
    require(args.prefix.resolve().is_relative_to(root) and args.out.resolve().is_relative_to(root),
            'Archive/output outside project data')
    args.out.mkdir(exist_ok=False)
    source_files = [Path(__file__).resolve(), Path(__file__).parents[2]/'physmorph/compute.py',
                    Path(__file__).parents[2]/'physmorph/pipeline/settlement.py',
                    Path(__file__).with_name('coverage_paths.py'), Path(__file__).with_name('full_horizon.py')]
    code = {str(p): sha(p) for p in source_files}
    with cuda_execution('cuda:0'):
        report, data = analyze(args.prefix)
        host_data = to_host(data)
    path = args.out/'particles.npz'
    with path.open('xb') as stream:
        host_np.savez_compressed(stream, **host_data)
    require(all(sha(Path(k)) == v for k, v in code.items()), 'Analyzer code changed')
    report.update(code=code, particles_sha256=sha(path))
    with (args.out/'result.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
