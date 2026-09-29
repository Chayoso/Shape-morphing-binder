"""P324: observe zero learned control before/after the real runner handoff.

Capture and replay are separate processes. Nothing returned to the optimizer or
runner is replaced. A prepared successor is required; no final state is invented.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as host_np
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import array_api as np, cuda_execution, to_array, to_host, KDTree
from scripts.probes.coverage_paths import require, sha
from scripts.probes.full_horizon import array_digest


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def bound_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def verify_files(files):
    for path, expected in files.items():
        require(sha(Path(path)) == expected, 'Consumed source changed: ' + str(path))


class WithdrawalCapture:
    def __init__(self, folder, attempts=(5, 19, 27)):
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=False)
        self.attempts = tuple(attempts)
        require(len(set(attempts)) == len(attempts) and all(i >= 0 for i in attempts),
                'Source attempt indices must be unique and nonnegative')
        self.records = {}

    def save(self, tr, step, index, role, extra=None):
        from physmorph.mpm.withdrawal import OwnedWithdrawal
        owned = OwnedWithdrawal.capture(tr, step)
        arrays = to_host(owned.arrays())
        if extra:
            arrays.update(to_host(extra))
        path = self.folder / f'attempt_{index:03d}_{role}.npz'
        with path.open('xb') as stream:
            host_np.savez_compressed(stream, **arrays)
        record = dict(attempt=index, role=role, metadata=owned.metadata(),
                      path=path.name, sha256=sha(path),
                      x0_sha256=array_digest(arrays['x0']))
        self.records[(index, role)] = record
        return record

    def wrap(self, original):
        def observed(x0, prm, cfg, *args, **kwargs):
            from physmorph.pipeline import optimizer
            index = int(kwargs['win_index'])
            if index not in self.attempts and index - 1 not in self.attempts:
                return original(x0, prm, cfg, *args, **kwargs)
            require(not any(getattr(cfg, name, False) for name in
                    ('opt_material', 'commit_pic', 'shift_sub', 'local_dress_iters',
                     'settle_pin_follow', 'settle_pin_yield', 'settle_pin_kkt',
                     'rest_commit', 'settle_commit', 'freeze_arrived')),
                    'P324 capture requires raw endpoints and the declared fixed-material handoff')
            require(kwargs.get('on_rollout') is None, 'Do not replace another rollout observer')
            constructed = []
            constructor = optimizer.Trajectory

            def remember(*pos, **kw):
                tr = constructor(*pos, **kw)
                constructed.append(tr)
                return tr

            def terminal(tr, promoted, win_index):
                require(win_index == index, 'Wrong terminal callback attempt')
                require(len(constructed) == 1 and tr is constructed[0],
                        'Unexpected validated rollout ownership')
                if index in self.attempts:
                    require(torch.equal(wp.to_torch(tr.x[-1]), promoted),
                            'Raw endpoint and promoted endpoint differ')
                    self.save(tr, tr.T, index, 'pre', dict(
                        head_last_step=to_array(tr.x[-1], copy=True) - to_array(tr.x[-2]),
                        head_start=to_array(tr.x[0], copy=True)))

            with patch.object(optimizer, 'Trajectory', remember):
                output = original(x0, prm, cfg, *args, **{**kwargs, 'on_rollout': terminal})
            require(len(constructed) == 1, 'Expected one prepared evaluation trajectory')
            # The constructor precedes OT gate preparation. Own the input only
            # after that preparation/solve; fixed materials and x[0] are unchanged.
            if index - 1 in self.attempts:
                self.save(constructed[0], 0, index, 'post')
            return output
        return observed

    def finish(self, trace):
        by_attempt = {row['animation']: row for row in trace['attempts']}
        pairs, missing = [], []
        for source in self.attempts:
            before = self.records.get((source, 'pre'))
            after = self.records.get((source + 1, 'post'))
            row = by_attempt.get(source)
            if row is None or not row['committed'] or before is None or after is None:
                missing.append(dict(source_attempt=source, head_committed=bool(row and row['committed']),
                                    captured_head=before is not None, captured_successor=after is not None))
                continue
            successor = by_attempt[source + 1]
            require(row['end_frame'] == successor['start_frame'], 'Nonadjacent physical handoff')
            require(before['x0_sha256'] == after['x0_sha256'] == successor['x0_sha256'],
                    'Raw terminal position changed across the captured handoff')
            pairs.append(dict(source_attempt=source, next_attempt=source + 1,
                              end_frame=row['end_frame'], successor_committed=successor['committed'],
                              arrival_sidecar=row['sidecar'], arrival_sha256=row['sha256'],
                              pre=before, post=after))
        return dict(pairs=pairs, missing=missing, requested_source_attempts=list(self.attempts),
                    index_units='zero-based optimizer attempts, not accepted ordinals',
                    last_accepted_attempt=trace['actual_last_accepted'],
                    termination=trace['termination'],
                    scope='Actual next prepared handoff; final commit without a successor is not observed')


def capture(args):
    from physmorph.pipeline import runner
    from scripts.probes import full_horizon
    code_root = Path(__file__).resolve().parents[2]
    own_code = {str(code_root / name): sha(code_root / name) for name in
                ('scripts/probes/control_withdrawal.py', 'docs/control_withdrawal_p324.md')}
    observer = WithdrawalCapture(args.out.with_name(args.out.name + '_withdrawal'))
    cli = ['full_horizon.py', '--root', str(args.root), '--arm', 'raw', '--out', str(args.out)]
    with patch.object(runner, 'optimize_window', observer.wrap(runner.optimize_window)), \
            patch.object(sys, 'argv', cli):
        full_horizon.main()
    trace_path = args.out.with_suffix('.rest_trace.json')
    trace, digest = bound_json(trace_path)
    result = observer.finish(trace)
    verify_files(own_code)
    result.update(trace=str(trace_path), trace_sha256=digest, code=own_code, code_root=str(code_root),
                  utc=datetime.now(timezone.utc).isoformat())
    write_json(args.out.with_suffix('.withdrawal_capture.json'), result)


def tensor(array, device):
    return torch.as_tensor(array, device=device)


def coast_arrays(tr, head_last_step, spacing):
    """Per-material-ID readout; no renderer, no stopping threshold."""
    x = torch.stack([wp.to_torch(v) for v in tr.x]).double()
    steps = x[1:] - x[:-1]
    length2 = steps.square().sum(-1)
    previous = torch.cat((tensor(head_last_step, x.device).double()[None], steps[:-1]))
    dot = (steps * previous).sum(-1)
    floor_squared = (1e-4 * spacing) ** 2  # Existing P317 reversal eligibility, not a rest rule.
    valid_pair = (length2 > floor_squared) & (previous.square().sum(-1) > floor_squared)
    dets = torch.stack([torch.linalg.det(wp.to_torch(v).double()) for v in tr.F])
    require(bool(torch.isfinite(x).all() & torch.isfinite(dets).all()), 'Nonfinite passive rollout')
    fields = dict(net_squared=(x[-1] - x[0]).square().sum(-1) / spacing ** 2,
                  step_squared_mean=length2.mean(0) / spacing ** 2,
                  last_step_squared=length2[-1] / spacing ** 2,
                  stored_speed_squared=wp.to_torch(tr.v[-1]).double().square().sum(-1),
                  geometric_speed_squared=length2[-1] / tr.prm.dt ** 2,
                  reversals=((dot < 0) & valid_pair).sum(0),
                  reversal_pairs=valid_pair.sum(0), boundary_reversal=((dot[0] < 0) & valid_pair[0]),
                  boundary_pairs=valid_pair[0], min_detF=dets.min(0).values,
                  max_displacement_squared=(x - x[0]).square().sum(-1).max(0).values / spacing ** 2)
    return fields, x


def coast_health(tr):
    """Observe complete state and existing runner bounds without repairing it."""
    x = torch.stack([wp.to_torch(v) for v in tr.x])
    pin = wp.to_torch(tr.pin) > .5
    fields = {}
    for name in ('x', 'v', 'C', 'F') + (('Fg',) if tr.track_geom else ()):
        state = torch.stack([wp.to_torch(v) for v in getattr(tr, name)])
        fields['nonfinite_' + name] = (~torch.isfinite(state)).sum().double()
        if name in ('v', 'C'):
            fields['nonzero_pinned_' + name] = ((state[1:] != 0).reshape(tr.T, tr.N, -1).any(-1)
                                                 & pin[None]).sum().double()
    det = torch.stack([torch.linalg.det(wp.to_torch(v).double()) for v in tr.F])
    fields['nonpositive_detF'] = (det <= 0).sum().double()
    fields['min_detF'] = torch.nan_to_num(det, nan=-torch.inf).min()
    fields['pinned_position_changes'] = (((x != x[0]).any(-1)) & pin[None]).sum().double()
    dmin = np.asarray(tr.prm.grid_min, np.float32)
    dmax = dmin + tr.prm.dx * np.array([tr.prm.nx, tr.prm.ny, tr.prm.nz], np.float32)
    low, high = (tensor(value, x.device) for value in (dmin + 2 * tr.prm.dx, dmax - 2 * tr.prm.dx))
    fields['outside_runner_bounds'] = ((x < low) | (x > high)).any(-1).sum().double()
    values = torch.stack(list(fields.values())).cpu().tolist()
    report = {k: (v if host_np.isfinite(v) else None) for k, v in zip(fields, values)}
    report['valid'] = all(v == 0 for k, v in report.items() if k != 'min_detF') and report['min_detF'] is not None
    return report


def summarize_cohorts(fields, cohorts):
    labels = dict(net_squared='net_rms_sp', step_squared_mean='step_rms_sp',
                  last_step_squared='last_step_rms_sp', stored_speed_squared='stored_speed_rms_wu_s',
                  geometric_speed_squared='geometric_speed_rms_wu_s',
                  max_displacement_squared='max_displacement_sp')
    report = {}
    for name, mask in cohorts.items():
        count = mask.sum().double()
        denominator = count.clamp_min(1)
        packet = [count]
        keys = list(fields)
        for key in keys:
            values = fields[key].double()
            if key == 'min_detF':
                value = torch.where(mask, values, torch.inf).min()
            elif key.startswith('max_'):
                value = torch.where(mask, values, 0.).max().sqrt()
            elif key.endswith('squared') or key.endswith('squared_mean'):
                value = ((values * mask).sum() / denominator).sqrt()
            else:
                value = (values * mask).sum()
            packet.append(torch.where(count > 0, value, 0.))
        values = torch.stack(packet).cpu().tolist()
        names = [labels.get(key, key) for key in keys]
        report[name] = dict(count=int(values[0]), values=(dict(zip(names, values[1:])) if values[0] else None))
    return report


def load_snapshot(folder, record, device):
    from physmorph.mpm.withdrawal import OwnedWithdrawal
    path = folder / record['path']
    require(sha(path) == record['sha256'], 'Captured state hash mismatch')
    with host_np.load(path, allow_pickle=False) as data:
        arrays = {k: to_array(data[k]) for k in data.files if k not in ('head_last_step', 'head_start')}
        extra = {k: to_array(data[k]) for k in ('head_last_step', 'head_start') if k in data.files}
    return OwnedWithdrawal.from_arrays(arrays, record['metadata'], device=device), arrays, extra


def analyze(args):
    from physmorph import metrics
    from physmorph.pipeline.settlement import accepted_arrivals
    capture_path = args.source.with_suffix('.withdrawal_capture.json')
    captured, capture_digest = bound_json(capture_path)
    trace, trace_digest = bound_json(captured['trace'])
    require(trace_digest == captured['trace_sha256'], 'Trace changed since capture')
    protocol_path = args.source.with_suffix('.protocol.json')
    protocol, protocol_digest = bound_json(protocol_path)
    require(protocol_digest == trace['protocol_sha256'], 'Producer protocol changed')
    require(Path(captured['code_root']).resolve() == Path(__file__).resolve().parents[2],
            'Replay must use the frozen producer code')
    consumed = {str(capture_path): capture_digest, str(protocol_path): protocol_digest,
                str(captured['trace']): trace_digest, **captured['code'],
                **protocol['code'], **protocol['inputs'], **trace['output_sha256']}
    verify_files(consumed)
    config_path = args.source.with_suffix('.json')
    config, config_digest = bound_json(config_path)
    require(config_digest == trace['result_sha256'], 'Result configuration is not the observed run')
    folder = args.source.with_name(args.source.name + '_withdrawal')
    output = args.out
    output.mkdir(exist_ok=False)
    source_path = args.root / 'repro/current_pair/source_render_full_dt_iso_nn.npz'
    source_digest = sha(source_path)
    require(protocol['inputs'].get(str(source_path)) == source_digest,
            'Analysis geometry is not the producer input')
    with host_np.load(source_path, allow_pickle=False) as data:
        src, tgt = data['src'], data['tgt']
    report = dict(capture_sha256=capture_digest, config_sha256=config_digest,
                  input_sha256=source_digest, pairs=[], missing=captured['missing'],
                  discretization=dict(N=len(src), T=config['config']['T'], mpm=config['mpm'],
                                      loss_res=config['config']['loss_res'], iters=config['config']['iters']),
                  scope='One T-step passive window; no optimized controls, no adoption, no natural-rest certificate',
                  control_policy='Zero future dFc and u; no body force control. Existing relaxation/bonds/pins remain.',
                  arrival_policy='accepted_arrivals at the source endpoint with its own frozen full plan/radius; not rest',
                  rendering_influence='No rendering loss in passive replay; originating optimization report is retained',
                  render_report_sha256=sha(args.source.with_name(args.source.name + '.render_influence.json')),
                  reducer_units='net/step/max displacement RMS or maximum in source spacing; speed RMS wu/s; negative-dot counts; min detF',
                  reversals='Both step lengths exceed 1e-4 source spacing, as P317; negative dots are not an oscillation certificate')
    with cuda_execution('cuda:0'):
        src, tgt = to_array(src), to_array(tgt)
        spacing = float(np.median(KDTree(src).query(src, k=2)[0][:, 1]))
        target_spacing = float(np.median(KDTree(tgt).query(tgt, k=2)[0][:, 1]))
        report.update(source_spacing=spacing, target_spacing=target_spacing)
        for pair in captured['pairs']:
            pre, a, extra = load_snapshot(folder, pair['pre'], 'cuda:0')
            post, b, _ = load_snapshot(folder, pair['post'], 'cuda:0')
            consumed.update({str(folder / pair[k]['path']): pair[k]['sha256'] for k in ('pre', 'post')})
            oldpin, newpin = tensor(a['pin'], 'cuda') > .5, tensor(b['pin'], 'cuda') > .5
            require(not bool((oldpin & ~newpin).any()), 'Unexpected pin release in declared policy')
            cohorts = dict(common_free=~newpin & ~oldpin, newly_pinned=newpin & ~oldpin,
                           old_pins=oldpin, initially_free=~oldpin)
            arrival_path = args.source.with_name(args.source.name + '_cohorts') / pair['arrival_sidecar']
            require(sha(arrival_path) == pair['arrival_sha256'], 'Arrival reference changed')
            consumed[str(arrival_path)] = pair['arrival_sha256']
            with host_np.load(arrival_path, allow_pickle=False) as arrival:
                plan, radius = to_array(arrival['plan']), float(arrival['radius'])
            arrived = tensor(accepted_arrivals(a['x0'], plan, radius), 'cuda')
            cohorts.update(arrived_common_free=cohorts['common_free'] & arrived,
                           transit_common_free=cohorts['common_free'] & ~arrived)
            delta_names, delta_packet = [], []
            for name in sorted(a.keys() & b.keys()):
                require(a[name].shape == b[name].shape, 'State layout changed across handoff: ' + name)
                difference = tensor(b[name], 'cuda').double() - tensor(a[name], 'cuda').double()
                delta_names.append(name)
                delta_packet.extend((difference.square().mean().sqrt(), difference.abs().max()))
            deltas = torch.stack(delta_packet).cpu().tolist()
            item = dict(source_attempt=pair['source_attempt'], next_attempt=pair['next_attempt'],
                        end_frame=pair['end_frame'], successor_committed=pair['successor_committed'],
                        state_policy_deltas={k: dict(rms=deltas[2*i], max_abs=deltas[2*i+1])
                                             for i,k in enumerate(delta_names)},
                        metadata_before=pre.metadata(), metadata_after=post.metadata())
            start_distance = KDTree(a['x0']).query(tgt, k=1)[0]
            item['start'] = dict(sil_iou=metrics.sil_iou(a['x0'], tgt),
                                target_coverage_2sp=float((start_distance <= 2 * target_spacing).mean()))
            for label, owned in (('pre', pre), ('post', post)):
                tr = owned.trajectory(persistent=True)
                require(tr.capture(), 'Expected captured CUDA forward')
                tr.run()
                health = coast_health(tr)
                if not health['valid']:
                    path = output / f"attempt_{pair['source_attempt']:03d}_{label}_invalid.npz"
                    with path.open('xb') as stream:
                        host_np.savez_compressed(stream, **{name: to_host(torch.stack([wp.to_torch(v)
                            for v in getattr(tr, name)])) for name in
                            ('x', 'v', 'C', 'F') + (('Fg',) if tr.track_geom else ())})
                    item[label] = dict(health=health, metrics=None, path=path.name, sha256=sha(path))
                    del tr
                    continue
                fields, positions = coast_arrays(tr, extra['head_last_step'], spacing)
                final = to_array(positions[-1].float())
                distances = KDTree(final).query(tgt, k=1)[0]
                item[label] = dict(health=health, cohorts=summarize_cohorts(fields, cohorts),
                    sil_iou=metrics.sil_iou(final, tgt),
                    target_coverage_2sp=float((distances <= 2 * target_spacing).mean()))
                path = output / f"attempt_{pair['source_attempt']:03d}_{label}.npz"
                with path.open('xb') as stream:
                    host_np.savez_compressed(stream, positions=to_host(positions.float()),
                        **to_host(fields), **{'cohort_' + k: to_host(v) for k,v in cohorts.items()})
                item[label].update(path=path.name, sha256=sha(path))
                del tr, positions, fields
            report['pairs'].append(item)
            print(json.dumps(dict(source_attempt=pair['source_attempt'],
                pre=item['pre'].get('cohorts', {}).get('common_free', item['pre']['health']),
                post=item['post'].get('cohorts', {}).get('common_free', item['post']['health']))), flush=True)
        require(sha(source_path) == source_digest, 'Source changed during analysis')
    verify_files(consumed)
    report['consumed_hashes'] = consumed
    write_json(output / 'report.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('capture', 'analyze'))
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    require(args.out.resolve().is_relative_to(args.root.resolve()), 'Output must stay under project data')
    if args.mode == 'capture':
        capture(args)
    else:
        require(args.source is not None, 'Analysis needs the captured run prefix')
        analyze(args)


if __name__ == '__main__':
    main()
