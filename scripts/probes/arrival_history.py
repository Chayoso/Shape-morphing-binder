"""Read-only, accepted-window individual arrival history on corrected P300.

This is a new realization of the unchanged cap24 policy, not reconstructed
per-ID evidence for a previous archive. Production numerical work stays CUDA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
QUALITY_PROBE_SHA = '0b593c6f445c2f602498e7ce41dc3d3761752420c069cffb524531d7ca651fa2'
LIMIT_BYTES = 350_000_000
JSON_RESERVE = 9_000_000
ARCHIVE_OVERHEAD_RESERVE = 1_000_000
FLOAT_FIELDS = ('path_wu', 'step_square_wu2', 'max_step_wu', 'max_excursion_wu',
                'raw_final_square_wu2', 'pic_jump_square_wu2')
INT_FIELDS = ('steps', 'eligible_reversal_pairs', 'reversed_pairs', 'windows')
EVENT_FIELDS = ('geometric_departures', 'geometric_reentries', 'plan_departures', 'plan_reentries')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def budget(n=300000, windows=24):
    require(type(n) is int and n > 0 and type(windows) is int and windows > 0, 'Invalid budget dimensions')
    per_window = n*(3*3*4+3)  # plan/raw/promoted float32 Nx3; three bool masks.
    initial = n*3*4
    final = n*(2*(8*len(FLOAT_FIELDS)+4*len(INT_FIELDS))+12+4*(3+len(EVENT_FIELDS))+5)
    total = windows*per_window+initial+final+JSON_RESERVE+ARCHIVE_OVERHEAD_RESERVE
    require(total <= LIMIT_BYTES, 'Complete evidence exceeds the 350 MB limit')
    return dict(windows=windows, particles=n, per_window_array_bytes=per_window,
                initial_array_bytes=initial, final_counter_array_bytes=final,
                json_reserve_bytes=JSON_RESERVE, archive_overhead_reserve_bytes=ARCHIVE_OVERHEAD_RESERVE,
                preflight_max_bytes=total, hard_limit_bytes=LIMIT_BYTES)


def own(value, device, dtype=None):
    return torch.as_tensor(value, device=device, dtype=dtype).detach().clone()


def arrival_mask(x, plan, radius):
    """Use the actual settlement predicate on the caller's compute backend."""
    from physmorph.compute import to_array
    from physmorph.pipeline.settlement import accepted_arrivals
    return own(accepted_arrivals(to_array(x), to_array(plan), radius), x.device, torch.bool)


def motion_stats(vectors, mask, spacing):
    values = vectors[mask].double().norm(dim=1)
    return dict(particles=int(mask.sum()), rms_wu=float(values.square().mean().sqrt()) if values.numel() else None,
                rms_sp=float((values/spacing).square().mean().sqrt()) if values.numel() else None,
                max_wu=float(values.max()) if values.numel() else None,
                max_sp=float(values.max()/spacing) if values.numel() else None)


class Accumulator:
    """Only outer-accepted windows may update these persistent per-ID counters."""

    def __init__(self, initial, spacing, dt):
        require(initial.ndim == 2 and initial.shape[1] == 3 and initial.dtype == torch.float32,
                'Expected float32 (N,3) material positions')
        require(math.isfinite(spacing) and spacing > 0 and math.isfinite(dt) and dt > 0, 'Invalid physical units')
        self.n, self.device = len(initial), initial.device
        self.spacing, self.dt = float(spacing), float(dt)
        self.initial = initial.detach().clone()
        self.last_endpoint = self.initial.clone()
        self.first_arrival = torch.full((self.n,), -1, dtype=torch.int32, device=self.device)
        self.first_endpoint_arrival = torch.zeros(self.n, dtype=torch.int32, device=self.device)
        self.first_start_arrived = torch.zeros_like(self.first_arrival)
        self.anchor = torch.zeros_like(self.initial)
        self.events = {name: torch.zeros(self.n, dtype=torch.int32, device=self.device) for name in EVENT_FIELDS}
        self.last_end_arrived = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.previous_delta = torch.zeros_like(initial)
        self.previous_observed = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.last_start_pin = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.banks = {group: dict(
            **{name: torch.zeros(self.n, dtype=torch.float64, device=self.device) for name in FLOAT_FIELDS},
            **{name: torch.zeros(self.n, dtype=torch.int32, device=self.device) for name in INT_FIELDS})
            for group in ('start_free', 'start_pinned')}
        self.accepted = 0

    def accept(self, candidate):
        path, promoted, pin = (candidate[key] for key in ('path', 'promoted', 'pin'))
        plan, radius = candidate['plan'], candidate['radius']
        start_actual, start_geometric, end_arrived = (candidate[key] for key in
            ('start_arrived', 'start_geometric', 'end_arrived'))
        require(path.ndim == 3 and path.shape[1:] == self.initial.shape and len(path) >= 2,
                'Invalid owned physical path')
        require(torch.equal(path[0], self.last_endpoint), 'Accepted windows do not share the same endpoint/start state')
        require(not bool((self.last_start_pin & ~pin).any()), 'Pin release is outside this monotone-pin diagnostic')
        require(torch.equal(path[:, pin], path[0, pin].expand(len(path), -1, -1))
                and torch.equal(promoted[pin], path[0, pin]), 'Window-start pin moved')
        commit, steps = self.accepted+1, len(path)-1
        # Initial qualification belongs to the source anchor and includes W1.
        # It remains separately identifiable from accepted-endpoint admissions.
        if self.accepted == 0:
            self.first_arrival[start_geometric] = 0
            self.anchor[start_geometric] = path[0, start_geometric]
            self.last_end_arrived = start_geometric.detach().clone()
        seen = self.first_arrival >= 0
        new = ~seen & end_arrived
        self.first_start_arrived[(self.first_start_arrived == 0) & start_actual] = commit
        transitions = dict(
            geometric_departures=seen & start_geometric & ~end_arrived,
            geometric_reentries=seen & ~start_geometric & end_arrived,
            plan_departures=seen & self.last_end_arrived & ~start_geometric,
            plan_reentries=seen & ~self.last_end_arrived & start_geometric)
        for name, mask in transitions.items():
            self.events[name] += mask.to(torch.int32)
        row = dict(commit=commit, attempt=candidate['attempt'], radius_wu=radius, radius_sp=radius/self.spacing,
                   start_arrived=int(start_actual.sum()), geometric_start_arrived=int(start_geometric.sum()),
                   actual_vs_geometric_start_mismatch=int((start_actual != start_geometric).sum()),
                   end_arrived=int(end_arrived.sum()), newly_arrived=int(new.sum()),
                   initial_arrived=int((self.first_arrival == 0).sum()),
                   previously_arrived=int(seen.sum()), still_outside_after_prior_arrival=int((seen & ~end_arrived).sum()),
                   start_pinned=int(pin.sum()), new_start_pins=int((pin & ~self.last_start_pin).sum()),
                   transitions={name: int(mask.sum()) for name, mask in transitions.items()}, groups={})
        raw_final, pic_jump = path[-1]-path[-2], promoted-path[-1]
        saved_final = promoted-path[-2]
        for group, group_mask in (('start_free', seen & ~pin), ('start_pinned', seen & pin)):
            bank = self.banks[group]
            bank['windows'] += group_mask.to(torch.int32)
            bank['raw_final_square_wu2'] += raw_final.double().square().sum(1)*group_mask
            bank['pic_jump_square_wu2'] += pic_jump.double().square().sum(1)*group_mask
            row['groups'][group] = dict(particles=int(group_mask.sum()),
                raw_final=motion_stats(raw_final, group_mask, self.spacing),
                pic_jump=motion_stats(pic_jump, group_mask, self.spacing),
                saved_final=motion_stats(saved_final, group_mask, self.spacing))

        # The promoted endpoint replaces the final raw state in the saved path.
        # Previous delta survives accepted-window boundaries; rejects never enter.
        step_energy = {name: 0. for name in self.banks}
        step_path = {name: 0. for name in self.banks}
        reversed_window = {name: 0 for name in self.banks}
        eligible_window = {name: 0 for name in self.banks}
        previous = path[0]
        for phase in range(1, steps+1):
            x = promoted if phase == steps else path[phase]
            delta = x-previous
            length = delta.double().norm(dim=1)
            previous_length = self.previous_delta.double().norm(dim=1)
            eligible = seen & self.previous_observed & (length > 1e-4*self.spacing) & (previous_length > 1e-4*self.spacing)
            reverse = eligible & ((delta.double()*self.previous_delta.double()).sum(1) < 0)
            excursion = (x-self.anchor).double().norm(dim=1)
            for group, mask in (('start_free', seen & ~pin), ('start_pinned', seen & pin)):
                bank = self.banks[group]
                bank['path_wu'] += length*mask
                bank['step_square_wu2'] += length.square()*mask
                bank['steps'] += mask.to(torch.int32)
                bank['max_step_wu'] = torch.maximum(bank['max_step_wu'], torch.where(mask, length, 0.))
                bank['max_excursion_wu'] = torch.maximum(bank['max_excursion_wu'], torch.where(mask, excursion, 0.))
                bank['eligible_reversal_pairs'] += (eligible & mask).to(torch.int32)
                bank['reversed_pairs'] += (reverse & mask).to(torch.int32)
                step_energy[group] += float(length[mask].square().sum())
                step_path[group] += float(length[mask].sum())
                reversed_window[group] += int((reverse & mask).sum())
                eligible_window[group] += int((eligible & mask).sum())
            self.previous_delta = delta.detach().clone()
            self.previous_observed = seen.clone()
            previous = x
        for group, values in row['groups'].items():
            observations = values['particles']*steps
            values.update(saved_steps=observations, saved_path_sum_wu=step_path[group],
                          saved_step_rms_wu=math.sqrt(step_energy[group]/observations) if observations else None,
                          saved_step_rms_sp=math.sqrt(step_energy[group]/observations)/self.spacing if observations else None,
                          reversed_pairs=reversed_window[group], eligible_reversal_pairs=eligible_window[group],
                          reversal_fraction=reversed_window[group]/eligible_window[group] if eligible_window[group] else None)
        self.first_arrival[new] = commit
        self.first_endpoint_arrival[(self.first_endpoint_arrival == 0) & end_arrived] = commit
        self.anchor[new] = promoted[new]
        self.last_endpoint = promoted.detach().clone()
        self.last_end_arrived = end_arrived.detach().clone()
        self.last_start_pin = pin.detach().clone()
        self.accepted = commit
        return row

    def arrays(self):
        arrays = dict(first_arrival_commit=self.first_arrival, first_endpoint_arrival_commit=self.first_endpoint_arrival,
                      first_start_arrived_commit=self.first_start_arrived,
                      first_arrival_anchor=self.anchor, **self.events)
        arrays.update({group+'__'+name: value for group, bank in self.banks.items() for name, value in bank.items()})
        return arrays

    def summary(self):
        observed = sum(bank['steps'] for bank in self.banks.values()) > 0
        arrived = self.first_arrival >= 0
        def group_summary(bank, mask):
            total = int(bank['steps'][mask].sum())
            pairs = int(bank['eligible_reversal_pairs'][mask].sum())
            observed_ids = (bank['steps'] > 0) & mask
            rms = float((bank['step_square_wu2'][mask].sum()/total).sqrt()) if total else None
            return dict(observed_ids=int(observed_ids.sum()), saved_steps=total,
                path_sum_wu=float(bank['path_wu'][mask].sum()),
                step_rms_wu=rms, step_rms_sp=rms/self.spacing if rms is not None else None,
                max_step_wu=float(bank['max_step_wu'][observed_ids].max()) if total else None,
                max_anchor_excursion_wu=float(bank['max_excursion_wu'][observed_ids].max()) if total else None,
                eligible_reversal_pairs=pairs, reversed_pairs=int(bank['reversed_pairs'][mask].sum()),
                reversal_fraction=int(bank['reversed_pairs'][mask].sum())/pairs if pairs else None)
        groups = {group: group_summary(bank, arrived) for group, bank in self.banks.items()}
        def cohort_summary(mask):
            measured = mask & observed
            net = (self.last_endpoint-self.anchor).double().norm(dim=1)[measured]
            path = sum(bank['path_wu'] for bank in self.banks.values())[measured]
            moving = path > 1e-4*self.spacing
            return dict(particles=int(mask.sum()), observed_ids=int((mask & observed).sum()),
                        unobserved_ids=int((mask & ~observed).sum()),
                        start_free_steps=int(self.banks['start_free']['steps'][mask].sum()),
                        start_pinned_steps=int(self.banks['start_pinned']['steps'][mask].sum()),
                        net_anchor_observed_ids=int(measured.sum()),
                        net_anchor_rms_wu=float(net.square().mean().sqrt()) if net.numel() else None,
                        net_over_path_median=float(torch.quantile(net[moving]/path[moving], .5)) if bool(moving.any()) else None,
                        net_over_path_ids=int(moving.sum()),
                        groups={group: group_summary(bank, mask) for group, bank in self.banks.items()})
        return dict(accepted_commits=self.accepted, particles=self.n, first_arrival_count=int(arrived.sum()),
                    initial_arrived_count=int((self.first_arrival == 0).sum()),
                    first_endpoint_arrival_count=int((self.first_endpoint_arrival > 0).sum()),
                    never_arrived_count=int((~arrived).sum()), observed_after_first_arrival=int((arrived & observed).sum()),
                    unobserved_after_first_arrival=int((arrived & ~observed).sum()),
                    outside_at_last_endpoint_after_prior_arrival=int((arrived & ~self.last_end_arrived).sum()),
                    groups=groups, cohorts=dict(initial_arrived=cohort_summary(self.first_arrival == 0),
                        later_endpoint_arrived=cohort_summary(self.first_arrival > 0)),
                    event_totals={name: int(value.sum()) for name, value in self.events.items()})


class EvidenceWriter:
    """Uncompressed exact sidecars; bound both expected and actual output bytes."""
    def __init__(self, out, limits):
        self.out, self.limits, self.bytes = Path(out), limits, 0
        self.json_bytes, self.archive_overhead_bytes = 0, 0
        self.out.mkdir(parents=True, exist_ok=False)

    def save_bytes(self, filename, data):
        require(self.bytes+len(data) <= self.limits['hard_limit_bytes'], 'Evidence disk budget exceeded')
        path = self.out/filename
        with path.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        self.bytes += len(data)
        return dict(path=str(path), bytes=len(data), sha256=sha(data))

    def json(self, filename, value):
        data = json.dumps(value, indent=2, allow_nan=False).encode('utf-8')
        require(self.json_bytes+len(data) <= JSON_RESERVE, 'JSON evidence exceeds reserved budget')
        self.json_bytes += len(data)
        return self.save_bytes(filename, data)

    def arrays(self, filename, arrays, metadata):
        import numpy as host_np
        from physmorph.compute import to_host
        buffer = io.BytesIO()
        host_np.savez(buffer, __meta__=json.dumps(metadata, allow_nan=False),
                      **{name: to_host(value) for name, value in arrays.items()})
        payload = sum(value.numel()*value.element_size() for value in arrays.values())
        self.archive_overhead_bytes += buffer.tell()-payload
        require(self.archive_overhead_bytes <= ARCHIVE_OVERHEAD_RESERVE, 'Archive headers exceed reserved budget')
        return self.save_bytes(filename, buffer.getvalue())


class Capture:
    """Two-stage tentative ownership and matching outer-commit admission."""
    def __init__(self, initial, spacing, dt, writer=None, max_windows=24, require_cuda=True):
        require(not require_cuda or initial.is_cuda, 'Production observation requires CUDA')
        self.accumulator = Accumulator(initial, spacing, dt)
        self.writer, self.max_windows = writer, max_windows
        self.pending, self.rows, self.discarded = None, [], []

    def observe(self, tr, promoted, win_index):
        import warp as wp
        require(self.pending is None, 'Duplicate rollout observer for one attempt')
        path = torch.stack([wp.to_torch(tr.x[t]).detach() for t in range(tr.T+1)])
        require(path.device == self.accumulator.device and path.dtype == torch.float32
                and path.shape[1:] == self.accumulator.initial.shape, 'Unexpected rollout identity/device/dtype')
        self.pending = dict(attempt=int(win_index)+1, path=path,
                            promoted=own(promoted, path.device),
                            pin=own(wp.to_torch(tr.pin), path.device) > .5, T=int(tr.T))

    def complete(self, stats):
        if self.pending is None:
            return
        p = self.pending
        owned = stats.get('owned_endpoint')
        require(owned is not None and all(torch.equal(getattr(owned, key), value) for key, value in
                (('start', p['path'][0]), ('raw', p['path'][-1]), ('pin', p['pin']), ('promoted', p['promoted']))),
                'Owned endpoint and observed path/pins/promoted state disagree')
        require(stats.get('plan_img') is not None and stats.get('arrived_mask') is not None,
                'Actual frozen plan and start arrival mask required')
        radius = float(stats['pace_r'])
        require(math.isfinite(radius) and radius > 0, 'Invalid policy arrival radius')
        p.update(plan=own(stats['plan_img'], p['path'].device, torch.float32), radius=radius,
                 start_arrived=own(stats['arrived_mask'], p['path'].device, torch.bool))
        require(p['plan'].shape == p['promoted'].shape and p['start_arrived'].shape == p['pin'].shape,
                'Invalid plan or material-mask shape')
        require(all(bool(torch.isfinite(p[name]).all()) for name in ('path', 'promoted', 'plan')),
                'Nonfinite observer state')
        p['start_geometric'] = arrival_mask(p['path'][0], p['plan'], radius)
        p['end_arrived'] = arrival_mask(p['promoted'], p['plan'], radius)

    def wrap(self, original):
        def wrapped(*args, **kwargs):
            require('on_rollout' not in kwargs, 'Observer already installed')
            # A rejected or null attempt cannot leak into a subsequent call.
            self.pending = None
            try:
                result = original(*args, on_rollout=self.observe, **kwargs)
                self.complete(result[-1])
                return result
            except BaseException:
                self.pending = None
                raise
        return wrapped

    def commit(self, attempt, x, F, v, record):
        del F, v
        if not record.get('frame_end') or record.get('null_commit') or record.get('held'):
            self.discarded.append(dict(attempt=int(attempt)+1, reason='not an accepted physical commit'))
            self.pending = None
            return
        p = self.pending
        require(p is not None and p['attempt'] == int(attempt)+1 and 'end_arrived' in p,
                'Accepted commit lacks matching completed observer')
        require(self.accumulator.accepted < self.max_windows, 'Accepted window budget exceeded')
        require(torch.equal(p['promoted'], torch.as_tensor(x, device=p['promoted'].device, dtype=p['promoted'].dtype)),
                'Outer commit differs from owned promoted endpoint')
        row = self.accumulator.accept(p)
        row.update(frame_end=int(record['frame_end']), T=p['T'], dt=self.accumulator.dt,
                   native_spacing_wu=self.accumulator.spacing, pins_exact=True, endpoint_exact=True)
        if self.writer is not None:
            row['sidecar'] = self.writer.arrays(f'accepted_{row["commit"]:03d}.npz',
                dict(plan=p['plan'], raw=p['path'][-1], promoted=p['promoted'],
                     start_arrived=p['start_arrived'], end_arrived=p['end_arrived'], start_pin=p['pin']),
                {key: row[key] for key in ('commit', 'attempt', 'frame_end', 'T', 'dt', 'native_spacing_wu', 'radius_wu')})
        self.rows.append(row)
        self.pending = None


def run(quality, quality_sha256, out):
    """New unchanged-policy realization; original corrected-run JSON is evidence."""
    import numpy as host_np
    from scripts.probes import constitutive_quality as cq
    from physmorph.compute import cuda_execution, to_array
    from physmorph.pipeline import PipelineConfig, runner
    from physmorph.mpm.state import MPMParams
    from scripts.probes.render_influence import count_optimizer_attempts

    qbytes = quality.read_bytes()
    require(sha(qbytes) == quality_sha256, 'Quality JSON hash differs')
    q = json.loads(qbytes)
    require(q.get('requested_windows') == 24 and q['probe_sha256'] == QUALITY_PROBE_SHA
            == sha(Path(cq.__file__).read_bytes()), 'Exact corrected cap24 quality report required')
    loaded = cq.snapshot(ROOT)
    treatment = q['source_treatment']
    helpers = {name: sha((ROOT/name).read_bytes()) for name in cq.HELPERS}
    cq.validate_snapshots(treatment['old'], treatment['new'], [treatment['helper_sha256'], helpers, helpers])
    require(loaded['digest'] == cq.SOURCE_DIGESTS[1] and loaded['files'] == treatment['new']['files'],
            'Corrected numerical source differs')
    require(Path(sys.modules['physmorph'].__file__).resolve().parent.parent == ROOT, 'Imported source outside inspected snapshot')
    metadata_path = Path(q['metadata_path'])
    metadata_bytes = metadata_path.read_bytes()
    require(sha(metadata_bytes) == cq.METADATA_SHA, 'Original source metadata differs')
    metadata = json.loads(metadata_bytes)
    reference = Path(q['config']['target_reference'])
    require(sha(reference.read_bytes()) == cq.REFERENCE_SHA, 'Prepared target reference differs')
    config = cq.validate_recipe([q['config']]*2, [q['mpm']]*2, metadata,
                               PipelineConfig.__dataclass_fields__, reference, windows=24)
    require(config.get('motion_accounting') is True and not config.get('grad_dump'), 'Unexpected observer recipe')
    source_path = metadata_path.with_name(metadata_path.stem+'_'+cq.ARM+'.npz')
    for path in (metadata_path, reference, source_path):
        require(path.resolve().is_relative_to('/data'), 'Production inputs must resolve below /data')
    before = cq.file_identity(source_path)
    with host_np.load(source_path, allow_pickle=False) as archive:
        source, target = archive['src'], archive['tgt']
    require(cq.file_identity(source_path) == before, 'Source archive changed while reading')
    input_hashes = cq.validate_inputs(source, target)
    reference_run_bytes = Path(q['candidate']['prefix']+'.json').read_bytes()
    require(sha(reference_run_bytes) == q['candidate']['run_json_sha256'], 'Original corrected run JSON differs')
    reference_run = json.loads(reference_run_bytes)
    cq.validate_run_binding(reference_run, cq.SOURCE_DIGESTS[1])
    require(reference_run['config'] == config and reference_run['mpm'] == q['mpm'], 'Reference recipe mismatch')
    limits = budget()
    writer = EvidenceWriter(out, limits)
    protocol = dict(start_utc=datetime.now(timezone.utc).isoformat(), quality_path=str(quality),
        quality_sha256=quality_sha256, quality_probe_sha256=QUALITY_PROBE_SHA,
        source_path=str(source_path), source_identity=before, input_hashes=input_hashes,
        source_metadata_sha256=cq.METADATA_SHA, target_reference_sha256=cq.REFERENCE_SHA,
        comparison_run_json_sha256=sha(reference_run_bytes), numerical_source_sha256=loaded['digest'],
        numerical_files=loaded['files'], helper_sha256=helpers, probe_sha256=sha(Path(__file__).read_bytes()),
        config=config, mpm=q['mpm'], budget=limits, native_spacing_wu=q['native_spacing'],
        new_realization=True, scope='Read-only observer on unchanged corrected cap24 recipe; no bitwise replay assumption')
    protocol_evidence = writer.json('protocol.json', protocol)
    prm, cfg = MPMParams(**q['mpm']), PipelineConfig(**config)
    with cuda_execution('cuda'):
        initial = torch.as_tensor(to_array(source), device='cuda').clone()
        capture = Capture(initial, q['native_spacing'], prm.dt, writer=writer)
    writer.arrays('initial.npz', dict(x0=capture.accumulator.initial), dict(source_sha256=input_hashes['source_sha256']))
    started = time.perf_counter()
    with patch.object(runner, 'optimize_window', capture.wrap(runner.optimize_window)):
        result = runner.run_pipeline(source, target, prm, cfg, on_commit=capture.commit)
    require(not any(result['guards'].values()), 'Physical guard fired; observational result is invalid')
    # The public callback crosses an existing host-I/O boundary. It is not a
    # new CPU numerical path; commit checks and all counters run on the device.
    with cuda_execution('cuda'):
        arrays = capture.accumulator.arrays()
        final_pin = own(to_array(result['pinned']), 'cuda', torch.bool)
        final_pin_at = own(to_array(result['pinned_at']), 'cuda', torch.int32)
        arrays.update(final_pin=final_pin, final_pin_at_attempt=final_pin_at)
        final_arrays = writer.arrays('per_particle.npz', arrays, dict(
            spacing_wu=q['native_spacing'], dt=prm.dt, array_definitions='See result.json definitions'))
        summary = capture.accumulator.summary()
        summary['pins'] = dict(final_pinned=int(final_pin.sum()),
            observed_in_start_pinned_window=int((capture.accumulator.banks['start_pinned']['steps'] > 0).sum()),
            final_admitted_without_later_physical_window=int((final_pin & ~capture.accumulator.last_start_pin).sum()),
            definition='Final pinned_at uses one-based optimizer attempt, not accepted ordinal; on_commit precedes new pin admission. Final-admission rows have no later physical observation.')
    accepted = [row for row in result['history'] if row.get('frame_end') and not row.get('null_commit') and not row.get('held')]
    require(len(accepted) == capture.accumulator.accepted, 'Observer/history admission mismatch')
    reference_records = [row for row in reference_run['history'] if row.get('frame_end')
                         and not row.get('null_commit') and not row.get('held')]
    keys = ('Jmin_traj', 'arrived_end_frac', 'pinned_frac', 'lambda', 'outer_render', 'd_vol', 'd_sil')
    comparison = [dict(commit=i+1, observed_attempt=int(a['animation'])+1,
        reference_attempt=int(b['animation'])+1,
        values={key: dict(observed=a.get(key), reference=b.get(key),
                         difference=a[key]-b[key] if isinstance(a.get(key), (int, float))
                         and isinstance(b.get(key), (int, float)) else None) for key in keys})
        for i, (a, b) in enumerate(zip(accepted, reference_records))]
    output = dict(protocol=protocol_evidence, seconds=time.perf_counter()-started,
        end_utc=datetime.now(timezone.utc).isoformat(), summary=summary, windows=capture.rows,
        callback_discard_records=capture.discarded,
        admission_counts=dict(optimizer_attempts=count_optimizer_attempts(result['history']),
            accepted_physical_commits=len(accepted),
            nonaccepted_optimizer_attempts=count_optimizer_attempts(result['history'])-len(accepted),
            callback_discard_count=len(capture.discarded),
            scope='Nonaccepted attempts come from final optimizer history; discard list covers callbacks only. Some rejected/null attempts never call on_commit.'),
        per_particle=final_arrays, guards=result['guards'],
        history=result['history'], deliver_n=result['deliver_n'], n_held=result['n_held'], truncation=result['truncation'],
        actual_accepted_scope=True, comparison=dict(reference_accepted=len(reference_records),
            observed_accepted=len(accepted), matched_scalars=comparison,
            interpretation='Fresh realization; scalar differences are descriptive, no bitwise or causal identity claim'),
        torch_peak_bytes=torch.cuda.max_memory_allocated(), definitions=definitions())
    # Report payload bytes separately to avoid a self-referential file hash/size.
    output['sidecar_bytes_before_result'] = writer.bytes
    writer.json('result.json', output)
    print(json.dumps(dict(out=str(out), total_evidence_bytes=writer.bytes, accepted=len(accepted),
                          unobserved_after_first_arrival=summary['unobserved_after_first_arrival'])), flush=True)
    return output


def definitions():
    return dict(
        arrival='Initial source qualification is first_arrival_commit=0 and anchored at source; never qualified=-1; later first accepted promoted arrival uses its positive accepted commit. Radius=max(leash_r,loss-cell width), approximately one cell, not an exact optimum/rest tolerance. Plans/radii may change.',
        first_sample='Initial-arrived IDs are observed from W1. Later endpoint-sampled first arrival can lag a crossing by up to T*dt only if qualification persists until that endpoint; an entry and exit between endpoints can be missed entirely. Its anchor is that accepted endpoint and motion starts in the NEXT accepted window. first_endpoint_arrival_commit separately records first accepted-end qualification,0=none.',
        start_arrived='Actual optimizer start mask and first_start_arrived_commit are separate from endpoint first_arrival_commit. Geometric start predicate is recomputed from owned x0/plan/radius; threshold-rounding mismatches are reported.',
        membership='Every previously arrived ID remains observed even outside later plans; no escapee censoring. Never-arrived IDs do not enter post-arrival motion. Final admissions without later physical samples are unobserved.',
        events='Plan departures/reentries compare previous endpoint membership with current geometric-start membership at identical positions. Geometric departures/reentries compare current start/end under ONE plan. Events are endpoint sampled, not continuous crossing counts.',
        pins='start_free and start_pinned partition each observed window by its actual start pins. New pins admitted after this commit first affect the next window. No pin release allowed; start-pin positions must be exact across the full raw path and promoted endpoint.',
        saved_motion='T position differences: raw phases1..T-1, then promoted minus raw x[T-1]. Path=sum norms; step_square=sum squared norms; max_step=max norm; max_excursion=max distance from first-arrival anchor. No momentum-velocity substitution.',
        reversals='Both adjacent saved displacements must exceed1e-4 native source spacing, with negative dot product. Last displacement persists across accepted windows; rejects/null/held copies do not enter. Pair assigned by destination window start pin.',
        final_split='raw_final=x[T]-x[T-1]; pic_jump=promoted-x[T]; saved_final=their sum. Separate per-window RMS/max and per-ID squared sums; windows is their denominator.',
        counters='Float64 sums/maxima and int32 counts, raw world units. Per-ID RMS=sqrt(step_square_wu2/steps), or null at0steps; /source spacing gives sp. Total observed physical duration=steps*dt; pins have a separate denominator.',
        net='Endpoint-minus-first-arrival-anchor norm, over IDs with at least one later observed physical step; unobserved arrivals excluded. Net/path pools each ID\'s free and pinned periods and is not a pin-conditioned movement measure.',
        sidecars='Full frozen plan, raw x[T], promoted and start/end-arrival/start-pin masks per accepted window; initial x0 once. Previous promoted is next x0. Masks independently recomputable except actual start mask whose original displacement arithmetic may differ at threshold rounding.',
        evidence_limit='Raw trajectories exist only as temporary owned GPU tensors. Final per-ID motion sufficient statistics cannot reconstruct or independently recompute every physical step; plan/endpoint arrival and PIC jumps can be independently checked from sidecars.',
        stopping='All actual accepted commits are observed, even if delivered output later trims to an earlier best state. Copied suffixes are excluded; converged metadata does not mean physical rest.',
        limit='No new objective, controls, damping, pins, thresholds, loss calls or renderer. This is an observer on a fresh realization, not a no-hole or no-oscillation pass.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--quality-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.quality, args.out):
        require(path.resolve().is_relative_to('/data'), 'Production paths must resolve below /data')
    for name in ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH'):
        value = os.environ.get(name)
        require(value and Path(value).resolve().is_relative_to('/data'), 'Explicit /data cache required: '+name)
    require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
    sys.path.insert(0, str(ROOT))
    run(args.quality.resolve(), args.quality_sha256, args.out.resolve())


if __name__ == '__main__':
    main()
