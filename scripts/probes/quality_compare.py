"""P292/P293/P294 matched raw-state quality audit; run on hyde06 with CUDA only.

The baseline/candidate must share inputs, MPM parameters and exact simulation code.
Default: only body_rprop may change. Explicit stress_taper and commit_pic_off
interventions permit their one flag plus cap 60 -> 8 and compare a common prefix.
The shared-PIC objective prefix permits only its objective flag, with cap8 in both arms.
The geometric-rest prefix additionally fixes T=20, dt=1/240 and a positive kinetic weight.
The geometric-variance prefix compares the whole adaptive policy at cap8 with
positive temporal variance weight and matched read-only motion accounting.
Its full mode uses the same policy guard at cap60 and audits each delivered endpoint.
P320 layer_projection_off_full requires cap300 and separates the complete archive
clock, each delivery-retained endpoint and the common retained accepted interval.
Numerical geometry, cohorts and motion use CUDA;
archive/hash/JSON I/O use the host. No renderer or optimization-loss operator is read.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

import numpy as host_np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import array_api as np, cuda_execution, KDTree, to_array, to_host
from physmorph.metrics import sil_iou, target_extent
from scripts.probes.morph_raw_qa import audit
from scripts.probes.render_influence import load_run, channel_summary, count_optimizer_attempts

wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
FULL_INTERVENTIONS = ('body_rprop', 'commit_pic_off_full', 'render_arrival_handoff',
                      'geometric_rest_full', 'geometric_variance_full', 'layer_projection_off_full')
PREFIX_INTERVENTIONS = ('stress_taper', 'commit_pic_off', 'commit_pic_objective_prefix',
                        'geometric_rest_prefix', 'geometric_variance_prefix', 'shared_pic_off_prefix',
                        'pin_admission_off_prefix')
P303_INTERVENTIONS = ('shared_pic_off_prefix', 'pin_admission_off_prefix')
GEOMETRIC_INTERVENTIONS = ('geometric_rest_prefix', 'geometric_rest_full')
VARIANCE_PREFIX = 'geometric_variance_prefix'
VARIANCE_INTERVENTIONS = (VARIANCE_PREFIX, 'geometric_variance_full')
LAYER_FULL = 'layer_projection_off_full'
RAW_ENDPOINT_INTERVENTIONS = (*P303_INTERVENTIONS, LAYER_FULL)


def stats(values, include_rms=False):
    if not values.size:
        return None
    result = dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                  max=float(np.max(values)))
    if include_rms:
        result['rms'] = float(np.sqrt(np.mean(np.asarray(values, np.float64)**2)))
    return result


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact_stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns,
                inode=value.st_ino, device=value.st_dev)


def bind_layer_artifact(path):
    before = artifact_stat(path)
    digest = file_digest(path)
    if artifact_stat(path) != before:
        raise ValueError('P320 artifact changed while hashing: '+str(path))
    return dict(sha256=digest, stat=before)


def verify_layer_artifacts(runs):
    for run in runs:
        for suffix, binding in run['artifact_bindings'].items():
            if bind_layer_artifact(run['prefix']+suffix) != binding:
                raise ValueError('P320 artifact changed during analysis: '+run['prefix']+suffix)


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
    elif intervention == 'render_arrival_handoff':
        valid = (set(changes) == {'render_paced_arrived'}
                 and not ca.get('render_paced_arrived', False) and cb.get('render_paced_arrived') is True
                 and all(c.get('stop_after_windows') == 60 and c.get('commit_pic') is False
                         and c.get('outer_render_committed') is True and c.get('render_paced') is True
                         and c.get('lambda_auto', 0.) > 0 for c in (ca, cb)))
    elif intervention == LAYER_FULL:
        valid = (set(changes) == {'layer_ctrl', 'layer_relax'}
                 and all(ca.get(k) is True and cb.get(k) is False for k in changes)
                 and all(c.get('stop_after_windows') == 300 and c.get('animations') == 300
                         and c.get('T') == 20 and c.get('iters') == 8 and c.get('loss_res') == 36
                         and c.get('commit_pic') is False and c.get('commit_pic_objective') is False
                         and c.get('shift_sub') is False and c.get('outer_render_committed') is True
                         and c.get('body_ctrl') is True and c.get('motion_accounting') is True
                         and c.get('lambda_auto', 0.) > 0 and host_np.isfinite(c['lambda_auto'])
                         and (c.get('render_until', 0) <= 0 or c.get('render_until', 0) >= 300)
                         and not any(c.get(k, False) for k in ('surface_gs_loss', 'render_F_geom',
                             'geometric_rest', 'geometric_variance', 'reattach', 'settle_commit',
                             'assim_consensus', 'settle_pin_yield', 'settle_pin_follow', 'settle_pin_kkt'))
                         and c.get('lg_sweeps', 0) == 0 and c.get('w_grow', 0) == 0
                         and c.get('local_dress_iters', 0) == 0
                         for c in (ca, cb)))
    elif intervention == 'commit_pic_objective_prefix':
        valid = (set(changes) == {'commit_pic_objective'}
                 and not ca.get('commit_pic_objective', False) and cb.get('commit_pic_objective') is True
                 and all(c.get('stop_after_windows') == 8 and c.get('commit_pic') is True
                         and c.get('shift_sub') is False and c.get('outer_render_committed') is True
                         and c.get('lambda_auto', 0.) > 0
                         and (c.get('render_until', 0) <= 0 or c.get('render_until', 0) >= 8)
                         for c in (ca, cb)))
    elif intervention == 'shared_pic_off_prefix':
        valid = (set(changes) == {'commit_pic', 'commit_pic_objective'}
                 and all(ca.get(k) is True and cb.get(k) is False
                         for k in ('commit_pic', 'commit_pic_objective'))
                 and all(c.get('stop_after_windows') == 24 and c.get('T') == 20
                         and c.get('iters') == 8 and c.get('loss_res') == 36
                         and c.get('shift_sub') is False and c.get('outer_render_committed') is True
                         and c.get('motion_accounting') is True and c.get('lambda_auto', 0.) > 0
                         and host_np.isfinite(c['lambda_auto'])
                         and (c.get('render_until', 0) <= 0 or c.get('render_until', 0) >= 24)
                         and not any(c.get(k, False) for k in ('surface_gs_loss', 'geometric_rest',
                             'geometric_variance', 'reattach', 'settle_commit', 'assim_consensus',
                             'settle_pin_yield', 'settle_pin_follow', 'settle_pin_kkt'))
                         and c.get('lg_sweeps', 0) == 0 and c.get('w_grow', 0) == 0
                         for c in (ca, cb)))
    elif intervention == 'pin_admission_off_prefix':
        valid = (set(changes) == {'settle_pin'} and ca.get('settle_pin') is True
                 and cb.get('settle_pin') is False
                 and all(c.get('stop_after_windows') == 24 and c.get('T') == 20
                         and c.get('iters') == 8 and c.get('loss_res') == 36
                         and c.get('commit_pic') is False and c.get('commit_pic_objective') is False
                         and c.get('shift_sub') is False and c.get('outer_render_committed') is True
                         and c.get('motion_accounting') is True and c.get('lambda_auto', 0.) > 0
                         and host_np.isfinite(c['lambda_auto'])
                         and (c.get('render_until', 0) <= 0 or c.get('render_until', 0) >= 24)
                         and not any(c.get(k, False) for k in ('surface_gs_loss', 'geometric_rest',
                             'geometric_variance', 'reattach', 'settle_commit', 'assim_consensus',
                             'settle_pin_yield', 'settle_pin_follow', 'settle_pin_kkt', 'freeze_arrived', 'settle_eta'))
                         and c.get('lg_sweeps', 0) == 0 and c.get('w_grow', 0) == 0
                         for c in (ca, cb)))
    elif intervention in VARIANCE_INTERVENTIONS:
        cap = 8 if intervention == VARIANCE_PREFIX else 60
        def positive_finite(value):
            return (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and host_np.isfinite(value) and value > 0)
        valid = (set(changes) == {'geometric_variance'}
                 and ca.get('geometric_variance', False) is False and cb.get('geometric_variance') is True
                 and all(c.get('stop_after_windows') == cap and c.get('T') == 20
                         and c.get('commit_pic') is True and c.get('commit_pic_objective') is True
                         and c.get('shift_sub') is False and c.get('outer_render_committed') is True
                         and c.get('geometric_rest', False) is False and c.get('motion_accounting') is True
                         and positive_finite(c.get('w_kin_var')) and positive_finite(c.get('lambda_auto'))
                         and c.get('phys_loss') in ('auto', 'ot_pace', 'ot_shape')
                         and (c.get('render_until', 0) <= 0 or c.get('render_until', 0) >= cap)
                         for c in (ca, cb)))
    elif intervention in GEOMETRIC_INTERVENTIONS:
        cap = 8 if intervention == 'geometric_rest_prefix' else 60
        valid = (set(changes) == {'geometric_rest'}
                 and ca.get('geometric_rest', False) is False and cb.get('geometric_rest') is True
                 and all(c.get('stop_after_windows') == cap and c.get('T') == 20
                         and c.get('commit_pic') is True and c.get('commit_pic_objective') is True
                         and c.get('shift_sub') is False and c.get('outer_render_committed') is True
                         and c.get('phys_loss') in ('auto', 'ot_pace', 'ot_shape')
                         and host_np.isfinite(c.get('w_kin', 0.)) and c.get('w_kin', 0.) > 0
                         and c.get('lambda_auto', 0.) > 0
                         and (c.get('render_until', 0) <= 0 or c.get('render_until', 0) >= cap)
                         for c in (ca, cb)))
    else:
        raise ValueError(f'Unknown intervention: {intervention}')
    if not valid:
        raise ValueError(f'Unexpected {intervention} configuration changes: {changes}')
    return changes


def checked_mpm_parameters(baseline, candidate, intervention):
    if baseline != candidate:
        raise ValueError('MPM discretisation mismatch')
    if (intervention in GEOMETRIC_INTERVENTIONS or intervention in VARIANCE_INTERVENTIONS
            or intervention in RAW_ENDPOINT_INTERVENTIONS) and baseline.get('dt') != 1/240:
        raise ValueError(f'{intervention} requires dt=1/240')


def checked_layer_archive(prefix):
    """Check the bounded-I/O contract before the legacy loader can decompress frames."""
    path = Path(str(prefix)+'_render_full_dt_iso_nn.npz')
    with zipfile.ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if entry.filename == 'frames.npy']
        if len(entries) != 1 or entries[0].compress_type != zipfile.ZIP_STORED:
            raise ValueError('P320 requires one uncompressed frames.npy member for bounded mmap I/O')
    with host_np.load(path, allow_pickle=False) as data:
        value = data['deliver_n']
        if value.shape != () or value.dtype.kind not in 'iu':
            raise ValueError('P320 archive delivery count must be an integer scalar')
        return int(value)


def checked_layer_delivery(run, declared_delivery):
    """Reconstruct the complete clock before selecting delivery-retained commits.

    Only metadata is interpreted here. Frame equality, geometry and pin membership
    are validated on CUDA in checked_runs; no full trajectory/F array is loaded.
    """
    arm, frames = run['arm'], run['frames']
    if run['source'].shape != (300000, 3) or run['target'].shape != (300000, 3):
        raise ValueError('P320 requires the original N300k source and target layouts')
    if frames.ndim != 3 or frames.shape[1:] != run['source'].shape or frames.dtype != host_np.float32:
        raise ValueError('P320 raw frame layout differs from material IDs')
    if (not 1 <= declared_delivery <= len(frames) or arm.get('deliver_n') != declared_delivery
            or run['delivered'] != declared_delivery):
        raise ValueError('P320 delivery count mismatch; silent clamping is forbidden')
    cursor, next_animation, held = 0, 0, 0
    actual, holds, events, attempt_rows = [], [], [], []
    for row in arm['history']:
        if row.get('animation') != next_animation:
            raise ValueError('P320 history lacks a contiguous attempt/held clock')
        if 'c2f_render_res' in row:
            if row.get('frame_end') or row.get('held') or row.get('null_commit'):
                raise ValueError('P320 C2F event cannot contain a physical frame span')
            events.append(dict(row))
            continue
        next_animation += 1
        end = row.get('frame_end')
        null, rejected = bool(row.get('null_commit')), bool(row.get('outer_rejected'))
        if row.get('held'):
            if end or null or rejected:
                raise ValueError('P320 held suffix cannot be an accepted/rejected/null solve')
            held += 1
            holds.append((cursor, cursor+1))
            cursor += 1
            continue
        if held:
            raise ValueError('P320 optimizer attempt appears after the held suffix')
        committed = bool(end and not null and not rejected and row.get('outer_accepted', 1))
        if end and not committed:
            raise ValueError('P320 rejected/null row claims an accepted frame endpoint')
        if committed:
            if end != cursor+20+1 or row.get('accepted', 0) <= 0:
                raise ValueError('P320 accepted frame span differs from T20 or lacks an inner acceptance')
            cursor = int(end)-1
            actual.append(row)
        elif null and not rejected:
            holds.append((cursor, cursor+1))
            cursor += 1
        elif not (rejected or row.get('grad_converged')):
            raise ValueError('P320 noncommitted solve has no recorded disposition')
        attempt_rows.append(dict(attempt=int(row['animation'])+1, committed=committed,
                                 outer_rejected=rejected, null_commit=null,
                                 grad_converged=bool(row.get('grad_converged'))))
    if cursor != len(frames)-1 or held != arm.get('n_held') or next_animation > 300:
        raise ValueError('P320 archive clock, held count or configured horizon differs')
    retained = [row for row in actual if int(row['frame_end']) <= declared_delivery]
    if not retained:
        raise ValueError('P320 has no delivery-retained accepted commit')
    truncation = arm.get('truncation')
    if truncation is None:
        if declared_delivery != len(frames):
            raise ValueError('P320 delivery truncation is not recorded')
    elif (truncation.get('frames_kept') != declared_delivery
          or truncation.get('frames_dropped') != len(frames)-declared_delivery
          or truncation.get('best_animation') != int(retained[-1]['animation'])+1
          or int(retained[-1]['frame_end']) != declared_delivery):
        raise ValueError('P320 best-state truncation does not end at its recorded accepted commit')
    if run['records'] != retained:
        raise ValueError('P320 loaded accepted rows differ from the validated delivery clock')
    run['actual_records'], run['hold_pairs'] = actual, holds
    run['delivery_scope'] = dict(
        actual_archive_frames=len(frames), deliver_n=declared_delivery,
        actual_accepted_commits=len(actual), delivery_retained_accepted_commits=len(retained),
        actual_last_accepted_attempt=int(actual[-1]['animation'])+1,
        actual_last_accepted_frame=int(actual[-1]['frame_end'])-1,
        delivered_last_accepted_attempt=int(retained[-1]['animation'])+1,
        delivered_last_accepted_frame=int(retained[-1]['frame_end'])-1,
        delivery_rows_after_last_accepted=declared_delivery-int(retained[-1]['frame_end']),
        held_suffix_rows=held, null_hold_rows=len(holds)-held,
        truncation=truncation, c2f_events=events, attempts=attempt_rows,
        interpretation='Common clock counts only delivery-retained accepted rollouts; '
                       'null/rejected/C2F/held rows are not motion steps; ordinary stopping is not rest')


def checked_arrival_evidence(run):
    """Serialized auto is allowed only with a recorded OT resolution and accepted arrival evidence."""
    mode = run['arm']['config'].get('phys_loss')
    evidence = dict(serialized_mode=mode, resolved_mode=mode, log_sha256=None, resolver_line=None)
    if mode == 'auto':
        path = Path(str(run['prefix'])+'.log')
        data = path.read_bytes()
        lines = [line for line in data.decode('utf-8').splitlines() if line.startswith('[v2] phys_loss auto:')]
        if len(lines) != 1:
            raise ValueError('Serialized auto requires exactly one recorded loss-resolution event')
        match = re.fullmatch(r'\[v2\] phys_loss auto: [0-9.]+% of the source particles sit in target-empty '
                             r'cells -> (ot_pace|ot_shape)(?: \+ cell-wise hand-off)?', lines[0])
        if match is None:
            raise ValueError('Serialized auto did not resolve to a full-plan OT arrival mode')
        evidence.update(resolved_mode=match.group(1), log_sha256=hashlib.sha256(data).hexdigest(),
                        resolver_line=lines[0])
    elif mode not in ('ot_pace', 'ot_shape'):
        raise ValueError('Full-plan OT arrival mode required')
    records = run['records']
    if not records or any(row.get('pin_arrival_evidence') != 'accepted_full_plan'
                          or not isinstance(row.get('arrived_end_frac'), (int, float))
                          or not host_np.isfinite(row['arrived_end_frac'])
                          or not 0 <= row['arrived_end_frac'] <= 1 for row in records):
        raise ValueError('Every accepted commit must record full-plan arrival evidence')
    evidence['accepted_commits_checked'] = len(records)
    return evidence


def checked_arrival_modes(runs):
    evidence = [checked_arrival_evidence(run) for run in runs]
    if len({item['resolved_mode'] for item in evidence}) != 1:
        raise ValueError('Both arms must resolve to the same full-plan OT arrival mode')
    return evidence


def checked_runs(baseline, candidate, intervention):
    bindings = ([{suffix: bind_layer_artifact(str(prefix)+suffix)
                  for suffix in ('.json', '.npz', '_render_full_dt_iso_nn.npz')}
                 for prefix in (baseline, candidate)] if intervention == LAYER_FULL else None)
    deliveries = ([checked_layer_archive(prefix) for prefix in (baseline, candidate)]
                  if intervention == LAYER_FULL else None)
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
    checked_mpm_parameters(a['meta']['provenance']['mpm'], b['meta']['provenance']['mpm'], intervention)
    for index, (run, config) in enumerate(zip(runs, configs)):
        if intervention == LAYER_FULL:
            checked_layer_delivery(run, deliveries[index])
            run['artifact_bindings'] = bindings[index]
            run['artifact_hashes'] = {suffix: binding['sha256'] for suffix, binding in bindings[index].items()}
        if intervention in RAW_ENDPOINT_INTERVENTIONS:
            receipt_path = Path(run['prefix']+'.pin_schema.json')
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if file_digest(run['prefix']+'_render_full_dt_iso_nn.npz') != receipt['normalized_archive_sha256']:
                    raise ValueError('Normalized archive no longer matches its schema receipt')
                if file_digest(run['prefix']+'.json') != receipt['original_metadata_sha256']:
                    raise ValueError('Schema normalization changed simulation metadata')
                run['archive_schema'] = dict(receipt_sha256=file_digest(receipt_path),
                                             receipt=receipt)
            if (run['meta']['guards'] != run['arm']['guards'] or any(run['arm']['guards'].values())
                    or run['meta']['config'] != config or run['meta']['history'] != run['arm']['history']):
                raise ValueError('Invalid raw-endpoint run metadata or a physical guard fired')
            if not config['commit_pic']:
                for row in run.get('actual_records', run['records']):
                    components = row['motion_accounting']['cohorts']['all']['window_component_rms_wu']
                    if any(components[name] != 0. for name in ('commit_other', 'commit_pic', 'commit_shift')):
                        raise ValueError('Raw-endpoint candidate has an external position correction')
        if config.get('compute_backend') != 'cuda':
            raise ValueError('Both simulations must use the strict CUDA backend')
        if int(config.get('archive_stride', 1)) != 1:
            raise ValueError('Exact raw-step audit requires archive_stride=1')
        if any(config.get(k) for k in ('settle_pin_yield', 'settle_pin_follow', 'settle_pin_kkt')):
            raise ValueError('This audit requires monotone pin admission')
        if not run['records']:
            raise ValueError('At least one accepted commit is required')
        if len(run['source']) != len(run['frames'][0]):
            raise ValueError('Material particle IDs must be preserved')
        run['physical_indices'] = accepted_raw_indices(run['records'], config['T'])
        run['interior_hold_frames'] = run['physical_indices'][-1]+1-len(run['physical_indices'])
        if intervention in RAW_ENDPOINT_INTERVENTIONS:
            prm = run['meta']['provenance']['mpm']
            with cuda_execution('cuda'):
                lo = to_array(prm['grid_min']) + 2*prm['dx']
                hi = to_array(prm['grid_min']) + prm['dx']*to_array([prm['nx'], prm['ny'], prm['nz']]) - 2*prm['dx']
                validation_indices = (accepted_raw_indices(run['actual_records'], config['T'])
                                      if intervention == LAYER_FULL else run['physical_indices'])
                for index in validation_indices:
                    state = to_array(run['frames'][index])
                    if not bool((np.isfinite(state) & (state >= lo) & (state <= hi)).all()):
                        raise ValueError(f'Invalid raw position at frame {index}')
                with host_np.load(run['prefix']+'.npz') as compact:
                    if not bool(np.array_equal(to_array(compact['final']), to_array(run['frames'][-1]))):
                        raise ValueError('Compact/raw final state differs')
                    f = to_array(compact['F'])
                    if not bool(np.isfinite(f).all()) or not bool((np.linalg.det(f) > 0).all()):
                        raise ValueError('Invalid compact final deformation state')
                if not bool(np.array_equal(to_array(run['source']), to_array(run['frames'][0]))):
                    raise ValueError('Raw archive does not begin at the exact source')
                if intervention == LAYER_FULL:
                    pins, when = to_array(run['pins']), to_array(run['pin_at'])
                    if (pins.shape != (len(run['source']),) or when.shape != pins.shape
                            or not bool(np.array_equal(pins, when >= 0))
                            or not bool(np.isin(when[pins], np.asarray([
                                int(row['animation'])+1 for row in run['actual_records']])).all())):
                        raise ValueError('P320 pin admission does not belong to actual accepted commits')
                    for before, after in run['hold_pairs']:
                        if not bool(np.array_equal(to_array(run['frames'][before]), to_array(run['frames'][after]))):
                            raise ValueError('P320 null/held archive row changed positions')
    if (intervention in GEOMETRIC_INTERVENTIONS or intervention in VARIANCE_INTERVENTIONS
            or intervention in ('shared_pic_off_prefix', LAYER_FULL)):
        for run, evidence in zip(runs, checked_arrival_modes(runs)):
            run['arrival_mode_evidence'] = evidence
    return runs, changes


def scoped_runs(runs, intervention):
    if intervention in FULL_INTERVENTIONS:
        scope = dict(kind='full runs', endpoint_claim=(
            'last accepted states retained in each arm delivery; later accepted states '
            'can be omitted by best-state truncation'))
        if intervention == LAYER_FULL:
            scope.update(common_clock='delivery-retained accepted commits only',
                         configured_windows=300, common_accepted_commits=min(len(r['records']) for r in runs),
                         delivery_clock_validated=all('delivery_scope' in r for r in runs),
                         arm_delivery_scopes=[r.get('delivery_scope') for r in runs],
                         endpoints='own delivered endpoint and actual last accepted endpoint reported separately')
        return runs, scope
    requested = 24 if intervention in P303_INTERVENTIONS else 8
    common = min(requested, *(len(run['records']) for run in runs))
    scoped = []
    for original in runs:
        run = dict(original)
        run['records'] = original['records'][:common]
        run['delivered'] = int(run['records'][-1]['frame_end'])
        run['physical_indices'] = accepted_raw_indices(run['records'], run['arm']['config']['T'])
        run['interior_hold_frames'] = run['physical_indices'][-1]+1-len(run['physical_indices'])
        scoped.append(run)
    return scoped, dict(kind='common accepted prefix', requested_commits=requested, analyzed_commits=common,
                        original_commits=[len(run['records']) for run in runs],
                        original_delivered_frames=[run['delivered'] for run in runs],
                        endpoint_claim='prefix endpoint only; no final quality or convergence inference',
                        metadata='Original archives, run metadata and configurations remain unchanged')


def render_reference_history(run, intervention):
    """Host metadata; retain rejected/null attempts and distinguish delivery trims."""
    last_attempt = int(run['records'][-1]['animation'])+1
    result = []
    for record in run['arm']['history']:
        if 'animation' not in record or 'c2f_render_res' in record or record.get('held'):
            continue
        attempt = int(record['animation'])+1
        if intervention not in FULL_INTERVENTIONS and attempt > last_attempt:
            continue
        accepted = bool(record.get('frame_end') and not record.get('null_commit'))
        if intervention == LAYER_FULL:
            accepted = accepted and not record.get('outer_rejected') and bool(record.get('outer_accepted', 1))
        result.append(dict(attempt=attempt, accepted=accepted,
                           delivered=bool(accepted and int(record['frame_end']) <= run['delivered']),
                           null_commit=bool(record.get('null_commit')),
                           **{key: record.get(key) for key in (
                               'render_target_kind', 'render_arrival_count', 'render_arrival_trigger',
                               'render_arrival_trigger_attempt', 'render_arrival_trigger_commit')}))
    return result


def fixed_material_density(x, tree, ids, radius):
    """Same source-selected IDs, including those outside the moving top region."""
    if not len(ids):
        return dict(particles=0, density=None, under_half=None, y_gt_2_3_frac=None)
    counts = tree.query_ball_point(x[ids], radius, return_length=True)-1
    return dict(particles=len(ids), density=float(counts.mean()/8),
                under_half=float((counts < 4).mean()),
                y_gt_2_3_frac=float((x[ids, 1] > 2.3).mean()))


def geometry_curve(run, target, target_tree, extent, target_spacing, radius, tip,
                   fixed_source_ids=None, *, records=None):
    top_target = target[:, 1] > 2.3
    curve = []
    for ordinal, record in enumerate(run['records'] if records is None else records, 1):
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
                    'endpoint_contract', 'commit_from_accepted', 'pic_null_share', 'geometric_rest',
                    'body_step_node_mean', 'body_step_transit_nodes_frac',
                    'body_step_transit_min', 'body_step_arrived_median', 'body_update_modes_rms',
                    'body_rms_wu', 'body_terminal_rms_wu', 'body_step_scale',
                    'render_target_kind', 'render_arrival_count', 'render_arrival_trigger',
                    'render_arrival_trigger_attempt', 'render_arrival_trigger_commit'):
            row[key] = record.get(key)
        if fixed_source_ids is not None:
            row['source_upper_surface_density'] = fixed_material_density(x, tree, fixed_source_ids, radius)
            row['source_upper_surface_density']['pinned_frac'] = (
                float(admitted_mask(run, int(record['animation'])+1)[fixed_source_ids].mean())
                if len(fixed_source_ids) else None)
            row['geometric_variance'] = record.get('geometric_variance')
            row['motion_accounting'] = record.get('motion_accounting')
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


def handoff_intervals(trigger_commit, common):
    """The trigger commit used paced guidance; only subsequent solves are fixed."""
    if not 1 <= trigger_commit <= common:
        return {}
    intervals = {}
    if trigger_commit > 1:
        intervals['pre_trigger'] = [max(1, trigger_commit-10), trigger_commit]
    if common > trigger_commit:
        intervals['post_trigger'] = [trigger_commit, min(common, trigger_commit+10)]
    return intervals


def cohort_motion(run, ids, common, spacing, first_commit=None, include_rms=False):
    if not len(ids) or common < 3:
        return None
    first = max(1, common-10) if first_commit is None else int(first_commit)
    if not 1 <= first < common <= len(run['curve']):
        raise ValueError('Invalid common material motion interval')
    rows = run['curve'][first-1:common]
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
    summarize_values = lambda values: stats(values, include_rms=include_rms)
    return dict(commit_range=[rows[0]['commit'], rows[-1]['commit']],
                frame_range=[rows[0]['frame'], rows[-1]['frame']],
                raw_physical_states=len(raw_indices),
                excluded_interior_hold_frames=rows[-1]['frame']-rows[0]['frame']+1-len(raw_indices),
                pinned_frac_start=float(admitted_mask(run, rows[0]['attempt'])[ids].mean()),
                pinned_frac_end=float(admitted_mask(run, rows[-1]['attempt'])[ids].mean()),
                step_sp=summarize_values(lengths.ravel()/spacing),
                normal_sp_per_commit=summarize_values(np.abs(signed[:, valid]).ravel()/spacing),
                tangent_sp_per_commit=summarize_values(tangent[:, valid].ravel()/spacing),
                reversal_fraction=float(reversal[active].mean()) if active.any() else None,
                reversed_pairs=int(reversal[active].sum()), moving_pairs=int(active.sum()),
                moving_material_points=int(moving.sum()),
                net_over_path=summarize_values(efficiency),
                normal_basis='arm-specific frozen 33NN centroid-offset normal at common interval endpoint',
                normal_valid_points=int(valid.sum()),
                raw_step_sp=summarize_values(raw_lengths.ravel()/spacing),
                normal_sp_per_raw_step=summarize_values(np.abs(raw_signed[:, valid]).ravel()/spacing),
                tangent_sp_per_raw_step=summarize_values(raw_tangent[:, valid].ravel()/spacing),
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
            run['curve'] = geometry_curve(run, target, target_tree, target_extent(target), target_spacing, radius, tip,
                                           source_ids if intervention in VARIANCE_INTERVENTIONS
                                           or intervention in RAW_ENDPOINT_INTERVENTIONS else None)
            if intervention == LAYER_FULL:
                if run['actual_records'][-1] == run['records'][-1]:
                    run['actual_endpoint'] = dict(run['curve'][-1])
                else:
                    actual_view = dict(run, records=run['actual_records'])
                    run['actual_endpoint'] = geometry_curve(actual_view, target, target_tree,
                        target_extent(target), target_spacing, radius, tip, source_ids,
                        records=run['actual_records'][-1:])[0]
                    run['actual_endpoint']['commit'] = len(run['actual_records'])
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
            cohorts[name] = dict(**metadata, arms={arm: cohort_motion(run, ids, common, spacing,
                                                                      include_rms=(intervention in VARIANCE_INTERVENTIONS
                                                                                   or intervention in RAW_ENDPOINT_INTERVENTIONS))
                                                  for arm, run in zip(names, runs)})
        handoff_bands = None
        if intervention == 'render_arrival_handoff':
            trigger = next((row for row in runs[1]['curve'] if row.get('render_arrival_trigger')), None)
            if trigger:
                intervals = handoff_intervals(trigger['commit'], common)
                handoff_bands = dict(trigger_commit=trigger['commit'], trigger_attempt=trigger['attempt'],
                                     cohort=free_meta,
                                     definition='same outcome-selected common endpoint-free IDs; trigger endpoint finishes paced solve, subsequent intervals begin fixed solves; each band has its own arm-specific endpoint normal basis',
                                     intervals={label: dict(commit_range=bounds, arms={
                                         arm: cohort_motion(run, free_ids, bounds[1], spacing, first_commit=bounds[0])
                                         for arm, run in zip(names, runs)}) for label, bounds in intervals.items()})
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
                  equal_accepted_commits=pairs, first_chamfer_threshold_crossings=progress, cohorts=cohorts,
                  handoff_common_material_bands=handoff_bands)
    if intervention in VARIANCE_INTERVENTIONS or intervention == LAYER_FULL:
        result['matched_free_motion_status'] = (
            'available_descriptive' if free_meta['sampled_count'] and common >= 3 else
            'inconclusive_empty_common_free_cohort' if not free_meta['sampled_count'] else
            'inconclusive_fewer_than_three_common_commits')
    if intervention in VARIANCE_INTERVENTIONS:
        result['definitions'].update(
            comparison='whole adaptive-lambda policy; no fixed-lambda causal inference or primitive-gate promotion',
            source_upper_density='same pre-treatment source_upper_surface sampled IDs at each accepted endpoint; target median r8, self excluded, count/8; includes IDs below y=2.3 and pinned IDs',
            moving_top_density='top_density/top_under_half condition on each current cloud y>2.3; membership can differ across arms and windows',
            fixed_target_top='top_target_near_frac/gap use the SAME target IDs with y>2.3; coverage is not watertightness',
            phase_scope='paired archive phase audit starts at commit1 and covers windows2..common (at most8); W1 excluded and separately checked by cap1 integration',
            final_phase='phase20 is the raw final step plus PIC remap, not PIC alone',
            motion_accounting='separate raw-step/PIC telemetry uses per-arm, per-window arrival/pin cohorts; not matched-ID causal evidence',
            prefix_limit='lower motion may reflect slower progress or more pins; compare geometry/arrival, common progress and pin fractions; no rest/repair conclusion')
        if intervention == 'geometric_variance_full':
            first = max(1, common-10)
            result['definitions']['phase_scope'] = (
                f'paired archive phase audit starts at accepted commit{first} and covers '
                f'windows{first+1}..{common}; at most the last10 common windows, W1 excluded')
            result['definitions']['full_run_limit'] = result['definitions'].pop('prefix_limit')
            result['definitions']['endpoints'] = (
                'each arm uses its delivered accepted endpoint and delivery-scoped raw audit; '
                'later accepted states may be excluded by truncation; matched phase motion '
                 'ends at the common delivered accepted count, not necessarily either endpoint')
    if intervention == LAYER_FULL:
        first = max(1, common-10)
        result['definitions'].update(
            comparison='whole adaptive/stopping policy with both layer_ctrl and layer_relax disabled; '
                       'not an isolated layer-kernel causal effect or a fixed-lambda comparison',
            endpoints='own delivery-retained accepted endpoint, actual last accepted endpoint and '
                      'common delivery-retained accepted endpoint are distinct; held rows are not rest evidence',
            phase_scope=f'accepted commit{first} through{common}, at most10 common retained windows; W1 excluded',
            final_phase='raw final rollout step; PIC and shift are disabled in both arms',
            ordinary_stop='rejection/plateau or a configured cap does not certify natural rest or absence of holes')
    result['dependencies'] = {
        name: hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()
        for name in ('scripts.probes.render_influence', 'scripts.probes.morph_raw_qa')}
    for name, run in zip(names, runs):
        with cuda_execution('cuda'):
            prefix_pin = scoped_pin_motion(run, spacing) if intervention not in FULL_INTERVENTIONS else None
        result[name] = dict(prefix=run['prefix'], seconds=run['meta'].get('seconds'),
                           runtime_scope='complete original run, not the analyzed prefix',
                           commits=len(run['records']),
                           attempts=count_optimizer_attempts(
                               run['arm']['history'],
                               through_animation=(int(run['records'][-1]['animation'])
                                                  if intervention not in FULL_INTERVENTIONS else None)),
                           attempts_scope=('optimizer calls through last analyzed accepted endpoint'
                                           if intervention not in FULL_INTERVENTIONS else
                                           'optimizer calls in complete original run, including after delivery'),
                           original_attempts=count_optimizer_attempts(run['arm']['history']),
                           history_records=len(run['arm']['history']),
                           history_records_scope='complete original history, including non-optimizer metadata',
                           guards=run['arm']['guards'], guards_scope='complete original run',
                           arrival_mode_evidence=run.get('arrival_mode_evidence'),
                           archive_schema=run.get('archive_schema'),
                           curve=run['curve'], tip_history=run['tip_history'],
                           min_accepted_detF=min(row['Jmin_traj'] for row in run['curve'] if row['Jmin_traj'] is not None),
                           interior_hold_frames=run['interior_hold_frames'],
                           gradient_summary=channel_summary(run['records']),
                           scoped_pin_motion=prefix_pin,
                           raw_audit=audit(Path(run['prefix']), compute_backend='cuda') if intervention in FULL_INTERVENTIONS else None)
        result[name]['render_reference_history'] = render_reference_history(run, intervention)
        if intervention == LAYER_FULL:
            result[name].update(delivery_scope=run['delivery_scope'],
                                artifact_hashes=run['artifact_hashes'],
                                artifact_bindings=run['artifact_bindings'],
                                delivered_endpoint=run['curve'][-1], actual_endpoint=run['actual_endpoint'])
        if run['interior_hold_frames'] and result[name]['raw_audit'] and result[name]['raw_audit']['tail_unpinned_surface']:
            result[name]['raw_audit']['tail_unpinned_surface']['time_basis_warning'] = (
                'Legacy audit includes interior null holds; use corrected common-cohort raw-step motion for this comparison')
    if intervention == LAYER_FULL:
        verify_layer_artifacts(runs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({'out': str(out), 'config_changes': changes, 'source_cohort_n': source_meta['sampled_count']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--intervention', choices=(*FULL_INTERVENTIONS, *PREFIX_INTERVENTIONS), default='body_rprop')
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.out, args.intervention)
