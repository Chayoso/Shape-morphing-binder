"""P300 cap6/cap24 raw-state comparison of the exact old/new corotated adjoints.

Numerical metrics run in the strict CUDA context. Host work is input/provenance
I/O and JSON metadata only. Existing same-code comparison guards are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ARM = 'render_full_dt_iso_nn'
SOURCE_SHA = '71eb14d38c2efb41379a12b0ce017430e093883cbce51188265ed9e2948d0e34'
TARGET_SHA = '8de5081712e5fb55f7c1c281e0bbe787ba8c2e948bc6c3635d2d01ac45dd8ed7'
METADATA_SHA = '0accf3fc9a93ecf4baf9450dc86ac304f784ee88bbc8559e8d8a84edf793b48d'
REFERENCE_SHA = '9c1f205945b4b68b330b902f59570b3f9ccb9d8eca8e0ebc553e035f642620a7'
SOURCE_DIGESTS = (
    'dd10bf7f5dac3c224069dab033e73d6a58d4edf38b5552d167cdbd17f601abce',
    '16a8bc6e72092abd81858eac4eda01b6c235af5e2ce4641a3c2ee0eb738167ec',
)
APPROVED_BLOBS = {
    'physmorph/__init__.py': (
        '28646b4d7dcd5df716ca638834c5c6635d7b0c4a8ee57230e05c3f8e20d90db9',
        '0e4acbe277745c4fce5402209f50def6b4592c9aaf76939270697d0c6de209bc'),
    'physmorph/mpm/constitutive.py': (
        '83a8b6adb6c21794db8a833d2fec04b1be949fb5ca597c872d4e0749e1a83097',
        '166080b825e19fe79a19ee613f35cedac750d5ef465a371fa8a1c80435c7bd2b'),
}
HELPERS = {
    'scripts/probes/gpu_pipeline.py': 'e1064b858c23abd2b6d360ffc4cc4ee09b9133c8a891e7d01006e20aca6457f0',
    'scripts/probes/quality_compare.py': '098da49eae06db205f603c692a7b64c75f8d7cf5eb095488cac6518c0dcc2925',
    'scripts/probes/raw_phase.py': '89e1388ea36e4ab298ad9d658f73cae3d8a1f748f07bc7c4f4e814a2140b0c13',
    'scripts/probes/render_influence.py': 'bc937261ed9229880c745137fa92a5dd1e82c3d3a48f71b5ecf790c7d02a7dd2',
    'scripts/probes/morph_raw_qa.py': '4a861b74bc740b8dbcf65638aa9a648d76666478fde65c3ad5a4fa1c0c1f3952',
    'scripts/probes/pic_endpoint_quality.py': '72296dc59659acb7ab31141e432646d856d08ff9ac6119d852c18b791003ab62',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(root):
    """Same ordered path/NUL/byte aggregate as the actual gpu_pipeline driver."""
    files = sorted((root/'physmorph').rglob('*.py'))
    require(files, 'Missing frozen physmorph source')
    blobs = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
    digest = sha(b''.join(name.encode()+b'\0'+data for name, data in blobs.items()))
    return dict(root=str(root), digest=digest, files={name: sha(data) for name, data in blobs.items()})


def validate_snapshots(old, new, helpers):
    require((old['digest'], new['digest']) == SOURCE_DIGESTS, 'Unapproved frozen source digest')
    a, b = old['files'], new['files']
    require(set(a) == set(b), 'Frozen source membership differs')
    changed = {name for name in a if a[name] != b[name]}
    require(changed == set(APPROVED_BLOBS), 'Expected exactly the two approved source changes')
    for name, pair in APPROVED_BLOBS.items():
        require((a[name], b[name]) == pair, 'Unapproved source blob pair: '+name)
    require(len(helpers) == 3, 'Old/new/loaded helper manifests required')
    require(all(values == HELPERS for values in helpers), 'Driver or numerical helper bytes changed')
    return dict(old=old, new=new, changed_files=sorted(changed), helper_sha256=HELPERS,
                interpretation='Intentional derivative treatment, not numerical source equivalence')


def validate_run_binding(meta, digest):
    require(meta.get('code_sha256') == digest == meta.get('provenance', {}).get('code_hash'),
            'Run does not match its own frozen source')
    arm = meta['arms'][ARM]
    require(meta['config'] == arm['config'], 'Run/arm config metadata differs')
    require(meta['mpm'] == meta['provenance']['mpm'], 'Run/provenance MPM differs')
    require(meta['history'] == arm['history'], 'Run/arm histories differ')
    require(meta['guards'] == arm['guards'] and not any(meta['guards'].values()), 'State guard fired or metadata differs')


def validate_windows(windows):
    require(type(windows) is int and windows in (6, 24), 'Only cap6 or cap24 is supported')


def expected_config(metadata, fields, target_reference, windows=6):
    validate_windows(windows)
    config = {k: v for k, v in metadata['arms'][ARM]['config'].items() if k in fields}
    config.update(compute_backend='cuda', stop_after_windows=windows, iters=8,
                  target_reference=str(target_reference), motion_accounting=True,
                  outer_render_committed=True, commit_pic_objective=True,
                  shift_sub=False, geometric_variance=False)
    return config


def validate_recipe(configs, mpms, metadata, fields, target_reference, windows=6):
    expected = expected_config(metadata, fields, target_reference, windows)
    require(configs[0] == configs[1] == expected, f'Configs differ or depart from the original cap{windows} recipe')
    require(mpms[0] == mpms[1] == metadata['provenance']['mpm'], 'MPM/input metadata mismatch')
    require(mpms[0]['dt'] == 1/240 and mpms[0]['dx'] == .3062907543956724,
            'Unexpected discretization')
    require(expected.get('T') == 20 and expected.get('loss_res') == 36 and expected.get('commit_pic') is True
            and expected.get('archive_stride') == 1 and expected.get('body_ctrl') is True,
            'Unsupported original recipe')
    require(not any(expected.get(k, False) for k in ('geometric_rest', 'render_F_geom', 'rest_commit',
            'settle_pin_yield', 'settle_pin_follow', 'settle_pin_kkt')), 'Unsupported state/pin-release mode')
    return expected


def array_identity(values):
    """Array bytes are a provenance boundary, not host geometry computation."""
    require(values.shape == (300000, 3) and str(values.dtype) == 'float32', 'Expected original float32 N300k input')
    return sha(values.tobytes())


def validate_inputs(source, target):
    hashes = (array_identity(source), array_identity(target))
    require(hashes == (SOURCE_SHA, TARGET_SHA), 'Original source/target array hashes differ')
    return dict(source_sha256=hashes[0], target_sha256=hashes[1])


def scope_history(history, delivered, archived, steps=20, windows=6):
    """Exclude copied suffixes from physical observation; keep stopping metadata."""
    validate_windows(windows)
    accepted = [r for r in history if r.get('frame_end') and not r.get('null_commit') and not r.get('held')]
    records = [r for r in accepted if int(r['frame_end']) <= delivered]
    require(0 < delivered <= archived and records, 'No delivered accepted physical endpoint')
    require(all(0 <= int(r['animation']) < windows for r in accepted), f'History exceeds cap{windows} attempts')
    indices = [0]
    for row in records:
        end = int(row['frame_end'])
        start = end-steps
        require(start > indices[-1] and end <= archived, 'Overlapping or invalid physical spans')
        indices.extend(range(start, end))
    end = int(records[-1]['frame_end'])
    return dict(records=records, physical_indices=indices, delivered=end,
                interior_hold_frames=end-len(indices),
                endpoint_scope=dict(original_delivered_frames=delivered, archived_frames=archived,
                    physical_endpoint_frame=end-1, held_suffix_excluded=delivered-end,
                    delivered_accepted_commits=len(records), actual_accepted_commits=len(accepted),
                    actual_last_accepted_frame=int(accepted[-1]['frame_end'])-1,
                    later_accepted_geometry_measured=False if len(accepted) > len(records) else True,
                    definition='Geometry ends at delivered accepted state; copied suffix is not physical observation'))


def matching_motion_status(count, common):
    return ('inconclusive_empty_common_free_cohort' if count == 0 else
            'inconclusive_fewer_than_three_common_commits' if common < 3 else 'available_descriptive')


def comparison_interval(common, windows):
    validate_windows(windows)
    require(type(common) is int and 1 <= common <= windows, 'Invalid common accepted count for selected cap')
    return (1 if windows == 6 else max(1, common-10), common)


def file_identity(path):
    stat = path.stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def file_record(path):
    before = file_identity(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    require(file_identity(path) == before, 'Evidence file changed during hashing')
    return dict(path=str(path), sha256=digest.hexdigest(), **before)


def phase_summary(run, ids, common, spacing, first_commit=1):
    """Saved phases only; reuse the reviewed phase indexing/distribution helpers."""
    from physmorph.compute import array_api as np, to_array
    from scripts.probes.raw_phase import phase_frame_indices, distribution
    if matching_motion_status(len(ids), common) != 'available_descriptive':
        return None
    indices = phase_frame_indices(run['records'], first_commit, common, 20)
    positions = np.stack([np.stack([to_array(run['frames'][frame])[ids] for frame in window])
                          for window in indices])
    moves = positions[:, 1:]-positions[:, :-1]
    lengths = np.linalg.norm(moves, axis=-1)
    stats = lambda values: distribution(values, include_rms=True)
    flat = moves.reshape(-1, len(ids), 3)
    size = lengths.reshape(-1, len(ids))
    active = (size[1:] > 1e-4*spacing) & (size[:-1] > 1e-4*spacing)
    reverse = (flat[1:]*flat[:-1]).sum(-1) < 0
    destination = np.arange(1, len(flat)) % 20+1
    groups = {}
    for name, mask in (('into_final_phase', destination == 20),
                       ('from_final_to_next_phase1', destination == 1),
                       ('interior_phases', (destination != 20) & (destination != 1))):
        valid = active[mask]
        count = int(valid.sum())
        reversed_count = int((reverse[mask] & valid).sum())
        groups[name] = dict(eligible_pairs=count, reversed_pairs=reversed_count,
                            fraction=reversed_count/count if count else None)
    path = lengths.sum((0, 1))
    moving = path > 1e-4*spacing
    return dict(accepted_interval=[first_commit, common], audited_windows=[first_commit+1, common], window1_excluded=True,
        frame_indices=indices, first_19_sp=stats((lengths[:, :-1]/spacing).ravel()),
        final_phase_sp=stats((lengths[:, -1]/spacing).ravel()),
        phases=[dict(phase=i+1, displacement_sp=stats((lengths[:, i]/spacing).ravel())) for i in range(20)],
        final_path_share=float(lengths[:, -1].sum()/path.sum()) if float(path.sum()) else None,
        net_over_path=stats(np.linalg.norm(positions[-1, -1]-positions[0, 0], axis=1)[moving]/path[moving]),
        reversal_groups=groups,
        definition=('W2..common; phase20 is final raw physical/layer step PLUS PIC. W1->W2 terminal reversal is not measured.'
                    if first_commit == 1 else
                    f'W{first_commit+1}..{common}; phase20 is final raw physical/layer step PLUS PIC. '
                    f'W{first_commit}->W{first_commit+1} terminal reversal is not measured.'))


def raw_checks(run):
    from physmorph.compute import array_api as np, to_array
    for index in run['physical_indices']:
        x = to_array(run['frames'][index])
        require(x.shape == (300000, 3) and bool(np.isfinite(x).all()), 'Nonfinite or malformed raw frame')
    return dict(finite=True, physical_frames=len(run['physical_indices']),
                frame_indices=run['physical_indices'], scope='Every accepted physical position row; no renderer')


def compare(old_root, new_root, baseline, candidate, source_metadata, target_reference, out, windows=6):
    validate_windows(windows)
    require(not out.exists(), 'Output exists; preserve earlier evidence')
    trees = [snapshot(root) for root in (old_root, new_root)]
    helper_sets = [{name: sha((root/name).read_bytes()) for name in HELPERS}
                   for root in (old_root, new_root, ROOT)]
    provenance = validate_snapshots(*trees, helper_sets)
    require(snapshot(ROOT)['digest'] == trees[1]['digest'], 'Loaded numerical tree must be the approved new snapshot')
    metadata_bytes = source_metadata.read_bytes()
    require(sha(metadata_bytes) == METADATA_SHA, 'Original c291 metadata hash differs')
    metadata = json.loads(metadata_bytes)
    require(sha(target_reference.read_bytes()) == REFERENCE_SHA, 'Prepared target reference differs')
    from physmorph.compute import array_api as np, cuda_execution, KDTree, to_array
    from physmorph.pipeline import PipelineConfig
    from physmorph.metrics import target_extent, hole_frac
    from scripts.probes.render_influence import load_run, count_optimizer_attempts, channel_summary
    from scripts.probes.quality_compare import (geometry_curve, bounded_ids, admitted_mask,
        cohort_motion, scoped_pin_motion, checked_arrival_modes, tip_history, render_reference_history)
    require(Path(sys.modules['physmorph'].__file__).resolve().parent.parent == ROOT,
            'Imported numerical package is outside the inspected audit root')
    runs = [load_run(prefix) for prefix in (baseline, candidate)]
    for run, tree in zip(runs, trees):
        validate_run_binding(run['meta'], tree['digest'])
        validate_inputs(run['source'], run['target'])
        require(Path(run['meta']['provenance']['source_archive']).with_suffix('.json').resolve() == source_metadata,
                'Driver used a different source metadata path')
        require(int(run['arm']['deliver_n']) == run['delivered'], 'Delivered archive/metadata mismatch')
        run['evidence'] = [file_record(Path(run['prefix']+suffix))
                           for suffix in ('.json', '.log', '_'+ARM+'.npz')]
        run.update(scope_history(run['arm']['history'], run['delivered'], len(run['frames']), windows=windows))
    config = validate_recipe([r['arm']['config'] for r in runs], [r['meta']['mpm'] for r in runs],
                             metadata, PipelineConfig.__dataclass_fields__, target_reference, windows)
    require(all(count_optimizer_attempts(r['arm']['history']) <= windows for r in runs), f'Attempt budget exceeds cap{windows}')
    arrival = checked_arrival_modes(runs)
    names = ('baseline', 'candidate')
    result = dict(scope=f'One cap{windows} whole-optimizer comparison; no final quality, rest, hole repair or promotion claim',
        source_treatment=provenance, probe_sha256=sha(Path(__file__).read_bytes()),
        input_hashes=dict(source=SOURCE_SHA, target=TARGET_SHA, metadata=METADATA_SHA, target_reference=REFERENCE_SHA),
        metadata_path=str(source_metadata), config_changes={}, config=config, mpm=runs[0]['meta']['mpm'],
        n=300000, T=20, loss_res=36, requested_windows=windows, no_policy_promotion=True)
    with cuda_execution('cuda'):
        source, target = to_array(runs[0]['source']), to_array(runs[0]['target'])
        source_tree, target_tree = KDTree(source), KDTree(target)
        spacing = float(np.median(source_tree.query(source, k=2)[0][:, 1]))
        target_spacing = float(np.median(target_tree.query(target, k=2)[0][:, 1]))
        radius = float(np.median(target_tree.query(target, k=9)[0][:, 8]))
        extent, tip = target_extent(target), target[target[:, 1].argmax()]
        counts = source_tree.query_ball_point(source, 2*spacing, return_length=True)
        source_mask = (counts < .6*np.median(counts)) & (source[:, 1] >= (source[:, 1].min()+source[:, 1].max())/2)
        source_ids, source_cohort = bounded_ids(source_mask)
        initial_chamfer = float(target_tree.query(source)[0].mean()+source_tree.query(target)[0].mean())
        for name, run, arrival_evidence in zip(names, runs, arrival):
            finite = raw_checks(run)
            run['curve'] = geometry_curve(run, target, target_tree, extent, target_spacing, radius, tip, source_ids)
            for row in run['curve']:
                row['hole_frac'] = hole_frac(to_array(run['frames'][row['frame']]), extent)
            pin = scoped_pin_motion(run, spacing)
            require(pin['moved_particles_exact'] == 0, 'Previously admitted pin moved')
            pin['physical_observation_scope'] = 'Through last delivered accepted physical endpoint; suffix excluded. Last-endpoint admissions lack later physical observation.'
            minimums = [r.get('Jmin_traj') for r in run['records'] if r.get('Jmin_traj') is not None]
            require(len(minimums) == len(run['records']) and all(math.isfinite(v) and v > 0 for v in minimums),
                    'Missing or invalid accepted trajectory orientation evidence')
            result[name] = dict(prefix=run['prefix'], run_json_sha256=sha(Path(run['prefix']+'.json').read_bytes()),
                own_delivered_endpoint=run['curve'][-1], curve=run['curve'], endpoint_scope=run['endpoint_scope'],
                seconds=run['meta']['seconds'], attempts=count_optimizer_attempts(run['arm']['history']),
                history_records=len(run['arm']['history']), guards=run['arm']['guards'],
                converged=run['arm']['converged'], truncation=run['arm']['truncation'], n_held=run['arm']['n_held'],
                interior_hold_frames=run['interior_hold_frames'], raw_checks=finite, scoped_pin_motion=pin,
                min_accepted_trajectory_detF=min(minimums), arrival_mode_evidence=arrival_evidence,
                gradient_summary=channel_summary(run['records']), tip_history=tip_history(run, tip),
                render_reference_history=render_reference_history(run, f'constitutive_cap{windows}'),
                artifact_provenance=run['evidence'])
        common = min(len(run['records']) for run in runs)
        first_commit, _ = comparison_interval(common, windows)
        boundary, free = np.zeros(300000, bool), np.ones(300000, bool)
        for run in runs:
            row = run['curve'][common-1]
            x = to_array(run['frames'][row['frame']])
            counts = KDTree(x).query_ball_point(x, 2*spacing, return_length=True)
            boundary |= counts < .6*np.median(counts)
            free &= ~admitted_mask(run, row['attempt'])
        free_ids, free_cohort = bounded_ids(boundary & free)
        cohorts = {}
        for label, ids, cohort in (('fixed_source_upper_surface', source_ids, source_cohort),
                                    ('common_endpoint_free_both', free_ids, free_cohort)):
            cohorts[label] = dict(**cohort, status=matching_motion_status(len(ids), common),
                arms={name: cohort_motion(run, ids, common, spacing, first_commit=first_commit, include_rms=True)
                      for name, run in zip(names, runs)},
                phases={name: phase_summary(run, ids, common, spacing, first_commit=first_commit)
                        for name, run in zip(names, runs)})
        selected_commits = (3, 6) if windows == 6 else (3, 6, 12, 18, 24)
        progress_fractions = (.75, .5, .25) if windows == 6 else (.75, .5, .25, .225, .22, .215, .2, .15, .1)
        result.update(native_spacing=spacing, target_spacing=target_spacing, density_radius=radius,
            target_extent=float(extent), target_tip_n=int((np.linalg.norm(target-tip, axis=1) < .25).sum()),
            common_accepted_commits=common, motion_commit_interval=[first_commit, common],
            cohorts=cohorts, initial_chamfer=initial_chamfer,
            matched_endpoints=[dict(commit=i, baseline=runs[0]['curve'][i-1], candidate=runs[1]['curve'][i-1])
                               for i in sorted({1, common} | {i for i in selected_commits if i <= common})],
            progress_crossings=[dict(chamfer_fraction=fraction, threshold=initial_chamfer*fraction,
                **{name: next((row for row in run['curve'] if row['chamfer'] <= initial_chamfer*fraction), None)
                   for name, run in zip(names, runs)}) for fraction in progress_fractions])
    result['definitions'] = dict(
        fixed_source='Original source upper sparse boundary, sorted IDs sampled once at most20000; includes pinned IDs. Same IDs both arms.',
        common_free='Outcome-selected sparse-boundary union at common accepted endpoint, intersect unpinned in BOTH arms. Not an after-arrival cohort.',
        density='Target median k9 distance[:,8], exclude self; count/8 and count<4. Fixed source IDs remain fixed below/above y2.3.',
        top='Target y>2.3 IDs fixed; current top density has changing source membership. Tip is radius.25 about target max-y.',
        phase='W2..common only; saved phase20 includes final raw physical/layer step PLUS PIC. No W1->W2 terminal reversal measured.',
        accounting='History motion_accounting separates raw/PIC on per-arm dynamic cohorts; not same-ID causal attribution.',
        direction='Normal/tangent bases are arm-specific and frozen at common endpoint; vector RMS/net/path use same IDs.',
        policy='Whole adaptive optimizer/code intervention: corrected gradients can change lambda, PCGrad, controls and pins. Not a fixed-lambda comparison.',
        stopping='converged can mean policy plateau/reject stop; copied suffix does not establish physical rest.',
        limitation='One pair cannot remove CUDA optimizer variability; early density/coverage are not watertightness and lower motion may be slower transport.')
    if windows == 24:
        result['definitions'].update(
            phase=f'Common accepted interval{first_commit}..{common}; W{first_commit+1}..{common}, at most last10 common windows. W1 excluded; phase20=raw final physical/layer step+PIC.',
            endpoints='Full delivered accepted curves and own delivered endpoints retained; motion ends at common delivered count, which may differ from either actual last accepted state.',
            progress='Crossings use each full retained curve from its first accepted commit; lower motion in the common tail is not certified after-arrival rest.',
            history='Attempt count covers complete original history; render-reference rows stop at the last delivered accepted endpoint. Later accepted geometry is unmeasured when delivery truncates.')
    for run in runs:
        for evidence in run['evidence']:
            require(file_identity(Path(evidence['path'])) == {key: evidence[key]
                    for key in ('device', 'inode', 'bytes', 'mtime_ns')}, 'Evidence changed during audit')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-root', 'new-root', 'baseline', 'candidate', 'source-metadata', 'target-reference', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--windows', type=int, choices=(6, 24), default=6)
    args = parser.parse_args()
    for value in vars(args).values():
        if isinstance(value, Path):
            require(value.resolve().is_relative_to('/data'), 'All production paths must resolve below /data')
    for name in ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH'):
        value = os.environ.get(name)
        require(value and Path(value).resolve().is_relative_to('/data'), 'Explicit /data cache required: '+name)
    sys.path.insert(0, str(ROOT))
    import torch
    require(torch.cuda.is_available(), 'CUDA required; no CPU numerical fallback')
    result = compare(**{key: value.resolve() if isinstance(value, Path) else value for key, value in vars(args).items()})
    print(json.dumps(dict(out=str(args.out), common=result['common_accepted_commits'],
                          status=result['cohorts']['common_endpoint_free_both']['status'])), flush=True)


if __name__ == '__main__':
    main()
