"""Motion/appearance diagnostic for the existing mixed60 studio Gaussian video.

Run on hyde06 only. Positions, attribute reconstruction and statistics run on CUDA;
archive/hash/JSON I/O stay on the host. The frozen renderer and input are verified
before measurement. This CLI is derived from, but is not byte-identical to, the
original executed diagnostic; both hashes are recorded separately.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import textwrap
import time

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def stats(values):
    if not values.numel():
        return None
    # Quantiles are exact nearest-rank selections, on CUDA (no NumPy sample path).
    count = values.numel()
    return dict(mean=float(values.mean()), median=float(values.kthvalue(max(1, math.ceil(.5*count))).values),
                p95=float(values.kthvalue(max(1, math.ceil(.95*count))).values),
                max=float(values.max()), changed_frac=float((values != 0).float().mean()))


def readings(values, selected):
    count = int(selected.sum())
    return dict(particle_pairs=count, **{name: stats(value[selected]) for name, value in values.items()})


ROOT = Path('/data/relcfd/chayo/physmorph_v2')
ORIGINAL_MEASUREMENT_SHA256 = 'aca52ad8dd294663e85ff60c7da2bc10186814df21db5da51dc68a6450de39f6'
SOURCE_SHA256 = '7896e4fcb5f95020c559c3fb12a570c1e2106c062f0c2887c230986f2aee336e'
RUN_SHA256 = '0accf3fc9a93ecf4baf9450dc86ac304f784ee88bbc8559e8d8a84edf793b48d'
META_SHA256 = 'f917ea721f8a56f61a86e546c7a9471e324a680ce01c63354613da91d796d3c4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=ROOT/'work/gpu_refactor/render1080/repo')
    parser.add_argument('--snapshot-hashes', type=Path, default=ROOT/'work/gpu_refactor/render1080/snapshot_sha256.json')
    parser.add_argument('--npz', type=Path, default=ROOT/'output/c291/c291_bunny_mixed60_render_full_dt_iso_nn.npz')
    parser.add_argument('--run-json', type=Path, default=ROOT/'output/c291/c291_bunny_mixed60.json')
    parser.add_argument('--render-meta', type=Path, default=ROOT/'work/gpu_refactor/render4k/mixed60_studio_4k.json')
    parser.add_argument('--source-sha256', default=SOURCE_SHA256)
    parser.add_argument('--run-json-sha256', default=RUN_SHA256)
    parser.add_argument('--render-meta-sha256', default=META_SHA256)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    SNAPSHOT, SOURCE, RUN, META = args.snapshot, args.npz, args.run_json, args.render_meta
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = json.loads(args.snapshot_hashes.read_text())
    required = ('scripts/render_splat_photoreal.py', 'physmorph/render/knn_gpu.py', 'physmorph/render/settled.py')
    for name in required:
        if name not in manifest:
            raise ValueError(f'Missing frozen dependency hash: {name}')
    snapshot_root = SNAPSHOT.resolve()
    for name, expected in manifest.items():
        path = (snapshot_root/name).resolve()
        if not path.is_relative_to(snapshot_root) or sha(path) != expected:
            raise ValueError(f'Frozen snapshot hash mismatch: {name}')
    source_hash = sha(SOURCE)
    if source_hash != args.source_sha256:
        raise ValueError('Source archive hash mismatch')
    if sha(RUN) != args.run_json_sha256 or sha(META) != args.render_meta_sha256:
        raise ValueError('Run or video metadata hash mismatch')
    meta = json.loads(META.read_text())
    if Path(meta['source']).resolve() != SOURCE.resolve():
        raise ValueError('Render metadata refers to a different source archive')
    if meta['script_sha256'] != manifest['scripts/render_splat_photoreal.py']:
        raise ValueError('Renderer hash differs from the video metadata')
    # Import the frozen renderer after provenance checks. Its helpers must resolve
    # inside that snapshot rather than the concurrently edited working checkout.
    sys.path.insert(0, str(snapshot_root))
    script = snapshot_root/'scripts/render_splat_photoreal.py'
    spec = importlib.util.spec_from_file_location('frozen_renderer', script)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    if not torch.cuda.is_available():
        raise RuntimeError('Run this diagnostic on hyde06 with CUDA; no CPU fallback')
    device = torch.device('cuda:0')
    run = json.loads(RUN.read_text())['arms']['render_full_dt_iso_nn']
    archive = np.load(SOURCE, allow_pickle=True)
    frames = archive['frames']
    indices = meta['raw_frame_indices']
    count = min(len(frames), int(archive['deliver_n']))
    starts_np = renderer.pin_start_frames(np.asarray(archive['pinned'], bool), archive['pinned_at'], run['history'], run['config'])
    renderer.validate_pins_cuda(frames, starts_np, count, device)
    starts = torch.as_tensor(starts_np, device=device)
    settled = renderer.SettledAppearance(starts_np, device)
    target = torch.as_tensor(np.asarray(archive['tgt'], np.float32), device=device)
    center = target.mean(0)
    radius = float((target-center).norm(dim=1).max())
    target_d, _ = renderer.knn_self_torch(target, 9)
    spacing = float(target_d[:, 1].median())
    coverage_radius = float(target_d[:, 8].median())
    normals_from_density = renderer.DensityNormals(center, radius, spacing)

    # Reuse the literal attribute loop from the deployed script; no approximate reimplementation.
    text = script.read_text()
    begin = text.index('            distances, neighbors = knn_self_torch(x, 33)')
    end_text = '            normals, sigma, support = settled.apply(raw_index, x, normals, sigma, support)'
    end = text.index(end_text, begin) + len(end_text)
    segment = textwrap.dedent(text[begin:end])
    scope = dict(vars(renderer))
    scope.update(device=device, spacing=spacing, coverage_radius=coverage_radius,
                 normals_from_density=normals_from_density, settled=settled)
    exec('def attributes(x, raw_index):\n' + textwrap.indent(segment, '    ') + '\n    return normals, sigma, support\n', scope)
    attributes = scope['attributes']

    # Fixed material cohorts, selected geometrically at raw480 before the reported tip regression.
    reference_raw = 480
    reference = torch.as_tensor(np.asarray(frames[reference_raw], np.float32), device=device)
    ymin, ymax = float(target[:, 1].min()), float(target[:, 1].max())
    height = ymax-ymin
    thresholds = dict(head=ymin+.55*height, upper_ears=ymin+.75*height)
    cohorts = dict(all=torch.ones(len(reference), dtype=torch.bool, device=device),
                   head=reference[:, 1] >= thresholds['head'],
                   upper_ears=reference[:, 1] >= thresholds['upper_ears'])
    periods = dict(all_nonhold=(0, 780), growth=(0, 336), late=(480, 780), last=(660, 780))
    metric_names = ['position_sp', 'normal_degrees', 'normal_vector_l2', 'sigma_relative', 'support_absolute']
    pools = {(p, c, g): {name: [] for name in metric_names} for p in periods for c in cohorts for g in ['pinned', 'unpinned', 'new_pin']}
    reversal = {(p, c): [0, 0] for p in periods for c in cohorts}
    rows = []
    previous = previous_delta = previous_unpinned = previous_delta_from = None
    ever_support = {name: torch.zeros(len(reference), dtype=torch.bool, device=device) for name in periods}
    ever_active = {name: torch.zeros(len(reference), dtype=torch.bool, device=device) for name in periods}
    with torch.inference_mode():
        for step, raw in enumerate(indices):
            x = torch.as_tensor(np.asarray(frames[raw], np.float32), device=device)
            normal, sigma, support = attributes(x, raw)
            if previous is not None:
                oldraw, oldx, oldnormal, oldsigma, oldsupport = previous
                delta = x-oldx
                equal_normals = (normal == oldnormal).all(1)
                angle = torch.rad2deg(torch.atan2(torch.linalg.cross(normal, oldnormal).norm(dim=1), (normal*oldnormal).sum(1)))
                angle = torch.where(equal_normals, torch.zeros_like(angle), angle)
                values = dict(position_sp=delta.norm(dim=1)/spacing,
                              normal_degrees=angle, normal_vector_l2=(normal-oldnormal).norm(dim=1),
                              sigma_relative=(sigma-oldsigma).abs()/oldsigma.clamp_min(1e-12),
                              support_absolute=(support-oldsupport).abs())
                groups = dict(pinned=starts <= oldraw, unpinned=starts > raw, new_pin=(starts > oldraw) & (starts <= raw))
                row = dict(raw_from=oldraw, raw_to=raw, saved_frame_gap=raw-oldraw, extra_hold=(raw==781), groups={})
                for cohort, mask in cohorts.items():
                    row['groups'][cohort] = {group: readings(values, mask & selection) for group, selection in groups.items()}
                rows.append(row)
                for period, (lo, hi) in periods.items():
                    if oldraw < lo or raw > hi:
                        continue
                    ever_active[period] |= groups['pinned']
                    ever_support[period] |= groups['pinned'] & (values['support_absolute'] != 0)
                    for cohort, mask in cohorts.items():
                        for group, selection in groups.items():
                            pool = pools[(period, cohort, group)]
                            for name, value in values.items():
                                pool[name].append(value[mask & selection])
                        # Direction reversals are kinematic evidence only, not proof of a periodic mode.
                        if previous_delta is not None and previous_delta_from >= lo:
                            moving = mask & groups['unpinned'] & previous_unpinned & (values['position_sp'] >= .01) & (previous_delta.norm(dim=1)/spacing >= .01)
                            reversal[(period, cohort)][0] += int(((previous_delta*delta).sum(1)[moving] < 0).sum())
                            reversal[(period, cohort)][1] += int(moving.sum())
                previous_delta = delta
                previous_delta_from = oldraw
                previous_unpinned = groups['unpinned']
            previous = raw, x, normal, sigma, support
            if step % 12 == 0:
                print(json.dumps(dict(stage='frame', index=step, raw=raw, seconds=time.perf_counter()-started)), flush=True)

    summary = {}
    for period in periods:
        summary[period] = {}
        for cohort, mask in cohorts.items():
            result = {}
            for group in ['pinned', 'unpinned', 'new_pin']:
                values = {name: torch.cat(v) if v else torch.empty(0, device=device) for name, v in pools[(period, cohort, group)].items()}
                result[group] = dict(particle_pairs=values['position_sp'].numel(), **{name: stats(value) for name, value in values.items()})
            reversed_count, eligible_count = reversal[(period, cohort)]
            result['unpinned_reversal'] = dict(reversed_particle_triples=reversed_count, eligible_particle_triples=eligible_count,
                                               fraction=reversed_count/max(1,eligible_count), minimum_each_displacement_sp=.01)
            result['pinned_unique'] = dict(active_particles=int((ever_active[period] & mask).sum()),
                                            support_changed_particles=int((ever_support[period] & mask).sum()))
            summary[period][cohort] = result

    result = dict(measurement_version='2-cli', original_measurement_script_sha256=ORIGINAL_MEASUREMENT_SHA256, measurement_script_sha256=sha(__file__), source=str(SOURCE), source_sha256=source_hash, run_json_sha256=sha(RUN), render_metadata_sha256=sha(META), snapshot_manifest_sha256=sha(args.snapshot_hashes),
                  renderer_sha256=sha(script), literal_attribute_segment_sha256=hashlib.sha256(segment.encode()).hexdigest(),
                  dependencies={str(p.relative_to(SNAPSHOT)):sha(p) for p in [SNAPSHOT/'physmorph/render/knn_gpu.py', SNAPSHOT/'physmorph/render/settled.py']},
                  discretization=dict(N=len(target), T=20, dt=1/240, dx=.3062907544, loss_res=36, spacing=spacing),
                  cohort_definition=dict(reference_raw=reference_raw, reference='fixed raw-position material IDs', target_ymin=ymin, target_ymax=ymax,
                                         y_thresholds=thresholds, particles={k:int(v.sum()) for k,v in cohorts.items()},
                                         warning='upper_ears is a height-defined region, not an exhaustive thickness classifier'),
                  raw_indices=indices, validated_raw_frames=count, periods=periods,
                  summary_units=dict(position_sp='renderer target median nearest-neighbour spacings per selected-frame transition, normally12 saved frames; not source-native spacings', normal_degrees='absolute angular change; exactly equal vectors explicitly zeroed', normal_vector_l2='direct L2 difference, exactly0 for bitwise-identical normals',
                                     sigma_relative='absolute radius change divided by previous radius', support_absolute='absolute support in0..1; opacity=.92support'),
                  semantics=dict(pinned='active before both endpoints', unpinned='inactive at both endpoints', new_pin='admitted within interval, reported separately',
                                 quantiles='nearest-rank exact over pooled particle-transition samples; not mean of frame quantiles',
                                 hold='raw780->781 is saved in rows but excluded from summary periods',
                                 reversal='negative dot of successive displacement vectors; both>=0.01sp; not proof of periodic oscillation',
                                 causality='correlated changes do not assign pixel-flicker causality; no image counterfactual was rendered'),
                  summary=summary, per_transition=rows, seconds=time.perf_counter()-started)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(stage='done', seconds=result['seconds'], output=str(args.out))), flush=True)


if __name__ == '__main__':
    main()
