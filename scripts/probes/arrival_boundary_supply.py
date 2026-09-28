"""P301 saved raw-T versus promoted supply; CUDA geometry, no simulation."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.probes import constitutive_quality as cq
from scripts.probes import transient_supply as ts

ARRIVAL_SHA = '6f6d1f53d0facac22eb579126e5b363bfe33599d18a1792d4c73ff5cbbb8cfca'
SUPPLY_SHA = '2f74dcbe6d9017b26ca104a02d321a0b38934714bdbf265e0af86969b403d818'


def verified_bytes(path, expected_sha, expected_bytes=None):
    before = cq.file_identity(path)
    data = path.read_bytes()
    cq.require(cq.file_identity(path) == before and cq.sha(data) == expected_sha
               and (expected_bytes is None or len(data) == expected_bytes), 'Evidence identity/hash/size differs: '+str(path))
    return data


def boundary_summary(counts, gaps):
    """Three rows: prior accepted start, raw x[T], promoted. No interior phase."""
    from physmorph.compute import array_api as np, to_array
    counts, gaps = to_array(counts), to_array(gaps)
    cq.require(counts.ndim == gaps.ndim == 2 and len(counts) == len(gaps) == 3
               and bool(np.isfinite(counts).all()) and bool(np.isfinite(gaps).all())
               and bool((counts >= 0).all()) and bool((gaps >= 0).all()), 'Invalid three-state supply samples')
    supported, covered = counts >= 4, gaps <= 2
    def event(mask, eligible, values, source=False):
        n, denominator = int(mask.sum()), int(eligible.sum())
        v = values[mask]
        return dict(count=n, eligible_ids=denominator, fraction=n/denominator if denominator else None,
            minimum_count=int(v.min()) if n and source else None,
            zero_count_ids=int((v == 0).sum()) if source else None,
            maximum_gap_sp=float(v.max()) if n and not source else None,
            p95_gap_sp=float(np.percentile(v, 95)) if n and not source else None)
    source_eligible, target_eligible = supported[0] & supported[2], covered[0] & covered[2]
    return dict(states={name: ts._summary(counts[i], gaps[i]) for i, name in enumerate(('previous_start', 'raw_T', 'promoted'))},
        source_pic_loss=event(supported[1] & ~supported[2], supported[1], counts[2], True),
        source_pic_gain=event(~supported[1] & supported[2], ~supported[1], counts[1], True),
        target_pic_loss=event(covered[1] & ~covered[2], covered[1], gaps[2]),
        target_pic_gain=event(~covered[1] & covered[2], ~covered[1], gaps[1]),
        source_raw_deficit_between_supported_endpoints=event(source_eligible & ~supported[1], source_eligible, counts[1], True),
        target_raw_deficit_between_covered_endpoints=event(target_eligible & ~covered[1], target_eligible, gaps[1]))


def audit(result, result_sha256, out):
    cq.require(not out.exists(), 'Preserve existing output')
    record = json.loads(verified_bytes(result, result_sha256))
    pr = record['protocol']; protocol_path = Path(pr['path'])
    cq.require(protocol_path.resolve().is_relative_to('/data'), 'Protocol must resolve below /data')
    protocol = json.loads(verified_bytes(protocol_path, pr['sha256'], pr['bytes']))
    qpath = Path(protocol['quality_path'])
    cq.require(qpath.resolve().is_relative_to('/data'), 'Quality evidence must resolve below /data')
    q = json.loads(verified_bytes(qpath, protocol['quality_sha256']))
    ts.validate_quality(q, cq.sha(Path(cq.__file__).read_bytes()))
    cq.require(protocol['probe_sha256'] == ARRIVAL_SHA and protocol['quality_probe_sha256'] == q['probe_sha256']
               and cq.sha(Path(ts.__file__).read_bytes()) == SUPPLY_SHA,
               'Unreviewed capture or numerical helper')
    helpers = {name: cq.sha((ROOT/name).read_bytes()) for name in cq.HELPERS}
    loaded = cq.snapshot(ROOT)
    cq.require(helpers == protocol['helper_sha256'] == cq.HELPERS and loaded['digest'] == cq.SOURCE_DIGESTS[1]
               == protocol['numerical_source_sha256'] and loaded['files'] == protocol['numerical_files'], 'Numerical provenance differs')
    cq.require(protocol['config'] == q['config'] and protocol['mpm'] == q['mpm']
               and protocol['native_spacing_wu'] == q['native_spacing'] and not any(record['guards'].values()), 'Recipe/units/guards differ')
    accepted = [r for r in record['history'] if r.get('frame_end') and not r.get('null_commit') and not r.get('held')]
    rows, evidence = record['windows'], []
    cq.require(len(rows) == len(accepted) == record['summary']['accepted_commits'] > 0, 'Admission scope differs')
    # Verify every payload before loading any sidecar arrays.
    for i, (row, history) in enumerate(zip(rows, accepted), 1):
        item = row['sidecar']; path = Path(item['path'])
        cq.require(row['commit'] == i and row['attempt'] == int(history['animation'])+1
                   and row['frame_end'] == history['frame_end'] and row['T'] == 20
                   and path.resolve() == (result.parent/f'accepted_{i:03d}.npz').resolve(), 'Sidecar admission/path mismatch')
        actual = cq.file_record(path)
        cq.require(actual['sha256'] == item['sha256'] and actual['bytes'] == item['bytes'], 'Sidecar hash/size differs')
        evidence.append(actual)
    from physmorph.compute import array_api as np, KDTree, cuda_execution, to_array, to_host
    from scripts.probes.quality_compare import bounded_ids
    import numpy as host_np
    cq.require(Path(sys.modules['physmorph'].__file__).resolve().parent.parent == ROOT, 'Numerical package outside inspected root')
    source_path = Path(protocol['source_path'])
    cq.require(source_path.resolve().is_relative_to('/data'), 'Source must resolve below /data')
    cq.require(cq.file_identity(source_path) == protocol['source_identity'], 'Original input archive identity differs')
    with host_np.load(source_path, allow_pickle=False) as archive:
        source, target = archive['src'], archive['tgt']
    cq.require(cq.validate_inputs(source, target) == protocol['input_hashes'], 'Original input array hashes differ')
    cq.require(cq.file_identity(source_path) == protocol['source_identity'], 'Input archive changed during read')
    output = dict(result_path=str(result), result_sha256=result_sha256, protocol=pr,
        quality_path=str(qpath), quality_sha256=protocol['quality_sha256'], probe_sha256=cq.sha(Path(__file__).read_bytes()),
        capture_probe_sha256=ARRIVAL_SHA, supply_helper_sha256=SUPPLY_SHA, numerical_source_sha256=loaded['digest'],
        sidecar_evidence=evidence, mpm=q['mpm'], T=20, n=300000, native_spacing=q['native_spacing'],
        target_spacing=q['target_spacing'], density_radius=q['density_radius'], windows=[], no_policy_promotion=True)
    with cuda_execution('cuda'):
        source, target = to_array(source), to_array(target)
        counts = KDTree(source).query_ball_point(source, 2*q['native_spacing'], return_length=True)
        ids, meta = bounded_ids((counts < .6*np.median(counts)) & (source[:, 1] >= (source[:, 1].min()+source[:, 1].max())/2))
        cq.require(meta['ids_sha256'] == ts.SOURCE_IDS_SHA and len(ids) == 6712, 'Fixed source IDs differ')
        target_ids = np.flatnonzero(target[:, 1] > 2.3)
        cq.require(len(target_ids) == 15312, 'Fixed target IDs differ')
        output['cohorts'] = dict(source=meta, target_count=len(target_ids), target_ids_sha256=cq.sha(to_host(target_ids).tobytes()))
        target = target[target_ids]
        sample = lambda x: ts.frame_sample(x, ids, target, q['density_radius'], q['target_spacing'])
        previous = sample(source)
        for row, item in zip(rows, evidence):
            with host_np.load(item['path'], allow_pickle=False) as archive:
                metadata = json.loads(str(archive['__meta__']))
                cq.require(all(metadata[key] == row[key] for key in ('commit', 'attempt', 'frame_end', 'T', 'dt', 'native_spacing_wu', 'radius_wu')), 'Sidecar metadata differs')
                raw, promoted = to_array(archive['raw']), to_array(archive['promoted'])
            cq.require(raw.shape == promoted.shape == (300000, 3), 'Material ID count differs')
            raw_sample, promoted_sample = sample(raw), sample(promoted)
            values = (previous, raw_sample, promoted_sample)
            output['windows'].append(dict(commit=row['commit'], attempt=row['attempt'], frame_end=row['frame_end'],
                **boundary_summary(np.stack([v[0] for v in values]), np.stack([v[1] for v in values]))))
            previous = promoted_sample
    for item in evidence:
        cq.require(cq.file_identity(Path(item['path'])) == {key: item[key] for key in ts.IDENTITY_KEYS}, 'Sidecar changed during audit')
    output['definitions'] = dict(states='Prior accepted promoted start (source for W1), raw x[T], promoted endpoint only; no x[T-1]/phase19 or full-window minimum is measured.',
        support='Same fixed6712source and15312target IDs as quality24. Count inclusive target-k9 radius minus self; supported count>=4, density=count/8. Fixed-target coverage gap<=2 target spacings.',
        operator='Loss/gain compares the actual raw and promoted positions on the same IDs; direct final-position-operator geometry only, not its later dynamical effect.',
        eligibility='PIC loss denominator=raw supported/covered IDs; gain denominator=raw unsupported/uncovered IDs. Raw-deficit denominator=supported/covered at BOTH prior accepted start and promoted end.',
        severity='Only actual event IDs contribute: losses use promoted adverse values, gains/raw deficits use raw adverse values. Empty extrema/percentiles are null; empty event and source-zero counts are0. Absolute early uncovered regions may be unfinished transport.',
        scope='Fresh P301 realization, all actual accepted commits including any later delivery-trimmed tail. Not the earlier P300 pair; no watertightness, rest or repair pass.')
    data = json.dumps(output, indent=2, allow_nan=False).encode()
    cq.require(len(data) < 2_000_000, 'JSON budget exceeded')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.result, args.out):
        cq.require(path.resolve().is_relative_to('/data'), 'Production paths must resolve below /data')
    for name in ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH'):
        value = os.environ.get(name)
        cq.require(value and Path(value).resolve().is_relative_to('/data'), 'Explicit /data cache required: '+name)
    sys.path.insert(0, str(ROOT))
    import torch
    cq.require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
    value = audit(args.result.resolve(), args.result_sha256, args.out.resolve())
    print(json.dumps(dict(out=str(args.out), accepted=len(value['windows']))), flush=True)


if __name__ == '__main__':
    main()
