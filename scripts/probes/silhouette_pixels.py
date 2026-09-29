"""Archive-only P314 raw-mask localization and fixed-histogram CUDA parity.

No MPM replay, renderer, optimization, adoption, or metric threshold change.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

import numpy as host_np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import (array_api as np, cuda_execution, cuda_module,
                              ndimage, to_array, to_host)
from physmorph.metrics import _splat_body, target_extent, hole_frac
from physmorph.metric_splat import fixed_footprint_counts
from physmorph.pipeline.render_loss import make_views
from scripts.probes.coverage_paths import observations, require, sha


def projection(x, res, theta, phi, extent):
    # Preserve the audited metric's FP32 basis and operation order.
    right = np.asarray([math.cos(theta), 0.0, -math.sin(theta)], np.float32)
    up = np.asarray([-math.sin(phi)*math.sin(theta), math.cos(phi),
                     -math.sin(phi)*math.cos(theta)], np.float32)
    p = np.stack([x @ right, x @ up], 1)
    rel = (p + extent) / (2 * extent) * res
    ij = np.floor(rel).astype(np.int64)
    return ij, (ij >= 0).all(1) & (ij < res).all(1)


def legacy_counts(ij, valid, res):
    ij = ij[valid]
    flat = np.zeros(res * res, np.float64)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            i2 = np.clip(ij[:, 0] + ox, 0, res-1)
            j2 = np.clip(ij[:, 1] + oy, 0, res-1)
            flat += np.bincount(i2*res+j2, minlength=res*res)
    return flat.reshape(res, res)


def transitions(base, candidate, target):
    """Separate TP loss/gain and FP gain/removal; never infer changes from net IoU."""
    require(base.shape == candidate.shape == target.shape, 'Mask layout mismatch')
    holes = lambda mask: ndimage.binary_fill_holes(mask) & ~mask
    hb, hc = holes(base), holes(candidate)
    return dict(lost_tp=int((base & ~candidate & target).sum()),
                gained_tp=int((~base & candidate & target).sum()),
                gained_fp=int((~base & candidate & ~target).sum()),
                removed_fp=int((base & ~candidate & ~target).sum()),
                new_internal_hole_pixels=int((hc & ~hb).sum()),
                removed_internal_hole_pixels=int((hb & ~hc).sum()))


def supplier_ids(ij, valid, pixel):
    # Centers are inside the image; clipped border taps do not introduce a
    # center farther than one pixel. This returns unique IDs, not tap counts.
    return np.flatnonzero(valid & (np.abs(ij - pixel).max(1) <= 1))


def analyze(obs, clouds, expected_rows, res=128):
    require(len(clouds) == len(expected_rows) == 6, 'Expected original3/origin/control/treatment')
    N = len(obs['x0']); target = obs['target']; extent = target_extent(target)
    for x in clouds:
        require(x.shape == (N, 3) and x.dtype == np.float32 and bool(np.isfinite(x).all()),
                'Invalid endpoint')
        require(bool((x[obs['pins']] == obs['x0'][obs['pins']]).all()), 'Pinned endpoint changed')
    views = make_views(8, (0., .5, -.5)); all_masks=[]; all_counts=[]; all_ious=[]; view_rows=[]
    pixels=[]; ptr=[0]; suppliers=[]
    for v, (theta, phi) in enumerate(views):
        counts=[]; masks=[]; projections=[]
        for x in clouds + [target]:
            ij, valid = projection(x, res, theta, phi, extent)
            count = fixed_footprint_counts(ij, valid, res)
            require(bool((count == legacy_counts(ij, valid, res)).all()), 'Count parity failed')
            mask = _splat_body(x, res, theta, phi, extent)
            require(bool((mask == (count > 0)).all()), 'Existing metric mask parity failed')
            projections.append((ij, valid)); counts.append(count); masks.append(mask)
        counts, masks = np.stack(counts), np.stack(masks)
        target_mask = masks[-1]
        intersections = (masks[:6] & target_mask).sum((1, 2))
        unions = (masks[:6] | target_mask).sum((1, 2))
        ious = np.where(unions > 0, intersections / np.maximum(unions, 1), 1.)
        changed = np.argwhere((masks[:6] != masks[0]).any(0))
        for pixel in changed:
            pixels.append(np.concatenate((np.asarray([v], np.int64), pixel)))
            for ij, valid in projections[:6]:
                ids = supplier_ids(ij, valid, pixel)
                suppliers.append(ids); ptr.append(ptr[-1] + len(ids))
        view_rows.append(dict(view=v, theta=theta, phi=phi,
            intersection=to_host(intersections).tolist(), union=to_host(unions).tolist(),
            iou=to_host(ious).tolist(), changed_pixels=len(changed),
            baseline_ambiguous_pixels=int((masks[:3] != masks[0]).any(0).sum()),
            comparisons={str(c):[transitions(masks[b], masks[c], target_mask) for b in range(3)]
                         for c in range(3, 6)}))
        all_masks.append(masks); all_counts.append(counts); all_ious.append(ious)
    masks=np.stack(all_masks); counts=np.stack(all_counts)
    ious=np.stack(all_ious).mean(0)
    for i, row in enumerate(expected_rows):
        require(abs(float(ious[i])-row['geometry']['sil_iou']) <= 2*host_np.finfo(float).eps,
                'Recorded raw IoU failed closure at endpoint '+str(i))
    ids=np.concatenate(suppliers) if suppliers else np.empty(0, np.int64)
    material_ids=np.unique(ids)
    cohorts=np.where(obs['pins'][material_ids], 0, np.where(obs['start_arrived'][material_ids], 1, 2))
    data=dict(masks=masks, counts=counts,
              pixels=np.stack(pixels) if pixels else np.empty((0, 3), np.int64),
              supplier_ptr=np.asarray(ptr, np.int64), supplier_ids=ids,
              material_ids=material_ids, material_cohorts=cohorts,
              material_positions=np.stack([x[material_ids] for x in clouds]),
              target=target, extent=np.asarray(extent), views=np.asarray(views))
    summary=dict(N=N, resolution=res, extent=extent, views=view_rows, raw_iou=to_host(ious).tolist(),
                 changed_view_pixels=len(pixels), material_ids=to_host(material_ids).tolist(),
                 material_cohorts=to_host(cohorts).tolist(),
                 cohort_labels=['start_pinned','start_arrived_free','remaining_free'],
                 endpoint_hole_frac_160=[hole_frac(x, extent) for x in clouds],
                 baseline_masks_identical=bool((masks[:, :3] == masks[:, :1]).all()),
                 exact_count_mask_parity=True, recorded_iou_closure=True)
    return summary, data


def histogram_cuda_gate(ij, valid, res):
    """Bounded warmed comparison; capture tests this helper, not the full runtime."""
    cp=cuda_module(); stream=cp.cuda.get_current_stream()
    expected=legacy_counts(ij,valid,res)
    for _ in range(3): fixed_footprint_counts(ij,valid,res)
    stream.synchronize()
    stream.begin_capture()
    captured=fixed_footprint_counts(ij,valid,res)
    graph=stream.end_capture(); graph.launch(stream); stream.synchronize()
    require(bool((captured == expected).all()), 'Captured footprint counts differ')
    # Probe duplicates, border clipping, excluded rows, empty and all-excluded.
    edge=to_array(host_np.array([[0,0],[0,0],[res-1,res-1],[-1,0],[res,res]],host_np.int64))
    good=(edge>=0).all(1)&(edge<res).all(1)
    require(bool((fixed_footprint_counts(edge,good,res)==legacy_counts(edge,good,res)).all()),
            'CUDA duplicate/border parity failed')
    require(not bool(fixed_footprint_counts(edge,np.zeros(5,np.bool_),res).any()), 'Excluded center counted')
    require(not bool(fixed_footprint_counts(edge[:0],good[:0],res).any()), 'Empty footprint counted')
    times={'legacy':[], 'fixed':[]}
    for repeat in range(6):
        for label in (('legacy','fixed') if repeat%2==0 else ('fixed','legacy')):
            op=legacy_counts if label=='legacy' else fixed_footprint_counts
            stream.synchronize(); start=time.perf_counter()
            value=op(ij,valid,res); stream.synchronize()
            times[label].append(time.perf_counter()-start)
            require(bool((value==expected).all()), 'Timed histogram parity failed')
    return dict(graph_capture_replay_exact=True, edge_empty_checks=True, wall_seconds=times,
                scope='Warmed histogram helper only; projection, I/O, MPM and raster extension excluded')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    require(args.out.resolve().is_relative_to(args.root.resolve()), 'Output outside project data')
    args.out.mkdir(exist_ok=False)
    run=args.root/'work/p303/support_repair1';folder=run/'support_repair'
    result_path=run/'result.json';protocol_path=run/'protocol.json'
    result_bytes=result_path.read_bytes(); protocol_bytes=protocol_path.read_bytes()
    result_sha=sha256(result_bytes).hexdigest(); protocol_sha=sha256(protocol_bytes).hexdigest()
    require(result_sha=='6108b1950b13c831baf2dd0391d2200e702a4fb586e72a5af0c1782209d2a24e','Wrong P314 result')
    result=json.loads(result_bytes);protocol=json.loads(protocol_bytes);e=result['extension']
    require(protocol_sha==result['protocol_sha256'],'Wrong P314 protocol')
    names=[f'baseline/repeat_{i}.npz' for i in range(3)] + [
        'origin/terminal05_origin_2.npz',
        'silhouette/update1_half8_correction0_endpoint.npz',
        'support/update1_half8_correction1_endpoint.npz']
    rows=e['baseline']['rows']+[e['origin']['terminal_record']]
    for arm,label in [('silhouette','update1_half8_correction0'),('support','update1_half8_correction1')]:
        rows.append(next(row for row in e['arms'][arm]['trials'] if row['label']==label))
    inputs={str(folder/name):e['sidecars'][name] for name in names+['live_window.npz']}
    inputs.update({str(result_path):result_sha,str(protocol_path):protocol_sha})
    code_root=Path(__file__).resolve().parents[2]
    files=['scripts/probes/silhouette_pixels.py','scripts/probes/coverage_paths.py',
           'physmorph/metric_splat.py','physmorph/metrics.py','physmorph/compute.py',
           'physmorph/pipeline/render_loss.py','docs/silhouette_pixels_p315.md',
           'scripts/ops/run_p303_probe.sh','scripts/ops/cuda_python.py']
    code={str(code_root/name):sha(code_root/name) for name in files}
    for name in ('physmorph/metrics.py','physmorph/compute.py','physmorph/pipeline/render_loss.py'):
        require(code[str(code_root/name)]==protocol['code'][name], 'Original metric dependency changed: '+name)
    require(all(sha(Path(k))==v for k,v in inputs.items()),'Input checksum mismatch')
    spec=dict(start_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,code=code,endpoint_order=names,
              mpm=protocol['mpm'],config=protocol['config'],no_forward=True,no_adoption=True,
              scope='Post-hoc saved endpoint mask/supplier localization; not all-phase or visible-hole certification')
    own_protocol=args.out/'protocol.json';own_protocol.write_text(json.dumps(spec,indent=2,allow_nan=False))
    with cuda_execution('cuda:0'):
        obs=observations(folder/'live_window.npz');clouds=[]
        require(len(obs['x0'])==result['N']==300000 and result['T']==20
                and obs['dt']==protocol['mpm']['dt'], 'Unexpected discretisation')
        for name in names:
            with host_np.load(folder/name,allow_pickle=False) as archive:
                clouds.append(to_array(archive['x'] if 'x' in archive else archive['positions'][-1]))
        summary,data=analyze(obs,clouds,rows)
        ij,valid=projection(clouds[0],128,*make_views(8)[0],summary['extent'])
        summary['histogram_gate']=histogram_cuda_gate(ij,valid,128)
        path=args.out/'pixels.npz';host_np.savez_compressed(path,**to_host(data))
    require(all(sha(Path(k))==v for k,v in {**inputs,**code}.items()),'Inputs/code changed during analysis')
    summary.update(protocol_sha256=sha(own_protocol),sidecars={path.name:sha(path)},inputs_code_unchanged=True,
        scope=spec['scope'],rendering=dict(role='Inherited P314 only; no rendering intervention',
            lambda_render=result['lambda_render'],evidence=result['render_influence']))
    (args.out/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(dict(done=True,changed_view_pixels=summary['changed_view_pixels'],
                         material_count=len(summary['material_ids']),raw_iou=summary['raw_iou'])),flush=True)


if __name__=='__main__': main()
