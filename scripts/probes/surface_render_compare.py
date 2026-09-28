"""Matched cap1 P302 raw-state comparison, GPU numerics; never a rest/holes gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as host_np
import warp as wp
wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
from physmorph.compute import cuda_execution, to_array, array_api as np
from physmorph.metrics import chamfer, sil_iou
from scripts.probes.morph_raw_qa import frames_array


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--repeat', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    paths = [Path(v) for v in (args.control, args.candidate, args.repeat)]
    report_bytes = [p.with_suffix('.json').read_bytes() for p in paths]
    reports = [json.loads(raw) for raw in report_bytes]
    reference = reports[0]
    for i, report in enumerate(reports):
        changes = {k for k in set(reference['config']) | set(report['config'])
                   if reference['config'].get(k) != report['config'].get(k)}
        assert changes == ({'surface_gs_weight'} if i == 1 else set()), changes
        assert report['mpm'] == reference['mpm'] and report['code_sha256'] == reference['code_sha256']
        assert len(report['history']) == 1 and report['history'][0]['outer_accepted'] == 1
        assert report['history'][0]['surface_render_model'] == reference['history'][0]['surface_render_model']
    assert reference['config']['surface_gs_weight'] == 0 and reports[1]['config']['surface_gs_weight'] == 1
    frames, rows, evidence = [], [], []
    with cuda_execution('cuda:0'):
        source = target = None
        for path, report, captured_json in zip(paths, reports, report_bytes):
            archive_path = str(path)+'_render_full_dt_iso_nn.npz'
            hashes = {suffix: file_sha256(str(path)+suffix) for suffix in
                      ('.json', '.npz', '_render_full_dt_iso_nn.npz')}
            assert hashes['.json'] == hashlib.sha256(captured_json).hexdigest()
            with host_np.load(archive_path) as archive:
                src, tgt = to_array(archive['src']), to_array(archive['tgt'])
                count = int(archive['deliver_n'])
                f_samples = to_array(archive['F_samples'])
                f_indices = archive['F_sample_idx']
            if source is None:
                source, target = src, tgt
            assert bool(np.array_equal(src, source) & np.array_equal(tgt, target))
            raw = frames_array(archive_path)
            assert count == len(raw), 'cap1 comparison must have no held/undelivered suffix'
            x = to_array(raw)
            assert len(x) == int(report['config']['T'])+1
            assert report['config']['stop_after_windows'] == 1
            assert report['history'][0]['frame_end'] == count
            assert bool(np.isfinite(x).all() & np.isfinite(src).all() & np.isfinite(tgt).all()
                        & np.isfinite(f_samples).all())
            assert bool(np.array_equal(x[0], src)), 'archive start is not the claimed source'
            final_indices = host_np.flatnonzero(f_indices == count-1)
            assert len(final_indices) == 1
            with host_np.load(path.with_suffix('.npz')) as compact:
                final, final_f, commits = (to_array(compact[k]) for k in ('final', 'F', 'commits'))
                assert bool(np.array_equal(final, x[-1]) &
                            np.array_equal(final_f, f_samples[int(final_indices[0])]))
                assert len(commits) == 1 and bool(np.array_equal(commits[0], x[-1]))
            # Bind the exact bytes before and after reading; fail on concurrent replacement.
            assert hashes == {suffix: file_sha256(str(path)+suffix) for suffix in hashes}
            evidence.append(dict(prefix=str(path), sha256=hashes))
            frames.append(x)
            row = report['history'][0]
            movement = x[1:]-x[:-1]
            rows.append(dict(name=path.name, seconds=report['seconds'], peak_torch_bytes=report['torch_peak_bytes'],
                guards=report['guards'], accepted=row['accepted'], rejected=row['rejected'],
                lambda_render=row['lambda'], d_vol=row['d_vol'], d_cic=row['d_render'],
                surface_components=row['surface_render'], Jmin_traj=row['Jmin_traj'],
                endpoint_chamfer_wu=chamfer(x[-1], target), endpoint_binary_sil_iou=sil_iou(x[-1], target),
                saved_step_rms_wu=float(np.sqrt(np.square(movement).sum(2).mean())),
                scope='Entire first window including travelers; not settled motion or a holes metric'))
        rms = lambda value: float(np.sqrt(np.square(value).sum(-1).mean()))
        result = dict(discretization=dict(N=len(source), T=reference['config']['T'], dt=reference['mpm']['dt'],
                dx=reference['mpm']['dx'], loss_res=reference['config']['loss_res'], iters=reference['config']['iters']),
            code_sha256=reference['code_sha256'], rows=rows,
            input_evidence=evidence, helper_sha256=file_sha256(__file__),
            metric_source_sha256=file_sha256(Path(__file__).resolve().parents[2]/'physmorph/metrics.py'),
            candidate_vs_control_endpoint_rms_wu=rms(frames[1][-1]-frames[0][-1]),
            repeat_vs_control_endpoint_rms_wu=rms(frames[2][-1]-frames[0][-1]),
            candidate_vs_control_path_rms_wu=rms(frames[1]-frames[0]),
            repeat_vs_control_path_rms_wu=rms(frames[2]-frames[0]),
            limitations='One cap1 realization plus one control repeat, not a noise distribution. '
                'Only GS weight differs; existing CIC/PBR guidance remains in both. Lambda adapts. '
                'No all-frame holes/rest/high-resolution appearance certificate.')
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
