"""Hyde06 CUDA backend integration probe on immutable mixed60 inputs."""
import argparse
import faulthandler
import dataclasses
import hashlib
import gc
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import warp as wp
wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
from physmorph.compute import cuda_execution, to_array, to_host, KDTree, ndimage, warp_array
from physmorph.pipeline import PipelineConfig, run_pipeline
from physmorph.mpm.state import MPMParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/relcfd/chayo/physmorph_v2')
    parser.add_argument('--backend', default='cuda')
    parser.add_argument('--windows', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--primitives', action='store_true')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--archive', action='store_true', help='Save raw frames and canonical render metadata')
    parser.add_argument('--confirm', action='store_true', help='Existing start/end pin confirmation diagnostic')
    parser.add_argument('--body-rprop', action='store_true', help='Transit-protected displacement-mode RPROP')
    parser.add_argument('--taper-sp', type=float, default=None, help='Existing stress taper with the mixed body controller')
    parser.add_argument('--motion-accounting', action='store_true', help='Read-only accepted hybrid displacement decomposition')
    parser.add_argument('--no-commit-pic', action='store_true', help='Ablate the existing endpoint PIC correction only')
    parser.add_argument('--outer-render-committed', action='store_true', help='Fixed-target promoted-state outer render gate')
    parser.add_argument('--out', required=True)
    parser.add_argument('--trace_seconds', type=int, default=0)
    args = parser.parse_args()
    if args.trace_seconds:
        faulthandler.dump_traceback_later(args.trace_seconds, repeat=True)
    wp.init()
    if args.primitives:
        from scipy.spatial import cKDTree
        x = np.random.default_rng(8).normal(size=(131, 3)).astype(np.float32)
        expected = cKDTree(x).query(x, k=9)
        with cuda_execution('cuda:0'):
            cp_x = to_array(x)
            d, i = KDTree(cp_x).query(cp_x, k=9)
            np.testing.assert_allclose(to_host(d), expected[0], atol=1e-10)
            np.testing.assert_array_equal(to_host(i), expected[1])
            w = warp_array(cp_x, wp.vec3, 'cuda:0')
            saved = to_array(w, copy=True)
            wp.to_torch(w).zero_()
            np.testing.assert_array_equal(to_host(saved), x)
            lab, n = ndimage.label(to_array(np.eye(4)))
            assert n == 4
        print('CUDA primitives PASS', flush=True)
        return
    root = Path(args.root)
    prefix = root / 'output/c291/c291_bunny_mixed60'
    metadata = json.loads(prefix.with_suffix('.json').read_text())
    with np.load(str(prefix) + '_render_full_dt_iso_nn.npz') as data:
        src, tgt = data['src'], data['tgt']
    config = metadata['arms']['render_full_dt_iso_nn']['config']
    config = {k: v for k, v in config.items() if k in PipelineConfig.__dataclass_fields__}
    config.update(compute_backend=args.backend, stop_after_windows=args.windows, iters=args.iters,
                  target_reference=str(root / 'work/gpu_refactor/target_reference.npz'))
    if args.physical:
        config['lambda_auto'] = 0.
    if args.confirm:
        config['settle_pin_confirm'] = True
    if args.body_rprop:
        config['body_rprop'] = True
    if args.motion_accounting:
        config['motion_accounting'] = True
    if args.no_commit_pic:
        config['commit_pic'] = False
    if args.outer_render_committed:
        config['outer_render_committed'] = True
    if args.taper_sp is not None:
        if not np.isfinite(args.taper_sp) or args.taper_sp < 0:
            parser.error('--taper-sp must be finite and nonnegative')
        config['ctrl_taper_sp'] = args.taper_sp
    cfg = PipelineConfig(**config)
    prm = MPMParams(**metadata['provenance']['mpm'])
    start = time.monotonic()
    memory = []
    code_root = Path(__file__).resolve().parents[2]
    source_files = sorted((code_root / 'physmorph').rglob('*.py'))
    digest = hashlib.sha256(b''.join(p.relative_to(code_root).as_posix().encode() + b'\0' + p.read_bytes()
                                   for p in source_files)).hexdigest()
    for repeat in range(args.repeat):
        if repeat:
            del result
            gc.collect()
        result = run_pipeline(src, tgt, prm, PipelineConfig(**config))
        torch.cuda.synchronize()
        gc.collect()
        import cupy as cp
        memory.append(dict(repeat=repeat, torch_allocated=torch.cuda.memory_allocated(),
                           torch_reserved=torch.cuda.memory_reserved(), cupy_used=cp.get_default_memory_pool().used_bytes(),
                           cupy_reserved=cp.get_default_memory_pool().total_bytes()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    commits = [result['frames'][row['frame_end']-1] for row in result['history'] if row.get('frame_end')]
    np.savez_compressed(out.with_suffix('.npz'), final=result['frames'][-1], F=result['F_frames'][-1],
                        commits=np.stack(commits))
    record = dict(code_sha256=digest, memory=memory, seconds=time.monotonic()-start, config=config, mpm=dataclasses.asdict(prm),
                  history=result['history'], guards=result['guards'],
                  torch_peak_bytes=torch.cuda.max_memory_allocated())
    if args.archive:
        ids = sorted({0, len(result['frames'])-1} | {int(r['frame_end'])-1 for r in result['history'] if r.get('frame_end')})
        np.savez(str(out)+'_render_full_dt_iso_nn.npz', src=src, tgt=tgt, frames=np.stack(result['frames']),
                 deliver_n=result['deliver_n'], pinned=result['pinned'], pinned_at=result['pinned_at'],
                 Fp=result['Fp'], F_samples=np.stack([result['F_frames'][i] for i in ids]), F_sample_idx=ids)
        record['provenance'] = dict(source_archive=str(prefix), mpm=dataclasses.asdict(prm), code_hash=digest,
                                    compute_backend=args.backend)
        record['arms'] = {'render_full_dt_iso_nn': dict(config=config, history=result['history'],
                          guards=result['guards'], metrics={}, deliver_n=result['deliver_n'],
                          converged=result['converged'], truncation=result['truncation'], n_held=result['n_held'])}
    out.with_suffix('.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({'seconds': record['seconds'], 'guards': record['guards']}), flush=True)


if __name__ == '__main__':
    main()
