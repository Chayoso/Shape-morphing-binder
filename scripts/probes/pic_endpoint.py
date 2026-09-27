"""Hyde06-only numerical/timing probe of the shared XPIC endpoint, no simulation.

Reads only the original mixed60 NPZ's src member. Numerical comparisons,
random fields, reductions and finite differences run on CUDA. Host operations
are input/configuration I/O, byte hashes, clocks and scalar JSON telemetry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as host_np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SOURCE_SHA = '71eb14d38c2efb41379a12b0ce017430e093883cbce51188265ed9e2948d0e34'
FORWARD_ATOL = 3e-6
FORWARD_RTOL = 3e-5
DOT_NORM_ATOL = 2e-6
FD_ATOL = 1e-7
FD_RTOL = 1e-6
FD_EPS = 1e-5
SEED = 292
REPEATS = 3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cuda_timed(call):
    """Whole-call wall time after synchronizing; Torch allocated memory only."""
    torch.cuda.synchronize()
    gc.collect()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return value, dict(seconds=seconds, torch_allocated_before_bytes=before,
                       torch_allocated_after_bytes=torch.cuda.memory_allocated(),
                       torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       torch_peak_increment_bytes=torch.cuda.max_memory_allocated()-before,
                       torch_reserved_bytes=torch.cuda.memory_reserved())


def close(a, b, atol=FORWARD_ATOL, rtol=FORWARD_RTOL):
    delta = (a-b).abs()
    allowed = atol + rtol*b.abs()
    return dict(passed=bool((delta <= allowed).all()), max_abs_error=float(delta.max()),
                rms_error=float(delta.double().square().mean().sqrt()),
                max_tolerance_ratio=float((delta/allowed).max()))


def metadata_sibling(source):
    suffix = '_render_full_dt_iso_nn.npz'
    if source.name.endswith(suffix):
        return source.with_name(source.name[:-len(suffix)] + '.json')
    return source.with_suffix('.json')


def run_probe(source, metadata, out):
    for path in (source, metadata, out):
        if not path.is_relative_to(Path('/data')):
            raise ValueError('All probe input/output paths must resolve under /data')
    if out.exists():
        raise FileExistsError(out)
    caches = {name: os.environ.get(name) for name in
              ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH')}
    for name, cache in caches.items():
        if not cache or not Path(cache).resolve().is_relative_to(Path('/data')):
            raise ValueError(f'{name} must be an explicit /data directory')
    if not torch.cuda.is_available():
        raise RuntimeError('This probe requires CUDA; no CPU fallback')
    import warp as wp
    wp.config.kernel_cache_dir = caches['WARP_CACHE_PATH']
    from physmorph.compute import cuda_execution, cuda_module
    from physmorph.mpm.endpoint_filter import FixedEndpointFilter
    from physmorph.mpm.gridfilter import grid_project

    before = source.stat()
    metadata_bytes = metadata.read_bytes()
    meta = json.loads(metadata_bytes)
    mpm = meta.get('provenance', {}).get('mpm') or meta.get('mpm')
    if not mpm:
        raise ValueError('Source metadata lacks MPM discretisation')
    with host_np.load(source) as archive:
        src_host = archive['src']
    if src_host.shape != (300000, 3) or src_host.dtype != host_np.float32:
        raise ValueError('Expected original mixed60 float32 300000x3 source')
    source_sha = hashlib.sha256(src_host.tobytes()).hexdigest()
    if source_sha != SOURCE_SHA:
        raise ValueError('Source particles differ from the immutable original mixed60 input')
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source archive changed while reading')
    dx, origin = float(mpm['dx']), mpm['grid_min']
    dims = tuple(int(mpm[k]) for k in ('nx', 'ny', 'nz'))
    result = dict(start_utc=datetime.now(timezone.utc).isoformat(), source=str(source),
                  source_member='src only; no trajectory/target member loaded',
                  source_array_sha256=source_sha, source_bytes=after.st_size,
                  source_mtime_ns=after.st_mtime_ns, metadata=str(metadata),
                  metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                  cache_directories=caches, mpm=mpm, N=len(src_host), order=5,
                  seed=SEED, repeats=REPEATS, dtype='float32', masses='equal unit masses',
                  tolerances=dict(forward_atol=FORWARD_ATOL, forward_rtol=FORWARD_RTOL,
                                  dot_norm_atol=DOT_NORM_ATOL, fd_atol=FD_ATOL,
                                  fd_rtol=FD_RTOL, fd_epsilon=FD_EPS),
                  code_sha256={str(path.relative_to(ROOT)):sha(path) for path in (
                      Path(__file__).resolve(), ROOT/'physmorph/mpm/endpoint_filter.py',
                      ROOT/'physmorph/mpm/gridfilter.py', ROOT/'physmorph/compute.py')},
                  timing_scope='legacy: preparation+filter+legacy telemetry+CuPy result copy; new: preparation separate from reusable apply; not identical API work',
                  memory_scope='Torch allocated/reserved only; CuPy allocator bytes separately sampled, not a total device peak',
                  no_physical_quality_claim=True)

    with cuda_execution('cuda'):
        cp = cuda_module()
        x0 = torch.as_tensor(src_host, device='cuda').clone()
        del src_host
        generator = torch.Generator(device=x0.device).manual_seed(SEED)
        d = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        g = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        torch.cuda.synchronize()
        result['device'] = dict(name=torch.cuda.get_device_name(), torch=torch.__version__,
                                cuda=torch.version.cuda, cupy=cp.__version__, warp=wp.__version__,
                                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        result['legacy_calls'] = []
        legacy_reference = None
        for repeat in range(REPEATS):
            (legacy_array, stats), timing = cuda_timed(
                lambda: grid_project(d, x0, dx, origin, dims, device='cuda', order=5))
            legacy = torch.as_tensor(legacy_array, device=x0.device)
            if legacy_reference is None:
                legacy_reference = legacy.clone()
            else:
                timing['versus_first_call'] = close(legacy, legacy_reference)
            timing['legacy_stats'] = stats
            timing['cupy_pool_used_bytes'] = cp.get_default_memory_pool().used_bytes()
            timing['cupy_pool_total_bytes'] = cp.get_default_memory_pool().total_bytes()
            result['legacy_calls'].append(timing)
            del legacy, legacy_array

        prepared, result['new_preparation'] = cuda_timed(
            lambda: FixedEndpointFilter(x0, dx, origin, dims, order=5))
        filtered, result['new_first_apply'] = cuda_timed(lambda: prepared.apply_H(d))
        result['forward'] = close(filtered, legacy_reference)
        result['new_repeated_apply'] = []
        for _ in range(REPEATS):
            repeated, timing = cuda_timed(lambda: prepared.apply_H(d))
            timing['versus_first_call'] = close(repeated, filtered)
            del repeated
            torch.cuda.synchronize()
            timing['torch_allocated_after_release_bytes'] = torch.cuda.memory_allocated()
            result['new_repeated_apply'].append(timing)

        transpose, result['new_transpose_apply'] = cuda_timed(lambda: prepared.apply_HT(g))
        # FP32 operator evaluation, FP64 dot products and norm reductions on GPU.
        lhs = (filtered.double()*g.double()).sum()
        rhs = (d.double()*transpose.double()).sum()
        norm_product = d.double().norm()*g.double().norm()
        dot_error = (lhs-rhs).abs()/norm_product.clamp_min(1e-30)
        result['dot_product'] = dict(lhs=float(lhs), rhs=float(rhs),
                                   norm_product=float(norm_product), normalized_error=float(dot_error),
                                   passed=bool(dot_error <= DOT_NORM_ATOL))
        pins = torch.arange(len(x0), device=x0.device) % 17 == 0
        raw = x0 + d*(.01*dx)
        endpoint, result['new_endpoint_apply'] = cuda_timed(lambda: prepared.endpoint(raw, pins))
        result['pins'] = dict(count=int(pins.sum()),
                              exact_bits=bool(torch.equal(endpoint[pins].view(torch.int32),
                                                          x0[pins].view(torch.int32))),
                              max_displacement=float((endpoint[pins]-x0[pins]).abs().max()))

        # Nearby actual source IDs provide a nontrivial coupled stencil. A small
        # nonuniform-mass double test independently differentiates a scalar loss
        # by finite differences; it does not call apply_HT to form the oracle.
        ids = (x0-x0[0]).square().sum(1).topk(12, largest=False).indices
        small_x = x0[ids].double()
        small_m = torch.linspace(.4, 2.1, len(ids), device=x0.device, dtype=torch.float64)
        small = FixedEndpointFilter(small_x, dx, origin, dims, m=small_m, order=5)
        small_pins = torch.arange(len(ids), device=x0.device) % 3 == 0
        raw_small = (small_x + .03*dx*torch.randn(small_x.shape, dtype=torch.float64,
                                                device=x0.device, generator=generator)).requires_grad_()
        direction = torch.randn(small_x.shape, dtype=torch.float64, device=x0.device, generator=generator)
        direction = direction/direction.norm()
        test_covector = torch.randn(small_x.shape, dtype=torch.float64, device=x0.device, generator=generator)
        loss = (small.endpoint(raw_small, small_pins)*test_covector).sum()
        grad, = torch.autograd.grad(loss, raw_small)
        analytic = (grad*direction).sum()
        with torch.no_grad():
            plus = (small.endpoint(raw_small+FD_EPS*direction, small_pins)*test_covector).sum()
            minus = (small.endpoint(raw_small-FD_EPS*direction, small_pins)*test_covector).sum()
            finite_difference = (plus-minus)/(2*FD_EPS)
        error = (analytic-finite_difference).abs()
        fd_tolerance = FD_ATOL+FD_RTOL*finite_difference.abs()
        result['double_directional_finite_difference'] = dict(
            N=len(ids), source_selection='12 nearest original source IDs to source[0], GPU topk',
            pinned=int(small_pins.sum()), mass_range=[.4,2.1],
            analytic=float(analytic), finite_difference=float(finite_difference),
            abs_error=float(error), allowed_error=float(fd_tolerance),
            passed=bool(error <= fd_tolerance))

        repeats_pass = all(r['versus_first_call']['passed'] for r in result['new_repeated_apply'])
        legacy_repeats_pass = all(r.get('versus_first_call', {'passed':True})['passed']
                                  for r in result['legacy_calls'])
        result['passed'] = bool(result['forward']['passed'] and result['dot_product']['passed']
                                and result['pins']['exact_bits']
                                and result['double_directional_finite_difference']['passed']
                                and repeats_pass and legacy_repeats_pass)
    result['end_utc'] = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(out=str(out), passed=result['passed'])), flush=True)
    if not result['passed']:
        raise RuntimeError('Endpoint GPU numerical gate failed; see JSON')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    metadata = (args.metadata or metadata_sibling(source)).resolve()
    run_probe(source, metadata, args.out.resolve())


if __name__ == '__main__':
    main()
