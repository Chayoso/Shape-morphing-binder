"""Hyde06 CUDA probe of previous-position bridges; no optimization or rest weight.

The immutable mixed60 source supplies a fixed 64-particle subset. CUDA evaluates
all selection, volume estimation, dynamics, derivatives and numerical checks.
The subset/material controls are a bridge test, not a physical-quality experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as host_np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.probes.pic_endpoint import SOURCE_SHA, metadata_sibling, sha, cuda_timed

N = 64
T = 3
SEED = 293
FORWARD_ATOL, FORWARD_RTOL = 3e-6, 3e-5
GRAD_ATOL, GRAD_RTOL = 2e-6, 2e-4
FD_ATOL, FD_RTOL = 3e-3, .025


def compare(a, b, *, gradient=False):
    atol, rtol = (GRAD_ATOL, GRAD_RTOL) if gradient else (FORWARD_ATOL, FORWARD_RTOL)
    error = (a-b).abs()
    allowed = atol + rtol*b.abs()
    return dict(passed=bool((error <= allowed).all()), max_abs_error=float(error.max()),
                reference_max=float(b.abs().max()),
                max_tolerance_ratio=float((error/allowed).max()))


def all_checks(value):
    if isinstance(value, dict):
        return all(all_checks(v) for k, v in value.items() if k != 'passed') and value.get('passed', True)
    if isinstance(value, list):
        return all(all_checks(v) for v in value)
    return True


def run_probe(source, metadata, out):
    for path in (source, metadata, out):
        if not path.is_relative_to(Path('/data')):
            raise ValueError('All paths must resolve under /data')
    if out.exists():
        raise FileExistsError(out)
    caches = {name: os.environ.get(name) for name in
              ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH')}
    for name, value in caches.items():
        if not value or not Path(value).resolve().is_relative_to(Path('/data')):
            raise ValueError(f'{name} must be an explicit /data directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU fallback')
    import warp as wp
    wp.config.kernel_cache_dir = caches['WARP_CACHE_PATH']
    from physmorph.compute import cuda_execution, cuda_module, to_array
    from physmorph.mpm.function import PersistentAdjoint, RolloutSpec, warp_mpm_ext_with_previous
    from physmorph.mpm.state import MPMParams
    from physmorph.mpm.traj import compute_rest_volumes

    stat0 = source.stat()
    metadata_bytes = metadata.read_bytes()
    meta = json.loads(metadata_bytes)
    prov = meta.get('provenance', {})
    mpm = prov.get('mpm') or meta.get('mpm')
    if not mpm:
        raise ValueError('Original metadata must specify MPM discretisation')
    with host_np.load(source) as archive:
        src = archive['src']
    source_sha = hashlib.sha256(src.tobytes()).hexdigest()
    if src.shape != (300000, 3) or src.dtype != host_np.float32 or source_sha != SOURCE_SHA:
        raise ValueError('Expected immutable original mixed60 float32 300k source')
    stat1 = source.stat()
    if (stat0.st_dev, stat0.st_ino, stat0.st_size, stat0.st_mtime_ns) != (
            stat1.st_dev, stat1.st_ino, stat1.st_size, stat1.st_mtime_ns):
        raise RuntimeError('Source changed during member read')
    prm = MPMParams(**mpm)
    young, poisson = float(prov['young']), float(prov['poisson'])
    lam = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    result = dict(start_utc=datetime.now(timezone.utc).isoformat(), source=str(source),
                  source_member='src only', source_array_sha256=source_sha,
                  source_bytes=stat1.st_size, source_mtime_ns=stat1.st_mtime_ns,
                  metadata=str(metadata), metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                  cache_directories=caches, full_source_N=len(src), probe_N=N, probe_T=T,
                  mpm=mpm, lam=lam, mu=mu, seed=SEED,
                  subset_rule='stable CUDA argsort of squared distance to original source[0], first64',
                  rest_volumes='estimated once on CUDA for this subset, shared by both bridges; not original full-body volumes',
                  controls='seeded dFc .002; layer u .001*dx, self-neighbor normal+y, relaxation0; two body modes .001*dx; pins index%11==0',
                  tolerances=dict(forward_atol=FORWARD_ATOL, forward_rtol=FORWARD_RTOL,
                                  gradient_atol=GRAD_ATOL, gradient_rtol=GRAD_RTOL,
                                  fd_atol=FD_ATOL, fd_rtol=FD_RTOL, fd_eps_wu=.001*prm.dx),
                  code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (
                      Path(__file__).resolve(), ROOT/'scripts/probes/pic_endpoint.py',
                      ROOT/'physmorph/mpm/function.py', ROOT/'physmorph/mpm/traj.py',
                      ROOT/'physmorph/mpm/kernels.py', ROOT/'physmorph/mpm/state.py',
                      ROOT/'physmorph/mpm/step.py', ROOT/'physmorph/mpm/constitutive.py',
                      ROOT/'physmorph/pipeline/body_control.py', ROOT/'physmorph/compute.py')},
                  no_rest_objective_or_physical_quality_claim=True)

    with cuda_execution('cuda'):
        cp = cuda_module()
        full = torch.as_tensor(src, device='cuda')
        ids = torch.argsort((full-full[0]).square().sum(1), stable=True)[:N]
        x0 = full[ids].clone()
        result['source_ids'] = ids.tolist()  # Small immutable provenance I/O only.
        del full, src
        gen = torch.Generator(device=x0.device).manual_seed(SEED)
        vol0 = compute_rest_volumes(to_array(x0), 1., prm, 'cuda')
        pins = torch.arange(N, device=x0.device) % 11 == 0
        normals = torch.zeros_like(x0); normals[:, 1] = 1
        layer = (to_array(torch.ones(N, device=x0.device)), to_array(normals),
                 to_array(torch.arange(N, device=x0.device, dtype=torch.int32)[:, None]),
                 to_array(torch.ones(N, 1, device=x0.device)), 0.)
        spec = RolloutSpec(to_array(x0), 1., lam, mu, prm, T, device='cuda', vol0=vol0,
                           pin=to_array(pins.float()), layer=layer, body_ctrl=True, body_modes=2)
        base = (.002*torch.randn(T, N, 3, 3, device=x0.device, generator=gen),
                .001*prm.dx*torch.randn(N, device=x0.device, generator=gen),
                .001*prm.dx*torch.randn(2*N, 3, device=x0.device, generator=gen))
        weight = torch.randn(N, 3, device=x0.device, generator=gen)
        def ordinary(leaves, current_spec=spec):
            return warp_mpm_ext_with_previous(leaves[0], current_spec,
                                             u_t=leaves[1], body_t=leaves[2])
        def loss(outputs, kind):
            if kind == 'previous':
                return ((outputs[-1]-x0)*weight).sum()/prm.dx
            if kind == 'step':
                return ((outputs[0]-outputs[-1])/prm.dx).square().sum()
            if kind == 'terminal':
                return ((outputs[0]-x0)*weight).sum()/prm.dx
            return (outputs[4]*(prm.dt/prm.dx)).square().sum()

        def parity_case():
            a = tuple(t.clone().requires_grad_() for t in base)
            b = tuple(t.clone().requires_grad_() for t in base)
            ref = ordinary(a)
            adj = PersistentAdjoint(spec, previous_position=True)
            got = adj.apply_with_previous(b[0], u_t=b[1], body_t=b[2])
            checks = dict(captured_forward=adj.g_fwd is not None, captured_backward=adj.g_bwd is not None,
                          forward={name:compare(g, r) for name, g, r in zip(
                              ('xT', 'FT', 'vT', 'FgT', 'V', 'x_previous'), got, ref)}, gradients={})
            checks['capture'] = dict(passed=checks['captured_forward'] and checks['captured_backward'])
            first_previous = None
            for iteration, kind in enumerate(('previous', 'step', 'terminal', 'velocity', 'previous')):
                gr = torch.autograd.grad(loss(ref, kind), a, retain_graph=True)
                gg = torch.autograd.grad(loss(got, kind), b, retain_graph=True)
                entry = {name:compare(g, r, gradient=True) for name, g, r in zip(('dFc', 'u', 'body'), gg, gr)}
                if kind == 'previous' and first_previous is None:
                    first_previous = tuple(g.clone() for g in gg)
                    entry['reference_signal'] = dict(
                        passed=all(bool(g.abs().max() > 1e-9) for g in gr),
                        max_abs={name:float(g.abs().max()) for name, g in zip(('dFc', 'u', 'body'), gr)})
                    entry['last_control_zero'] = dict(passed=bool(torch.count_nonzero(gg[0][-1]) == 0),
                                                     max_abs=float(gg[0][-1].abs().max()))
                elif kind == 'previous':
                    entry['repeat'] = {name:compare(g, old, gradient=True) for name, g, old in
                                       zip(('dFc', 'u', 'body'), gg, first_previous)}
                checks['gradients'][f'{iteration}_{kind}'] = entry
            checks['pins'] = dict(passed=bool(torch.equal(got[-1][pins], x0[pins])
                                              and torch.equal(got[0][pins], x0[pins])
                                              and torch.count_nonzero(got[2][pins]) == 0))
            # Independent FD uses ordinary forward evaluations and the captured
            # previous-output derivative; no manually constructed seed oracle.
            checks['finite_difference'] = {}
            for index, name in ((1, 'u'), (2, 'body')):
                gradient = first_previous[index]
                flat = int(gradient.abs().argmax())
                plus, minus = [t.clone() for t in base], [t.clone() for t in base]
                eps = .001*prm.dx
                plus[index].reshape(-1)[flat] += eps
                minus[index].reshape(-1)[flat] -= eps
                with torch.no_grad():
                    fd = (loss(ordinary(plus), 'previous')-loss(ordinary(minus), 'previous'))/(2*eps)
                an = gradient.reshape(-1)[flat]
                allowed = FD_ATOL+FD_RTOL*fd.abs()
                checks['finite_difference'][name] = dict(
                    flat_index=flat, analytic=float(an), finite_difference=float(fd),
                    abs_error=float((an-fd).abs()), allowed_error=float(allowed),
                    passed=bool((an.abs() > 10*FD_ATOL) & ((an-fd).abs() <= allowed)))
            # Replay after all old-context gradients, then reject their reuse.
            saved_previous = got[-1].detach().clone()
            adj.apply_with_previous(b[0]*.7, u_t=b[1], body_t=b[2])
            stale_rejected = False
            try:
                torch.autograd.grad(loss(got, 'previous'), b)
            except RuntimeError as error:
                if 'stale persistent forward' not in str(error):
                    raise
                stale_rejected = True
            checks['stale_context'] = dict(passed=stale_rejected and torch.equal(got[-1], saved_previous))
            return checks

        result['checks'], result['parity_timing'] = cuda_timed(parity_case)
        # T=1 constant previous position, captured path, no unsupported body pulse.
        one = replace(spec, T=1, body_ctrl=False, body_modes=1, layer=None)
        d1 = base[0][:1].clone().requires_grad_()
        r1 = warp_mpm_ext_with_previous(d1, one)
        adj1 = PersistentAdjoint(one, previous_position=True)
        p1 = adj1.apply_with_previous(d1)
        ga, = torch.autograd.grad(((r1[0]-r1[-1])/prm.dx).square().sum(), d1)
        gb, = torch.autograd.grad(((p1[0]-p1[-1])/prm.dx).square().sum(), d1)
        result['checks']['T1'] = dict(
            constant=dict(passed=not p1[-1].requires_grad and not r1[-1].requires_grad
                          and torch.equal(p1[-1], x0) and torch.equal(r1[-1], x0)),
            forward=compare(p1[0], r1[0]), gradient=compare(gb, ga, gradient=True))
        result['device'] = dict(name=torch.cuda.get_device_name(), torch=torch.__version__,
                                cuda=torch.version.cuda, cupy=cp.__version__, warp=wp.__version__,
                                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        result['memory_scope'] = 'timing helper reports Torch allocator only, not total device peak'
        result['passed'] = bool(all_checks(result['checks']))
    result['end_utc'] = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(out=str(out), passed=result['passed'])), flush=True)
    if not result['passed']:
        raise RuntimeError('Previous-position CUDA gate failed; see JSON')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    run_probe(source, (args.metadata or metadata_sibling(source)).resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
