"""Diagnostic-only CUDA repeat study; retains the failed v1 strict gate.

The fixed 64-particle subset is a differentiation test, not a morph-quality run.
The source's dx/dt/material are retained; subset rest volumes are recomputed once.
All selection, neighborhood construction, dynamics and checks execute on CUDA.
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
from scripts.probes.geometric_rest import compare, all_checks, GRAD_ATOL, GRAD_RTOL

N, T, K, SEED = 64, 3, 8, 297
FD_ATOL, FD_RTOL = 3e-3, .025
FD_FACTORS = (1., 2.)
ADJOINT_REPEATS = 8
KINDS = ('path', 'first', 'merge', 'terminal', 'velocity')
V1_PROBE_SHA = '304e8dbb0d585a6f63e499b600b35e8678cb3afe3f7cb2b7a71b6347cba56eb4'


def vector_difference(a, b):
    """Descriptive full-vector checks; no replacement acceptance predicate."""
    aa, bb = a.detach().double().flatten(), b.detach().double().flatten()
    delta = aa-bb
    la, lb, ld = aa.norm(), bb.norm(), delta.norm()
    ra, rb, rd = la/aa.numel()**.5, lb/bb.numel()**.5, ld/aa.numel()**.5
    # Keep near-zero relative/direction readouts unresolved at the existing atol.
    resolved = bool(torch.minimum(ra, rb) > GRAD_ATOL)
    cosine = (torch.dot(aa, bb)/(la*lb)).clamp(-1., 1.) if resolved else None
    tolerance = GRAD_ATOL+GRAD_RTOL*bb.abs()
    bad = delta.abs() > tolerance
    worst = int((delta.abs()/tolerance).argmax())
    return dict(elements=aa.numel(), max_abs_error=float(delta.abs().max()),
        error_rms=float(rd), error_l2=float(ld), lhs_rms=float(ra), reference_rms=float(rb),
        relative_and_direction_resolved=resolved,
        relative_l2=float(ld/lb) if resolved else None,
        cosine=float(cosine) if resolved else None,
        angle_degrees=float(torch.acos(cosine)*180/torch.pi) if resolved else None,
        signed_parallel_error_fraction=float(torch.dot(delta, bb)/(lb*lb)) if resolved else None,
        strict_coordinate_pass=not bool(bad.any()), strict_coordinate_failures=int(bad.sum()),
        strict_max_tolerance_ratio=float((delta.abs()/tolerance).max()),
        strict_worst_index=worst, strict_worst_lhs=float(aa[worst]),
        strict_worst_reference=float(bb[worst]), strict_worst_allowance=float(tolerance[worst]))


def repeat_study(ref, got, a, b, loss, channels):
    """Calibrate consecutively first; held-out calls never enlarge that context."""
    saved = [tuple(value.detach().clone() for value in outputs) for outputs in (ref, got)]
    calibration = {}
    for kind in KINDS:
        ordinary, captured = [], []
        for _ in range(ADJOINT_REPEATS):
            ordinary.append(tuple(g.detach().clone() for g in torch.autograd.grad(
                loss(ref, kind), a, retain_graph=True)))
            captured.append(tuple(g.detach().clone() for g in torch.autograd.grad(
                loss(got, kind), b, retain_graph=True)))
        calibration[kind] = ordinary, captured
    report = dict(repeats=ADJOINT_REPEATS,
        scope='one ordinary and one captured forward; only adjoints repeat; implementations interleaved with the same seed kind consecutively',
        reduction_precision='float64 from owned float32 gradients, including diagnostic coordinate tolerance arithmetic; unchanged v1 strict checks remain separately recorded',
        interpretation='empirical sample variability and mean difference, not confidence bounds or proof of atomic causation; no relaxed acceptance',
        relative_resolution='both vector RMS > unchanged v1 gradient_atol; otherwise relative/direction is null',
        noise_signal_context='within-method max pair RMS divided by that method mean-gradient RMS; compare descriptively to unchanged v1 gradient_rtol',
        unchanged_gradient_atol=GRAD_ATOL, unchanged_gradient_rtol=GRAD_RTOL,
        calibration={}, held_out=[])
    means, contexts = {}, {}
    for kind, (ordinary, captured) in calibration.items():
        means[kind], contexts[kind], report['calibration'][kind] = {}, {}, {}
        for channel, name in enumerate(channels):
            rows = [torch.stack([grad[channel] for grad in samples]) for samples in (ordinary, captured)]
            averages = [row.double().mean(0) for row in rows]
            means[kind][name] = averages
            data = {}
            noise_context = []
            for label, row, average in zip(('ordinary', 'captured'), rows, averages):
                pairs = [dict(i=i, j=j, **vector_difference(row[i], row[j]))
                    for i in range(ADJOINT_REPEATS) for j in range(i)]
                pair_rms = max(p['error_rms'] for p in pairs)
                signal_rms = float(average.square().mean().sqrt())
                data[label] = dict(pairs=pairs, max_pair_error_rms=pair_rms,
                    max_pair_coordinate_error=max(p['max_abs_error'] for p in pairs),
                    mean_gradient_rms=signal_rms,
                    noise_to_mean_rms=pair_rms/signal_rms if signal_rms > GRAD_ATOL else None)
                noise_context.append(pair_rms)
            contexts[kind][name] = noise_context
            data['cross_pairs'] = [dict(ordinary=i, captured=j,
                **vector_difference(rows[1][j], rows[0][i]))
                for i in range(ADJOINT_REPEATS) for j in range(ADJOINT_REPEATS)]
            data['sample_mean_difference'] = vector_difference(averages[1], averages[0])
            report['calibration'][kind][name] = data
    # This ordered held-out cycle is never included in calibration statistics.
    for position, kind in enumerate((*KINDS, 'path')):
        ordinary = torch.autograd.grad(loss(ref, kind), a, retain_graph=True)
        captured = torch.autograd.grad(loss(got, kind), b, retain_graph=True)
        row = dict(position=position, kind=kind, channels={})
        for i, name in enumerate(channels):
            entry = dict(cross=vector_difference(captured[i], ordinary[i]))
            for j, (label, gradient) in enumerate(zip(('ordinary', 'captured'), (ordinary[i], captured[i]))):
                context = contexts[kind][name][j]
                deviation = vector_difference(gradient, means[kind][name][j])
                deviation['calibrated_max_pair_rms'] = context
                deviation['error_over_calibrated_max_pair_rms'] = (
                    deviation['error_rms']/context if context > 0 else None)
                deviation['outside_observed_pair_rms'] = deviation['error_rms'] > context
                entry[label+'_vs_calibrated_mean'] = deviation
            row['channels'][name] = entry
        report['held_out'].append(row)
    report['returned_forward_values_unchanged'] = all(torch.equal(old, new)
        for before, outputs in zip(saved, (ref, got)) for old, new in zip(before, outputs))
    return report


def run_probe(source, metadata, out, evidence):
    for path in (source, metadata, out, evidence):
        if not path.is_relative_to(Path('/data')):
            raise ValueError('All paths must resolve under /data')
    if out.exists():
        raise FileExistsError(out)
    evidence_bytes = evidence.read_bytes()
    original = json.loads(evidence_bytes)
    if original.get('passed') is not False:
        raise ValueError('Expected the preserved failed v1 evidence')
    old_code = original['code_sha256']
    if old_code.get('scripts/probes/position_sequence.py') != V1_PROBE_SHA:
        raise ValueError('Unexpected v1 probe identity')
    numerical = {p.relative_to(ROOT).as_posix() for p in (ROOT/'physmorph').rglob('*.py')}
    if numerical != {name for name in old_code if name.startswith('physmorph/')}:
        raise ValueError('Numerical source path set changed from v1')
    for name, expected in old_code.items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Frozen v1 dependency changed: {name}')
    caches = {name: os.environ.get(name) for name in
              ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH')}
    if any(not value or not Path(value).resolve().is_relative_to(Path('/data')) for value in caches.values()):
        raise ValueError('Warp/CuPy/CUDA caches must be explicit /data directories')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU fallback')
    import warp as wp
    wp.config.kernel_cache_dir = caches['WARP_CACHE_PATH']
    from physmorph.compute import cuda_execution, cuda_module, to_array
    from physmorph.mpm.function import (
        PersistentAdjoint, RolloutSpec, warp_mpm_ext,
        warp_mpm_ext_with_positions, warp_mpm_ext_with_previous,
    )
    from physmorph.mpm.state import MPMParams
    from physmorph.mpm.traj import compute_rest_volumes

    stat0 = source.stat()
    metadata_bytes = metadata.read_bytes()
    meta = json.loads(metadata_bytes)
    prov = meta.get('provenance', {})
    mpm = prov.get('mpm') or meta.get('mpm')
    if not mpm:
        raise ValueError('Original metadata must specify MPM discretisation')
    with host_np.load(source, allow_pickle=False) as archive:
        src = archive['src']
    source_sha = hashlib.sha256(src.tobytes()).hexdigest()
    if src.shape != (300000, 3) or src.dtype != host_np.float32 or source_sha != SOURCE_SHA:
        raise ValueError('Expected immutable original mixed60 float32 300k source')
    if source_sha != original['source_array_sha256'] or hashlib.sha256(metadata_bytes).hexdigest() != original['metadata_sha256']:
        raise ValueError('Source or original recipe metadata differs from v1')
    stat1 = source.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(stat0) != identity(stat1):
        raise RuntimeError('Source changed during member read')
    prm = MPMParams(**mpm)
    young, poisson = float(prov['young']), float(prov['poisson'])
    lam = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    code = sorted((ROOT/'physmorph').rglob('*.py')) + [Path(__file__).resolve(),
        ROOT/'scripts/probes/position_sequence.py', ROOT/'scripts/probes/geometric_rest.py', ROOT/'scripts/probes/pic_endpoint.py']
    result = dict(start_utc=datetime.now(timezone.utc).isoformat(), source=str(source),
        source_member='src only', source_array_sha256=source_sha,
        source_bytes=stat1.st_size, source_mtime_ns=stat1.st_mtime_ns,
        metadata=str(metadata), metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        code_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in code}, cache_directories=caches,
        full_source_N=300000, probe_N=N, probe_T=T, mpm=mpm, young=young, poisson=poisson,
        lam=lam, mu=mu, seed=SEED,
        subset_rule='stable CUDA argsort of distance squared to source[0], first64',
        rest_volumes='estimated once on CUDA for subset; shared by all bridges, not full-body source volumes',
        layer='all-particle +y normals; 8 nearest nonself neighbors; inverse-distance normalized weights; relaxation 1/T; u gate .2..1',
        T1_scope='API/adjoint edge case: body disabled, frozen T3 layer data retained (relaxation fraction 1/3); u fraction follows T1',
        initial_state='seeded nonzero v=.001 dx/dt, C=.001/dt, F=I+.001 noise, Fp=I+.0005 noise, eta=.1..0.3',
        controls='dFc .002; u and two body modes .001 dx; pins index%11==0, pin_mode0',
        tolerances=dict(forward_atol=3e-6, forward_rtol=3e-5,
                        gradient_atol=2e-6, gradient_rtol=2e-4,
                        forward_units='x/X divided by dx, v/V multiplied by dt/dx, F/Fg unchanged',
                        fd_atol=FD_ATOL, fd_rtol=FD_RTOL,
                        fd_eps_dfc=.002, fd_eps_u_body=.001*prm.dx,
                        fd_factors=list(FD_FACTORS),
                        fd_resolution='both eps: numerator >10x max(repeated plus/minus/base loss variation, float64 reduction allowance); |analytic| >10*fd_atol'),
        no_objective_change_or_physical_quality_claim=True)
    result['diagnostic_amendment'] = dict(version=2, original_evidence=str(evidence),
        original_evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        original_strict_passed=False, original_checks=original['checks'],
        protocol='calibrate eight consecutive same-kind adjoints per implementation on unchanged forward, then held-out path/first/merge/terminal/velocity/path cycle; all v1 strict gates retained',
        acceptance='diagnostic only; passed still means the original strict checks, never a noise-adjusted substitute')

    with cuda_execution('cuda'):
        cp = cuda_module()
        full = torch.as_tensor(src, device='cuda')
        ids = torch.argsort((full-full[0]).square().sum(1), stable=True)[:N]
        x0 = full[ids].clone()
        result['source_ids'] = ids.tolist()  # Small immutable provenance I/O.
        del full, src
        gen = torch.Generator(device=x0.device).manual_seed(SEED)
        def noise(*shape):
            return torch.randn(*shape, device=x0.device, generator=gen)
        vol0 = compute_rest_volumes(to_array(x0), 1., prm, 'cuda')
        pins = torch.arange(N, device=x0.device) % 11 == 0
        distance = torch.cdist(x0, x0)
        distance.fill_diagonal_(float('inf'))
        nbr = torch.argsort(distance, dim=1, stable=True)[:, :K]
        weights = distance.gather(1, nbr).clamp_min(torch.finfo(x0.dtype).eps*prm.dx).reciprocal()
        weights /= weights.sum(1, keepdim=True)
        normals = torch.zeros_like(x0); normals[:, 1] = 1
        ug = torch.linspace(.2, 1., N, device=x0.device)
        layer = (to_array(torch.ones(N, device=x0.device)), to_array(normals),
                 to_array(nbr.to(torch.int32)), to_array(weights), 1/T, None, 0., to_array(ug))
        eye = torch.eye(3, device=x0.device)[None].expand(N, -1, -1)
        spec = RolloutSpec(to_array(x0), 1., lam, mu, prm, T, device='cuda', vol0=vol0,
            pin=to_array(pins.float()), layer=layer, body_ctrl=True, body_modes=2,
            v0=to_array(.001*prm.dx/prm.dt*noise(N, 3)), C0=to_array(.001/prm.dt*noise(N, 3, 3)),
            F0=to_array(eye+.001*noise(N, 3, 3)), Fp=to_array(eye+.0005*noise(N, 3, 3)),
            Fg0=to_array(eye.clone()), eta=to_array(torch.linspace(.1, .3, N, device=x0.device)))
        base = (.002*noise(T, N, 3, 3), .001*prm.dx*noise(N), .001*prm.dx*noise(2*N, 3))
        path_weight, merge_weight = noise(T, N, 3), noise(4, N, 3)
        channels = ('dFc', 'u', 'body')
        def ordinary(leaves, current_spec=spec):
            return warp_mpm_ext_with_positions(leaves[0], current_spec, u_t=leaves[1], body_t=leaves[2])
        def loss(outputs, kind):
            if kind == 'path':
                return (((outputs[-1]-x0[None])/prm.dx).double()*path_weight).sum()
            if kind == 'first':
                return (((outputs[-1][0]-x0)/prm.dx).double()*path_weight[0]).sum()
            if kind == 'terminal':
                return (((outputs[0]-x0)/prm.dx).double()*path_weight[-1]).sum()
            if kind == 'velocity':
                return (outputs[4].double()*(prm.dt/prm.dx)).square().sum()
            if kind == 'merge':
                return (((outputs[0]-x0).double()*merge_weight[0] +
                         (outputs[-1][-1]-x0).double()*merge_weight[1])/prm.dx).sum() + (
                         (outputs[2].double()*merge_weight[2] + outputs[4][-1].double()*merge_weight[3])
                         *(prm.dt/prm.dx)).sum()
            raise ValueError(kind)

        def parity_case():
            a = tuple(t.clone().requires_grad_() for t in base)
            b = tuple(t.clone().requires_grad_() for t in base)
            ref = ordinary(a)
            adj = PersistentAdjoint(spec, position_sequence=True)
            got = adj.apply_with_positions(b[0], u_t=b[1], body_t=b[2])
            factors = (1/prm.dx, 1., prm.dt/prm.dx, 1., prm.dt/prm.dx, 1/prm.dx)
            checks = dict(capture=dict(passed=adj.g_fwd is not None and adj.g_bwd is not None),
                forward={name:compare(g*f, r*f) for name, g, r, f in zip(
                    ('xT', 'FT', 'vT', 'FgT', 'V', 'X'), got, ref, factors)}, gradients={})
            result['adjoint_noise_diagnostic'] = repeat_study(ref, got, a, b, loss, channels)
            first_path = None
            for iteration, kind in enumerate(('path', 'first', 'merge', 'terminal', 'velocity', 'path')):
                gr = torch.autograd.grad(loss(ref, kind), a, retain_graph=True)
                gg = torch.autograd.grad(loss(got, kind), b, retain_graph=True)
                entry = {name:compare(g, r, gradient=True) for name, g, r in zip(channels, gg, gr)}
                if kind == 'path' and first_path is None:
                    first_path = tuple(g.clone() for g in gg)
                    entry['signal'] = dict(passed=all(bool(g.abs().max() > 10*FD_ATOL) for g in gr),
                                          max_abs={name:float(g.abs().max()) for name, g in zip(channels, gr)})
                elif kind == 'path':
                    entry['repeat'] = {name:compare(g, old, gradient=True) for name, g, old in zip(channels, gg, first_path)}
                if kind == 'first':
                    entry['future_control_zero'] = dict(passed=bool(torch.count_nonzero(gg[0][1:]) == 0))
                checks['gradients'][f'{iteration}_{kind}'] = entry
                if kind == 'merge':
                    # Independent five-output oracle: combine duplicate seeds before the bridge.
                    leaf = tuple(t.clone().requires_grad_() for t in base)
                    old = warp_mpm_ext(leaf[0], spec, u_t=leaf[1], body_t=leaf[2])
                    merged = (((old[0]-x0).double()*(merge_weight[0]+merge_weight[1]))/prm.dx).sum()
                    merged += (old[2].double()*(merge_weight[2]+merge_weight[3])*(prm.dt/prm.dx)).sum()
                    oracle = torch.autograd.grad(merged, leaf)
                    entry['merged_seed_oracle'] = {name:compare(g, r, gradient=True) for name, g, r in zip(channels, gg, oracle)}
            pinned_grads = torch.autograd.grad(got[-1][:, pins].sum(), b, retain_graph=True)
            checks['pins'] = dict(passed=bool(torch.equal(got[-1][:, pins], x0[pins].expand(T, -1, -1))
                and torch.count_nonzero(got[2][pins]) == 0 and all(torch.count_nonzero(g) == 0 for g in pinned_grads)))
            delta = got[-1]-torch.cat((x0[None], got[-1][:-1]), 0)
            u_step = (ug*base[1]*(~pins)/T)[:, None]*normals
            relaxation = delta-prm.dt*got[4]-u_step[None]
            checks['nontrivial_relaxation'] = dict(passed=bool(relaxation.abs().max() > 1e-5*prm.dx),
                                                   max_abs_wu=float(relaxation.abs().max()))
            previous = warp_mpm_ext_with_previous(base[0], spec, u_t=base[1], body_t=base[2])
            checks['previous_api'] = compare(got[-1][-2]/prm.dx, previous[-1]/prm.dx)

            checks['finite_difference'] = {}
            with torch.no_grad():
                baseline = [loss(ordinary(base), 'path') for _ in range(2)]
            for index, name in enumerate(channels):
                gradient = first_path[index]
                flat = int(gradient.abs().argmax())
                an = gradient.reshape(-1)[flat]
                entries = []
                for factor in FD_FACTORS:
                    eps = factor*(.002 if index == 0 else .001*prm.dx)
                    plus, minus = [t.clone() for t in base], [t.clone() for t in base]
                    plus[index].reshape(-1)[flat] += eps
                    minus[index].reshape(-1)[flat] -= eps
                    with torch.no_grad():
                        fp = [loss(ordinary(plus), 'path') for _ in range(2)]
                        fm = [loss(ordinary(minus), 'path') for _ in range(2)]
                    numerator = (fp[0]+fp[1]-fm[0]-fm[1])/2
                    fd = numerator/(2*eps)
                    noise_floor = torch.stack(((baseline[0]-baseline[1]).abs(),
                        (fp[0]-fp[1]).abs(), (fm[0]-fm[1]).abs(),
                        64*torch.finfo(torch.float64).eps*torch.stack(fp+fm+baseline).abs().max())).max()
                    allowed = FD_ATOL+FD_RTOL*fd.abs()
                    resolved = bool((numerator.abs() > 10*noise_floor) & (an.abs() > 10*FD_ATOL))
                    entries.append(dict(eps=eps, flat_index=flat, analytic=float(an), finite_difference=float(fd),
                        numerator=float(numerator), numerical_loss_floor=float(noise_floor), resolved=resolved,
                        abs_error=float((an-fd).abs()), allowed_error=float(allowed),
                        passed=resolved and bool((an-fd).abs() <= allowed)))
                checks['finite_difference'][name] = entries

            # A prepared sequence object can still use legacy apply; its sX must clear.
            saved = tuple(output.detach().clone() for output in got)
            changed = tuple((1.2*t).clone().requires_grad_() for t in base)
            legacy = adj.apply(changed[0], u_t=changed[1], body_t=changed[2])
            gl = torch.autograd.grad(legacy[0].square().sum(), changed)
            c = tuple(t.detach().clone().requires_grad_() for t in changed)
            fresh = warp_mpm_ext(c[0], spec, u_t=c[1], body_t=c[2])
            gc = torch.autograd.grad(fresh[0].square().sum(), c)
            checks['legacy_seed_reset'] = dict(cleared=dict(passed=bool(torch.count_nonzero(adj.sX) == 0)),
                gradients={name:compare(g, r, gradient=True) for name, g, r in zip(channels, gl, gc)})
            stale = False
            try:
                torch.autograd.grad(loss(got, 'path'), b)
            except RuntimeError as error:
                if 'stale persistent forward' not in str(error):
                    raise
                stale = True
            checks['stale_and_ownership'] = dict(
                changed_forward_max_wu=float((legacy[0]-saved[0]).abs().max()),
                passed=stale and bool(torch.count_nonzero(legacy[0]-saved[0]) > 0) and all(
                    torch.equal(before, after) for before, after in zip(saved, got)))
            return checks

        result['checks'], result['timing'] = cuda_timed(parity_case)
        one = replace(spec, T=1, body_ctrl=False, body_modes=1)
        a1, b1 = base[0][:1].clone().requires_grad_(), base[0][:1].clone().requires_grad_()
        r1 = warp_mpm_ext_with_positions(a1, one, u_t=base[1])
        adj1 = PersistentAdjoint(one, position_sequence=True)
        p1 = adj1.apply_with_positions(b1, u_t=base[1])
        ga, = torch.autograd.grad((r1[0].double()+r1[-1][0].double()).square().sum(), a1)
        gb, = torch.autograd.grad((p1[0].double()+p1[-1][0].double()).square().sum(), b1)
        c1 = base[0][:1].clone().requires_grad_()
        old1 = warp_mpm_ext(c1, one, u_t=base[1])
        gc1, = torch.autograd.grad((2*old1[0].double()).square().sum(), c1)
        result['checks']['T1'] = dict(
            exact=dict(passed=r1[-1].requires_grad and p1[-1].requires_grad
                       and torch.equal(r1[-1][0], r1[0]) and torch.equal(p1[-1][0], p1[0])),
            capture=dict(passed=adj1.g_fwd is not None and adj1.g_bwd is not None),
            forward=compare(p1[-1]/prm.dx, r1[-1]/prm.dx), gradient=compare(gb, ga, gradient=True),
            merged_seed_oracle=compare(gb, gc1, gradient=True))
        result['device'] = dict(name=torch.cuda.get_device_name(), torch=torch.__version__,
            cuda=torch.version.cuda, cupy=cp.__version__, warp=wp.__version__,
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        result['memory_scope'] = 'timing helper reports Torch allocator, not total device peak'
        result['passed'] = bool(all_checks(result['checks']))
    result['end_utc'] = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(out=str(out), passed=result['passed'])), flush=True)
    if not result['passed']:
        raise RuntimeError('Position-sequence CUDA gate failed; see JSON')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-evidence', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    run_probe(source, (args.metadata or metadata_sibling(source)).resolve(), args.out.resolve(), args.baseline_evidence.resolve())


if __name__ == '__main__':
    main()
