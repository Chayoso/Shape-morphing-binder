"""P335 failure localization using frozen producer code and archived inputs.

CUDA only; no optimizer, candidate selection, commit, tolerance change or
full-handoff derivative claim. Synthetic boundary swaps are diagnostics only.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'File changed while hashing: ' + str(path))
    return dict(bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=digest.hexdigest())


def write_json(path, value):
    def finite_json(item):
        if isinstance(item, dict):
            return {key: finite_json(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite_json(value) for value in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    payload = json.dumps(finite_json(value), indent=2, allow_nan=False).encode()
    require(len(payload) <= 16_000_000, 'Diagnostic JSON exceeds bounded output allowance')
    with path.open('xb') as stream:
        stream.write(payload)


def compare(actual, reference, unit, cohorts, top=8):
    """Numerics remain on device; only bounded scalar/ID evidence is copied."""
    import torch
    require(actual.shape == reference.shape and actual.dtype == reference.dtype
            and actual.device == reference.device, 'Comparison layout mismatch')
    require(actual.dtype == torch.float32, 'The frozen gate is FP32')
    a, b = actual.detach().reshape(len(actual), -1), reference.detach().reshape(len(reference), -1)
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    if not finite:
        return dict(passed=False, finite=False)
    error = (a-b).abs()
    ratio = error/(32*torch.finfo(b.dtype).eps*(unit+b.abs()))
    per_id, component = ratio.max(dim=1)
    ids = torch.topk(per_id, min(top, len(per_id))).indices
    rows = []
    for particle in ids.tolist():
        k = int(component[particle])
        rows.append(dict(id=particle, component=k, actual=float(a[particle, k]),
            reference=float(b[particle, k]), ratio=float(per_id[particle]),
            cohort=next(name for name, mask in cohorts.items() if bool(mask[particle]))))
    return dict(passed=bool((ratio <= 1).all()), finite=True, bit_exact=torch.equal(a, b),
        native_scale=unit, max_abs=float(error.max()), max_tolerance_ratio=float(ratio.max()),
        rms=float((a.double()-b.double()).square().mean().sqrt()),
        failing_particles=int((per_id > 1).sum()), worst=rows,
        cohorts={name: dict(count=int(mask.sum()), failing=int(((per_id > 1) & mask).sum()),
            max_ratio=float(per_id[mask].max()) if bool(mask.any()) else None)
            for name, mask in cohorts.items()})


def phase_comparisons(actual, reference, units, cohorts):
    require(all(len(actual[key]) == len(reference[key]) for key in units),
            'Phase count mismatch')
    return {key: [compare(a, b, units[key], cohorts) for a, b in zip(actual[key], reference[key])]
            for key in units}


def tensor_hash(value):
    """Full numerical output binding; host transfer is archival I/O only."""
    array = value.detach().contiguous().cpu().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest())


def sequence_hashes(values):
    return {key: [tensor_hash(value) for value in phases] for key, phases in values.items()}


def solver_library_bindings():
    """Loaded Linux library bytes, not a third-party implementation audit."""
    maps = Path('/proc/self/maps')
    if not maps.is_file():
        return {}
    paths = {Path(line.split()[-1]) for line in maps.read_text().splitlines()
             if 'libcusolver' in line and line.split()[-1].startswith('/')}
    return {str(path): identity(path) for path in sorted(paths)}


def ordinary_handoff(F, Fp, old, new, cfg):
    from physmorph.compute import to_array
    from physmorph.plasticity import assimilate_elastic
    F, Fp = to_array(F), to_array(Fp)
    old, new = to_array(old), to_array(new)
    result = assimilate_elastic(F, Fp, eta=cfg.assim, smin=cfg.assim_smin,
                                smax=cfg.assim_smax, isochoric=cfg.assim_iso)
    if cfg.settle_pin_assim:
        result[old] = Fp[old]
        if bool(new.any()):
            # Preserve actual subset size and therefore production SVD dispatch.
            result[new] = assimilate_elastic(F[new], result[new], eta=1.,
                smin=cfg.assim_smin, smax=cfg.assim_smax, isochoric=False)
    return result


def fp64_handoff(F, Fp, old, new, cfg):
    """Precision counterfactual: each call computes double and stores FP32.

    Independent forward expression, using the frozen svd3 implementation. This
    is neither an alternate production implementation nor a new derivative.
    """
    import torch
    from physmorph.plasticity.cuda_svd import svd3
    require(F.shape == Fp.shape and F.ndim == 3 and F.shape[1:] == (3, 3)
            and F.device == Fp.device, 'FP64 handoff requires matching state tensors')
    require(all(mask.dtype == torch.bool and mask.shape == F.shape[:1] and mask.device == F.device
                for mask in (old, new)) and not bool((old & new).any()), 'Invalid frozen pin masks')
    F, Fp = F.double(), Fp.double()

    def assimilate(f, p, eta, iso):
        if eta <= 0:
            return p.clone()
        inverse, info = torch.linalg.inv_ex(p, check_errors=False)
        torch._assert_async((info == 0).all(), 'Singular FP64 counterfactual Fp')
        elastic = f @ inverse
        _, stretch, vh = svd3(elastic)
        power = stretch.clamp_min(1e-3) ** eta
        if iso:
            power = power / power.prod(1, keepdim=True) ** (1. / 3.)
        increment = vh.transpose(1, 2) @ torch.diag_embed(power) @ vh
        increment = torch.where((torch.linalg.det(elastic) > 1e-6)[:, None, None],
                                increment, torch.eye(3, dtype=f.dtype, device=f.device))
        u, band, vh = svd3(increment @ p)
        band = band.clamp(cfg.assim_smin, cfg.assim_smax)
        if iso:
            # Retain both production clamps and the same 50 bisections.
            low, high = math.log(cfg.assim_smin), math.log(cfg.assim_smax)
            logs = band.log()
            target = torch.zeros_like(logs[:, 0]).clamp(3*low+1e-6, 3*high-1e-6)
            left, right = logs.min(1).values-high-1e-3, logs.max(1).values-low+1e-3
            for _ in range(50):
                middle = .5*(left+right)
                above = (logs-middle[:, None]).clamp(low, high).sum(1) > target
                left, right = torch.where(above, middle, left), torch.where(above, right, middle)
            band = (logs-(.5*(left+right))[:, None]).clamp(low, high).exp()
        return u @ torch.diag_embed(band) @ vh

    # Match the ordinary runner's stored Fp state between its two calls.
    result = assimilate(F, Fp, cfg.assim, cfg.assim_iso).float()
    if cfg.settle_pin_assim:
        result = result.clone()
        result[old] = Fp[old].float()
        if bool(new.any()):
            result[new] = assimilate(F[new], result[new].double(), 1., False).float()
    return result


def diagnose(args, protocol, receipt, report):
    import numpy as host_np
    import torch
    import warp as wp
    from physmorph.compute import cuda_execution, cuda_module, to_array
    from physmorph.mpm.withdrawal import OwnedWithdrawal
    from physmorph.pipeline import PipelineConfig
    from physmorph.pipeline.frozen_body_window import FrozenBodyWindow
    from physmorph.pipeline.post_assimilation_window import PostAssimilationWindow
    cfg = PipelineConfig(**protocol['effective_config'])
    original_phys_loss = cfg.phys_loss
    require(cfg.phys_loss in ('auto', 'ot_pace'), 'Unexpected producer physical loss')
    cfg.phys_loss = 'ot_pace'  # Native runner resolution; no optimizer is rerun.
    require(cfg.compute_backend == 'cuda' and str(cfg.device).startswith('cuda'), 'CUDA only')
    report['config_resolution'] = dict(archived=original_phys_loss, replay=cfg.phys_loss,
        scope='Explicit native auto-to-ot_pace resolution; all other archived fields retained')
    metadata = receipt['receipt']['window_21_prepared_metadata']
    report['discretization'] = dict(N=metadata['N'], T=metadata['T'], prm=metadata['prm'])
    with cuda_execution(cfg.device), torch.no_grad():
        report['runtime'] = dict(torch=torch.__version__, torch_cuda=torch.version.cuda,
            warp=wp.__version__, cupy=cuda_module().__version__,
            device=torch.cuda.get_device_name(), joint_capture=True,
            independent_coast='ordinary no-grad Trajectory, persistent=True')
        report['solver_libraries_before'] = solver_library_bindings()
        with host_np.load(args.native/'window_21_prepared.npz', allow_pickle=False) as archive:
            successor = OwnedWithdrawal.from_arrays(dict(archive), metadata, device=cfg.device)
        actual = successor.arrays()
        owner, _ = FrozenBodyWindow.load(args.native/'prepared_owner.npz', cfg.device)
        old = torch.as_tensor(to_array(owner.spec.pin), device=cfg.device) > .5
        pins = torch.as_tensor(actual['pin'], device=cfg.device) > .5
        new = pins & ~old
        require(not bool((old & ~pins).any()) and bool(new.any()), 'Invalid old/new pin witness')
        cohorts = dict(old_pinned=old, newly_pinned=new, surviving_free=~pins)
        initial_Fp = torch.as_tensor(to_array(owner.spec.Fp), device=cfg.device).clone()
        with host_np.load(args.native/'window_20_prepared.npz', allow_pickle=False) as archive:
            actual_old_Fp = torch.as_tensor(archive['Fp'], device=cfg.device)
            actual_old_pins = torch.as_tensor(archive['pin'], device=cfg.device) > .5
        require(torch.equal(actual_old_Fp, initial_Fp) and torch.equal(actual_old_pins, old),
                'Owner Fp/pins differ from actual prepared W20 inputs')
        report['owner_actual_prepared_Fp_pins_exact'] = True
        del actual_old_Fp, actual_old_pins
        report['cohort_bindings'] = {key: dict(count=int(value.sum()), **tensor_hash(value))
                                     for key, value in cohorts.items()}
        units = dict(x=owner.spec.prm.dx, v=owner.spec.prm.dx/(owner.spec.T*owner.spec.prm.dt),
                     C=1/(owner.spec.T*owner.spec.prm.dt), F=1.)
        model = None
        try:
            model = PostAssimilationWindow(owner, successor, cfg, capture=True)
            coeff = owner.coefficients
            values = model.evaluate(coeff[:, 3:], coeff[:, :3])
            report['joint_health'] = values['health']
            require(values['valid'] and values['pins_exact'], 'Joint replay health failure')
            joint = {key: tuple(row.detach().clone() for row in values[saved])
                     for key, saved in (('x', 'coast_X'), ('v', 'coast_V'),
                                        ('C', 'coast_C'), ('F', 'coast_F'))}
            joint = {key: tuple(row.reshape(-1, 3, 3) if key in ('F', 'C') else row
                               for row in rows) for key, rows in joint.items()}
            joint_Fp = values['coast_Fp'].detach().clone()
            joint_F = values['F'].detach().reshape(-1, 3, 3).clone()
            report['joint_output_hashes'] = sequence_hashes(joint)
            report['joint_Fp_hash'] = tensor_hash(joint_Fp)
            report['joint_boundary_vs_actual'] = {
                key: compare(joint[key][0], torch.as_tensor(actual[key+'0'], device=cfg.device),
                             units[key], cohorts) for key in units}
            report['joint_boundary_vs_actual']['Fp'] = compare(joint_Fp,
                torch.as_tensor(actual['Fp'], device=cfg.device), 1., cohorts)
            with host_np.load(args.native/'window_20_identity.npz', allow_pickle=False) as archive:
                accepted_F = torch.as_tensor(archive['F'], device=cfg.device).reshape(-1, 3, 3)
            report['joint_F_vs_accepted'] = compare(joint_F, accepted_F, 1., cohorts)
            report['assimilation'] = {}
            if args.precision_comparison:
                precise_Fp = {}
                report['precision_comparison'] = dict(
                    scope='Synthetic FP64 assimilation then FP32 coast inputs; NOT an actual-native witness',
                    arithmetic='Each call: FP64 inv_ex/svd3/powers/clamps/50-step log projection, then FP32 store; second call reuploads stored Fp as FP64',
                    admission='Same original Fp and frozen disjoint old/new pins; actual new subset only',
                    maps={})
            for label, F, expected in (('same_joint_F', joint_F, joint_Fp),
                                      ('exact_accepted_F', accepted_F,
                                       torch.as_tensor(actual['Fp'], device=cfg.device))):
                production = torch.as_tensor(ordinary_handoff(F, initial_Fp, old, new, cfg),
                                             device=cfg.device)
                if label == 'same_joint_F':
                    joint_ordinary_Fp = production.clone()
                report['assimilation'][label] = dict(
                    comparison=compare(production, expected, 1., cohorts),
                    production_hash=tensor_hash(production), expected_hash=tensor_hash(expected),
                    first_call_N=len(F), new_subset_N=int(new.sum()),
                    dispatch='Ordinary full first call; restore old pins; second call on new subset only')
                if args.precision_comparison:
                    precise = fp64_handoff(F, initial_Fp, old, new, cfg)
                    require(bool(torch.isfinite(precise).all()), 'Nonfinite FP64 counterfactual')
                    precise_Fp[label] = precise
                    report['precision_comparison']['maps'][label] = dict(
                        per_call_fp32_state_hash=tensor_hash(precise),
                        cast_vs_ordinary_fp32=compare(precise_Fp[label], production, 1., cohorts),
                        cast_vs_original_boundary=compare(precise_Fp[label], expected, 1., cohorts))
                    del precise
            if args.precision_comparison:
                delta64 = precise_Fp['same_joint_F']-precise_Fp['exact_accepted_F']
                delta32 = joint_ordinary_Fp-production  # Last loop call is exact accepted F.
                report['precision_comparison']['Fp_response'] = dict(
                    definition='Joint-F minus accepted-F handoff response with identical original Fp/pins',
                    cast_pair=compare(precise_Fp['same_joint_F'], precise_Fp['exact_accepted_F'], 1., cohorts),
                    ordinary_pair=compare(joint_ordinary_Fp, production, 1., cohorts),
                    response_difference=compare(delta64, delta32, 1., cohorts),
                    cohorts={key: dict(count=int(mask.sum()),
                        fp64_cast_response_norm=float(delta64[mask].double().norm()),
                        fp32_response_norm=float(delta32[mask].double().norm()))
                        for key, mask in cohorts.items()})
                del delta64, delta32
            del values, coeff, joint_F, accepted_F, initial_Fp, production
        finally:
            if model is not None:
                model.close()
            owner.close()
        del model, owner
        gc.collect()
        torch.cuda.empty_cache()

        def replay(replacements):
            arrays = actual.copy()  # Other arrays are read-only; from_arrays owns all data.
            arrays.update({key: to_array(value) for key, value in replacements.items()})
            owned = OwnedWithdrawal.from_arrays(arrays, metadata, device=cfg.device)
            tr = owned.trajectory(persistent=True)
            tr.rollout()
            result = {key: tuple(wp.to_torch(row).detach().clone() for row in getattr(tr, key))
                      for key in units}
            del tr, owned
            return result

        reference = replay({})
        report['actual_coast_output_hashes'] = [sequence_hashes(reference)]
        report['joint_vs_actual_coast'] = phase_comparisons(joint, reference, units, cohorts)
        report['actual_repeat_vs_first'] = []
        for _ in range(1, args.actual_repeats):
            repeated = replay({})
            report['actual_coast_output_hashes'].append(sequence_hashes(repeated))
            report['actual_repeat_vs_first'].append(phase_comparisons(repeated, reference, units, cohorts))
            del repeated
        state_only = {key+'0': joint[key][0] for key in units}
        report['boundary_swap_diagnostics'] = {}
        for label, replacement in (
                ('joint_Fp_only', dict(Fp=joint_Fp)),
                ('joint_x_v_C_F_only', state_only),
                ('all_joint_boundary', dict(state_only, Fp=joint_Fp))):
            result = replay(replacement)
            report['boundary_swap_diagnostics'][label] = dict(
                scope='Synthetic input isolation only; NOT an actual-handoff witness',
                replaced_fields=list(replacement), output_hashes=sequence_hashes(result),
                versus_actual=phase_comparisons(result, reference, units, cohorts),
                versus_joint=phase_comparisons(result, joint, units, cohorts))
            if label == 'all_joint_boundary':
                all_joint = result
            else:
                del result
        result = replay(dict(state_only, Fp=joint_ordinary_Fp))
        report['boundary_swap_diagnostics']['joint_state_ordinary_Fp'] = dict(
            scope='Synthetic input isolation only; NOT an actual-handoff witness',
            replaced_fields=[*state_only, 'Fp=ordinary_handoff(same joint F)'],
            output_hashes=sequence_hashes(result),
            versus_actual=phase_comparisons(result, reference, units, cohorts),
            versus_joint=phase_comparisons(result, joint, units, cohorts),
            versus_all_joint=phase_comparisons(result, all_joint, units, cohorts))
        del result, all_joint, joint_ordinary_Fp
        if args.precision_comparison:
            accepted_precise = replay(dict(Fp=precise_Fp['exact_accepted_F']))
            joint_precise = replay(dict(state_only, Fp=precise_Fp['same_joint_F']))
            report['precision_comparison']['coasts'] = dict(
                accepted_boundary=dict(output_hashes=sequence_hashes(accepted_precise),
                    versus_actual=phase_comparisons(accepted_precise, reference, units, cohorts)),
                joint_boundary=dict(output_hashes=sequence_hashes(joint_precise),
                    versus_joint=phase_comparisons(joint_precise, joint, units, cohorts),
                    versus_actual=phase_comparisons(joint_precise, reference, units, cohorts)),
                joint_vs_accepted_precision_pair=phase_comparisons(joint_precise, accepted_precise, units, cohorts))
            del accepted_precise, joint_precise, precise_Fp
        def summarize(rows):
            return {key: dict(failing_phases=[i for i, row in enumerate(phases) if not row['passed']],
                max_tolerance_ratio=max(row['max_tolerance_ratio'] for row in phases)
                    if all(row['finite'] for row in phases) else None)
                for key, phases in rows.items()}
        report['summary'] = dict(
            joint_vs_actual=summarize(report['joint_vs_actual_coast']),
            actual_repeats=[summarize(row) for row in report['actual_repeat_vs_first']],
            swaps={key: {comparison: summarize(value[comparison])
                         for comparison in ('versus_actual', 'versus_joint', 'versus_all_joint')
                         if comparison in value}
                   for key, value in report['boundary_swap_diagnostics'].items()},
            interpretation='Localization measurements only; no automatic attribution to FP32 noise or kernel error')
        if args.precision_comparison:
            report['summary']['precision_pair'] = summarize(
                report['precision_comparison']['coasts']['joint_vs_accepted_precision_pair'])
        report['completed'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True, help='Producer tree matching native source hashes')
    parser.add_argument('--native', type=Path, required=True, help='Retained failed native1 output')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--actual-repeats', type=int, default=2, choices=(2, 3, 4))
    parser.add_argument('--precision-comparison', action='store_true',
                        help='Add two synthetic FP64-assimilation/FP32-coast counterfactuals')
    args = parser.parse_args()
    args.code, args.native, args.out = (path.resolve() for path in (args.code, args.native, args.out))
    require(all(path.is_relative_to(args.root.resolve()) for path in (args.code, args.native, args.out)),
            'All code/input/output must be under the server data root')
    require(args.out != args.native and not args.out.is_relative_to(args.native), 'Separate diagnostic output required')
    args.out.mkdir(exist_ok=False)
    report = dict(schema='p335_closure_diagnosis_v1', completed=False,
        scope='Read-only numerical localization; failed native gate remains failed',
        rule='32*FP32_eps*(native_scale+abs(reference)); elementwise; unchanged',
        start_utc=datetime.now(timezone.utc).isoformat(), arguments={k: str(v) for k, v in vars(args).items()})
    started, bindings = time.perf_counter(), {}
    try:
        cache_paths = {key: os.environ.get(key, '') for key in
                       ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH')}
        require(all(value and Path(value).resolve().is_relative_to(args.root.resolve())
                    for value in cache_paths.values()), 'All CUDA caches must be explicitly under the data root')
        report['cache_paths'] = cache_paths
        input_paths = [args.native/name for name in ('protocol.json', 'result.json',
            'prepared_owner.npz', 'window_20_identity.npz', 'window_21_prepared.npz',
            'window_20_prepared.npz')]
        bindings = {str(path): identity(path) for path in input_paths}
        protocol = json.loads((args.native/'protocol.json').read_text())
        receipt = json.loads((args.native/'result.json').read_text())
        require(bindings[str(args.native/'protocol.json')]['sha256'] == receipt['protocol_sha256'],
                'Producer protocol receipt mismatch')
        for path in input_paths[2:]:
            expected = receipt['files'][path.name]
            require(all(bindings[str(path)][k] == expected[k] for k in ('bytes', 'sha256')),
                    'Producer archive receipt mismatch: '+path.name)
        physics_paths = sorted((args.code/'physmorph').rglob('*.py'))
        source_paths = [*physics_paths, args.code/'VERSION', args.code/'scripts/ops/cuda_python.py',
                        args.code/'scripts/ops/gpu_env.sh', Path(__file__).resolve()]
        bindings.update({str(path): identity(path) for path in source_paths})
        producer = {name.split('/physmorph/', 1)[1]: value for name, value in protocol['bindings'].items()
                    if '/physmorph/' in name}
        require(producer and set(producer) == {str(p.relative_to(args.code/'physmorph')).replace('\\', '/')
                for p in physics_paths}, 'Frozen producer source inventory mismatch')
        for relative, expected in producer.items():
            require(bindings[str(args.code/'physmorph'/relative)]['sha256'] == expected['sha256'],
                    'Frozen producer source mismatch: '+relative)
        report['bindings_before'] = bindings
        report['code_version'] = (args.code/'VERSION').read_text().strip()
        report['producer_code_binding'] = 'All physmorph source bytes match archived native protocol; VERSION separately bound'
        write_json(args.out/'protocol.json', report)
        preimported = {name: str(Path(module.__file__).resolve())
                      for name, module in list(sys.modules.items())
                      if (name == 'physmorph' or name.startswith('physmorph.'))
                      and getattr(module, '__file__', None)}
        require(all(Path(path).is_relative_to(args.code) and path in bindings
                    for path in preimported.values()), 'Preimported physics is outside the bound frozen source')
        report['preimported_frozen_sources'] = preimported
        sys.path.insert(0, str(args.code))
        diagnose(args, protocol, receipt, report)
    except Exception as error:
        report['failure'] = dict(type=type(error).__name__, message=str(error))
    finally:
        imported = {name: str(Path(module.__file__).resolve()) for name, module in list(sys.modules.items())
                    if name.startswith('physmorph') and getattr(module, '__file__', None)}
        report['imported_sources'] = imported
        report['imports_from_frozen_code'] = all(Path(path).is_relative_to(args.code) for path in imported.values())
        report['solver_libraries_after'] = solver_library_bindings()
        report['solver_library_scope'] = ('Loaded libcusolver file hashes only, not a dependency/source audit. '
            'Libraries absent from the before map have post-run bindings only.')
        report['solver_libraries_preexisting_unchanged'] = all(
            report['solver_libraries_after'].get(path) == value
            for path, value in report.get('solver_libraries_before', {}).items())
        try:
            after = {name: identity(Path(name)) for name in bindings}
        except Exception as error:
            after = {}
            report['binding_check_failure'] = dict(type=type(error).__name__, message=str(error))
        report['bindings_after'] = after
        report['bindings_unchanged'] = bindings == after
        report['completed'] = bool(report['completed'] and report['bindings_unchanged']
                                   and report['imports_from_frozen_code']
                                   and report['solver_libraries_preexisting_unchanged'])
        report['elapsed_seconds'] = time.perf_counter()-started
        write_json(args.out/'result.json', report)
        write_json(args.out/'output_receipt.json', {p.name: identity(p) for p in sorted(args.out.glob('*.json'))})
    require(report['completed'], 'Diagnostic incomplete; inspect retained failure receipt')


if __name__ == '__main__':
    main()
