"""P329: noncommitting W20/inner8 prepared withdrawal capability measurement."""
import argparse
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime, timezone
import gc
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as host_np
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution, cuda_module, to_array, to_host, KDTree, array_api as np
from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, runner
from physmorph.pipeline.frozen_withdrawal_window import FrozenWithdrawalWindow
from scripts.probes.reference_swap import require, sha


LIMIT = 3_000_000_000
SCOPE = ('Read-only original-control pre-assimilation withdrawal; no proposal, '
         'new objective, postcommit derivative, rest/shape certificate or adopted state')


def safe_json(value):
    if isinstance(value, dict): return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False), encoding='utf-8')


def identity(path):
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed while hashing: '+str(path))
    return dict(bytes=after.st_size, mtime_ns=after.st_mtime_ns, inode=after.st_ino, sha256=digest)


def verify_bindings(bindings):
    require(all(identity(Path(k)) == v for k, v in bindings.items()), 'Bound input/source changed')


def validate_recipe(metadata):
    cfg = PipelineConfig(**metadata['config'])
    prm = MPMParams(**metadata['mpm'])
    require(cfg.T == 20 and cfg.iters == 8 and cfg.loss_res == 36 and
            prm.dt == 1/240 and prm.dx == .3062907543956724, 'Unexpected discretization/budget')
    require(cfg.compute_backend == 'cuda' and str(cfg.device).startswith('cuda') and
            cfg.body_ctrl and cfg.body_terminal_ctrl and cfg.layer_ctrl and cfg.layer_relax and
            cfg.lambda_auto > 0 and cfg.render_res == 64 and cfg.loss_units == 'density',
            'Expected raw mixed-body active CIC/PBR recipe')
    require(not any(getattr(cfg, key) for key in ('commit_pic', 'commit_pic_objective', 'shift_sub',
            'opt_material', 'geometric_rest', 'geometric_variance', 'grad_dump', 'render_F_geom',
            'use_gauss_loss', 'surface_gs_loss', 'continuity', 'settle_pin_kkt')),
            'Unsupported objective/commit/renderer policy')
    cfg.stop_after_windows = 20  # The sole recipe override; ordinary earlier stops remain active.
    return cfg, prm


def closure(actual, expected, native_scale):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype and actual.device == expected.device,
            'Closure layout/dtype/device mismatch')
    require(native_scale > 0 and math.isfinite(native_scale), 'Invalid closure unit')
    with torch.no_grad():
        error = (actual.detach()-expected.detach()).abs()
        tolerance = 32*torch.finfo(expected.dtype).eps*(native_scale+expected.detach().abs())
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        return dict(passed=finite and bool((error <= tolerance).all()), finite=finite,
                    native_scale=native_scale, max_abs=float(error.max()),
                    max_tolerance_ratio=float((error/tolerance).max()),
                    rule='32*FP32_eps*(native_scale+abs(reference)); elementwise')


def scalar_closure(actual, expected, dtype):
    tolerance = 32*torch.finfo(dtype).eps*max(abs(expected), 1e-12)
    error = abs(actual-expected)
    return dict(passed=math.isfinite(actual) and math.isfinite(expected) and error <= tolerance,
                actual=actual, expected=expected, absolute_difference=error, tolerance=tolerance)


def head_closure(values, accepted, spec):
    return {key: closure(values[key], accepted[key].reshape_as(values[key]), unit) for key, unit in
            (('positions', spec.prm.dx), ('V', spec.prm.dx/(spec.T*spec.prm.dt)),
             ('F', 1.), ('C', 1/(spec.T*spec.prm.dt)))}


def private_closure(values, archive_path, spec):
    """Read archived witnesses one field at a time; numerical comparison on device."""
    result = {}
    with host_np.load(archive_path, allow_pickle=False) as archive:
        for key, unit in (('positions', spec.prm.dx), ('V', spec.prm.dx/(spec.T*spec.prm.dt)),
                          ('F', 1.), ('C', 1/(spec.T*spec.prm.dt)), ('Fg', 1.)):
            expected = torch.as_tensor(archive[key], device=values[key].device)
            result[key] = closure(values[key], expected, unit)
    return result


def coast_losses(values, mask, dt):
    """Population mean speeds squared; includes boundary-to-first step, excludes V0."""
    if not bool(mask.any()): return None
    X = values['coast_X'][:, mask].double()
    V = values['coast_V'][1:, mask].double()
    return dict(geometric_step_mean_square=((X[1:]-X[:-1])/dt).square().sum(-1).mean(),
                stored_speed_mean_square=V.square().sum(-1).mean())


def finite_coast_observations(rows, steps):
    keys = ('volume', 'silhouette', 'pbr', 'render', 'weighted_render')
    return (isinstance(rows, list) and [row.get('phase') for row in rows] == [0, steps//2, steps]
            and all(type(row.get(key)) in (int, float) and math.isfinite(row[key])
                    for row in rows for key in keys))


def tensor_bytes(tree):
    if torch.is_tensor(tree): return tree.numel()*tree.element_size()
    if hasattr(tree, 'nbytes'): return int(tree.nbytes)
    if isinstance(tree, dict): return sum(tensor_bytes(v) for v in tree.values())
    if isinstance(tree, (tuple, list)): return sum(tensor_bytes(v) for v in tree)
    return 0


def memory_snapshot(device):
    if torch.device(device).type != 'cuda': return dict(device='cpu')
    free, total = torch.cuda.mem_get_info(device)
    result = dict(device=str(device), free_bytes=free, total_bytes=total,
                  torch_allocated=torch.cuda.memory_allocated(device),
                  torch_reserved=torch.cuda.memory_reserved(device),
                  torch_peak_allocated=torch.cuda.max_memory_allocated(device),
                  torch_peak_reserved=torch.cuda.max_memory_reserved(device),
                  scope='Torch peaks are process-lifetime allocator peaks; device free includes other allocations/users')
    pool = cuda_module().get_default_memory_pool()
    result.update(cupy_pool_used=pool.used_bytes(), cupy_pool_total=pool.total_bytes(),
                  process_device_peak_scope='External root-owned nvidia-smi sampled receipt; stage values are not peaks')
    return result


class Capture:
    def __init__(self, out, spacing, *, window=19, iteration=8):
        self.out, self.spacing = Path(out), float(spacing)
        self.window, self.iteration = window, iteration
        self.report = None
        self.sidecars = {}
        self.accepted_endpoint = None
        self.packet = None
        self.outer_accepted = False
        self.outer_record = None

    def reserve(self, size):
        existing = sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
        require(existing+size+1_048_576 <= LIMIT, 'P329 output would exceed 3GB')

    def save(self, name, arrays):
        self.reserve(tensor_bytes(arrays))
        path = self.out/name
        require(not path.exists(), 'Refusing to overwrite evidence')
        # Archive transfer only; every measurement and gradient stays on the input device.
        with path.open('xb') as stream:
            host_np.savez_compressed(stream, **{k: to_host(v) for k, v in arrays.items()})
        self.sidecars[name] = identity(path)
        self.reserve(0)

    def stage(self, label, device, start):
        if torch.device(device).type == 'cuda': torch.cuda.current_stream(device).synchronize()
        self.report['stages'][label] = dict(elapsed_seconds=time.perf_counter()-start,
                                           memory=memory_snapshot(device))

    def observe(self, index, packet):
        require(index == self.window and packet['iteration'] == self.iteration and self.report is None,
                'Unexpected or repeated checkpoint')
        self.packet = packet  # Released immediately after the original optimizer returns.
        owner = packet['rollout']
        dev = owner.coefficients.device
        accepted = {k: packet[k].detach().clone() for k in ('positions', 'V', 'F', 'C')}
        self.accepted_endpoint = dict(x=accepted['positions'][-1].clone(), v=accepted['V'][-1].clone(),
                                      F=accepted['F'].clone(), C=accepted['C'].clone())
        coefficients = packet['controls']['body'].detach().clone()
        require(torch.equal(coefficients, owner.coefficients), 'Prepared coefficient binding changed')
        masks = dict(start_free=~packet['pins'], start_arrived_free=~packet['pins'] & packet['start_arrived'])
        self.report = dict(scope=SCOPE, window=index+1, iteration=packet['iteration'],
            N=len(packet['x0']), T=owner.spec.T, dt=owner.spec.prm.dt, dx=owner.spec.prm.dx,
            spacing=self.spacing, lambda_render=packet['lambda_render'],
            pbr_weight=packet['reference'].pbr_weight, reference_kind=packet['reference'].kind,
            history=deepcopy(packet['history']), stages={}, errors=[],
            accepted_F_C_scope='Accepted X/V sequence; terminal F/C only; no accepted Fg available',
            coast_policy='Original Fp, fixed start pins/layer/bonds; learned controls withdrawn, passive projections remain',
            gradient_scope='Original coefficients only; no direction normalization, projection, proposal or lambda update')
        start = time.perf_counter()
        binding = packet['evaluate_merit'].binding_digest()
        self.report['merit_binding_before'] = binding
        self.save('accepted_head.npz', dict(**accepted, x0=packet['x0'], pins=packet['pins'],
            start_arrived=packet['start_arrived'], coefficients=coefficients,
            **{name+'_ids': torch.nonzero(mask).flatten() for name, mask in masks.items()}))
        owner_arrays = {f.name: getattr(owner.spec, f.name) for f in fields(owner.spec)}
        owner_arrays.update({k: getattr(owner, k) for k in ('idx', 'weights', 'gate', 'coefficients', 'stress', 'surface_u')})
        observations = dict(scope=SCOPE, reference=asdict(packet['reference']),
                            lambda_render=packet['lambda_render'], native_spacing=self.spacing)
        self.reserve(tensor_bytes(owner_arrays)+tensor_bytes(observations))
        owner_path = self.out/'prepared_owner.npz'
        owner.save(owner_path, observations)
        self.sidecars[owner_path.name] = identity(owner_path)
        self.reserve(0)
        self.stage('owned_input_archive', dev, start)
        model = None
        try:
            start = time.perf_counter()
            with torch.no_grad():
                original = owner.evaluate(coefficients[:, 3:], coefficients[:, :3])
                original['Fg'] = wp.to_torch(owner.adjoint.traj.Fg[owner.spec.T]).reshape(-1, 9).clone()
                self.save('private_head.npz', {k: original[k] for k in ('positions', 'V', 'F', 'C', 'Fg', 'body_energy')})
                self.report['original_closure'] = head_closure(original, accepted, owner.spec)
                self.report['original_health'] = dict(valid=original['valid'], pins_exact=original['pins_exact'],
                                                      min_det=original['min_det'])
                if original['valid']:
                    self.report['original_merit'] = packet['evaluate_merit'](original)
                    self.report['original_scalar_closure'] = {key: scalar_closure(
                        self.report['original_merit'][key], expected, coefficients.dtype) for key, expected in
                        (('merit', packet['history']['loss']), ('volume', packet['history']['d_vol']),
                         ('render', packet['history']['d_render']+
                          packet['reference'].pbr_weight*(packet['history']['d_pbr'] or 0.)))}
            del original
            owner.adjoint = None
            gc.collect()
            self.stage('private_original_released', dev, start)
            start = time.perf_counter()
            displacement = coefficients[:, :3].clone().requires_grad_()
            terminal = coefficients[:, 3:].clone().requires_grad_()
            with torch.enable_grad():
                model = FrozenWithdrawalWindow(owner)
                values = model.evaluate(terminal, displacement)
                self.report['joint_closure'] = head_closure(values, accepted, owner.spec)
                self.report['private_joint_closure'] = private_closure(values, self.out/'private_head.npz', owner.spec)
                self.report['joint_health'] = dict(values['health'], valid=values['valid'],
                                                  pins_exact=values['pins_exact'], min_det=values['min_det'])
                arrays = {key: values[key] for key in ('x', 'F', 'C', 'v', 'Fg', 'positions', 'V',
                                                      'coast_X', 'coast_V', 'coast_F', 'coast_Fg', 'coast_C', 'body_energy')}
                arrays.update(displacement=displacement, terminal=terminal)
                # Preserve witnesses even when the strict C closure fails.
                self.save('joint_state.npz', arrays)
                self.stage('joint_forward_and_state_archive', dev, start)
                self.report['cohorts'] = {}
                gradient_arrays = {}
                if values['valid']:
                    self.report['joint_merit'] = packet['evaluate_merit'](values)
                    terms = packet['reference'].terms(values['x'])
                    self.report['prepared_head'] = {k: float(v.detach()) for k, v in terms.items()}
                    self.report['prepared_head']['weighted_render'] = packet['lambda_render']*float(terms['render'].detach())
                    self.report['scalar_closure'] = {
                        'merit': scalar_closure(self.report['joint_merit']['merit'], packet['history']['loss'], coefficients.dtype),
                        'volume': scalar_closure(float(terms['volume'].detach()), packet['history']['d_vol'], coefficients.dtype),
                        'render': scalar_closure(float(terms['render'].detach()), packet['history']['d_render']+
                                   packet['reference'].pbr_weight*(packet['history']['d_pbr'] or 0.), coefficients.dtype)}
                    self.report['coast_prepared_observations'] = []
                    with torch.no_grad():
                        for phase in (0, owner.spec.T//2, owner.spec.T):
                            t = packet['reference'].terms(values['coast_X'][phase])
                            self.report['coast_prepared_observations'].append(dict(phase=phase,
                                **{k: float(v) for k, v in t.items()},
                                weighted_render=packet['lambda_render']*float(t['render'])))
                    self.report['coast_observation_scope'] = 'Frozen paced reference at boundary/mid/end; not independent raw coverage or all-phase quality'
                    del terms
                    start = time.perf_counter()
                    for name, mask in masks.items():
                        count = int(mask.sum())
                        row = dict(particles=count, losses=None, gradients=None)
                        self.report['cohorts'][name] = row
                        if not count: continue
                        losses = coast_losses(values, mask, packet['dt'])
                        row['losses'] = {k: float(v.detach()) for k, v in losses.items()}
                        row['gradients'] = {}
                        for label, loss in losses.items():
                            grads = torch.autograd.grad(loss, (displacement, terminal), retain_graph=True)
                            detail = {}
                            for channel, grad in zip(('displacement', 'terminal'), grads):
                                gradient_arrays[name+'__'+label+'__'+channel] = grad.detach().clone()
                                detail[channel] = dict(finite=bool(torch.isfinite(grad).all()),
                                    norm=float(grad.double().norm()), rms=float(grad.double().square().mean().sqrt()),
                                    absmax=float(grad.abs().max()), nonzero=bool((grad != 0).any()))
                            row['gradients'][label] = detail
                        X = values['coast_X'][:, mask].detach().double()
                        row['net_rms_sp'] = float((X[-1]-X[0]).square().sum(-1).mean().sqrt())/self.spacing
                        row['path_mean_sp'] = float((X[1:]-X[:-1]).norm(dim=-1).sum(0).mean())/self.spacing
                    self.save('coast_gradients.npz', gradient_arrays)
                    self.stage('future_only_backward', dev, start)
                self.report['merit_binding_after'] = packet['evaluate_merit'].binding_digest()
                self.report['merit_binding_unchanged'] = self.report['merit_binding_after'] == binding
        except (ValueError, AssertionError) as exc:
            self.report['errors'].append(dict(type=type(exc).__name__, message=str(exc)))
        finally:
            if model is not None: model.close()
            owner.adjoint = None
            self.report['sidecars'] = self.sidecars
            self.report['measurement_passed'] = self.measurement_passed()
            write_json(self.out/'checkpoint.json', self.report)

    def measurement_passed(self):
        r = self.report
        if r is None or r['errors'] or not r.get('merit_binding_unchanged'): return False
        for key in ('original_closure', 'joint_closure', 'private_joint_closure', 'scalar_closure', 'original_scalar_closure'):
            if key not in r or not all(v['passed'] for v in r[key].values()): return False
        if not all(r.get(k, {}).get('valid') for k in ('original_health', 'joint_health')): return False
        if not finite_coast_observations(r.get('coast_prepared_observations'), r['T']): return False
        cohorts = r.get('cohorts', {})
        return bool(cohorts) and all(row['particles'] > 0 and row['gradients'] and
            all(math.isfinite(loss) for loss in row['losses'].values()) and
            all(g['finite'] for channels in row['gradients'].values() for g in channels.values())
            for row in cohorts.values())

    def wrap(self, original):
        def wrapped(*args, **kwargs):
            if kwargs['win_index'] != self.window: return original(*args, **kwargs)
            result = original(*args, on_checkpoint=self.observe, checkpoint_iterations=(self.iteration,),
                              checkpoint_rollout=True, checkpoint_merit=True, **kwargs)
            if self.report is not None:
                self.report['optimizer_state_after_callback_exact'] = self.packet.get('optimizer_state_after_callback_exact', False)
                self.report['inner_stats'] = {key: result[-1].get(key) for key in
                    ('accepted', 'rejected', 'ls_exhausted', 'grad_converged', 'pace_bound')}
                end = result[2]
                self.report['original_return_preserved'] = {key: torch.equal(
                    torch.as_tensor(to_array(end[key]), device=value.device).reshape_as(value), value)
                    for key, value in self.accepted_endpoint.items() if key != 'x'}
                self.report['original_return_preserved']['x'] = torch.equal(
                    torch.as_tensor(to_array(result[0][-1]), device=self.accepted_endpoint['x'].device),
                    self.accepted_endpoint['x'])
                self.packet = None
            return result
        return wrapped

    def commit(self, index, x, F, v, record):
        if index != self.window: return
        self.outer_record = deepcopy(record)
        self.outer_accepted = bool(record.get('frame_end') and not record.get('null_commit') and
                                   not record.get('outer_rejected') and record.get('outer_accepted', 1))
        if self.outer_accepted and self.accepted_endpoint is not None:
            self.report['committed_original_endpoint'] = {key: torch.equal(
                torch.as_tensor(to_array(actual), device=self.accepted_endpoint[key].device).reshape_as(self.accepted_endpoint[key]),
                self.accepted_endpoint[key]) for key, actual in (('x', x), ('F', F), ('v', v))}


def passed_report(report, result):
    record = report.get('checkpoint') or {}
    returned, committed = record.get('original_return_preserved', {}), record.get('committed_original_endpoint', {})
    return bool(not report['failure'] and report['bindings_unchanged'] and report['outer_accepted'] and
        record.get('measurement_passed') and record.get('optimizer_state_after_callback_exact') and
        set(returned) == {'x', 'v', 'F', 'C'} and all(returned.values()) and
        set(committed) == {'x', 'F', 'v'} and all(committed.values()) and
        result is not None and not any(result['guards'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(args.out.resolve().is_relative_to(args.root.resolve()), 'Output outside project data')
    args.out.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    metadata_path = args.root/'work/p303/raw24a.json'
    source_path = args.root/'repro/current_pair/source_render_full_dt_iso_nn.npz'
    initial = identity(metadata_path)
    metadata_bytes = metadata_path.read_bytes()
    require(sha256(metadata_bytes).hexdigest() == initial['sha256'], 'Parsed source metadata changed')
    metadata = json.loads(metadata_bytes)
    cfg, prm = validate_recipe(metadata)
    paths = [metadata_path, source_path, Path(cfg.target_reference),
             *sorted((repo/'physmorph').rglob('*.py')),
             *(repo/p for p in ('scripts/probes/prepared_withdrawal.py', 'scripts/probes/reference_swap.py',
                'scripts/ops/run_p303_probe.sh', 'scripts/ops/cuda_python.py', 'docs/production_withdrawal_p329.md'))]
    bindings = {str(p): identity(p) for p in paths}
    require(bindings[str(metadata_path)] == initial, 'Metadata changed before binding')
    with host_np.load(source_path, allow_pickle=False) as archive: source, target = archive['src'], archive['tgt']
    require(source.shape == target.shape == (300000, 3), 'Unexpected native particle inputs')
    verify_bindings(bindings)
    protocol = dict(schema='prepared_withdrawal_p329_v1', start_utc=datetime.now(timezone.utc).isoformat(),
        scope=SCOPE, bindings=bindings, source_config=metadata['config'], effective_config=asdict(cfg),
        mpm=asdict(prm), overrides=dict(stop_after_windows=20), N=len(source), window=20, iteration=8,
        closure=dict(multiplier_eps=32, X='dx', V='dx/(T*dt)', F=1, C='1/(T*dt)',
                     scalar='32eps*max(abs(reference),1e-12)'), max_output_bytes=LIMIT)
    write_json(args.out/'protocol.json', protocol)
    result = None; failure = None; capture = None
    start = time.perf_counter()
    try:
        with cuda_execution(cfg.device):
            points = to_array(source)
            spacing = float(np.median(KDTree(points).query(points, k=2)[0][:, 1]))
        capture = Capture(args.out, spacing)
        with patch.object(runner, 'optimize_window', capture.wrap(runner.optimize_window)):
            result = runner.run_pipeline(source, target, prm, cfg, on_commit=capture.commit)
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
    stable = True
    try:
        verify_bindings(bindings)
        if capture:
            require(all(identity(args.out/k) == v for k, v in capture.sidecars.items()), 'Saved sidecar changed')
    except Exception as exc:
        stable = False
        failure = failure or dict(type=type(exc).__name__, message=str(exc))
    report = dict(protocol_sha256=sha(args.out/'protocol.json'), scope=SCOPE,
        elapsed_seconds=time.perf_counter()-start, failure=failure, bindings_unchanged=stable,
        checkpoint=None if capture is None else capture.report,
        outer_accepted=False if capture is None else capture.outer_accepted,
        outer_record=None if capture is None else capture.outer_record,
        completion_scope='Actual outer acceptance, separately from delivery retention and imposed held frames')
    if result is not None:
        from physmorph.pipeline.render_reporting import write_render_report
        report.update(history=result['history'], guards=result['guards'], termination=result.get('termination'),
                      deliver_n=result['deliver_n'], truncation=result['truncation'],
                      render_influence=write_render_report(args.out/'run', result['history'], asdict(cfg), asdict(prm), len(source)))
    report['passed'] = passed_report(report, result)
    require(sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())+1_048_576 <= LIMIT, 'Output exceeds 3GB')
    write_json(args.out/'result.json', report)
    require(report['passed'], 'P329 capability gate failed; evidence saved, no candidate was adopted')


if __name__ == '__main__': main()
