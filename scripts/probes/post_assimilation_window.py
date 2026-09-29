"""P335: actual prepared successor versus joint post-assimilation withdrawal.

Run the unchanged original W20/W21, retain their owned prepared inputs, then
measure the private joint forward/backward after production has returned.
"""
import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as host_np
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution, to_array
from physmorph.mpm.withdrawal import OwnedWithdrawal
from physmorph.pipeline import runner
from physmorph.pipeline.frozen_body_window import FrozenBodyWindow
from physmorph.pipeline.post_assimilation_window import PostAssimilationWindow
from physmorph.pipeline.render_reporting import write_render_report
from scripts.probes.window_selection import IdentityCapture
from scripts.probes.prepared_withdrawal import (
    identity, verify_bindings, validate_recipe, scalar_closure, closure,
    require, sha, coast_losses, memory_snapshot, tensor_bytes, safe_json)


SCOPE = ('Actual-prepared fixed-policy post-assimilation capability; no changed '
         'candidate, admission/preparation derivative, rest/hole/4K certificate')


class Capture(IdentityCapture):
    FAILURE_RESERVE = 65536

    def reserve(self, size):
        super().reserve(size+self.FAILURE_RESERVE)

    def finish(self, report):
        """Retain a structured failure even if a later report exceeds its cap."""
        try:
            self.write_json('result.json', report)
        except Exception as exc:
            bounded = dict(scope=SCOPE, passed=False, failure=report.get('failure') or
                dict(type=type(exc).__name__, message=str(exc)),
                finalization_failure=dict(type=type(exc).__name__, message=str(exc)),
                bindings_unchanged=report.get('bindings_unchanged'),
                detail='Full result exceeded or failed output finalization; existing evidence retained')
            payload = json.dumps(safe_json(bounded), allow_nan=False).encode('utf-8')
            require(len(payload) <= self.FAILURE_RESERVE, 'Failure receipt exceeds reserve')
            super().reserve(len(payload))
            with (self.out/'failure.json').open('xb') as stream:
                stream.write(payload)
            return False
        return report['passed']

    def wrap(self, original):
        def with_velocity_witness(*args, **kwargs):
            if kwargs['win_index'] == self.window:
                require(kwargs.get('on_rollout') is None, 'Unexpected existing rollout observer')
                def observed(tr, promoted, index):
                    require(index == self.window, 'Wrong velocity witness window')
                    self.save('accepted_velocity.npz', dict(
                        V=torch.stack([wp.to_torch(value).clone() for value in tr.v[1:]])))
                kwargs['on_rollout'] = observed
            return original(*args, **kwargs)
        return super().wrap(with_velocity_witness)

    def select(self, context):
        choice = super().select(context)
        owner = context._owner
        evaluator = context._evaluate_merit
        binding = evaluator.binding_digest()
        coefficients = owner.coefficients
        live_report = dict(binding_before=binding,
            scope='W20 callback only; evaluator is not retained or revived for joint replay')
        self.receipt['live_head_merit'] = live_report
        try:
            with torch.no_grad():
                values = owner.evaluate(coefficients[:, 3:], coefficients[:, :3], retain_full_state=True)
                require(values['valid'] and values['pins_exact'], 'Invalid original private head')
                merit = evaluator(values)
                scalar = scalar_closure(merit['merit'], context._loss, coefficients.dtype)
                live_report.update(scalar=scalar, terms=merit)
                require(scalar['passed'], 'Live original full merit differs from donor')
                reference = context.inspect(choice)['values']
                checks = {key: closure(values[key], reference[key].reshape_as(values[key]), unit)
                          for key, unit in (('positions', owner.spec.prm.dx), ('F', 1.),
                                            ('v', owner.spec.prm.dx/(owner.spec.T*owner.spec.prm.dt)),
                                            ('C', 1/(owner.spec.T*owner.spec.prm.dt)))}
                checks['F_sequence'] = closure(values['F_sequence'], reference['F_sequence'], 1.)
                with host_np.load(self.out/'accepted_velocity.npz', allow_pickle=False) as archive:
                    accepted_V = torch.as_tensor(archive['V'], device=coefficients.device)
                    checks['V'] = closure(values['V'], accepted_V,
                        owner.spec.prm.dx/(owner.spec.T*owner.spec.prm.dt))
                live_report['head_closure'] = checks
                require(all(row['passed'] for row in checks.values()), 'Private head differs from original')
                self.save('merit_head.npz', {key: values[key] for key in
                    ('x', 'F', 'C', 'v', 'V', 'positions', 'body_energy')})
            observations = dict(scope=SCOPE, full_merit=merit, scalar_closure=scalar,
                reference=asdict(context._reference), binding=binding)
            path = self.out/'prepared_owner.npz'
            owner_data = {item.name: getattr(owner.spec, item.name) for item in fields(owner.spec)}
            owner_data.update({key: getattr(owner, key) for key in
                               ('idx', 'weights', 'gate', 'coefficients', 'stress', 'surface_u')})
            self.reserve(tensor_bytes(owner_data)+tensor_bytes(observations)+1_048_576)
            owner.save(path, observations)
            self.files[path.name] = identity(path)
            self.reserve(0)
            after = evaluator.binding_digest()
            live_report['binding_after'] = after
            require(after == binding, 'Original merit binding changed')
        finally:
            owner.adjoint = None
            gc.collect()
        return choice

    def measure(self, cfg):
        """Called after ordinary pipeline return: no production graph is retained."""
        require(self.context is not None and self.context.closed, 'Production selection is still live')
        metadata = self.receipt['window_21_prepared_metadata']
        with host_np.load(self.out/'window_21_prepared.npz', allow_pickle=False) as archive:
            successor = OwnedWithdrawal.from_arrays(dict(archive), metadata, device=cfg.device)
        owner, observations = FrozenBodyWindow.load(self.out/'prepared_owner.npz', cfg.device)
        model = None
        report = dict(scope=SCOPE, stages={}, head_merit_scope=
            'Live W20 full-merit closure plus joint head-input closure; no joint full-merit recomputation',
            geometric_F_scope='Joint auxiliary geometric F only; no actual Fg claim when ordinary successor did not track it')
        started = time.perf_counter()
        try:
            model = PostAssimilationWindow(owner, successor, cfg)
            displacement = owner.coefficients[:, :3].clone().requires_grad_()
            terminal = owner.coefficients[:, 3:].clone().requires_grad_()
            with torch.enable_grad():
                values = model.evaluate(terminal, displacement)
                report['health'] = values['health']
                require(values['valid'] and values['pins_exact'], 'Joint head/coast is invalid')
                with host_np.load(self.out/'merit_head.npz', allow_pickle=False) as archive:
                    report['head_inputs'] = {}
                    for key in ('x', 'F', 'C', 'v', 'V', 'positions', 'body_energy'):
                        expected = torch.as_tensor(archive[key], device=cfg.device)
                        unit = (owner.spec.prm.dx if key in ('x', 'positions') else
                                owner.spec.prm.dx/(owner.spec.T*owner.spec.prm.dt)
                                if key in ('v', 'V') else 1/(owner.spec.T*owner.spec.prm.dt)
                                if key == 'C' else 1.)
                        row = closure(values[key], expected, unit)
                        row['bit_exact'] = torch.equal(values[key], expected)
                        report['head_inputs'][key] = row
                require(all(row['passed'] for row in report['head_inputs'].values()), 'Joint head input closure failed')
                arrays = successor.arrays()
                # Physical F, Fp and all carried initial state are checked separately.
                report['actual_boundary'] = {}
                for key, actual, unit in (
                    ('x0', values['coast_X'][0], owner.spec.prm.dx),
                    ('v0', values['coast_V'][0], owner.spec.prm.dx/(owner.spec.T*owner.spec.prm.dt)),
                    ('C0', values['coast_C'][0], 1/(owner.spec.T*owner.spec.prm.dt)),
                    ('F0', values['coast_F'][0].reshape(-1, 3, 3), 1.),
                    ('Fp', values['coast_Fp'], 1.)):
                    expected = torch.as_tensor(arrays[key], device=cfg.device)
                    report['actual_boundary'][key] = closure(actual, expected, unit)
                require(all(row['passed'] for row in report['actual_boundary'].values()), 'Actual handoff boundary differs')
                pins = torch.as_tensor(arrays['pin'], device=cfg.device) > .5
                require(torch.equal(values['coast_pins'], pins), 'Next-pin identity differs')
                report['next_pins'] = int(pins.sum())
                old_pins = torch.as_tensor(to_array(owner.spec.pin), device=cfg.device) > .5
                report['new_pins'] = int((pins & ~old_pins).sum())
                require(report['new_pins'] > 0, 'No actual new-pin witness')
                del arrays
                report['stages']['joint_forward'] = dict(seconds=time.perf_counter()-started,
                    memory=memory_snapshot(cfg.device))
                # The independent continuation uses actual successor x/F/Fp/v/C,
                # not the joint result or a manufactured endpoint substitution.
                with torch.no_grad():
                    independent = successor.trajectory(persistent=True)
                    independent.rollout()
                    report['independent_coast'] = {}
                    for name, key, unit in (
                        ('x', 'coast_X', owner.spec.prm.dx),
                        ('v', 'coast_V', owner.spec.prm.dx/(owner.spec.T*owner.spec.prm.dt)),
                        ('C', 'coast_C', 1/(owner.spec.T*owner.spec.prm.dt)),
                        ('F', 'coast_F', 1.)):
                        # One phase at a time limits extra full-sequence allocations.
                        rows = []
                        for phase, actual in enumerate(getattr(independent, name)):
                            reference = wp.to_torch(actual).reshape_as(values[key][phase])
                            rows.append(closure(values[key][phase], reference, unit))
                        report['independent_coast'][name] = rows
                    require(all(row['passed'] for rows in report['independent_coast'].values()
                                for row in rows), 'Actual independent coast differs from joint')
                    del independent
                mask = ~pins
                require(bool(mask.any()), 'No surviving free cohort')
                losses = coast_losses(values, mask, owner.spec.prm.dt)
                report['free_particles'] = int(mask.sum())
                report['free_losses'] = {key: float(value.detach()) for key, value in losses.items()}
                gradients = {}
                report['gradient'] = {}
                for name, loss in losses.items():
                    pair = torch.autograd.grad(loss, (displacement, terminal), retain_graph=True)
                    report['gradient'][name] = {}
                    for channel, gradient in zip(('displacement', 'terminal'), pair):
                        row = dict(finite=bool(torch.isfinite(gradient).all()),
                            norm=float(gradient.double().norm()), nonzero=bool((gradient != 0).any()))
                        require(row['finite'] and row['nonzero'], 'Invalid future coefficient gradient')
                        report['gradient'][name][channel] = row
                        gradients[name+'__'+channel] = gradient.detach().clone()
                self.save('coast_gradients.npz', gradients)
                self.save('joint_coast.npz', {key: values[key].detach() for key in
                    ('coast_X', 'coast_V', 'coast_F', 'coast_C', 'coast_Fp', 'coast_pins')})
                report['stages']['backward_and_archive'] = dict(seconds=time.perf_counter()-started,
                    memory=memory_snapshot(cfg.device))
                report['passed'] = True
        finally:
            if model is not None:
                model.close()
            owner.close()
            self.receipt['joint'] = report
        return report


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
    metadata_binding = identity(metadata_path)
    cfg, prm = validate_recipe(json.loads(metadata_path.read_text()))
    cfg.stop_after_windows = 21
    paths = [metadata_path, source_path, Path(cfg.target_reference),
             *sorted((repo/'physmorph').rglob('*.py')),
             *(repo/p for p in ('scripts/probes/post_assimilation_window.py',
               'scripts/probes/window_selection.py', 'scripts/probes/prepared_withdrawal.py',
               'scripts/probes/reference_swap.py', 'scripts/ops/run_p303_probe.sh',
               'scripts/ops/cuda_python.py', 'docs/post_assimilation_window_p335.md'))]
    bindings = {str(path): identity(path) for path in paths}
    require(bindings[str(metadata_path)] == metadata_binding, 'Metadata changed while reading')
    with host_np.load(source_path, allow_pickle=False) as archive:
        source, target = archive['src'], archive['tgt']
    require(source.shape == target.shape == (300000, 3), 'Unexpected native inputs')
    capture = Capture(args.out)
    capture.write_json('protocol.json', dict(schema='post_assimilation_window_p335_v1',
        start_utc=datetime.now(timezone.utc).isoformat(), scope=SCOPE, bindings=bindings,
        effective_config=asdict(cfg), mpm=asdict(prm), N=len(source),
        window=20, successor=21, max_output_bytes=3_000_000_000,
        boundary_and_coast_rule='32*FP32_eps*(native_scale+abs(reference)); elementwise'))
    result, failure = None, None
    started = time.perf_counter()
    try:
        with patch.object(runner, 'optimize_window', capture.wrap(runner.optimize_window)):
            result = runner.run_pipeline(source, target, prm, cfg, select_window=capture.select)
        require(capture.context is not None and capture.context.closed, 'Selection context not closed')
        gc.collect()
        with cuda_execution(cfg.device):
            capture.measure(cfg)
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
    stable = True
    try:
        verify_bindings(bindings)
        require(all(identity(args.out/key) == value for key, value in capture.files.items()), 'Evidence changed')
    except Exception as exc:
        stable = False
        failure = failure or dict(type=type(exc).__name__, message=str(exc))
    report = dict(scope=SCOPE, protocol_sha256=sha(args.out/'protocol.json'), failure=failure,
        bindings_unchanged=stable, elapsed_seconds=time.perf_counter()-started,
        receipt=capture.receipt, files=capture.files)
    if result is not None:
        windows = {row['animation']: row for row in result['history'] if 'accepted' in row}
        archive_exact = False
        try:
            if 19 in windows and windows[19].get('frame_end'):
                end = windows[19]['frame_end']
                begin = end-cfg.T-1
                with host_np.load(args.out/'window_20_identity.npz', allow_pickle=False) as evidence, \
                        cuda_execution(cfg.device):
                    archive_exact = all(torch.equal(
                        torch.as_tensor(host_np.stack(result[key][begin:end]), device=cfg.device),
                        torch.as_tensor(evidence[saved], device=cfg.device))
                        for key, saved in (('frames', 'X'), ('F_frames', 'F_sequence')))
        except Exception as exc:
            failure = failure or dict(type=type(exc).__name__, message=str(exc))
            report['failure'] = failure
        report.update(history=result['history'], guards=result['guards'], termination=result['termination'],
            archived_identity_exact=archive_exact,
            donor_outer_committed=bool(windows.get(19, {}).get('outer_accepted', 0)),
            successor_outer_committed=bool(windows.get(20, {}).get('outer_accepted', 0)))
        try:
            report['render_influence'] = write_render_report(args.out/'run', result['history'],
                asdict(cfg), asdict(prm), len(source), reserve_bytes=capture.reserve)
        except Exception as exc:
            failure = failure or dict(type=type(exc).__name__, message=str(exc))
            report['failure'] = failure
    report['passed'] = bool(failure is None and stable and capture.receipt.get('identity_exact')
        and capture.receipt.get('handoff') and capture.receipt.get('joint', {}).get('passed')
        and report.get('archived_identity_exact')
        and report.get('donor_outer_committed') and report.get('successor_outer_committed')
        and result is not None and not any(result['guards'].values()))
    completed = capture.finish(report)
    require(completed, 'P335 gate failed; original state is retained, no changed candidate admitted')


if __name__ == '__main__':
    main()
