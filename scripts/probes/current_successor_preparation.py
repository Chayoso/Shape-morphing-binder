"""P337 current original-head preparation versus its actual ordinary successor.

Identity-only native diagnostic. No prior-prefix estimate, changed candidate,
admission derivative, next-control preview, or natural-rest claim.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as host_np
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution, is_cuda_execution
from physmorph.mpm.withdrawal import OwnedWithdrawal
from physmorph.pipeline import runner
from physmorph.pipeline.render_reporting import write_render_report
from scripts.probes.post_assimilation_candidate import CandidateCapture, coast_health
from scripts.probes.prepared_withdrawal import identity, verify_bindings, validate_recipe, closure, require, sha, safe_json
from scripts.probes.window_selection import exact_tree


LIMIT = 8_000_000_000
RESERVE = 65536
SCOPE = ('Current original-head passive preparation identity; no changed control, '
         'admission derivative, next controlled optimizer preview, rest or quality promotion')


def compare_prepared(predicted, actual, prm):
    a, b = predicted.arrays(), actual.arrays()
    require(a.keys() == b.keys(), 'Prepared array schemas differ')
    pm, am = predicted.metadata(), actual.metadata()
    ignored_meta = {'source_body_control', 'scope'}
    metadata_exact = {key: value for key, value in pm.items() if key not in ignored_meta} == {
        key: value for key, value in am.items() if key not in ignored_meta}
    require(not pm['layer_F'] and not am['layer_F'], 'Passive gate qualification excludes layer_F')
    duration = pm['T']*prm.dt
    exact_fields = {'m', 'lam', 'mu', 'eta', 'vol', 'pin', 'layer_mask', 'layer_nbr', 'bond_nbr', 'bond_frag'}
    units = dict(x0=prm.dx, v0=prm.dx/duration, C0=1/duration, bond_rest=prm.dx)
    rows, gate = {}, {}
    for key in a:
        left, right = torch.as_tensor(a[key]), torch.as_tensor(b[key])
        require(left.shape == right.shape and left.dtype == right.dtype and left.device == right.device,
                'Prepared layout differs: '+key)
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        if key == 'layer_ug':
            gate = dict(finite=finite, exact=torch.equal(left, right), different_entries=int((left != right).sum()),
                        scope='Finite OT u gate retained in actual archive; neutral preview is qualified only by u=0 and layer_F=False')
            continue
        if key in exact_fields:
            rows[key] = dict(passed=finite and torch.equal(left, right), exact=torch.equal(left, right),
                             different_entries=int((left != right).sum()), comparison='exact')
        else:
            rows[key] = dict(**closure(left, right, units.get(key, 1.)), exact=torch.equal(left, right),
                             comparison='32eps native-scale closure')
    return dict(passed=metadata_exact and all(row['passed'] for row in rows.values()) and gate.get('finite', True),
                metadata_exact=metadata_exact, fields=rows, layer_ug=gate,
                source_control_scope='Actual controlled trajectory captured at step0; both replays withdraw dFc/u/body')


def compare_history(predicted, actual, original):
    mapping = dict(previous='displacement', scale='scale', reversals='reversals', frozen='frozen',
                   settled='settled', settled_at='settled_at', pins='pins')
    rows = {}
    for observed, expected in mapping.items():
        require(actual.get(observed) is not None and predicted.get(expected) is not None, 'Missing history '+observed)
        a, b = torch.as_tensor(actual[observed]), torch.as_tensor(predicted[expected])
        if observed == 'pins':
            a, b = a.bool(), b.bool()
        require(a.shape == b.shape and a.dtype == b.dtype and a.device == b.device, 'History layout differs: '+observed)
        rows[observed] = dict(passed=torch.equal(a, b), different_entries=int((a != b).sum()))
    a, b = torch.as_tensor(actual['neighbors']), torch.as_tensor(original['neighbors'])
    rows['neighbors'] = dict(passed=a.shape == b.shape and torch.equal(a, b),
                             scope='Same current source neighborhood, never recomputed from the endpoint')
    return dict(passed=all(row['passed'] for row in rows.values()), fields=rows,
                scale_apply_scope='Predicted array archived; actual runner snapshot exposes scale, not scale_apply',
                u_scale_scope='Not previewed; passive surface control is zero')


def final_gate(report):
    return bool(report.get('completed') and report.get('bindings_unchanged') and report.get('archives_exact')
        and report.get('outer_commits') == [True, True] and report.get('guards_clear')
        and report.get('receipt', {}).get('w20_identity') and report.get('receipt', {}).get('w21_identity')
        and report.get('history_comparison', {}).get('passed')
        and report.get('prepared_comparison', {}).get('passed')
        and report.get('coast_comparison', {}).get('passed'))


class CurrentPreparationCapture(CandidateCapture):
    def __init__(self, out):
        super().__init__(out, None, None, None, enabled=False)
        self.w21_context = self.w21_donor = None

    def configure_progress(self, cfg, prm, source):
        self.progress_config, self.progress_mpm = asdict(cfg), asdict(prm)
        self.source = source.copy()

    def reserve(self, size):
        used = sum(path.stat().st_size for path in self.out.iterdir() if path.is_file())
        require(used+size+RESERVE <= LIMIT, 'P337 evidence exceeds reserved8GB')

    def finish(self, report):
        try:
            self.write_json('result.json', report)
        except Exception as error:
            payload = json.dumps(safe_json(dict(scope=SCOPE, passed=False, completed=False,
                failure=dict(type=type(error).__name__, message=str(error)), prior_failure=report.get('failure'),
                optimizer_progress=self.optimizer_progress_receipt(),
                disposition='incomplete_original_identity_diagnostic', deliverable_promoted=False)), allow_nan=False).encode()
            require(len(payload) <= RESERVE, 'Failure receipt exceeds reserve')
            with (self.out/'failure.json').open('xb') as stream:
                stream.write(payload)
            return False
        return report['passed']

    def wrap(self, original):
        def with_w21_context(*args, **kwargs):
            if kwargs['win_index'] == self.window+1:
                kwargs['prepare_selection'] = True
            result = original(*args, **kwargs)
            if kwargs['win_index'] == self.window+1:
                self.w21_donor = (*result[:5], {key: value for key, value in result[5].items() if key != '_window_selection'})
            return result
        return super().wrap(with_w21_context)

    def select(self, context):
        require(is_cuda_execution(), 'Current preparation needs native CUDA execution')
        choice = context.original()
        inspection = context.inspect(choice)
        window = inspection['window']
        require(window in (self.window, self.window+1), 'Unexpected selection window')
        history = context.current_admission_history()
        require(all(value is not None for value in history.values()), 'Current initialized admission history missing')
        self.save(f'window_{window+1}_admission_history.npz', history)
        if window == self.window:
            choice = super().select(context)
            self.receipt.pop('estimate_scope', None)
            self.receipt['preparation_source'] = 'Current original head and current owned admission history; no prior-prefix estimate'
            before_preview, _ = context.resolve(choice)
            prepared, predicted, scope = context.preview_current_successor()
            self.save('predicted_successor.npz', prepared.arrays())
            self.receipt['predicted_metadata'] = prepared.metadata()
            self.receipt['prediction_scope'] = scope
            keys = ('displacement', 'scale', 'scale_apply', 'reversals', 'frozen', 'active', 'flip',
                    'settled', 'settled_at', 'newly', 'pins')
            self.save('predicted_admission_history.npz', {key: predicted[key] for key in keys})
            self.receipt['predicted_admission_telemetry'] = predicted['telemetry']
            self.receipt['new_pins'] = int(torch.as_tensor(predicted['newly']).sum())
            after, _ = context.resolve(choice)
            self.receipt['w20_identity'] = exact_tree(after, before_preview)
            require(self.receipt['w20_identity'], 'Preview changed original W20')
        else:
            require(self.w21_context is None and self.w21_donor is not None, 'Missing or repeated W21 context')
            self.w21_context = context
            result, report = context.resolve(choice)
            self.receipt['w21_identity'] = exact_tree(result, self.w21_donor) and not report['selected']
            require(self.receipt['w21_identity'], 'W21 original identity changed')
            self.w21_donor = None
        self.write_json('progress.json', dict(scope=SCOPE, receipt=self.receipt, files=self.files))
        return choice

    def load(self, name, cfg):
        with host_np.load(self.out/name, allow_pickle=False) as archive:
            return {key: torch.as_tensor(archive[key], device=cfg.device) for key in archive.files}

    def measure(self, cfg, prm):
        require(self.context is not None and self.context.closed and self.w21_context is not None
                and self.w21_context.closed, 'Both ordinary selection contexts must be closed')
        original = self.load('window_20_admission_history.npz', cfg)
        expected = self.load('predicted_admission_history.npz', cfg)
        observed = self.load('window_21_admission_history.npz', cfg)
        history = compare_history(expected, observed, original)
        predicted = OwnedWithdrawal.from_arrays(self.load('predicted_successor.npz', cfg), self.receipt['predicted_metadata'], device=cfg.device)
        actual = OwnedWithdrawal.from_arrays(self.load('window_21_prepared.npz', cfg), self.receipt['window_21_prepared_metadata'], device=cfg.device)
        for state in (predicted, actual):
            metadata = state.metadata()
            require(metadata['T'] == cfg.T and metadata['step'] == 0
                    and safe_json(metadata['prm']) == safe_json(asdict(prm))
                    and torch.device(metadata['device']) == torch.device(cfg.device),
                    'Prepared state disagrees with current configuration/parameters/device')
        prepared = compare_prepared(predicted, actual, prm)
        del original, expected, observed
        self.write_json('preparation_comparison.json', dict(history=history, prepared=prepared))
        coasts, health = {}, {}
        with torch.no_grad():
            for label, state in (('actual', actual), ('predicted', predicted)):
                tr = state.trajectory(persistent=True)
                tr.rollout()
                arrays = {key: torch.stack([wp.to_torch(value) for value in getattr(tr, name)]) for key, name in
                          (('coast_X', 'x'), ('coast_V', 'v'), ('coast_F', 'F'), ('coast_C', 'C'))}
                pins = wp.to_torch(tr.pin) > .5
                health[label] = coast_health(arrays, pins, prm)
                self.save(label+'_passive_coast.npz', arrays)
                coasts[label] = arrays
                del tr
                gc.collect()
            units = dict(coast_X=prm.dx, coast_V=prm.dx/(cfg.T*prm.dt), coast_F=1., coast_C=1/(cfg.T*prm.dt))
            rows = {key: [closure(a, b, unit) for a, b in zip(coasts['predicted'][key], coasts['actual'][key])]
                    for key, unit in units.items()}
            require(all(len(value) == cfg.T+1 for value in rows.values()), 'Incomplete passive phase comparison')
        return dict(history_comparison=history, prepared_comparison=prepared,
            coast_comparison=dict(passed=all(row['passed'] for field in rows.values() for row in field)
                and all(value['passed'] for value in health.values()), fields=rows, health=health,
                scope='Both passive replays retain their own complete prepared inputs; no boundary substitution'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(args.out.resolve().is_relative_to(args.root.resolve()), 'Output outside data root')
    require(all(os.environ.get(key) and Path(os.environ[key]).resolve().is_relative_to(args.root.resolve())
                for key in ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH')), 'CUDA caches must stay under data root')
    args.out.mkdir(exist_ok=False)
    capture = CurrentPreparationCapture(args.out)
    report = dict(scope=SCOPE, passed=False, completed=False, deliverable_promoted=False, no_candidate_search=True)
    result, bindings = None, {}
    started = time.perf_counter()
    try:
        repo = Path(__file__).resolve().parents[2]
        meta = args.root/'work/p303/raw24a.json'
        source_file = args.root/'repro/current_pair/source_render_full_dt_iso_nn.npz'
        before = identity(meta)
        cfg, prm = validate_recipe(json.loads(meta.read_text()))
        require(identity(meta) == before, 'Recipe changed while reading')
        cfg.stop_after_windows, cfg.assim_fp64 = 21, True
        paths = [meta, source_file, Path(cfg.target_reference), repo/'VERSION',
            *sorted((repo/'physmorph').rglob('*.py')), *sorted((repo/'scripts/probes').glob('*.py')),
            *(repo/name for name in ('scripts/ops/cuda_python.py', 'scripts/ops/gpu_env.sh',
                'scripts/ops/run_p303_probe.sh', 'docs/candidate_commit_contract.md',
                'docs/current_successor_preparation_p337.md'))]
        bindings = {str(path): identity(path) for path in paths}
        require(bindings[str(meta)] == before, 'Recipe changed before binding')
        capture.write_json('protocol.json', dict(scope=SCOPE, bindings=bindings,
            start_utc=datetime.now(timezone.utc).isoformat(), effective_config=asdict(cfg), mpm=asdict(prm),
            recipe_overrides=dict(stop_after_windows=21, assim_fp64=True), max_output_bytes=LIMIT,
            resolved_physics='auto resolves to ot_pace before live callback', prior_estimate=None,
            closure='32*FP32eps*(native_scale+abs(actual)); exact discrete/material/history checks'))
        report['protocol_sha256'] = sha(args.out/'protocol.json')
        with host_np.load(source_file, allow_pickle=False) as archive:
            source, target = archive['src'], archive['tgt']
        require(source.shape == target.shape == (300000, 3), 'Unexpected native particle count')
        capture.configure_progress(cfg, prm, source)
        with patch.object(runner, 'optimize_window', capture.wrap(runner.optimize_window)):
            result = runner.run_pipeline(source, target, prm, cfg, select_window=capture.select)
        gc.collect()
        with cuda_execution(cfg.device):
            report.update(capture.measure(cfg, prm))
            windows = {row['animation']: row for row in result['history'] if 'accepted' in row}
            report['outer_commits'] = [bool(windows.get(index, {}).get('outer_accepted')) for index in (19, 20)]
            checks = []
            for index, filename in ((19, 'selected_raw_head.npz'), (20, 'window_21_controlled.npz')):
                require(index in windows and windows[index].get('frame_end'), 'Missing ordinary archive clock')
                end = windows[index]['frame_end']
                saved = capture.load(filename, cfg)
                for name, key in (('frames', 'X'), ('F_frames', 'F_sequence')):
                    actual = torch.as_tensor(host_np.stack(result[name][end-cfg.T-1:end]), device=cfg.device)
                    checks.append(torch.equal(actual, saved[key]))
            report['archives_exact'] = all(checks)
        report.update(completed=True, guards_clear=not any(result['guards'].values()))
    except Exception as error:
        report.update(completed=False, failure=dict(type=type(error).__name__, message=str(error)))
    try:
        verify_bindings(bindings)
        require(all(identity(args.out/name) == value for name, value in capture.files.items()), 'Archived evidence changed')
        report['bindings_unchanged'] = True
    except Exception as error:
        report.update(completed=False, bindings_unchanged=False,
                      binding_failure=dict(type=type(error).__name__, message=str(error)))
    report.update(receipt=capture.receipt, files=capture.files, elapsed_seconds=time.perf_counter()-started,
                  optimizer_progress=capture.optimizer_progress_receipt())
    if result is not None:
        report.update(history=result['history'], guards=result['guards'], termination=result['termination'])
        try:
            report['render_influence'] = write_render_report(args.out/'run', result['history'],
                asdict(cfg), asdict(prm), len(source), reserve_bytes=capture.reserve)
        except Exception as error:
            report.update(completed=False, render_reporting_failure=dict(type=type(error).__name__, message=str(error)))
    report['passed'] = final_gate(report)
    report['disposition'] = 'original_identity_preparation_passed' if report['passed'] else 'original_identity_preparation_incomplete_or_failed'
    require(capture.finish(report), 'P337 original preparation gate incomplete or failed; bounded evidence retained')


if __name__ == '__main__':
    main()
