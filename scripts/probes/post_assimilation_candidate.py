"""P336 disposable candidate branch with a prior prepared-policy estimate.

Only the current live selection context supplies merit, gradients and reference.
An actual-policy failure rejects the experimental branch; it never rolls the
runner back after commit. No default, deliverable, rest or full-morph adoption.
"""
import argparse
from copy import deepcopy
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
from physmorph.compute import cuda_execution, is_cuda_execution, to_array
from physmorph.mpm.withdrawal import OwnedWithdrawal
from physmorph.pipeline import optimizer, runner, PipelineConfig
from physmorph.pipeline.endpoint_contract import endpoint_bounds, valid_endpoint
from physmorph.pipeline.render_reporting import write_render_report
from scripts.probes.prepared_withdrawal import identity, verify_bindings, validate_recipe, closure, require, sha, safe_json
from scripts.probes.window_selection import IdentityCapture, exact_tree
from scripts.probes.withdrawal_quality import RawWithdrawalQuality
from scripts.probes.withdrawal_search_core import allowance, energies, energy_comparison


LIMIT = 12_000_000_000
RESERVE = 65_536
ESTIMATE_REVISION = '504ea6a'
SCOPE = ('Conditional fixed-estimate search and one actual controlled successor; '
         'disposable branch, no default/deliverable adoption or matched-prefix causal claim')


def compatible_recipe(current, prm, protocol):
    """Normalize new default fields; prohibit silently changed recipe policies."""
    saved = PipelineConfig(**protocol['effective_config'])
    require(saved.assim_fp64 and current.assim_fp64, 'P336 requires the validated FP64-state recipe')
    require(saved.stop_after_windows == current.stop_after_windows == 21, 'Only the registered cap21 is supported')
    require(safe_json(asdict(current)) == safe_json(asdict(saved)) and safe_json(asdict(prm)) == safe_json(protocol['mpm']),
            'Current recipe differs from the validated estimate recipe')


def policy_comparison(estimate, actual, old_pins):
    """Mismatch is model evidence, separately labelled from raw-quality failure."""
    expected, observed = estimate.arrays(), actual.arrays()
    require(expected.keys() == observed.keys(), 'Prepared policy schemas differ')
    fields = {}
    for key in expected.keys()-{'x0', 'v0', 'C0', 'F0', 'Fp', 'Fg0'}:
        a, b = torch.as_tensor(observed[key]), torch.as_tensor(expected[key])
        require(a.shape == b.shape and a.dtype == b.dtype and a.device == b.device, 'Policy array layout differs: '+key)
        different = a != b
        fields[key] = dict(exact=torch.equal(a, b), different_entries=int(different.sum()))
        if a.is_floating_point():
            fields[key]['max_abs'] = float((a-b).abs().max())
    predicted, received = torch.as_tensor(expected['pin']).bool(), torch.as_tensor(observed['pin']).bool()
    require(not bool((old_pins & ~received).any()), 'Actual successor released old pins')
    cohorts = dict(old_pinned=old_pins, estimated_new=predicted & ~old_pins,
                   actual_new=received & ~old_pins, common_surviving_free=~predicted & ~received)
    return dict(array_differences=fields,
        metadata_exact=estimate.metadata() == actual.metadata(),
        pin_disagreements=int((predicted != received).sum()),
        estimated_only_new=int((predicted & ~received).sum()), actual_only_new=int((received & ~predicted).sum()),
        cohorts={key: int(mask.sum()) for key, mask in cohorts.items()},
        scope='Discrete membership and continuous prepared-policy differences; not by themselves proof of quality loss'), cohorts


def branch_decision(*, selected, outer_committed, successor_committed, archive_exact, guards_clear,
                    actual_check):
    """No successful search/commit can bypass later required observations."""
    required = ('health_passed', 'predicted_closure_passed', 'passive_raw_passed',
                'controlled_raw_passed', 'prepared_constraints_passed', 'common_motion_passed')
    later_passed = bool(actual_check and all(actual_check.get(key) is True for key in required))
    passed = bool(selected and outer_committed and successor_committed and archive_exact and guards_clear and later_passed)
    return dict(candidate_branch_gate_passed=passed,
        experimental_branch_rejected=bool(selected and not passed),
        disposition=('conditional_branch_passed_no_adoption' if passed else
                     'experimental_branch_rejected_no_rollback' if selected else 'original_result_retained'),
        deliverable_promoted=False, matched_prefix_causal_comparison=False,
        missing_actual_checks=[key for key in required if not actual_check or key not in actual_check])


def coast_health(coast, pins, prm):
    finite = all(bool(torch.isfinite(value).all()) for value in coast.values())
    exact = bool(torch.equal(coast['coast_X'][:, pins], coast['coast_X'][0, pins][None].expand(len(coast['coast_X']), -1, -1))
        and (coast['coast_V'][:, pins] == 0).all() and (coast['coast_C'][:, pins] == 0).all())
    minimum = float(torch.linalg.det(coast['coast_F'].reshape(*coast['coast_F'].shape[:2], 3, 3)).min())
    passed = finite and exact and minimum > 1e-4 and valid_endpoint(coast['coast_X'], endpoint_bounds(prm, coast['coast_X']))
    return dict(passed=bool(passed), finite=finite, pins_exact=exact, minimum_det=minimum,
                scope='Physical-state health; ordinary optimizer acceptance separately checks its stress path')


def fail_report(report, error):
    """A late reporting failure cannot retain an earlier successful disposition."""
    report.update(completed=False, failure=dict(type=type(error).__name__, message=str(error)),
        **branch_decision(selected=report.get('candidate_selected', False), outer_committed=False,
            successor_committed=False, archive_exact=False, guards_clear=False, actual_check=report.get('actual_check')))
    if not report.get('candidate_selected', False):
        report['disposition'] = 'driver_incomplete_no_candidate_adoption'


class CandidateCapture(IdentityCapture):
    def __init__(self, out, source, target, estimate, *, enabled=False):
        super().__init__(out)
        self.source, self.target, self.estimate, self.enabled = source, target, estimate, bool(enabled)
        self.quality = self.reference = self.search = None
        self.forward_records = []
        self.old_pins = self.arrived = None

    def reserve(self, size):
        used = sum(path.stat().st_size for path in self.out.iterdir() if path.is_file())
        require(used+size+RESERVE <= LIMIT, 'P336 evidence exceeds reserved12GB; incomplete branch rejected')

    def write_json(self, name, value):
        data = json.dumps(safe_json(value), indent=2, allow_nan=False).encode()
        path = self.out/name
        self.reserve(max(0, len(data)-(path.stat().st_size if path.exists() else 0)))
        path.write_bytes(data)

    def finish(self, report):
        try:
            self.write_json('result.json', report)
        except Exception as error:
            data = json.dumps(dict(completed=False, scope=SCOPE, deliverable_promoted=False,
                failure=dict(type=type(error).__name__, message=str(error)),
                prior_failure=report.get('failure'), experimental_branch_rejected=True,
                detail='Full result unavailable; retained evidence is incomplete, no rollback claimed'),
                allow_nan=False).encode()
            require(len(data) <= RESERVE, 'Failure receipt exceeds reserved bytes')
            with (self.out/'failure.json').open('xb') as stream:
                stream.write(data)
            return False
        return report['completed']

    def wrap(self, original):
        def witnessed(*args, **kwargs):
            index = kwargs['win_index']
            if index in (self.window, self.window+1):
                require(kwargs.get('on_rollout') is None, 'Unexpected existing rollout observer')
                def observe(tr, promoted, win):
                    require(win == index, 'Rollout clock mismatch')
                    arrays = {key: torch.stack([wp.to_torch(value) for value in getattr(tr, name)])
                              for key, name in (('X', 'x'), ('V', 'v'), ('F_sequence', 'F'), ('C_sequence', 'C'))}
                    self.save(f'window_{index+1}_controlled.npz', arrays)
                kwargs['on_rollout'] = observe
            return original(*args, **kwargs)
        return super().wrap(witnessed)

    def record(self, label, values, info):
        require(label and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in label), 'Invalid forward label')
        # Retain physical state and raw phases, without duplicate auxiliary Fg.
        keys = ('positions', 'V', 'F', 'C', 'F_sequence', 'body_energy',
                'coast_X', 'coast_V', 'coast_F', 'coast_C', 'coast_Fp', 'coast_pins')
        arrays = {key: values[key] for key in keys if key in values}
        require(not arrays.keys() & info['arrays'].keys(), 'Repeated evidence array key')
        arrays.update(info['arrays'])
        row = dict(label=label, archive=label+'.npz', **{key: value for key, value in info.items() if key != 'arrays'})
        self.forward_records.append(row)
        self.write_json('search_progress.json', dict(scope=SCOPE, forwards=self.forward_records))
        self.save(row['archive'], arrays)
        row['binding'] = self.files[row['archive']]
        self.write_json('search_progress.json', dict(scope=SCOPE, forwards=self.forward_records))

    def check_handoff(self):
        actual, head = self.successor, self.head
        old, pins = self.start['pin'].bool(), actual['pin'].bool()
        checks = {key: torch.equal(actual[key+'0'], head[key]) for key in ('x', 'F')}
        checks.update(v_free=torch.equal(actual['v0'][~pins], head['v'][~pins]),
            C_free=torch.equal(actual['C0'][~pins], head['C'][~pins]),
            pinned_v_zero=bool((actual['v0'][pins] == 0).all()), pinned_C_zero=bool((actual['C0'][pins] == 0).all()),
            old_pins_retained=bool(pins[old].all()))
        self.receipt['handoff'] = dict(checks=checks, new_pins=int((pins & ~old).sum()),
            surviving_free=int((~pins).sum()), scope='Actual ordinary handoff; zero new pins is reported, not hidden')
        require(all(checks.values()), 'Actual ordinary handoff differs from selected raw head')
        self.start = self.head = self.successor = None

    def select(self, context):
        require(is_cuda_execution() and self.context is None, 'Selection needs one native CUDA context')
        self.context = context
        original = context.original()
        before, _ = context.resolve(original)
        require(exact_tree(before, self.donor), 'Original result already differs from actual donor')
        self.receipt['original_identity_exact'] = True
        self.old_pins = self.start['pin'].bool().clone()
        self.arrived = torch.as_tensor(before[5]['arrived_mask']).bool().clone()
        self.save('current_cohorts.npz', dict(old_pins=self.old_pins, start_arrived=self.arrived,
                                            x0=torch.as_tensor(before[0][0])))
        choice = original
        if self.enabled:
            self.quality = RawWithdrawalQuality(to_array(self.source), to_array(self.target),
                to_array(before[0][0]), to_array(self.old_pins))
            # This is an owned frozen reference, not a retained merit closure.
            self.reference = deepcopy(context._reference)
            try:
                choice, self.search = context.search_post_assimilation(self.estimate,
                    record=self.record, raw_observe=self.quality.observe)
            except Exception as error:
                self.search = getattr(error, 'report', dict(status='error', candidate_found=False))
                self.write_json('search.json', self.search)
                raise
            finally:
                self.save('raw_baseline_envelope.npz', self.quality.archive_state())
            self.write_json('search.json', self.search)
        selected, selection = context.resolve(choice)
        if not selection['selected']:
            require(exact_tree(selected, before), 'No-candidate fallback changed the original result')
        inspection = context.inspect(choice if selection['selected'] else original)
        frames, Fs, end = selected[:3]
        self.head = {key: torch.as_tensor(end[key]).clone() for key in ('F', 'v', 'C')}
        self.head['x'] = torch.as_tensor(frames[-1]).clone()
        self.save('selected_raw_head.npz', dict(X=torch.stack([torch.as_tensor(x) for x in frames]),
            F_sequence=torch.stack([torch.as_tensor(F) for F in Fs]),
            coefficients=inspection['coefficients'], **self.head))
        self.receipt.update(selection_report=selection, no_candidate_original_exact=not selection['selected'],
            prediction_label='confirm_2' if selection['selected'] else
                'baseline_2' if self.search and len(self.search.get('baselines', [])) == 3 else None,
            estimate_scope='Prior identity successor policy; not this branch actual successor')
        self.donor = None
        return choice

    def measure_actual(self, cfg, prm):
        require(self.context is not None and self.context.closed, 'Live selection context must be closed')
        meta = self.receipt['window_21_prepared_metadata']
        with host_np.load(self.out/'window_21_prepared.npz', allow_pickle=False) as archive:
            actual = OwnedWithdrawal.from_arrays(dict(archive), meta, device=cfg.device)
        policy, cohorts = policy_comparison(self.estimate, actual, self.old_pins)
        self.receipt['actual_policy'] = policy
        cohorts['start_arrived_common_surviving_free'] = cohorts['common_surviving_free'] & self.arrived
        self.save('actual_policy_cohorts.npz', {key+'_ids': torch.nonzero(mask).flatten() for key, mask in cohorts.items()})
        if (not self.enabled or self.quality is None or self.quality.stable_covered is None
                or len(self.quality.baseline_labels) != 3 or not self.search.get('ceilings')):
            return dict(scope='No complete current baseline envelope; actual candidate gate not evaluated')
        prediction_label = self.receipt['prediction_label']
        require(prediction_label is not None, 'Missing registered prediction')
        with host_np.load(self.out/(prediction_label+'.npz'), allow_pickle=False) as archive:
            predicted = {key: torch.as_tensor(archive[key], device=cfg.device) for key in
                         ('coast_X', 'coast_V', 'coast_F', 'coast_C', 'coast_Fp')}
        with host_np.load(self.out/'selected_raw_head.npz', allow_pickle=False) as archive:
            head = torch.as_tensor(archive['X'][1:], device=cfg.device)
        with torch.no_grad():
            tr = actual.trajectory(persistent=True)
            tr.rollout()
            coast = {key: torch.stack([wp.to_torch(value) for value in getattr(tr, name)])
                     for key, name in (('coast_X', 'x'), ('coast_V', 'v'), ('coast_F', 'F'), ('coast_C', 'C'))}
            coast['coast_Fp'] = wp.to_torch(tr.Fp).clone()
            pins = wp.to_torch(tr.pin) > .5
            health = coast_health(coast, pins, prm)
            values = dict(positions=head, **coast, valid=health['passed'], pins_exact=health['pins_exact'])
            self.save('actual_passive_coast.npz', coast)
            units = dict(coast_X=prm.dx, coast_V=prm.dx/(cfg.T*prm.dt), coast_F=1., coast_C=1/(cfg.T*prm.dt))
            require(all(len(coast[key]) == len(predicted[key]) == cfg.T+1 and coast[key].numel() == predicted[key].numel()
                        for key in units), 'Actual/predicted coast layout differs')
            comparisons = {key: [closure(value, expected.reshape_as(value), units[key])
                for value, expected in zip(coast[key], predicted[key])] for key in units}
            fp_closure = closure(coast['coast_Fp'], predicted['coast_Fp'], 1.)
            passive_raw = self.quality.observe('actual_passive', values)
            current = energies(values, prm.dt)
            with host_np.load(self.out/'baseline_2.npz', allow_pickle=False) as archive:
                origin_energy = {key: torch.as_tensor(archive[key+'_coast_energy_per_id'], device=cfg.device).double()
                                 for key in ('geometric', 'stored')}
            motion_masks = {key: cohorts[key] for key in ('common_surviving_free', 'start_arrived_common_surviving_free')}
            per_id = energy_comparison(current, origin_energy, motion_masks)
            self.save('actual_passive_energy.npz',
                {**{key+'_energy_per_id': value for key, value in current.items()},
                 **{key+'_change_from_baseline2': current[key]-value for key, value in origin_energy.items()}})
            common = cohorts['common_surviving_free']
            motion = dict(common_particles=int(common.sum()), passed=False)
            if bool(common.any()):
                baseline = []
                for index in range(3):
                    with host_np.load(self.out/f'baseline_{index}.npz', allow_pickle=False) as archive:
                        baseline.append({key: float(torch.as_tensor(archive[key+'_coast_energy_per_id'], device=cfg.device)[common].double().mean())
                                         for key in ('geometric', 'stored')})
                geometric, stored = (float(current[key][common].mean()) for key in ('geometric', 'stored'))
                floor = min(row['geometric'] for row in baseline)
                cap = max(row['stored'] for row in baseline)
                motion.update(geometric=geometric, stored=stored, baselines=baseline,
                    passed=floor-geometric > allowance(floor) and stored <= cap+allowance(cap))
            terms = {key: float(value) for key, value in self.reference.terms(coast['coast_X'][-1]).items()}
            prepared = {key: terms[key] <= self.search['ceilings']['coast_'+key]
                        for key in ('volume', 'render', 'silhouette')}
            del tr
            with host_np.load(self.out/'window_21_controlled.npz', allow_pickle=False) as archive:
                controlled = {key: torch.as_tensor(archive[key], device=cfg.device) for key in ('X', 'V', 'F_sequence', 'C_sequence')}
            controlled_health = coast_health(dict(coast_X=controlled['X'], coast_V=controlled['V'],
                coast_F=controlled['F_sequence'], coast_C=controlled['C_sequence']), pins, prm)
            controlled_raw = self.quality.observe('actual_controlled_successor',
                dict(positions=head, coast_X=controlled['X'], valid=controlled_health['passed'],
                     pins_exact=controlled_health['pins_exact']))
            return dict(health_passed=health['passed'] and controlled_health['passed'],
                passive_health=health, controlled_health=controlled_health,
                predicted_closure=comparisons, Fp_closure=fp_closure,
                predicted_closure_passed=fp_closure['passed'] and all(row['passed'] for rows in comparisons.values() for row in rows),
                closure_scope='Conservative model-closure gate, separate from actual raw quality; differing policy geometry is not itself quality loss',
                passive_raw=passive_raw, passive_raw_passed=passive_raw['passed'],
                controlled_raw=controlled_raw, controlled_raw_passed=controlled_raw['passed'],
                prepared_coast=terms, prepared_constraints=prepared, prepared_constraints_passed=all(prepared.values()),
                common_motion=motion, common_motion_passed=motion['passed'],
                per_id_motion_vs_baseline2=per_id,
                per_id_scope='Observed individual changes on fixed common cohorts; aggregate improvement is not individual rest',
                reference_scope='Owned current W20 prepared reference; no expired complete-merit evaluator used',
                continuation_scope='One ordinary controlled successor; no same-prefix controlled comparator or full-horizon quality claim')


def bind_estimate(directory, cfg, prm, root):
    paths = [directory/name for name in ('protocol.json', 'result.json', 'window_21_prepared.npz')]
    bindings = {str(path): identity(path) for path in paths}
    protocol, result = (json.loads(path.read_text()) for path in paths[:2])
    require(result.get('passed') is True and result.get('bindings_unchanged') is True, 'Estimate native gate did not pass')
    require(sha(paths[0]) == result['protocol_sha256'], 'Estimate protocol digest mismatch')
    require(bindings[str(paths[2])]['sha256'] == result['files'][paths[2].name]['sha256'], 'Estimate archive digest mismatch')
    compatible_recipe(cfg, prm, protocol)
    producer_files = {name: value for name, value in protocol['bindings'].items() if '/physmorph/' in name}
    producer_roots = {Path(name.split('/physmorph/', 1)[0]) for name in producer_files}
    require(len(producer_roots) == 1, 'Ambiguous estimate source revision')
    producer = producer_roots.pop().resolve()
    require(producer.is_relative_to(root.resolve()), 'Estimate code outside data root')
    version = producer/'VERSION'
    require(version.read_text().strip().startswith(ESTIMATE_REVISION), 'Estimate is not validated native2 revision')
    bindings[str(version)] = identity(version)
    for name, expected in producer_files.items():
        bindings[name] = identity(Path(name))
        require(bindings[name]['sha256'] == expected['sha256'], 'Estimate producer source changed: '+name)
    return protocol, result, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--estimate', type=Path)
    parser.add_argument('--enable-candidate-search', action='store_true')
    args = parser.parse_args()
    estimate_path = args.estimate or args.root/'work/p303/p335_native2'
    require(all(path.resolve().is_relative_to(args.root.resolve()) for path in (args.out, estimate_path)), 'I/O outside data root')
    require(all(os.environ.get(key) and Path(os.environ[key]).resolve().is_relative_to(args.root.resolve())
                for key in ('WARP_CACHE_PATH', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH')), 'CUDA caches must stay under data root')
    args.out.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    meta_path = args.root/'work/p303/raw24a.json'
    source_path = args.root/'repro/current_pair/source_render_full_dt_iso_nn.npz'
    meta_binding = identity(meta_path)
    cfg, prm = validate_recipe(json.loads(meta_path.read_text()))
    require(identity(meta_path) == meta_binding, 'Recipe changed while reading')
    cfg.stop_after_windows, cfg.assim_fp64 = 21, True
    estimate_protocol, estimate_result, bindings = bind_estimate(estimate_path, cfg, prm, args.root)
    sources = [meta_path, source_path, Path(cfg.target_reference), repo/'VERSION', *sorted((repo/'physmorph').rglob('*.py')),
        *(repo/name for name in ('scripts/probes/post_assimilation_candidate.py', 'scripts/probes/window_selection.py',
          'scripts/probes/withdrawal_search_core.py', 'scripts/probes/withdrawal_quality.py', 'scripts/probes/prepared_withdrawal.py',
          'scripts/probes/reference_swap.py', 'scripts/ops/cuda_python.py', 'scripts/ops/gpu_env.sh',
          'scripts/ops/run_p303_probe.sh', 'docs/post_assimilation_selection_p336.md'))]
    bindings.update({str(path): identity(path) for path in sources})
    require(bindings[str(meta_path)] == meta_binding, 'Recipe changed before protocol binding')
    for path in (source_path, Path(cfg.target_reference)):
        require(bindings[str(path)]['sha256'] == estimate_protocol['bindings'][str(path)]['sha256'], 'Estimate source/target asset differs')
    with host_np.load(source_path, allow_pickle=False) as archive:
        source, target = archive['src'], archive['tgt']
    require(source.shape == target.shape == (300000, 3), 'Unexpected native particle count')
    with cuda_execution(cfg.device):
        with host_np.load(estimate_path/'window_21_prepared.npz', allow_pickle=False) as archive:
            estimate = OwnedWithdrawal.from_arrays(dict(archive), estimate_result['receipt']['window_21_prepared_metadata'], device=cfg.device)
    capture = CandidateCapture(args.out, source, target, estimate, enabled=args.enable_candidate_search)
    capture.write_json('protocol.json', dict(schema='post_assimilation_candidate_p336_v1', scope=SCOPE,
        start_utc=datetime.now(timezone.utc).isoformat(), bindings=bindings, effective_config=asdict(cfg), mpm=asdict(prm),
        estimate_revision=ESTIMATE_REVISION, estimate=str(estimate_path), candidate_search=args.enable_candidate_search,
        recipe_overrides=dict(stop_after_windows=21, assim_fp64=True),
        max_output_bytes=LIMIT, numerical_closure='32*FP32eps*(native_scale+abs(reference)); unchanged',
        estimate_scope='Separately run identity branch; no exact prefix equality or actual successor claim'))
    result, failure, actual_check = None, None, None
    started = time.perf_counter()
    try:
        with patch.object(runner, 'optimize_window', capture.wrap(runner.optimize_window)):
            result = runner.run_pipeline(source, target, prm, cfg, select_window=capture.select)
        gc.collect()
        with cuda_execution(cfg.device):
            actual_check = capture.measure_actual(cfg, prm)
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    stable = True
    try:
        verify_bindings(bindings)
        require(all(identity(args.out/name) == value for name, value in capture.files.items()), 'Candidate evidence changed')
    except Exception as error:
        stable = False
        failure = failure or dict(type=type(error).__name__, message=str(error))
    windows = {} if result is None else {row['animation']: row for row in result['history'] if 'accepted' in row}
    selected = capture.receipt.get('selection_report', {}).get('selected', False)
    archive_exact = False
    if result is not None and windows.get(19, {}).get('outer_accepted') and windows.get(20, {}).get('outer_accepted'):
        try:
            with cuda_execution(cfg.device):
                checks = []
                for index, name in ((19, 'selected_raw_head.npz'), (20, 'window_21_controlled.npz')):
                    end = windows[index]['frame_end']
                    with host_np.load(args.out/name, allow_pickle=False) as archive:
                        checks.extend(torch.equal(torch.as_tensor(host_np.stack(result[key][end-cfg.T-1:end]), device=cfg.device),
                                                  torch.as_tensor(archive[saved], device=cfg.device))
                                      for key, saved in (('frames', 'X'), ('F_frames', 'F_sequence')))
                archive_exact = all(checks)
        except Exception as error:
            failure = failure or dict(type=type(error).__name__, message=str(error))
    guards_clear = result is not None and not any(result['guards'].values())
    report = dict(scope=SCOPE, completed=bool(failure is None and stable and result is not None and
        capture.receipt.get('original_identity_exact') and capture.receipt.get('handoff')),
        failure=failure, elapsed_seconds=time.perf_counter()-started, protocol_sha256=sha(args.out/'protocol.json'),
        bindings_unchanged=stable, candidate_selected=selected, candidate_outer_committed=bool(windows.get(19, {}).get('outer_accepted') and selected),
        successor_outer_committed=bool(windows.get(20, {}).get('outer_accepted')), archive_exact=archive_exact,
        receipt=capture.receipt, search=capture.search, actual_check=actual_check, files=capture.files,
        **branch_decision(selected=selected, outer_committed=bool(windows.get(19, {}).get('outer_accepted')),
            successor_committed=bool(windows.get(20, {}).get('outer_accepted')), archive_exact=archive_exact,
            guards_clear=guards_clear and stable and failure is None, actual_check=actual_check))
    if result is not None:
        report.update(history=result['history'], guards=result['guards'], termination=result['termination'])
        try:
            report['render_influence'] = write_render_report(args.out/'run', result['history'], asdict(cfg), asdict(prm),
                                                            len(source), reserve_bytes=capture.reserve)
        except Exception as error:
            fail_report(report, error)
    require(capture.finish(report), 'P336 driver incomplete; branch evidence retained, no rollback or adoption')


if __name__ == '__main__':
    main()
