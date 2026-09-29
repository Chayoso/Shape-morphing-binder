"""P331 fresh-window joint body search; all production callbacks remain read-only."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as host_np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution, to_array, KDTree, array_api as np
from physmorph.pipeline import runner
from scripts.probes.prepared_withdrawal import (
    Capture, identity, verify_bindings, validate_recipe, safe_json,
    passed_report, require, sha,
)
from scripts.probes.withdrawal_search_core import run_search, SearchFailure
from scripts.probes.withdrawal_quality import RawWithdrawalQuality

LIMIT = 12_000_000_000
SCOPE = 'Read-only original-origin joint body candidate; pre-assimilation only; no adoption or rest/quality promotion'


def write_json(path, value, *, failure_only=False):
    """Reserve actual UTF-8 growth, including rewrites; keep a small error reserve."""
    data = (json.dumps(safe_json(value), indent=2, allow_nan=False)+'\n').encode('utf-8')
    existing = sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    replaced = path.stat().st_size if path.exists() else 0
    reserve = 0 if failure_only else 65_536
    require(existing-replaced+len(data)+reserve <= LIMIT, 'P331 JSON evidence exceeds12GB reservation')
    path.write_bytes(data)


class JointSearchCapture(Capture):
    def __init__(self, out, spacing, source, target, *, window=19, iteration=8):
        super().__init__(out, spacing, window=window, iteration=iteration)
        self.source, self.target = source, target
        self.search = None
        self.forward_records = []

    def reserve(self, size):
        existing = sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
        require(existing+size+2_097_152 <= LIMIT, 'P331 output would exceed12GB; search inconclusive')

    def record(self, label, values, info):
        require(label and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in label), 'Invalid forward label')
        arrays = {key: value for key, value in values.items() if torch.is_tensor(value)}
        extra = info.get('arrays', {})
        require(not arrays.keys() & extra.keys(), 'Repeated archive array name')
        arrays.update(extra)
        item = dict(label=label, archive=label+'.npz', **{k: v for k, v in info.items() if k != 'arrays'})
        self.forward_records.append(item)
        # Persist scalar/failure context even if the evidence reservation fails.
        write_json(self.out/'search_progress.json', dict(scope=SCOPE, forwards=self.forward_records))
        self.save(item['archive'], arrays)
        item['binding'] = self.sidecars[item['archive']]
        write_json(self.out/'search_progress.json', dict(scope=SCOPE, forwards=self.forward_records))

    def observe(self, index, packet):
        super().observe(index, packet)
        if not self.report['measurement_passed']:
            self.search = dict(status='prerequisite_failed', candidate_found=False)
            return
        quality = None
        binding = packet['evaluate_merit'].binding_digest()
        try:
            quality = RawWithdrawalQuality(to_array(self.source), to_array(self.target),
                                           to_array(packet['x0']), to_array(packet['pins']))
            self.search = run_search(packet, record=self.record, raw_observe=quality.observe)
        except SearchFailure as exc:
            self.search = dict(exc.report, status='search_failed', candidate_found=False,
                               error=dict(type=type(exc).__name__, message=str(exc)))
        except (ValueError, AssertionError) as exc:
            self.search = dict(status='search_failed', candidate_found=False,
                               error=dict(type=type(exc).__name__, message=str(exc)))
        finally:
            try:
                if quality is not None:
                    self.save('raw_baseline_envelope.npz', quality.archive_state())
            except (ValueError, OSError) as exc:
                self.search = dict(self.search or {}, status='search_failed', candidate_found=False,
                    archive_error=dict(type=type(exc).__name__, message=str(exc)))
            finally:
                unchanged = packet['evaluate_merit'].binding_digest() == binding
                self.search = self.search or dict(status='search_aborted', candidate_found=False)
                self.search.update(scope=SCOPE, merit_binding_unchanged=unchanged,
                                   forward_records=self.forward_records)
                self.report['merit_binding_unchanged'] &= unchanged
                self.report['sidecars'] = self.sidecars
                self.report['measurement_passed'] = self.measurement_passed()
                try:
                    write_json(self.out/'search.json', self.search)
                except ValueError as exc:
                    self.search = dict(status='search_failed', candidate_found=False,
                        error=dict(type=type(exc).__name__, message=str(exc)),
                        merit_binding_unchanged=unchanged, scope=SCOPE)
                    write_json(self.out/'search.json', self.search, failure_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(args.out.resolve().is_relative_to(args.root.resolve()), 'Output outside project data')
    args.out.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    meta_path = args.root/'work/p303/raw24a.json'
    source_path = args.root/'repro/current_pair/source_render_full_dt_iso_nn.npz'
    initial = identity(meta_path)
    raw = meta_path.read_bytes()
    require(sha256(raw).hexdigest() == initial['sha256'], 'Metadata changed before parsing')
    metadata = json.loads(raw)
    cfg, prm = validate_recipe(metadata)
    files = [meta_path, source_path, Path(cfg.target_reference), *sorted((repo/'physmorph').rglob('*.py')),
        *(repo/p for p in ('scripts/probes/joint_withdrawal_search.py',
            'scripts/probes/withdrawal_search_core.py', 'scripts/probes/withdrawal_quality.py',
            'scripts/probes/prepared_withdrawal.py', 'scripts/probes/reference_swap.py',
            'scripts/ops/run_p303_probe.sh', 'scripts/ops/cuda_python.py',
            'docs/joint_withdrawal_search_p331.md'))]
    bindings = {str(p): identity(p) for p in files}
    require(bindings[str(meta_path)] == initial, 'Metadata changed before binding')
    with host_np.load(source_path, allow_pickle=False) as archive:
        source, target = archive['src'], archive['tgt']
    require(source.shape == target.shape == (300000, 3), 'Unexpected native inputs')
    verify_bindings(bindings)
    protocol = dict(schema='joint_withdrawal_search_p331_v1', scope=SCOPE,
        start_utc=datetime.now(timezone.utc).isoformat(), bindings=bindings,
        source_config=metadata['config'], effective_config=asdict(cfg), mpm=asdict(prm),
        overrides=dict(stop_after_windows=20), N=len(source), window=20, iteration=8,
        max_output_bytes=LIMIT, original_repeats=3, halvings=11, confirmation_repeats=3,
        raw_phases='x0 + head1..T + coast1..T', no_adoption=True)
    write_json(args.out/'protocol.json', protocol)
    result = capture = failure = None
    start = time.perf_counter()
    try:
        with cuda_execution(cfg.device):
            points = to_array(source)
            spacing = float(np.median(KDTree(points).query(points, k=2)[0][:, 1]))
        capture = JointSearchCapture(args.out, spacing, source, target)
        with patch.object(runner, 'optimize_window', capture.wrap(runner.optimize_window)):
            result = runner.run_pipeline(source, target, prm, cfg, on_commit=capture.commit)
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
    stable = True
    try:
        verify_bindings(bindings)
        if capture:
            require(all(identity(args.out/k) == v for k, v in capture.sidecars.items()), 'Saved evidence changed')
    except Exception as exc:
        stable = False
        failure = failure or dict(type=type(exc).__name__, message=str(exc))
    report = dict(protocol_sha256=sha(args.out/'protocol.json'), scope=SCOPE,
        elapsed_seconds=time.perf_counter()-start, failure=failure, bindings_unchanged=stable,
        checkpoint=None if capture is None else capture.report,
        search=None if capture is None else capture.search,
        outer_accepted=False if capture is None else capture.outer_accepted,
        outer_record=None if capture is None else capture.outer_record, candidate_adopted=False)
    if result is not None:
        from physmorph.pipeline.render_reporting import summarize_render_influence
        report.update(history=result['history'], guards=result['guards'], termination=result.get('termination'),
            deliver_n=result['deliver_n'], truncation=result['truncation'],
            render_influence=summarize_render_influence(result['history'],
                dict(asdict(cfg), _particle_count=len(source)), asdict(prm)))
    report['original_capability_and_isolation_passed'] = passed_report(report, result)
    report['candidate_found'] = bool(report['original_capability_and_isolation_passed'] and
        capture.search and capture.search.get('candidate_found') and capture.search.get('merit_binding_unchanged'))
    try:
        write_json(args.out/'result.json', report)
    except ValueError as exc:
        write_json(args.out/'result.json', dict(failure=str(exc), candidate_found=False,
            candidate_adopted=False, scope=SCOPE, complete_result_archived=False), failure_only=True)
        raise
    require(report['original_capability_and_isolation_passed'], 'P331 original capability/isolation gate failed')
    require(capture.search is not None and capture.search.get('status') not in
            ('search_failed', 'search_aborted', 'prerequisite_failed'), 'P331 search incomplete; evidence saved')


if __name__ == '__main__':
    main()
