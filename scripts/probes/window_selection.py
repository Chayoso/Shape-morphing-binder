"""P332: exact original-result selection followed by the actual W21 handoff.

No changed candidate is admitted. Numerical comparisons stay on CUDA; host
transfers serialize evidence. The successor is the ordinary controlled solve.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as host_np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution, is_cuda_execution, to_host
from physmorph.mpm.withdrawal import OwnedWithdrawal
from physmorph.pipeline import optimizer, runner
from physmorph.pipeline.render_reporting import write_render_report
from scripts.probes.prepared_withdrawal import (
    identity, verify_bindings, validate_recipe, tensor_bytes, safe_json, require, sha)


LIMIT = 3_000_000_000
SCOPE = ('Original-result identity through the real handoff and controlled successor; '
         'not a candidate remedy, resumable fork, handoff derivative, rest or 4K quality gate')


def exact_tree(actual, expected):
    """No numerical tolerance for identity; compare arrays on their live device."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(exact_tree(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (type(actual) is type(expected) and len(actual) == len(expected)
                and all(exact_tree(a, b) for a, b in zip(actual, expected)))
    if hasattr(expected, 'shape') and hasattr(expected, 'dtype'):
        if not (hasattr(actual, 'shape') and actual.shape == expected.shape and actual.dtype == expected.dtype):
            return False
        a, b = torch.as_tensor(actual), torch.as_tensor(expected)
        return a.device == b.device and torch.equal(a, b)
    return actual == expected


class IdentityCapture:
    def __init__(self, out, window=19):
        self.out, self.window = Path(out), window
        self.files, self.receipt = {}, {}
        self.donor = self.context = self.start = self.head = self.successor = None

    def save(self, name, arrays):
        self.reserve(tensor_bytes(arrays)+1_048_576)
        path = self.out/name
        with path.open('xb') as stream:
            host_np.savez_compressed(stream, **{k: to_host(v) for k, v in arrays.items()})
        self.files[name] = identity(path)

    def reserve(self, size):
        used = sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
        require(used+size <= LIMIT, 'P332 exceeds reserved output bytes')

    def write_json(self, name, value):
        payload = json.dumps(safe_json(value), indent=2, allow_nan=False).encode('utf-8')
        self.reserve(len(payload))
        with (self.out/name).open('xb') as stream:
            stream.write(payload)

    def wrap(self, original):
        def observed(*args, **kwargs):
            index = kwargs['win_index']
            # The callback boundary is required only for the registered W20.
            kwargs['prepare_selection'] = index == self.window
            if index not in (self.window, self.window+1):
                return original(*args, **kwargs)
            made = []
            constructor = optimizer.Trajectory
            def remember(*pos, **kw):
                tr = constructor(*pos, **kw)
                made.append(tr)
                return tr
            with patch.object(optimizer, 'Trajectory', remember):
                result = original(*args, **kwargs)
            require(len(made) == 1, 'Expected one ordinary prepared trajectory')
            state = OwnedWithdrawal.capture(made[0], 0)
            arrays = state.arrays()
            self.save(f'window_{index+1}_prepared.npz', arrays)
            self.receipt[f'window_{index+1}_prepared_metadata'] = state.metadata()
            if index == self.window:
                require(result[5].get('_window_selection') is not None, 'W20 has no eligible final context')
                self.start = {k: torch.as_tensor(arrays[k]).clone() for k in ('Fp', 'pin')}
                self.donor = (*result[:5], {k: v for k, v in result[5].items() if k != '_window_selection'})
            else:
                require(self.context is not None and self.context.closed, 'Selection lease not closed before successor')
                self.successor = {k: torch.as_tensor(arrays[k]).clone()
                                  for k in ('x0', 'v0', 'F0', 'C0', 'Fp', 'pin')}
                self.receipt['successor_inner_accepted'] = result[5]['accepted']
                self.check_handoff()
            return result
        return observed

    def select(self, context):
        require(is_cuda_execution() and self.context is None, 'Unexpected callback context/count')
        self.context = context
        choice = context.original()
        inspection = context.inspect(choice)
        require(inspection['window'] == self.window and inspection['identity'], 'Wrong selected window')
        require(inspection['values']['positions'].is_cuda, 'Selection was converted to host')
        inspection['values']['positions'].zero_()
        inspection['values']['F_sequence'].zero_()
        inspection['coefficients'].fill_(99.)
        selected, report = context.resolve(choice)
        require(exact_tree(selected, self.donor), 'Identity changed original numerical state/history')
        frames, Fs, end, _, _, _ = selected
        self.head = {k: torch.as_tensor(end[k]).clone() for k in ('F', 'v', 'C')}
        self.head['x'] = torch.as_tensor(frames[-1]).clone()
        self.save('window_20_identity.npz', dict(
            X=torch.stack([torch.as_tensor(x) for x in frames]),
            F_sequence=torch.stack([torch.as_tensor(F) for F in Fs]), **self.head))
        self.receipt.update(identity_exact=True, selection_report=report,
                            identity_replay_count=0, inspection_mutation_isolated=True)
        self.donor = None
        return choice

    def check_handoff(self):
        next_state, head = self.successor, self.head
        require(next_state is not None and head is not None, 'Missing actual handoff side')
        old_pin, next_pin = self.start['pin'] > .5, next_state['pin'] > .5
        newly = next_pin & ~old_pin
        equal = {key: torch.equal(next_state[key+'0'], head[key]) for key in ('x', 'F')}
        equal.update(v_free=torch.equal(next_state['v0'][~next_pin], head['v'][~next_pin]),
                     C_free=torch.equal(next_state['C0'][~next_pin], head['C'][~next_pin]),
                     v_pinned_zero=bool((next_state['v0'][next_pin] == 0).all()),
                     C_pinned_zero=bool((next_state['C0'][next_pin] == 0).all()),
                     old_pins_retained=bool(next_pin[old_pin].all()))
        self.receipt['handoff'] = dict(checks=equal, new_pins=int(newly.sum()),
            surviving_free=int((~next_pin).sum()),
            Fp_changed_particles=int((next_state['Fp'] != self.start['Fp']).any(2).any(1).sum()),
            Fp_scope='Actual prepared successor after production assimilation and pin exceptions; not a frozen-Fp coast')
        require(all(equal.values()), 'Whole-state handoff mismatch')
        require(bool(newly.any()), 'Registered W20 lacks a positive new-pin witness')
        self.start = self.head = self.successor = None


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
    metadata = json.loads(metadata_path.read_text())
    cfg, prm = validate_recipe(metadata)
    cfg.stop_after_windows = 21
    paths = [metadata_path, source_path, Path(cfg.target_reference),
             *sorted((repo/'physmorph').rglob('*.py')),
             *(repo/p for p in ('scripts/probes/window_selection.py', 'scripts/probes/prepared_withdrawal.py',
                'scripts/probes/reference_swap.py', 'scripts/ops/run_p303_probe.sh',
                'scripts/ops/cuda_python.py', 'docs/window_selection_p332.md'))]
    bindings = {str(p): identity(p) for p in paths}
    require(bindings[str(metadata_path)] == metadata_binding, 'Metadata changed during reading')
    with host_np.load(source_path, allow_pickle=False) as archive:
        source, target = archive['src'], archive['tgt']
    require(source.shape == target.shape == (300000, 3), 'Unexpected native inputs')
    verify_bindings(bindings)
    capture = IdentityCapture(args.out)
    capture.write_json('protocol.json', dict(schema='window_selection_p332_v1',
        start_utc=datetime.now(timezone.utc).isoformat(), scope=SCOPE, bindings=bindings,
        effective_config=asdict(cfg), mpm=asdict(prm), N=len(source),
        window=20, successor=21, max_output_bytes=LIMIT, identity='exact; no tolerance'))
    result, failure = None, None
    start = time.perf_counter()
    try:
        with patch.object(runner, 'optimize_window', capture.wrap(runner.optimize_window)):
            result = runner.run_pipeline(source, target, prm, cfg, select_window=capture.select)
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
    stable = True
    try:
        verify_bindings(bindings)
        require(all(identity(args.out/k) == v for k, v in capture.files.items()), 'Evidence changed')
    except Exception as exc:
        stable = False
        failure = failure or dict(type=type(exc).__name__, message=str(exc))
    report = dict(scope=SCOPE, protocol_sha256=sha(args.out/'protocol.json'), failure=failure,
        bindings_unchanged=stable, elapsed_seconds=time.perf_counter()-start,
        receipt=capture.receipt, files=capture.files,
        context_closed=capture.context is not None and capture.context.closed)
    if result is not None:
        windows = {r['animation']: r for r in result['history'] if 'accepted' in r}
        archive_exact = False
        if 19 in windows and windows[19].get('frame_end'):
            end = windows[19]['frame_end']
            begin = end-cfg.T-1
            with host_np.load(args.out/'window_20_identity.npz', allow_pickle=False) as evidence, \
                    cuda_execution(cfg.device):
                archive_exact = all(torch.equal(
                    torch.as_tensor(host_np.stack(result[key][begin:end]), device=cfg.device),
                    torch.as_tensor(evidence[saved], device=cfg.device))
                    for key, saved in (('frames', 'X'), ('F_frames', 'F_sequence')))
        report.update(history=result['history'], guards=result['guards'], termination=result['termination'],
            archived_identity_exact=archive_exact,
            donor_outer_committed=bool(windows.get(19, {}).get('outer_accepted', 0)),
            successor_outer_committed=bool(windows.get(20, {}).get('outer_accepted', 0)),
            render_influence=write_render_report(args.out/'run', result['history'], asdict(cfg), asdict(prm), len(source),
                                                 reserve_bytes=capture.reserve))
    report['passed'] = bool(failure is None and stable and report['context_closed']
        and capture.receipt.get('identity_exact') and capture.receipt.get('handoff')
        and report.get('donor_outer_committed') and report.get('archived_identity_exact')
        and result is not None and not any(result['guards'].values()))
    capture.write_json('result.json', report)
    require(report['passed'], 'P332 capability gate failed; evidence saved, no changed candidate admitted')


if __name__ == '__main__':
    main()
