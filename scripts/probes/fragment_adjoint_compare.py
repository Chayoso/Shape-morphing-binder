"""P327: same-code raw runs differing only in reverse fragment-mask lifetime."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.mpm.function import PersistentAdjoint
from physmorph.mpm.traj import Trajectory
from scripts.probes.coverage_paths import require, sha


class FragmentAdjointObserver:
    """Own observation buffers in both arms, including the legacy allocation."""
    def __init__(self, mode):
        if mode not in ('legacy', 'retained'):
            raise ValueError('Unknown fragment adjoint mode')
        self.mode = mode
        self.attempt = None
        self.models = []
        self.rows = []

    def attach(self, tr):
        if not tr.bonds or tr.share_grid:
            return
        require(len({v.ptr for v in tr.frag_steps}) == tr.T, 'Expected retained-mask input code')
        tr._p327_retained = tuple(tr.frag_steps)  # Equal allocation/lifetime in both arms.
        tr._p327_observed = tuple(wp.empty_like(v) for v in tr.frag_steps)
        tr._p327_model = len(self.models)
        if self.mode == 'legacy':
            tr.frag_steps = [tr.frag_steps[0]]*tr.T
            tr.frag_step = tr.frag_steps[0]
        self.models.append(dict(model=tr._p327_model, attempt=self.attempt,
            N=tr.N, T=tr.T, device=str(tr.device),
            reverse_unique_buffers=len({a.ptr for a in tr.frag_steps}),
            retained_allocations=len({a.ptr for a in tr._p327_retained}),
            observation_buffers=len({a.ptr for a in tr._p327_observed})))

    def observe(self, adj):
        tr = adj.traj
        if not hasattr(tr, '_p327_observed'):
            return
        with torch.no_grad():
            masks = torch.stack([wp.to_torch(v) > .5 for v in tr._p327_observed])
            permanent = wp.to_torch(tr.bond_frag) > .5
            free = wp.to_torch(tr.pin) <= .5
            layer = wp.to_torch(tr.layer_mask) > .5 if tr.layer else torch.zeros_like(free)
            cohorts = (torch.ones_like(free), free, free & layer)
            fields = []
            for cohort in cohorts:
                fields.append(cohort.sum().reshape(1))
                for values in (masks, masks & ~permanent[None], masks != masks[-1:],
                               masks[1:] != masks[:-1]):
                    fields.append((values & cohort[None]).sum(1))
            # A single explicit scalar packet; no state array downloads or CPU geometry.
            packet = torch.cat(fields).cpu().tolist()
        offset, summaries = 0, {}
        for label in ('all', 'free', 'layer_free'):
            summaries[label] = dict(count=int(packet[offset])); offset += 1
            for key, length in (('active', tr.T), ('dynamic_only', tr.T),
                                ('different_from_final', tr.T), ('temporal_flips', tr.T-1)):
                summaries[label][key] = [int(v) for v in packet[offset:offset+length]]
                offset += length
        require(offset == len(packet), 'Fragment observation packet layout')
        self.rows.append(dict(forward=len(self.rows), model=tr._p327_model,
                              attempt=self.attempt, cohorts=summaries))

    def install(self, stack, runner):
        original_init, original_bonds = Trajectory.__init__, Trajectory._bond_args
        original_forward, original_solve = PersistentAdjoint.forward, runner.optimize_window
        def initialize(tr, *args, **kwargs):
            original_init(tr, *args, **kwargs)
            self.attach(tr)
        def bond_args(tr, step):
            values = original_bonds(tr, step)
            if hasattr(tr, '_p327_observed'):
                wp.copy(tr._p327_observed[step], values[2])
            return values
        def forward(adj):
            result = original_forward(adj)
            self.observe(adj)
            return result
        def solve(*args, **kwargs):
            previous = self.attempt
            self.attempt = int(kwargs['win_index'])
            try:
                return original_solve(*args, **kwargs)
            finally:
                self.attempt = previous
        stack.enter_context(patch.object(Trajectory, '__init__', initialize))
        stack.enter_context(patch.object(Trajectory, '_bond_args', bond_args))
        stack.enter_context(patch.object(PersistentAdjoint, 'forward', forward))
        stack.enter_context(patch.object(runner, 'optimize_window', solve))


def write_json(path, values):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(values, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/relcfd/chayo/physmorph_v2'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('legacy', 'retained'), required=True)
    args = parser.parse_args()
    require(args.out.resolve().is_relative_to(args.root.resolve()), 'Output outside project data')
    from physmorph.pipeline import runner
    from scripts.probes import full_horizon
    code_root = Path(__file__).resolve().parents[2]
    sources = {str(code_root / name): sha(code_root / name) for name in
               ('scripts/probes/fragment_adjoint_compare.py', 'docs/fragment_adjoint_p327.md')}
    protocol_path = args.out.with_suffix('.fragment_protocol.json')
    write_json(protocol_path, dict(mode=args.mode, code=sources,
        utc=datetime.now(timezone.utc).isoformat(),
        scope='Identical raw recipe and primal kernels; only reverse fragment-mask lifetime differs',
        observations='Every executed PersistentAdjoint forward, including attempted/repeated states; not accepted endpoints',
        no_withdrawal_objective=True, storage_reservation_bytes=30000000000))
    observer = FragmentAdjointObserver(args.mode)
    cli = ['full_horizon.py', '--root', str(args.root), '--arm', 'raw', '--out', str(args.out)]
    with ExitStack() as stack:
        observer.install(stack, runner)
        stack.enter_context(patch.object(sys, 'argv', cli))
        full_horizon.main()
    require(all(sha(Path(path)) == digest for path, digest in sources.items()), 'Wrapper source changed')
    require(bool(observer.models and observer.rows), 'No bonded adjoint observations were executed')
    require(all(row['N'] == 300000 and row['T'] == 20 for row in observer.models),
            'Unexpected production observation discretization')
    trace_path = args.out.with_suffix('.rest_trace.json')
    write_json(args.out.with_suffix('.fragment_activity.json'), dict(mode=args.mode,
        protocol_sha256=sha(protocol_path), rest_trace_sha256=sha(trace_path),
        models=observer.models, forwards=observer.rows, code_unchanged=True))


if __name__ == '__main__':
    main()
