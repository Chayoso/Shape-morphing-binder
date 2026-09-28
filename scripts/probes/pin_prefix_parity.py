"""CUDA first-two-window no-pin treatment versus same-policy repeat noise.

This does not establish a noise distribution or a rest/coverage result. It
checks whether the treatment diverges before either control admits a pin.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution, array_api as np, to_array
from scripts.probes.render_influence import load_run


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True)
    parser.add_argument('--repeat', required=True)
    parser.add_argument('--no-pin', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    paths = [Path(prefix+suffix) for prefix in (args.control, args.repeat, args.no_pin)
             for suffix in ('.json', '_render_full_dt_iso_nn.npz')]
    receipt = Path(args.no_pin+'.pin_schema.json')
    if receipt.exists():
        paths.append(receipt)
    identities = {str(path):digest(path) for path in paths}
    runs = [load_run(prefix) for prefix in (args.control, args.repeat, args.no_pin)]
    first = runs[0]['meta']
    for ordinal, run in enumerate(runs):
        meta = run['meta']
        arm = meta['arms']['render_full_dt_iso_nn']
        if any(meta[key] != arm[key] for key in ('config', 'history', 'guards')):
            raise ValueError('Inconsistent top-level/arm metadata')
        changes = {k for k in set(meta['config']) | set(first['config'])
                   if meta['config'].get(k) != first['config'].get(k)}
        assert changes == ({'settle_pin'} if ordinal == 2 else set())
        assert meta['config']['settle_pin'] is (ordinal != 2)
        assert meta['code_sha256'] == first['code_sha256'] and meta['mpm'] == first['mpm']
        assert not any(meta['guards'].values()) and meta['config']['compute_backend'] == 'cuda'
        assert meta['config']['T'] == 20 and meta['mpm']['dt'] == 1/240
        assert [r['frame_end'] for r in run['records'][:2]] == [21, 41]
        assert all(r.get('pinned_frac', 0.) == 0. for r in run['records'][:2])
    with cuda_execution('cuda'):
        for run in runs[1:]:
            assert bool(np.array_equal(to_array(run['source']), to_array(runs[0]['source'])))
            assert bool(np.array_equal(to_array(run['target']), to_array(runs[0]['target'])))
        rows = []
        for frame in range(41):
            states = [to_array(run['frames'][frame]) for run in runs]
            if not all(bool(np.isfinite(state).all()) for state in states):
                raise ValueError('Invalid prefix state')
            delta = [np.linalg.norm(state-states[0], axis=1) for state in states[1:]]
            rows.append(dict(frame=frame, repeat_rms_wu=float(np.sqrt((delta[0]**2).mean())),
                             treatment_rms_wu=float(np.sqrt((delta[1]**2).mean())),
                             repeat_max_wu=float(delta[0].max()), treatment_max_wu=float(delta[1].max())))
    result = dict(scope='first two windows before observed pin admission; one repeat is not a noise distribution',
                  code_sha256=first['code_sha256'], mpm=first['mpm'], N=len(runs[0]['source']), T=20,
                  loss_res=first['config']['loss_res'], rows=rows,
                  input_sha256=identities,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  metadata_sha256={run['prefix']:hashlib.sha256(Path(run['prefix']+'.json').read_bytes()).hexdigest()
                                   for run in runs})
    if identities != {str(path):digest(path) for path in paths}:
        raise ValueError('Inputs changed during parity analysis')
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
