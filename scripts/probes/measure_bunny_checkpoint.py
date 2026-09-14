"""Read-only development score of accepted checkpoints; never selects a control."""
import argparse
import io
import json
from pathlib import Path
import socket
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.bunny_response_benchmark import load_fixture, raw_quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    args = ap.parse_args()
    if socket.gethostname() != 'hyde06':
        raise SystemExit('benchmark measurements run on hyde06')
    data, meta = load_fixture(args.fixture)
    for name in args.runs:
        folder = Path(name)
        path = folder/'latest_state.npz'
        with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as a:
            x = a['x']
        print(json.dumps({'run': name, 'checkpoint_mtime': path.stat().st_mtime,
            'scope': 'development diagnostic, not last-iterate sealed evidence',
            'N': len(x), 'discretization': meta['discretization'],
            'quality': raw_quality(x, data['tgt'], meta['extent'])}), flush=True)


if __name__ == '__main__':
    main()
