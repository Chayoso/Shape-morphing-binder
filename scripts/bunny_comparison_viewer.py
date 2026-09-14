"""Serve reviewed server PNGs; no new simulation or rendering."""
import argparse
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physmorph.viewer.paired import PairedReplay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--physics', required=True)
    ap.add_argument('--guided', required=True)
    ap.add_argument('--port', type=int, default=8776)
    ap.add_argument('--label', default='Development comparison; final quality not approved')
    args = ap.parse_args()
    if socket.gethostname() != 'hyde06':
        raise SystemExit('Serve the benchmark on hyde06 through an SSH tunnel')
    viewer = PairedReplay(args.physics, args.guided, args.port, args.label)
    try:
        while True:
            time.sleep(1)
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
