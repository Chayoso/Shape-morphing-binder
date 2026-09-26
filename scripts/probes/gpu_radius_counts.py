"""Hyde06 radius-count regression on saved production clouds; no simulation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
from physmorph.compute import cuda_execution, KDTree, to_array, to_host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--native', action='store_true', help='reproduce upstream CuPy radius kernel failure')
    parser.add_argument('--queries', type=int, default=20000, help='0 checks every point')
    args = parser.parse_args()
    with np.load(args.archive) as data:
        source, target = data['src'], data['tgt']
    radius = float(np.median(cKDTree(target).query(target, k=9, workers=8)[0][:, -1]))
    for name, points in (('source', source), ('target', target)):
        sample = np.linspace(0, len(points)-1, min(args.queries or len(points), len(points)), dtype=np.int64)
        queries = points[sample]
        expected = cKDTree(points).query_ball_point(queries, radius, return_length=True, workers=8)
        start = time.monotonic()
        with cuda_execution('cuda'):
            import cupy as cp
            tree = KDTree(points)
            q = to_array(queries, np.float64)
            print(json.dumps(dict(stage='query', cloud=name, n=len(points), queries=len(q),
                                  radius=radius, native=args.native)), flush=True)
            if args.native:
                actual = tree.tree.query_ball_point(q, radius, return_length=True)
            else:
                actual = tree.query_ball_point(q, radius, return_length=True)
            cp.cuda.get_current_stream().synchronize()
            actual = to_host(actual)
        np.testing.assert_array_equal(actual, expected)
        print(json.dumps(dict(stage='PASS', cloud=name, seconds=time.monotonic()-start,
                              max_count=int(actual.max()), mean_count=float(actual.mean()))), flush=True)


if __name__ == '__main__':
    main()
