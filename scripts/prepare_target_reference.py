"""Prepare the existing Poisson shading input once; this asset stage uses CPU work.

The CUDA execution path loads this immutable artifact and never reconstructs a
Poisson mesh. This preserves the existing loss rather than replacing its normals.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warp as wp
if os.environ.get('WARP_CACHE_PATH'):
    wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']

from physmorph.input_assets import point_hash
from physmorph.pipeline.config import PipelineConfig, disc_ref_factor
from physmorph.render.surface_recon import target_surface_normals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', help='existing pipeline NPZ with tgt points')
    parser.add_argument('out')
    parser.add_argument('--disc_ref', action='store_true')
    parser.add_argument('--mass_ref_n', type=int, default=40000)
    args = parser.parse_args()
    with np.load(args.archive, allow_pickle=False) as data:
        points = np.ascontiguousarray(data['tgt'], np.float32)
    count = min(len(points), 20000)
    subset = points[np.random.default_rng(0).choice(len(points), count, replace=False)]
    factor = disc_ref_factor(len(points), PipelineConfig(disc_ref=args.disc_ref, mass_ref_n=args.mass_ref_n))
    spacing = float(np.median(cKDTree(subset).query(subset, k=9, workers=-1)[0][:, -1]))
    spacing *= (count / len(points)) ** (1.0 / 3.0)
    spacing *= factor
    normals, weights = target_surface_normals(points, spacing)
    metadata = {'schema': 1, 'target_sha256': point_hash(points), 'disc_ref_factor': factor,
                'method': 'existing target_surface_normals Poisson input', 'n': len(points)}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        np.savez_compressed(output, normals=normals, weights=weights, spacing=spacing,
                            metadata=json.dumps(metadata))
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({'path': str(path), **metadata}), flush=True)


if __name__ == '__main__':
    main()
