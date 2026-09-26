"""CPU asset preparation: immutable sampled geometry plus denoised shading input.

The strict CUDA CLI consumes these exact arrays, without resampling meshes under
its separate numerical dependencies. Existing outputs are never overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physmorph.input_assets import point_hash, file_hash
from physmorph.pipeline.config import PipelineConfig, disc_ref_factor
from physmorph.render.surface_recon import target_surface_normals
from physmorph.sampling import load_normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out')
    parser.add_argument('--src', default='assets/isosphere.obj')
    parser.add_argument('--tgt', required=True)
    parser.add_argument('--n', type=int, default=300000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sampler', choices=['replacement', 'stratified'], default='stratified')
    parser.add_argument('--disc_ref', action='store_true')
    parser.add_argument('--mass_ref_n', type=int, default=40000)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    if args.n < 9:
        raise ValueError('Input preparation requires at least 9 particles')
    source, source_volume = load_normalized(args.src, args.n, args.seed,
                                            return_volume=True, sample=args.sampler)
    target, target_volume = load_normalized(args.tgt, args.n, args.seed + 1,
                                            match_volume=source_volume, return_volume=True,
                                            sample=args.sampler)
    factor = disc_ref_factor(args.n, PipelineConfig(disc_ref=args.disc_ref,
                                                   mass_ref_n=args.mass_ref_n))
    count = min(args.n, 20000)
    subset = target[np.random.default_rng(0).choice(args.n, count, replace=False)]
    spacing = float(np.median(cKDTree(subset).query(subset, k=9, workers=-1)[0][:, -1]))
    spacing *= (count / args.n) ** (1 / 3) * factor
    normals, weights = target_surface_normals(target, spacing)
    metadata = {'schema': 1, 'input_schema': 1, 'n': args.n,
                'source_sha256': point_hash(source), 'target_sha256': point_hash(target),
                'disc_ref_factor': factor, 'method': 'existing target_surface_normals Poisson input',
                'preparation': {'source_path': str(Path(args.src).resolve()),
                                'target_path': str(Path(args.tgt).resolve()),
                                'source_mesh_sha256': file_hash(args.src),
                                'target_mesh_sha256': file_hash(args.tgt),
                                'seed': args.seed, 'sampler': args.sampler, 'sample': 'volume'}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        np.savez_compressed(stream, src=source, tgt=target, source_volume=source_volume,
                            target_volume=target_volume, normals=normals, weights=weights,
                            spacing=spacing, metadata=json.dumps(metadata))
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'path': str(out.resolve()), 'sha256': file_hash(out), **metadata}), flush=True)


if __name__ == '__main__':
    main()
