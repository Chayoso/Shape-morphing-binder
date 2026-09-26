"""Immutable denoised shading reference: asset I/O outside CUDA execution."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import numpy as np


def point_hash(points):
    return hashlib.sha256(np.ascontiguousarray(points, np.float32).tobytes()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_input_reference(path, count, *, source_path, target_path, seed, sampler, sample):
    """Load immutable sampled geometry, validating the CLI's input identity."""
    if count <= 0 or sample != 'volume':
        raise ValueError('Input bundles require positive particle count and volume sampling')
    payload = Path(path).read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata.get('input_schema') != 1 or metadata.get('n') != count:
            raise ValueError('Unsupported input bundle schema or particle count')
        source, target = data['src'].copy(), data['tgt'].copy()
        volumes = (float(data['source_volume']), float(data['target_volume']))
    if any(points.shape != (count, 3) or points.dtype != np.float32
           or not np.isfinite(points).all() for points in (source, target)):
        raise ValueError('Input bundle points must be finite float32 arrays of shape (N,3)')
    if any(not np.isfinite(v) or v <= 0 for v in volumes):
        raise ValueError('Input bundle volumes must be finite and positive')
    if (metadata.get('source_sha256') != point_hash(source)
            or metadata.get('target_sha256') != point_hash(target)):
        raise ValueError('Input bundle point hashes do not match its arrays')
    preparation = metadata.get('preparation', {})
    if any(preparation.get(key) != value for key, value in
           (('seed', seed), ('sampler', sampler), ('sample', sample))):
        raise ValueError('CLI sampling options conflict with the prepared input bundle')
    for key, mesh in (('source_mesh_sha256', source_path), ('target_mesh_sha256', target_path)):
        if preparation.get(key) != file_hash(mesh):
            raise ValueError('CLI mesh inputs conflict with the prepared input bundle')
    return {'source': source, 'target': target, 'volumes': volumes,
            'provenance': {'path': str(Path(path).resolve()),
                           'sha256': hashlib.sha256(payload).hexdigest(), 'metadata': metadata}}


def load_target_reference(path, target, factor):
    if not path:
        raise ValueError('CUDA denoised shading needs --target_reference prepared with scripts/prepare_target_reference.py')
    payload = Path(path).read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata.get('schema') != 1 or metadata.get('n') != len(target):
            raise ValueError('Unsupported prepared shading reference schema or point count')
        if metadata['target_sha256'] != point_hash(target) or metadata['disc_ref_factor'] != float(factor):
            raise ValueError('Prepared shading reference does not match target points/discretisation')
        normals, weights = data['normals'].copy(), data['weights'].copy()
        spacing = float(data['spacing'])
    if (normals.shape != np.shape(target) or weights.shape != (len(target),)
            or not np.isfinite(normals).all() or not np.isfinite(weights).all()
            or not np.isfinite(spacing) or spacing <= 0):
        raise ValueError('Prepared shading reference is invalid')
    return {'normals': normals, 'weights': weights, 'spacing': spacing, 'metadata': metadata,
            'provenance': {'path': str(Path(path).resolve()),
                           'sha256': hashlib.sha256(payload).hexdigest(),
                           'metadata': metadata, 'spacing': spacing}}
