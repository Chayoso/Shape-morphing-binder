"""Repair diagnostic NPZ optional-field schema without loading pickle or changing states.

Creates a new prefix; original evidence is untouched. Only scalar-object pin
members from a verified disabled-pin policy are replaced by typed empty masks.
Every other ZIP member is streamed byte-for-byte and hash-verified afterward.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(4*1024*1024), b''):
        h.update(block)
    return h.hexdigest()


def normalize(source, target):
    suffix = '_render_full_dt_iso_nn.npz'
    metadata_path = Path(str(source)+'.json')
    metadata = json.loads(metadata_path.read_text())
    cfg = metadata['config']
    arm = metadata['arms']['render_full_dt_iso_nn']
    if any(metadata[key] != arm[key] for key in ('config', 'history', 'guards')):
        raise ValueError('Inconsistent run/arm metadata')
    if (cfg.get('settle_pin') is not False or any(cfg.get(key) for key in (
            'freeze_arrived', 'settle_eta', 'rest_commit', 'settle_pin_yield', 'settle_pin_follow', 'settle_pin_kkt'))
            or any((row.get(key) or 0) > 0 for row in metadata['history']
                   for key in ('pinned_frac', 'frozen_frac', 'settled_frac'))
            or any(metadata['guards'].values())):
        raise ValueError('Only a verified disabled-pin policy can use this schema repair')
    outputs = [Path(str(target)+end) for end in ('.json', '.npz', '.log', suffix, '.pin_schema.json')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Use a new result prefix')
    replacements = ('pinned.npy', 'pinned_at.npy')
    original_hashes = {}
    with zipfile.ZipFile(str(source)+suffix) as old:
        if len(set(old.namelist())) != len(old.namelist()):
            raise ValueError('Duplicate archive member names')
        with old.open('src.npy') as stream:
            points = np.load(stream, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Invalid particle layout')
        with old.open('frames.npy') as stream:
            version = np.lib.format.read_magic(stream)
            if version != (1, 0):
                raise ValueError('Unexpected frame header version')
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            if len(shape) != 3 or shape[1:] != points.shape or dtype.hasobject:
                raise ValueError('Invalid frame layout')
        for name in replacements:
            with old.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise ValueError('Unexpected optional-array header version')
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                if shape != () or not dtype.hasobject:
                    raise ValueError('Only scalar-object optional pin fields may be replaced')
        with zipfile.ZipFile(str(target)+suffix, 'x', compression=zipfile.ZIP_STORED) as new:
            for info in old.infolist():
                with old.open(info.filename) as stream:
                    original_hashes[info.filename] = digest(stream)
                if info.filename in replacements:
                    data = (np.zeros(len(points), dtype=bool) if info.filename == 'pinned.npy'
                            else np.full(len(points), -1, dtype=np.int64))
                    stream = io.BytesIO()
                    np.save(stream, data, allow_pickle=False)
                    new.writestr(copy.copy(info), stream.getvalue())
                else:
                    with old.open(info.filename) as src, new.open(copy.copy(info), 'w') as dst:
                        shutil.copyfileobj(src, dst, length=4*1024*1024)
    with zipfile.ZipFile(str(target)+suffix) as new:
        new_hashes = {}
        for name in new.namelist():
            with new.open(name) as stream:
                new_hashes[name] = digest(stream)
            if name not in replacements and new_hashes[name] != original_hashes[name]:
                raise ValueError('Numeric simulation bytes changed: '+name)
    receipt = dict(scope='optional pin schema reconstructed from disabled policy; no simulated state changed',
                   source=str(source), target=str(target), N=len(points), replaced=list(replacements),
                   original_members=original_hashes, normalized_members=new_hashes,
                   original_metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with Path(str(source)+suffix).open('rb') as stream:
        receipt['original_archive_sha256'] = digest(stream)
    with Path(str(target)+suffix).open('rb') as stream:
        receipt['normalized_archive_sha256'] = digest(stream)
    receipt['pin_motion_interpretation'] = 'zero checked IDs; vacuous pin test, not stationary material'
    for end in ('.npz', '.log'):
        shutil.copy2(str(source)+end, str(target)+end)
    shutil.copy2(metadata_path, str(target)+'.json')
    Path(str(target)+'.pin_schema.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    args = parser.parse_args()
    normalize(args.source, args.target)
