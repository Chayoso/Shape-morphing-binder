"""Archive I/O only: canonical array bytes for completed P327 attempts 0..11."""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path('/data/relcfd/chayo/physmorph_v2/work/p303')
OUT = ROOT/'p327_prefix12_byte_identity.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(path):
    s = path.stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev)


bindings = {}
def read_bound(path):
    before = identity(path)
    data = path.read_bytes()
    if identity(path) != before:
        raise RuntimeError('File changed while reading: '+str(path))
    bindings[str(path)] = dict(sha256=digest(data), stat=before)
    return data


def archive(path):
    with np.load(io.BytesIO(read_bound(path)), allow_pickle=False) as values:
        if len(set(values.files)) != len(values.files):
            raise RuntimeError('Duplicate archive key')
        result = {}
        for key in values.files:
            a = values[key]
            data = a.tobytes(order='C')
            result[key] = (dict(shape=list(a.shape), dtype=a.dtype.str,
                c_contiguous=bool(a.flags.c_contiguous), fortran_contiguous=bool(a.flags.f_contiguous),
                canonical_bytes=len(data), array_sha256=digest(data)), data)
    if 'optimizer_raw_endpoint' not in result:
        raise RuntimeError('Missing raw endpoint')
    return result


if OUT.exists():
    raise FileExistsError('Preserve earlier runtime evidence: '+str(OUT))
protocols = {}
for mode in ('legacy', 'retained'):
    prefix = ROOT/('p327_'+mode+'1')
    protocols[mode] = {
        name: json.loads(read_bound(Path(str(prefix)+suffix)))
        for name, suffix in (('fragment', '.fragment_protocol.json'), ('full', '.protocol.json'))}
    if protocols[mode]['fragment']['mode'] != mode or protocols[mode]['full']['arm'] != 'raw':
        raise RuntimeError('Unexpected run mode')
declared_equal = {key: protocols['legacy']['full'][key] == protocols['retained']['full'][key]
                  for key in ('inputs', 'code', 'source_config', 'mpm')}
declared_equal['wrapper_code'] = protocols['legacy']['fragment']['code'] == protocols['retained']['fragment']['code']
if not all(declared_equal.values()):
    raise RuntimeError('Protocols declare different source/config/input identities')

rows = []
for attempt in range(12):
    loaded = {mode: archive(ROOT/('p327_'+mode+'1_cohorts')/f'attempt_{attempt:03d}.npz')
              for mode in ('legacy', 'retained')}
    keys = sorted(set(loaded['legacy']) | set(loaded['retained']))
    comparisons = {}
    for key in keys:
        a, b = loaded['legacy'].get(key), loaded['retained'].get(key)
        same_schema = a is not None and b is not None and all(a[0][k] == b[0][k] for k in ('shape', 'dtype'))
        comparisons[key] = dict(legacy=None if a is None else a[0], retained=None if b is None else b[0],
            same_shape_dtype=same_schema, canonical_bytes_equal=bool(same_schema and a[1] == b[1]))
    rows.append(dict(attempt_zero_based=attempt, arrays=comparisons,
                     raw_endpoint_bytes_equal=comparisons['optimizer_raw_endpoint']['canonical_bytes_equal']))

for path, expected in bindings.items():
    source = Path(path)
    if identity(source) != expected['stat'] or digest(source.read_bytes()) != expected['sha256']:
        raise RuntimeError('Bound source changed before report: '+path)
result = dict(created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=digest(Path(__file__).read_bytes()),
    script=str(Path(__file__).resolve()), numpy_version=np.__version__, attempts_zero_based=[0, 11],
    operation='NPZ decoding and canonical C-order array-byte identity; no numerical deltas, geometry, simulation or GPU work',
    scope='First12 completed attempted optimizer endpoints, not accepted-commit matching; candidate run may still be active',
    interpretation='A byte difference before legacy bug exposure disproves exact pre-exposure trajectory identity, but does not apportion causality. Equal bytes do not establish gradient identity.',
    declared_protocol_equal=declared_equal, protocol_input_hashes=protocols['legacy']['full']['inputs'],
    protocol_code_hashes=protocols['legacy']['full']['code'], wrapper_code_hashes=protocols['legacy']['fragment']['code'],
    source_bindings=bindings, source_identities_unchanged=True, rows=rows,
    first_raw_endpoint_difference=next((r['attempt_zero_based'] for r in rows if not r['raw_endpoint_bytes_equal']), None))
payload = json.dumps(result, indent=2, allow_nan=False).encode()
with OUT.open('xb') as stream:
    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
directory = os.open(OUT.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
print(json.dumps(dict(out=str(OUT), sha256=digest(payload),
    first_raw_endpoint_difference=result['first_raw_endpoint_difference'],
    raw_endpoint_equal=[r['raw_endpoint_bytes_equal'] for r in rows],
    equal_keys_by_attempt={str(r['attempt_zero_based']): [k for k, v in r['arrays'].items() if v['canonical_bytes_equal']] for r in rows})))
