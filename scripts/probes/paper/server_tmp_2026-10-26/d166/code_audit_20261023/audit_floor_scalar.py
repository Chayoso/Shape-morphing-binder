"""Read completed D167 floor scalar outputs without loading frames or new geometry."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json

B = Path('/data/relcfd/chayo/physmorph_v2')
D = B / 'output/gpu/codex_floor_json_20261023_1813'
D.mkdir(exist_ok=True)
old = json.loads((B / 'output/gpu/codex_floor_json_20261023_1808/audit.json').read_text())
rp = B / 'tmp/d166/d166_read.py'
spec = importlib.util.spec_from_file_location('reader', rp)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
sources = {}
for name, expected in old['sources'].items():
    data = (B / name).read_bytes()
    actual = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    assert actual == expected, (name, actual, expected)
    out = D / 'evidence' / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    sources[name] = actual
rows = []
rundir = B / 'output/gpu/d167/runs'
for mesh in reader.MESHES:
    default_path, _ = reader.gl().paths(mesh, 'V', 97)
    default = reader.run_rec(default_path[:-5])
    for arm, twin_code in [('F', 'T'), ('FP', 'TP')]:
        run = reader.run_rec(rundir / f'gal_{mesh}_{arm}_97')
        twin = reader.run_rec(reader.twin_prefix(rundir, mesh, twin_code, 97, 'gallery'))
        row = dict(mesh=mesh, seed=97, arm=arm, run=run, default=default, twin=twin,
                   complete=all(x is not None for x in [run, default, twin]))
        if row['complete']:
            ds, dc, dt = run['sil']-default['sil'], run['ch']-default['ch'], run['thin']-default['thin']
            assert twin['v_end'] > 0
            vr = run['v_end']/twin['v_end']
            row.update(delta_sil=ds, delta_chamfer=dc, delta_thin=dt, speed_ratio=vr,
                       gallery_tolerance_pass=ds >= -.003 and dc <= .002 and dt <= .02,
                       literal_geometry_no_worse=ds >= 0 and dc <= 0 and dt <= 0, rest_pass=vr <= 2)
        rows.append(row)
summary = {}
for arm in ['F', 'FP']:
    group = [r for r in rows if r['arm'] == arm and r['complete']]
    summary[arm] = dict(completed_rows=len(group),
                        gallery_tolerance_pass=[r['mesh'] for r in group if r['gallery_tolerance_pass']],
                        geometry_fail=[r['mesh'] for r in group if not r['gallery_tolerance_pass']],
                        rest_fail=[r['mesh'] for r in group if not r['rest_pass']])
result = dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
              status='COMPLETED_JSON_METRICS_ONLY', source_reader_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),
              correction='Legacy rc_NAME exit+integer-seconds parsing repaired; previous 18:08 audit preserved.',
              rows=rows, summary=summary, sources=sources, scope=old['scope'])
(D / 'audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'observed_at': result['observed_at'], 'summary': summary,
                  'bob': [r for r in rows if r['mesh'] == 'bob']}, indent=2))
