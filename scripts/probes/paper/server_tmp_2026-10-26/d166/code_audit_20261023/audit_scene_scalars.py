"""D167 exact delivered scalar metrics only; no new surface/image measurements."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json

B = Path('/data/relcfd/chayo/physmorph_v2')
D = B / 'output/gpu/codex_scene_json_20261023_1814'
D.mkdir(exist_ok=True)
rp = B / 'tmp/d166/d166_read.py'
spec = importlib.util.spec_from_file_location('reader', rp)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
sources = {}
def rec(prefix):
    prefix = Path(prefix)
    record = reader.run_rec(str(prefix))
    for p in [prefix.with_suffix('.json'), prefix.with_suffix('.rc'), prefix.parent / ('rc_'+prefix.name)]:
        if p.exists():
            data = p.read_bytes()
            sources[str(p.relative_to(B))] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return record

rows = []
rundir = B / 'output/gpu/d167/runs'
for arm in ['F', 'FP', 'S', 'SP', 'H', 'HP', 'R', 'RP']:
    twin_code = 'TP' if arm.endswith('P') else 'T'
    for scene in ['coin', 'bunny', 'band']:
        for seed in [97, 99]:
            run = rec(rundir / f'{scene}_{arm}_{seed}')
            twin = rec(reader.twin_prefix(rundir, scene, twin_code, seed, 'scenes'))
            row = dict(arm=arm, scene=scene, seed=seed, twin_code=twin_code,
                       run=run, twin=twin, complete=run is not None and twin is not None)
            if row['complete']:
                assert twin['v_end'] > 0
                row['speed_ratio'] = run['v_end']/twin['v_end']
                row['rest_pass'] = row['speed_ratio'] <= 2
            rows.append(row)
summary = {}
for arm in ['F', 'FP', 'S', 'SP', 'H', 'HP', 'R', 'RP']:
    group = [r for r in rows if r['arm'] == arm and r['complete']]
    worst = max(group, key=lambda r: r['speed_ratio']) if group else None
    summary[arm] = dict(completed_rows=len(group), rest_pass_count=sum(r['rest_pass'] for r in group),
                        max_speed_ratio=worst['speed_ratio'] if worst else None,
                        worst_case=f"{worst['scene']}/{worst['seed']}" if worst else None,
                        rest_fail=[f"{r['scene']}/{r['seed']}" for r in group if not r['rest_pass']])
for name, metadata in sources.items():
    assert hashlib.sha256((B/name).read_bytes()).hexdigest() == metadata['sha256'], name
result = dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
              status='COMPLETED_JSON_METRICS_ONLY', source_reader_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),
              scope='40k seeds97/99, exact delivered endpoint and successful rc; scalar rest criterion only. No authority, image, relief, retention or adoption verdict.',
              rows=rows, summary=summary, sources=sources)
(D/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'observed_at':result['observed_at'], 'summary':summary, 'sources':len(sources)}, indent=2))
