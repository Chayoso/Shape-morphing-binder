"""Read-only P337 producer preflight; no GPU/runtime launch and no independent-audit claim."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
COMMIT = '995f1f1ac57f8ec3c7c46b969a787dd4c24ccb35'
BASE = '/data/relcfd/chayo/physmorph_v2'
CODE = BASE+'/work/p303/code_current_preparation2'
REMOTE_PY = '/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

names = git('ls-tree', '-r', '--name-only', COMMIT).decode().splitlines()
selected = [p for p in names if (p.startswith(('physmorph/', 'scripts/probes/')) and p.endswith('.py'))]
selected += ['scripts/ops/cuda_python.py', 'scripts/ops/gpu_env.sh', 'scripts/ops/hyde06_env.sh',
             'scripts/ops/run_p303_probe.sh', 'docs/current_successor_preparation_p337.md',
             'docs/candidate_commit_contract.md', 'tests/test_preparation_geometry_cuda.py', 'tests/test_current_successor_cuda.py']
expected = {p: sha256(git('show', COMMIT+':'+p)).hexdigest() for p in sorted(set(selected))}
remote = '''import json,hashlib,subprocess,os
from pathlib import Path
from datetime import datetime,timezone
base=Path(BASE)
code=Path(CODE)
def identity(p):
    before=p.stat(); h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    after=p.stat()
    assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),str(p)
    return dict(bytes=after.st_size,sha256=h.hexdigest(),mtime_ns=after.st_mtime_ns,inode=after.st_ino)
sources={name:identity(code/name) for name in EXPECTED}
meta=base/'work/p303/raw24a.json'; binding=identity(meta); raw=json.loads(meta.read_text()); assert identity(meta)==binding
assets={str(p):identity(p) for p in (meta,base/'repro/current_pair/source_render_full_dt_iso_nn.npz',Path(raw['config']['target_reference']))}
runtime={str(p):identity(p) for p in (base/'work/p303/run_p337_v2.sh',base/'work/p303/sample_device_memory.py',code/'VERSION')}
used=int(subprocess.check_output(['du','-sb',str(base)]).split()[0])
print(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),version=(code/'VERSION').read_text().strip(),
 sources=sources,assets=assets,runtime=runtime,raw_config=raw['config'],mpm=raw['mpm'],used_bytes=used,
 native_output_exists=(base/'work/p303/p337_native1').exists(),free_filesystem_bytes=os.statvfs(base).f_bavail*os.statvfs(base).f_frsize)))
'''.replace('BASE',repr(BASE)).replace('CODE',repr(CODE)).replace('EXPECTED',repr(expected))
completed = subprocess.run(['ssh','-J','chayo@hyde01.dabh.io','chayo@hyde06.dabh.io',REMOTE_PY+' -'], input=remote.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
observed=json.loads(completed.stdout)
sys.path.insert(0,str(ROOT))
from dataclasses import asdict
from physmorph.pipeline.config import PipelineConfig
cfg=PipelineConfig(**observed['raw_config']); normalized=asdict(cfg)
cfg.stop_after_windows=21; cfg.assim_fp64=True
effective=asdict(cfg)
diff={k:dict(original=normalized[k],effective=v) for k,v in effective.items() if normalized[k]!=v}
source_ok=all(observed['sources'][p]['sha256']==h for p,h in expected.items())
prior=json.loads((ROOT/'docs/evidence/p335/p335_native2.protocol.json').read_text())
asset_ok=all(row['sha256']==prior['bindings'][path]['sha256'] for path,row in observed['assets'].items())
local_runtime=Path('C:/dev/physmorph_runtime/p303/run_p337_v2.sh')
runtime_ok=sha256(local_runtime.read_bytes()).hexdigest()==observed['runtime'][BASE+'/work/p303/run_p337_v2.sh']['sha256']
report=dict(schema='p337_producer_preflight_v1',scope='Producer read-only preflight, not independent audit or native numerical result',
 commit=COMMIT,frozen_code=CODE,observed=observed,git_source_sha256=expected,source_count=len(expected),
 sources_match_git=source_ok,assets_match_validated_recipe=asset_ok,runtime_matches_local=runtime_ok,
 config_changes=diff,effective_config=effective,only_registered_overrides=set(diff)<= {'stop_after_windows','assim_fp64'},
 no_borrowed_policy='Driver takes only root/out; original live context supplies preview. No prior successor archive is input.',
 reservation_bytes=8000000000,project_limit_bytes=100000000000,
 reservation_fits=observed['used_bytes']+8000000000<100000000000,
 launch_authorization='No launch performed. Root must separately require CUDA gate and independent receipt review.',
 source_scope='All frozen physmorph and top-level probe Python files plus wrapper/environment/docs/tests; inventory is not a dynamic import trace.')
report['passed']=source_ok and asset_ok and runtime_ok and report['only_registered_overrides'] and report['reservation_fits'] and observed['version']==COMMIT and not observed['native_output_exists']
report['producer_script_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
out=Path(__file__).with_name('p337_native1.producer_preflight.json')
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ('passed','source_count','sources_match_git','assets_match_validated_recipe','runtime_matches_local','config_changes','reservation_fits')}))
if not report['passed']:raise SystemExit(1)
