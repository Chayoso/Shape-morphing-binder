"""Read-only SSH collection of frozen P337 CUDA2 post-run evidence; no imports/runs."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
COMMIT = '995f1f1ac57f8ec3c7c46b969a787dd4c24ccb35'
REMOTE = '/data/relcfd/chayo/physmorph_v2/work/p303'
CODE = REMOTE + '/code_current_preparation2'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


tracked = set(git('ls-tree', '-r', '--name-only', COMMIT).decode().splitlines())
paths = {name for name in tracked if name.startswith('physmorph/')}
pending = ['tests/test_preparation_geometry_cuda.py', 'tests/test_current_successor_cuda.py']
while pending:
    name = pending.pop()
    if name in paths:
        continue
    paths.add(name)
    for node in ast.walk(ast.parse(git('show', COMMIT + ':' + name).decode('utf-8-sig'))):
        modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                   else [x.name for x in node.names] if isinstance(node, ast.Import) else [])
        for module in modules:
            candidates = [module.replace('.', '/') + '.py', 'tests/' + module.replace('.', '/') + '.py']
            for candidate in candidates:
                if candidate in tracked and candidate not in paths:
                    pending.append(candidate)
paths.update(['pytest.ini', 'scripts/ops/run_p303_probe.sh', 'scripts/ops/cuda_python.py',
              'scripts/ops/gpu_env.sh', 'scripts/ops/hyde06_env.sh'])
artifacts = ['p337_cuda2.' + suffix for suffix in
             ('log', 'xml', 'start', 'launch.log', 'memory.json', 'memory.log', 'memory.stop', 'exit')]
artifacts += ['run_p337_v2.sh', 'launch_p337_cuda2.sh', 'sample_device_memory.py']
remote_script = '''import base64,datetime,hashlib,json,subprocess
from pathlib import Path
CODE=Path(%r)
BASE=Path(%r)
PATHS=%r
ARTIFACTS=%r
def info(path):
    before=path.stat(); data=path.read_bytes(); after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    return dict(bytes=len(data),inode=after.st_ino,mtime_ns=after.st_mtime_ns,sha256=hashlib.sha256(data).hexdigest()),data
source={p:info(CODE/p)[0] for p in PATHS}
payload={}
artifact={}
for name in ARTIFACTS:
    artifact[name],data=info(BASE/name)
    payload[name]=base64.b64encode(data).decode()
library=Path('/data/relcfd/chayo/physmorph_v2/deps/gpu_pipeline/cupyx/scipy/ndimage/_measurements.py')
library_info,library_data=info(library)
payload['installed_cupyx_measurements.py.txt']=base64.b64encode(library_data).decode()
version,version_data=info(CODE/'VERSION')
tar,_=info(BASE/'p337_preparation_source2.tar')
print(json.dumps(dict(receipt=dict(collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    code=str(CODE),version=version_data.decode().strip(),version_binding=version,
    sources=source,artifacts=artifact,source_tar=tar,
    installed_library=dict(path=str(library),**library_info),
    project_apparent_bytes=int(subprocess.check_output(['du','-sb','/data/relcfd/chayo/physmorph_v2']).split()[0])),payload=payload)))
''' % (CODE, REMOTE, sorted(paths), artifacts)

response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
    '-J', 'chayo@hyde01.dabh.io', 'chayo@hyde06.dabh.io',
    '/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python', '-'],
    input=remote_script.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
received = json.loads(response.stdout)
receipt = received['receipt']
assert receipt['version'] == COMMIT
for name, entry in receipt['sources'].items():
    data = git('show', COMMIT + ':' + name)
    assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256'], name
for name, encoded in received['payload'].items():
    data = base64.b64decode(encoded)
    expected = receipt['installed_library'] if name == 'installed_cupyx_measurements.py.txt' else receipt['artifacts'][name]
    assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], name
    path = OUT/name
    if path.exists():
        assert path.read_bytes() == data, name
    else:
        path.write_bytes(data)
receipt['independent_git_source_comparison'] = dict(commit=COMMIT, files=len(receipt['sources']),
    exact_byte_matches=len(receipt['sources']), source='Git blobs, not mutable working tree')
target = OUT/'p337_cuda2.source_receipt.json'
assert not target.exists()
target.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(sources=len(receipt['sources']), artifacts=len(receipt['artifacts']),
    receipt_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    project_apparent_bytes=receipt['project_apparent_bytes'])))
