"""Sample nvidia-smi memory for one identified process; no CUDA computation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--pid', type=int, required=True)
p.add_argument('--gpu', type=int, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--stop', type=Path, required=True)
a = p.parse_args()
base = Path('/data/relcfd/chayo/physmorph_v2/work/p303').resolve()
assert a.out.resolve().is_relative_to(base) and a.stop.resolve().is_relative_to(base)
assert a.pid > 1 and a.gpu in range(4)
def identity():
    try:
        raw = Path(f'/proc/{a.pid}/stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        return fields[19]  # starttime: field22; fields begins at field3.
    except FileNotFoundError:
        return None
origin = identity()
assert origin is not None, 'Observed process is already absent'
def query(*fields, kind='gpu', gpu=None):
    args = ['nvidia-smi', '--query-'+kind+'='+','.join(fields), '--format=csv,noheader,nounits']
    if gpu is not None:
        args += ['--id='+str(gpu)]
    output = subprocess.run(args, check=True, capture_output=True, text=True, timeout=10).stdout
    return [[v.strip() for v in row.split(',')] for row in output.strip().splitlines() if row.strip()]
started = time.monotonic()
samples = []
errors = []
while not a.stop.exists() and identity() == origin:
    stamp = dict(utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic()-started)
    try:
        device, = query('uuid', 'memory.used', 'memory.total', gpu=a.gpu)
        processes = query('pid', 'gpu_uuid', 'used_gpu_memory', kind='compute-apps')
        mine = [row for row in processes if row[0] == str(a.pid) and row[1] == device[0]]
        stamp.update(device_used_MiB=int(device[1]), device_total_MiB=int(device[2]),
                     process_used_MiB=sum(int(row[2]) for row in mine), process_seen=bool(mine))
        samples.append(stamp)
    except (subprocess.SubprocessError, ValueError) as exc:
        errors.append(dict(**stamp, error=type(exc).__name__, message=str(exc)))
    time.sleep(.5)
report = dict(pid=a.pid, proc_starttime=origin, gpu=a.gpu, samples=samples, errors=errors,
    sleep_between_queries_s=.5, ended_utc=datetime.now(timezone.utc).isoformat(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    sampled_process_peak_MiB=max((r['process_used_MiB'] for r in samples), default=None),
    sampled_device_peak_MiB=max((r['device_used_MiB'] for r in samples), default=None),
    scope='Sampled device/process memory, including non-Torch ownership; sequential queries, not an exact high-water mark')
with a.out.open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps({k: v for k, v in report.items() if k not in ('samples', 'errors')}), flush=True)
