#!/bin/bash
set -eu
B=/data/relcfd/chayo/physmorph_v2
A=$B/tmp/d166/code_audit_20261023
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
export CUDA_VISIBLE_DEVICES= D166_CACHE_ONLY=1 OMP_NUM_THREADS=1
echo "waiting for existing readers; no GPU/metric recomputation"
while true; do
 if $PY - <<'PY'
from pathlib import Path
import os,sys
base=Path('/data/relcfd/chayo/physmorph_v2/output/gpu/d166/auto')
dirs=[base/j for j in ('63_assim10','64_assim08','65_cross_a10','66_assim025','67_stiff3')]
if not all((d/'verdict.json').exists() for d in dirs):sys.exit(1)
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid != os.getuid():continue
  args=p.joinpath('cmdline').read_bytes().split(b'\0')
  if any(str(d).encode() in args for d in dirs) and any(b'/d166/after.sh' in a or b'/d166/d166_read.py' in a for a in args):
   sys.exit(1)
 except OSError:pass
PY
 then
  for i in 1 2 3 4; do
   exec 9>"$B/tmp/d166/slots/$i.lock"
   if flock -n 9; then
    echo "refreshing cached summaries in existing slot $i"
    $PY "$A/refresh_cached.py" > "$A/refresh_after_active.log" 2>&1
    date -Is > "$A/refresh_after_active.done"
    exit 0
   fi
   exec 9>&-
  done
 fi
 sleep 30
done
