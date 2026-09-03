#!/usr/bin/env bash
# GPU placement rule (user directive 2026-09-03): sample hyde06 utilization; if ALL GPUs
# are saturated (>50% in every sample) print "hyde01"; else print "hyde06 <gpu ids <=50%>".
# Usage (from the workstation): bash scripts/ops/gpu_pick.sh   (needs ssh -J to hyde06)
SAMPLES=${SAMPLES:-5}
OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -J chayo@hyde01.dabh.io chayo@hyde06.dabh.io \
  "for i in \$(seq 1 $SAMPLES); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ';'; echo; sleep 3; done" 2>/dev/null)
[ -z "$OUT" ] && { echo "hyde01  # hyde06 unreachable"; exit 0; }
python3 - "$OUT" << 'PY'
import sys, collections
rows=[l for l in sys.argv[1].splitlines() if l.strip()]
util=collections.defaultdict(list)
for l in rows:
    for cell in l.strip(';').split(';'):
        i,u=cell.split(','); util[int(i)].append(float(u))
free=[i for i,us in sorted(util.items()) if max(us) <= 50.0]
if free: print("hyde06 " + " ".join(map(str,free)))
else: print("hyde01")
PY
