#!/bin/bash
# d174_run.sh CARD NAME REPO [pipeline_run args...] -- agent D174: one pipeline run alone on card CARD (free: < 2000 MiB,
# lock gpu_locks/gpuCARD.lock taken by mkdir, released by the trap), a 1 Hz utilisation sampler beside it, every log
# line stamped with the wall clock ($EPOCHREALTIME), the exit code and the seconds in output/gpu/d174/status.txt.
CARD=$1; NAME=$2; RP=$3; shift 3
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d174; LOCK=$B/gpu_locks/gpu$CARD.lock
[ "$CARD" = 2 ] && { echo "never gpu2"; exit 1; }
used=$(nvidia-smi -i $CARD --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || { echo "$(date '+%F %T') $NAME: card $CARD busy ($used MiB)" >> $O/status.txt; exit 1; }
mkdir $LOCK 2>/dev/null || { echo "$(date '+%F %T') $NAME: lock gpu$CARD held" >> $O/status.txt; exit 1; }
printf "agent D174\nrun %s (repo %s)\npid %s\nstart %s\n" $NAME $RP $$ "$(date '+%F %T %Z')" > $LOCK/info
nvidia-smi -i $CARD --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 1000 > $O/runs/$NAME.util.csv 2>/dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; rm -rf $LOCK' EXIT
trap 'exit 1' INT TERM HUP
export PHYSMORPH_RUN_REPO=$B/$RP PHYSMORPH_OUT=$B/output/gpu; source $B/$RP/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh; export TMPDIR=$B/tmp/d174/tmpdir CUDA_VISIBLE_DEVICES=$CARD PYTHONUNBUFFERED=1
cd $B/$RP
echo "$(date '+%F %T') $NAME start on card $CARD ($RP: $*)" >> $O/status.txt
t0=$(date +%s)
$PY scripts/pipeline_run.py "$@" --out $O/runs/$NAME < /dev/null 2>&1 | while IFS= read -r l; do printf '%s %s\n' "$EPOCHREALTIME" "$l"; done > $O/runs/$NAME.log
rc=${PIPESTATUS[0]}
echo $rc > $O/runs/$NAME.rc
echo "$(date '+%F %T') $NAME exit $rc ($(( $(date +%s) - t0 )) s)" >> $O/status.txt
