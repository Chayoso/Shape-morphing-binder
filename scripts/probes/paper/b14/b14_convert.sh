#!/bin/bash
# b14_convert.sh NAME TAG RUN DATA_DIR -- D145 B14: convert_b14.py on the CPU (pipeline frame, mesh properties, 300k
# sample -> archive), the end check on a card (physmorph's metrics are GPU-only: the job's own card while the pool's
# lock for this B14 run is held, else any free card claimed through the lock directory for the minute it takes), then
# the volume over time by the common routine (CPU). A free card is one unlocked and idle for 10+ min (the pool record).
NAME=$1; TAG=$2; RUN=$3; DATA=$4
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d145/b14; R=$B/baselines/b14; G=$B/output/gpu; L=$B/gpu_locks
say() { echo "$(date '+%Y-%m-%d %H:%M:%S %Z') $RUN $*" >> $R/status.txt; }
export PHYSMORPH_RUN_REPO=$B/repo_d145 PHYSMORPH_OUT=$R/out_env
source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null 2>&1
GPUV=${CUDA_VISIBLE_DEVICES:-}; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 TMPDIR=$R/cache/tmp
mkdir -p $R/out_env $G/d145_b14
say "convert start"; t0=$(date +%s)
$PY $W/convert_b14.py $NAME $TAG $R/recon/$RUN $DATA > $R/logs/${RUN}_convert.log 2>&1 || { say "FAILED convert"; exit 1; }
say "convert done $(( $(date +%s) - t0 )) s"
own=""
if [ -z "$GPUV" ] || ! head -2 $L/gpu$GPUV.lock/info 2>/dev/null | grep -q "run d145_b14/"; then
  while [ -z "$own" ]; do
    for g in 0 1 2 3; do
      [ -e $L/gpu$g.lock ] && continue
      [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 2000 ] || continue
      fs=$(cat $B/tmp/d145/free_since/gpu$g 2>/dev/null); [[ "$fs" =~ ^[0-9]+$ ]] && [ $(( $(date +%s) - fs )) -ge 600 ] || continue   # the 10-min idle rule
      if mkdir $L/gpu$g.lock 2>/dev/null; then
        printf "agent D145\nrun d145_b14/endcheck_%s (1 min)\npid %s\nstart %s\n" "$RUN" "$$" "$(date '+%Y-%m-%d %H:%M:%S %Z')" > $L/gpu$g.lock/info
        own=$g; GPUV=$g; break
      fi
    done
    [ -z "$own" ] && sleep 60
  done
  trap "rm -rf $L/gpu$own.lock" EXIT
fi
CUDA_VISIBLE_DEVICES=$GPUV OMP_NUM_THREADS=8 $PY $W/endcheck_b14.py $R/results/endcheck.jsonl $NAME $TAG >> $R/logs/${RUN}_convert.log 2>&1 || { say "FAILED endcheck"; exit 1; }
[ -n "$own" ] && { rm -rf $L/gpu$own.lock; trap - EXIT; }
OMP_NUM_THREADS=8 $PY $B/tmp/d145/vol145.py $R/results/vol.jsonl $NAME $G/d90/${NAME}_ind300k.npz \
  "$TAG=$G/d145_b14/${RUN}_frames12.npz" >> $R/logs/${RUN}_convert.log 2>&1 || { say "FAILED vol145"; exit 1; }
say "endcheck + volume done"
