#!/bin/bash
# d160_step1.sh -- D160 step 1, part B (agent D160): three diagnostic 40k re-runs of the full arm (bunny band / ears /
# turn, seed 97, repo_d160s, --term_dump --telemetry; measurement only) and d160_split.py on each, on ONE card by the
# lock protocol (a card among 3 / 1 / 0, never 2, with no lock, no other user's compute process, < 2000 MiB and < 5 %
# use at ten one-minute samples in a row; mkdir + info file; released by the trap).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d160/step1; W=$B/tmp/d160; T=$B/tmp; L=$B/gpu_locks
mkdir -p $O $W/tmpdir
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
S=$G/d160/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') [step1] $*" >> $S; }
CARD=""
declare -A idle
log "start pid $$ (waiting for a card idle 10 min)"
while [ -z "$CARD" ]; do
  for g in 3 1 0; do
    if [ -d $L/gpu$g.lock ]; then idle[$g]=0; continue; fi
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    util=$(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    napps=0
    for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader); do
      [ "$(ps -o user= -p $p 2>/dev/null | tr -d ' ')" = chayo ] || napps=$((napps + 1)); done
    if [ "$used" -lt 2000 ] && [ "$util" -lt 5 ] && [ "$napps" -eq 0 ]; then idle[$g]=$(( ${idle[$g]:-0} + 1 )); else idle[$g]=0; fi
    if [ "${idle[$g]}" -ge 10 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
      printf "agent D160\nrun d160/step1: 3 diagnostic 40k re-runs with term dumps + the split probe (~15 min)\npid %s\nstart %s\n" \
          "$$" "$(date '+%Y-%m-%d %H:%M:%S %Z')" > $L/gpu$g.lock/info
      CARD=$g; break
    fi
  done
  [ -z "$CARD" ] && sleep 60
done
trap 'rm -rf $L/gpu$CARD.lock; log "released gpu$CARD"' EXIT
trap 'exit 1' INT TERM HUP
export CUDA_VISIBLE_DEVICES=$CARD
log "claimed gpu$CARD"
for sc in band ears turn; do
  n=bunny40k_${sc}_L97_diag
  ( export PHYSMORPH_RUN_REPO=$B/repo_d160s PHYSMORPH_OUT=$G; source $B/repo_d160s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
    export TMPDIR=$W/tmpdir; cd $B/repo_d160s
    $PY scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 --appearance $sc --term_dump $O/dump_$sc \
        --telemetry --out $O/$n > $O/$n.log 2>&1 < /dev/null; echo $? > $O/rc_$n )
  log "run $n exit $(cat $O/rc_$n): $(grep -m1 'ARM render_full_dt_iso_nn:' $O/$n.log | cut -c1-120)"
  ( export PHYSMORPH_RUN_REPO=$B/repo_d160 PHYSMORPH_OUT=$G; source $B/repo_d160/scripts/ops/hyde06_env.sh > /dev/null 2>&1
    export TMPDIR=$W/tmpdir; cd $B/repo_d160
    $PY $W/d160_split.py $O/split.jsonl $sc 97 $O/dump_$sc $O/$n.json > $O/split_$sc.log 2>&1 < /dev/null; echo $? > $O/rc_split_$sc )
  log "split $sc exit $(cat $O/rc_split_$sc)"
done
log "done"
