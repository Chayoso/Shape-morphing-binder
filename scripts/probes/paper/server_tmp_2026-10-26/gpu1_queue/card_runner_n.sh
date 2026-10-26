#!/bin/bash
# card_runner_n.sh CARD -- tmp/d162/card_runner.sh with numeric job order over 2- and 3-digit names (NN_ / NNN_agent_name.sh;
# agent D166, 2026-10-23): takes the card's lock if the card is free (< 2000 MiB), runs jobs from tmp/gpu1_queue/ in
# numeric order, each claimed atomically by renaming it to NAME.sh.cCARD, with CUDA_VISIBLE_DEVICES=CARD; logs to
# runner.log; renames finished jobs to NAME.done; releases the lock after 30 min without jobs.
C=$1; B=/data/relcfd/chayo/physmorph_v2; Q=$B/tmp/gpu1_queue; LOCK=$B/gpu_locks/gpu$C.lock
log() { echo "$(date '+%F %T') [card $C] $*" >> $Q/runner.log; }
used=$(nvidia-smi -i $C --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || { log "busy ($used MiB): stop"; exit 1; }
mkdir $LOCK 2>/dev/null || { log "lock held: stop"; exit 1; }
printf "agent D166\nrun queue runner card %s (tmp/gpu1_queue, numeric order)\npid %s\nstart %s\n" $C $$ "$(date '+%F %T %Z')" > $LOCK/info
trap 'rm -rf $LOCK; log "released"' EXIT
trap 'exit 1' INT TERM HUP
log "holds gpu$C (card_runner_n, pid $$)"
idle=$(date +%s)
while true; do
  job=""
  for f in $(ls $Q 2>/dev/null | grep -E '^[0-9]{2,3}_.*\.sh$' | sort -n); do
    if mv $Q/$f $Q/$f.c$C 2>/dev/null; then job=$Q/$f; break; fi
  done
  if [ -z "$job" ]; then [ $(( $(date +%s) - idle )) -ge 1800 ] && break; sleep 30; continue; fi
  name=$(basename $job .sh); t0=$(date +%s); log "start $name"
  ( export CUDA_VISIBLE_DEVICES=$C; cd $B; bash $job.c$C ) > $Q/logs/$name.log 2>&1 < /dev/null; rc=$?
  oom=""; grep -q "out of memory" $Q/logs/$name.log && oom=" CUDA-OOM"
  log "end $name exit $rc$oom ($(( $(date +%s) - t0 )) s)"; mv $job.c$C $Q/$name.done; idle=$(date +%s)
done
