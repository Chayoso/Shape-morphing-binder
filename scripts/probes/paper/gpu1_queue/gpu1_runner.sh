#!/bin/bash
# gpu1_runner.sh -- the gpu1 queue runner (agent D162, the coordinator's order 2026-10-22 19:30): holds the gpu1 lock and
# runs the job scripts dropped in tmp/gpu1_queue/ (NN_agent_name.sh), one at a time in name order, with
# CUDA_VISIBLE_DEVICES=1; logs each job's start, end, seconds, exit code and a CUDA out-of-memory flag to runner.log;
# renames each finished script to NN_agent_name.done (its output in logs/NN_agent_name.log). With no job it waits up to
# 30 min for new ones, then releases the lock. HANDOVER_FROM=PID: take the lock over from this agent's own process on gpu1
# the moment it exits (fresh claims are impossible while another user fills all four cards). README.md: the contract.
B=/data/relcfd/chayo/physmorph_v2; Q=$B/tmp/gpu1_queue; L=$B/gpu_locks; LOCK=$L/gpu1.lock
mkdir -p $Q/logs
log() { echo "$(date '+%F %T') $*" >> $Q/runner.log; }
release() { [ -n "$HELD" ] && rm -rf $LOCK && log "runner: released gpu1"; HELD=""; }
trap 'release' EXIT
trap 'exit 1' INT TERM HUP
HELD=""
if [ -n "$HANDOVER_FROM" ]; then
  log "runner: waiting for pid $HANDOVER_FROM (this agent's chain on gpu1) to exit"
  while ps -p $HANDOVER_FROM > /dev/null 2>&1; do sleep 5; done
fi
for k in $(seq 1 60); do
  if mkdir $LOCK 2>/dev/null; then HELD=1; break; fi
  sleep 5
done
[ -n "$HELD" ] || { log "runner: could not take the gpu1 lock (someone else holds it): stop"; exit 1; }
printf "agent D162\nrun gpu1 queue runner (tmp/gpu1_queue, jobs from any agent)\npid %s\nstart %s\n" "$$" "$(date '+%F %T %Z')" > $LOCK/info
log "runner: holds gpu1 (pid $$)"
idle_since=$(date +%s)
while true; do
  job=$(ls $Q/[0-9][0-9]_*.sh 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then
    if [ $(( $(date +%s) - idle_since )) -ge 1800 ]; then log "runner: no job for 30 min: done"; break; fi
    sleep 30; continue
  fi
  name=$(basename $job .sh); out=$Q/logs/$name.log
  t0=$(date +%s); log "start $name"
  ( export CUDA_VISIBLE_DEVICES=1; cd $B; bash $job ) > $out 2>&1 < /dev/null
  rc=$?; t1=$(date +%s)
  oom=""; grep -q "CUDA out of memory\|OutOfMemoryError\|cudaErrorMemoryAllocation\|out of memory" $out && oom=" CUDA-OOM"
  log "end $name exit $rc$oom ($((t1 - t0)) s; log logs/$name.log)"
  mv $job $Q/$name.done
  idle_since=$(date +%s)
done
