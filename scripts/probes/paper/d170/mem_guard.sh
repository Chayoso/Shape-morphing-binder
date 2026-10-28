#!/bin/bash
# mem_guard.sh CARVE_B_PY RIPPLE_READER -- D170: the 300k relief readers of the ripples and of the bunny flank each hold ~95 GB
# (8.9M-vertex relief meshes); hyde06 has ~120 GB free. When carve_b's reader (python pid CARVE_B_PY) starts the bunny-flank
# read while the ripple read (pid RIPPLE_READER) still runs, stop it (SIGSTOP, our own process by its recorded parent) until
# the ripple read has exited, then continue it (SIGCONT). Logs to output/gpu/d170/status.txt.
PB=$1; PR=$2; O=/data/relcfd/chayo/physmorph_v2/output/gpu/d170
log() { echo "$(date '+%m-%d %H:%M:%S') [mem guard] $*" >> $O/status.txt; }
while ps -p $PB > /dev/null 2>&1; do
  ps -p $PR > /dev/null 2>&1 || { log "ripple reader $PR done: nothing to hold"; exit 0; }
  for c in $(ps -o pid= --ppid $PB); do
    if ps -o args= -p $c | grep -q "read_rel.json"; then
      kill -STOP $c && log "held the bunny-flank reader $c (SIGSTOP) while the ripple reader $PR runs"
      while ps -p $PR > /dev/null 2>&1; do sleep 30; done
      kill -CONT $c && log "released the bunny-flank reader $c (SIGCONT)"
      exit 0
    fi
  done
  sleep 10
done
