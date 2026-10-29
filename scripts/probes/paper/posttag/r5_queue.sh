#!/bin/bash
# r5_queue.sh TASKS SLOT -- D145 reading 5: one worker slot of a CPU task queue (at most three slots, each task nice 10
# inside its script). Each line of TASKS is a shell command; a slot claims the next unclaimed line by mkdir of
# TASKS.claim/N (atomic), runs it, records its exit status, and goes on; lines appended while it runs are picked up; with
# none left it waits (up to 10 h, polling every 2 min) while r5_gpu.sh (its PID in gpu.pid beside TASKS) is alive.
tasks=$1; slot=$2; C=$tasks.claim; mkdir -p $C
idle=0
while true; do
  n=$(grep -c . $tasks); got=0
  for i in $(seq 1 $n); do
    mkdir $C/$i 2>/dev/null || continue
    got=1
    cmd=$(sed -n "${i}p" $tasks)
    echo "slot $slot pid $$ task $i start $(date '+%m-%d %H:%M:%S'): ${cmd:0:140}" >> $tasks.log
    bash -c "$cmd"
    rc=$?
    echo $rc > $C/$i/rc
    echo "slot $slot task $i rc $rc end $(date '+%m-%d %H:%M:%S')" >> $tasks.log
    break
  done
  if [ $got = 0 ]; then
    if ps -p "$(cat $(dirname $tasks)/gpu.pid 2>/dev/null)" > /dev/null 2>&1 && [ $idle -lt 300 ]; then idle=$((idle + 1)); sleep 120; continue; fi
    break
  fi
  idle=0
done
echo "slot $slot done $(date '+%m-%d %H:%M:%S')" >> $tasks.log
