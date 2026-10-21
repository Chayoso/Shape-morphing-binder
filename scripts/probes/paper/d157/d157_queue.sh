#!/bin/bash
# d157_queue.sh TASKS SLOT -- D157 (agent D157): one worker slot of a CPU task queue (at most 3 slots run, nice 10).
# Each line of TASKS is a shell command; a worker claims the next unclaimed line by mkdir of TASKS.claim/N (atomic),
# runs it, records its exit status in TASKS.claim/N/rc, and goes on until no line is left (lines appended while it
# runs are picked up).
source /data/relcfd/chayo/physmorph_v2/tmp/d157/env.sh
tasks=$1; slot=$2; C=$tasks.claim; mkdir -p $C
cd $D157
while true; do
  n=$(grep -c . $tasks); got=0
  for i in $(seq 1 $n); do
    mkdir $C/$i 2>/dev/null || continue
    got=1
    cmd=$(sed -n "${i}p" $tasks)
    echo "slot $slot pid $$ task $i start $(date '+%m-%d %H:%M:%S'): ${cmd:0:120}" >> $tasks.log
    nice -n 10 bash -c "$cmd"
    rc=$?
    echo $rc > $C/$i/rc
    echo "slot $slot task $i rc $rc end $(date '+%m-%d %H:%M:%S')" >> $tasks.log
    break
  done
  [ $got = 0 ] && break
done
echo "slot $slot done $(date '+%m-%d %H:%M:%S')" >> $tasks.log
