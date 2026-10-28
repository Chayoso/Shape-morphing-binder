#!/bin/bash
# after.sh MODE JOB QUEUE_NAME[,QUEUE_NAME...] -- D170's detached CPU reader for one job: waits for every
# tmp/gpu1_queue/QUEUE_NAME.done, takes one of 4 CPU slots, runs d170_read.py MODE output/gpu/d170/auto/JOB, rewrites
# output/gpu/D170_SUMMARY.md, then runs tmp/d170/post.sh MODE JOB (deliverables, follow-up runs) if it exists.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d170; O=$B/output/gpu/d170; Q=$B/tmp/gpu1_queue
MODE=$1; JOB=$2; QNS=$3; D=$O/auto/$JOB
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=12 D157_THREADS=12 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d166:$B/deps/d157_lpips D170_WAIT_H=${D170_WAIT_H:-12}
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') [after $JOB] $*" >> $O/status.txt; }
mkdir -p $W/slots $D; cd $B/repo_d166
for qn in ${QNS//,/ }; do
  for i in $(seq 1 2160); do [ -e $Q/$qn.done ] && break; sleep 30; done
  [ -e $Q/$qn.done ] || { log "no $qn.done after 18 h: stop"; exit 1; }
done
log "jobs done ($QNS); reading"
while true; do
  for i in 1 2 3 4; do
    exec 9> $W/slots/$i.lock
    if flock -n 9; then
      s=$(date +%s); nice -n 10 $PY $W/d170_read.py $MODE $D > $D/read.out 2>&1; rc=$?
      log "read exit $rc ($(( $(date +%s) - s )) s): $(grep VERDICT $D/summary.md 2>/dev/null | cut -c1-400)"
      nice -n 10 $PY $W/d170_read.py aggregate $B/output/gpu/D170_SUMMARY.md >> $D/read.out 2>&1
      exec 9>&-
      [ -e $W/post.sh ] && bash $W/post.sh $MODE $JOB >> $D/post.out 2>&1
      exit $rc
    fi
    exec 9>&-
  done
  sleep 30
done
