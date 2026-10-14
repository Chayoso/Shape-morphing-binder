#!/bin/bash
# d147_job.sh GPU REPO TAG "command": claim the GPU through the lock directory, run the command in REPO's environment,
# release the lock on exit. Log: output/gpu/d147/TAG.log; status line in output/gpu/d147/jobs.log.
GPU=$1; REPO_DIR=$2; TAG=$3; CMD=$4
LOCKS=/data/relcfd/chayo/physmorph_v2/gpu_locks
LOCK=$LOCKS/gpu$GPU.lock
# D148 first (the coordinator, 2026-10-14): claim a free card only while D148 holds one, or after a free card has been
# left to D148 for 15 minutes (the time since this was first seen is kept in d148_free_since)
ST=/data/relcfd/chayo/physmorph_v2/output/gpu/d147/d148_free_since
if grep -qs "agent D148" $LOCKS/gpu*.lock/info; then
  rm -f $ST
else
  now=$(date +%s); [ -f $ST ] || echo $now > $ST
  [ $((now - $(cat $ST))) -ge 900 ] || exit 1
fi
mkdir "$LOCK" 2>/dev/null || exit 1
printf "agent D147\nrun d147/$TAG\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $LOCK/info
trap 'rm -rf "$LOCK"' EXIT
export PHYSMORPH_RUN_REPO=$REPO_DIR PHYSMORPH_OUT=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
source $REPO_DIR/scripts/ops/hyde06_env.sh
export CUDA_VISIBLE_DEVICES=$GPU
echo "${TAG}_start $(date '+%F %T') gpu$GPU" >> $PHYSMORPH_OUT/jobs.log
bash -c "$CMD" > $PHYSMORPH_OUT/$TAG.log 2>&1
rc=$?
echo "${TAG}_end $(date '+%F %T') exit $rc" >> $PHYSMORPH_OUT/jobs.log
exit 0
