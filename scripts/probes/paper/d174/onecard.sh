#!/bin/bash
# onecard.sh NAME REPO CMD... -- agent D174: wait for a free card (0 / 1 / 3, never 2; < 2000 MiB and its lock), run CMD
# from REPO with the repo's environment on that card, output to tmp/d174/NAME.out, release the lock.
NAME=$1; RP=$2; shift 2
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d174
CARD=""
while [ -z "$CARD" ]; do
  for c in 0 1 3; do
    u=$(nvidia-smi -i $c --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$u" -lt 2000 ] && mkdir $B/gpu_locks/gpu$c.lock 2>/dev/null; then CARD=$c; break; fi
  done
  [ -z "$CARD" ] && sleep 60
done
LOCK=$B/gpu_locks/gpu$CARD.lock
printf "agent D174\n%s (%s)\npid %s\nstart %s\n" $NAME $RP $$ "$(date '+%F %T %Z')" > $LOCK/info
trap 'rm -rf $LOCK' EXIT
trap 'exit 1' INT TERM HUP
export PHYSMORPH_RUN_REPO=$B/$RP PHYSMORPH_OUT=$B/output/gpu; source $B/$RP/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh; export TMPDIR=$B/tmp/d174/tmpdir CUDA_VISIBLE_DEVICES=$CARD PYTHONUNBUFFERED=1
cd $B/$RP
echo "$(date '+%F %T') $NAME start on card $CARD" >> $O/status.txt
t0=$(date +%s)
"$@" > $B/tmp/d174/$NAME.out 2>&1 < /dev/null
rc=$?
echo "$(date '+%F %T') $NAME exit $rc ($(( $(date +%s) - t0 )) s)" >> $O/status.txt
