#!/bin/bash
# d162c_B1.sh -- D162c (B) on gpu1 under this agent's own lock (held by the D162 gallery chain): every card carries another
# user's job, so no card can be claimed; this short regeneration (no rollout) runs beside the gallery run. Then the CPU read.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d162c; W=$B/tmp/d162
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $O/status.txt; }
[ -d $B/gpu_locks/gpu1.lock ] && grep -q "agent D162" $B/gpu_locks/gpu1.lock/info || { log "B: gpu1 is not this agent's; stop"; exit 1; }
log "B push on gpu1 under this agent's lock (the gallery chain's), beside its run"
export PHYSMORPH_RUN_REPO=$B/repo_d162b PHYSMORPH_OUT=$B/output/gpu
source $B/repo_d162b/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=1
cd $B/repo_d162b
$PY $W/d162c_push.py $O/push > $O/push.log 2>&1 < /dev/null; echo $? > $O/push.rc
log "B push exit $(cat $O/push.rc): $(grep -c 'd162c B\]' $O/push.log) runs"
export CUDA_VISIBLE_DEVICES=
nice -n 10 $PY $W/d162c_pushread.py $O/B.jsonl $O/push > $O/pushread.log 2>&1 < /dev/null; echo $? > $O/pushread.rc
log "B read exit $(cat $O/pushread.rc): $(wc -l < $O/B.jsonl) states"
