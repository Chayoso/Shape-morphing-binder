#!/bin/bash
# rgv_card.sh NAME CMD_FILE -- RGV's card wrapper on hyde06: waits for a free card under the agents' protocol (no
# gpu_locks/gpuN.lock and < 2000 MiB in use; claimed by mkdir with agent / run / pid / start inside, released by a
# trap; RGV holds at most one card before 19:00 and two after), then runs CMD_FILE with bash on it (CUDA_VISIBLE_DEVICES
# and G set to the card, the environment of hyde06_env.sh for the repository REPO, default repo_rgvr) and releases it.
# Status lines in output/gpu/rgv/status.txt.
NAME=$1; CMDS=$2
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; O=$B/output/gpu/rgv
mkdir -p $O
S=$O/status.txt
mine() { grep -l "agent RGV" $L/gpu*.lock/info 2>/dev/null | wc -l; }
claim() {
  local cap=1
  [ "$(date +%H)" -ge 19 ] && cap=2
  [ "$(mine)" -ge $cap ] && return 1
  for g in 0 1 2 3; do
    [ -e $L/gpu$g.lock ] && continue
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 2000 ] || continue
    if mkdir $L/gpu$g.lock 2>/dev/null; then
      printf "agent RGV\nrun rgv/$NAME\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
      echo $g; return 0
    fi
  done
  return 1
}
echo "$NAME waiting for a card $(date '+%m-%d %H:%M:%S')" >> $S
until g=$(claim); do sleep 90; done
trap "rm -rf $L/gpu$g.lock" EXIT
trap "exit 1" INT TERM HUP
echo "$NAME claimed gpu $g $(date '+%m-%d %H:%M:%S')" >> $S
export PHYSMORPH_RUN_REPO=${REPO:-$B/repo_rgvr} PHYSMORPH_OUT=$O OMP_NUM_THREADS=8
source $PHYSMORPH_RUN_REPO/scripts/ops/hyde06_env.sh > /dev/null
export CUDA_VISIBLE_DEVICES=$g G=$g
t0=$(date +%s)
bash $CMDS > $O/$NAME.cmd.log 2>&1 < /dev/null
rc=$?
echo "$NAME done rc $rc wall $(( $(date +%s) - t0 )) s $(date '+%m-%d %H:%M:%S')" >> $S
