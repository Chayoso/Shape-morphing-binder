#!/bin/bash
# rgv_idle.sh NAME CMD_FILE -- RGV's card wrapper on hyde06 (2026-10-15 rule): takes only a card that has had no
# gpu_locks/gpuN.lock and < 2000 MiB in use for 10+ minutes on end (D148 / D149 first), one card at most; claims it by
# mkdir with agent / run / pid / start inside (released by a trap), runs CMD_FILE in repo_rgvr's environment, releases.
NAME=$1; CMDS=$2
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; O=$B/output/gpu/rgv
S=$O/status.txt
declare -A since
echo "$NAME waiting for a card idle 10+ min $(date '+%m-%d %H:%M:%S')" >> $S
g=""
while [ -z "$g" ]; do
  if [ "$(grep -l "agent RGV" $L/gpu*.lock/info 2>/dev/null | wc -l)" -ge 1 ]; then
    since=()                                   # RGV holds a card: no idle clock runs meanwhile
  else
    now=$(date +%s)
    for c in 0 1 2 3; do
      if [ ! -e $L/gpu$c.lock ] && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c)" -lt 2000 ]; then
        [ -z "${since[$c]}" ] && since[$c]=$now
        if [ $(( now - ${since[$c]} )) -ge 600 ] && mkdir $L/gpu$c.lock 2>/dev/null; then
          printf "agent RGV\nrun rgv/$NAME\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$c.lock/info
          g=$c; break
        fi
      else
        since[$c]=""
      fi
    done
  fi
  [ -z "$g" ] && sleep 60
done
trap "rm -rf $L/gpu$g.lock" EXIT
trap "exit 1" INT TERM HUP
echo "$NAME claimed gpu $g $(date '+%m-%d %H:%M:%S')" >> $S
export PHYSMORPH_RUN_REPO=$B/repo_rgvr PHYSMORPH_OUT=$O OMP_NUM_THREADS=8
source $B/repo_rgvr/scripts/ops/hyde06_env.sh > /dev/null
export CUDA_VISIBLE_DEVICES=$g
t0=$(date +%s)
bash $CMDS > $O/$NAME.cmd.log 2>&1 < /dev/null
echo "$NAME done rc $? wall $(( $(date +%s) - t0 )) s $(date '+%m-%d %H:%M:%S')" >> $S
