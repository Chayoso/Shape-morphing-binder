#!/bin/bash
# rgv_h1.sh NAME -- the RGV videos on hyde01's render node (scripts/ops/hyde01_render.sh's pattern, by hand): waits for
# GPU 1 (no compute process on it and locks/gpu1.lock free; GPU 0 drives another user's desktop and is never used),
# claims it by mkdir with pid and job inside (released by a trap), then runs ~/physmorph_render/rgv/cmd_NAME.sh in the
# node's environment with the RGV code (~/physmorph_render/rgv/repo) and CUDA_VISIBLE_DEVICES=1. Status lines in
# ~/physmorph_render/rgv/status.txt. Everything stays under ~/physmorph_render (hyde01's /data is another, full fs).
R=$HOME/physmorph_render; G=1; L=$R/locks/gpu$G.lock; NAME=$1
S=$R/rgv/status.txt
mkdir -p $R/locks
busy() {
  local uuid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $G) || return 0
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"
}
echo "$NAME waiting for gpu $G $(date '+%m-%d %H:%M:%S')" >> $S
while true; do
  if [ -s $L/pid ] && ! kill -0 "$(cat $L/pid)" 2>/dev/null; then rm -rf $L; fi     # a dead holder's lock
  if ! busy && mkdir $L 2>/dev/null; then echo $$ > $L/pid; echo "rgv_$NAME" > $L/job; break; fi
  sleep 60
done
trap "rm -rf $L" EXIT
trap "exit 1" INT TERM HUP
echo "$NAME claimed gpu $G $(date '+%m-%d %H:%M:%S')" >> $S
export PHYSMORPH_RENDER_REPO=$R/rgv/repo
source $R/env.sh
export CUDA_VISIBLE_DEVICES=$G
t0=$(date +%s)
bash $R/rgv/cmd_$NAME.sh > $R/rgv/$NAME.log 2>&1 < /dev/null
echo "$NAME done rc $? wall $(( $(date +%s) - t0 )) s $(date '+%m-%d %H:%M:%S')" >> $S
