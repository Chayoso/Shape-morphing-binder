#!/bin/bash
# d151_suite.sh -- D151: the full suite on repo_d151 (refreeze-cand 169f251 + --layer_link) on the first free card
# (gpu_locks, mkdir; < 2 GB in use), the pinned refreeze references included; the exit status recorded, not a tail.
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; T=$B/tmp; log=$T/d151_suite.log
gpu=""
while [ -z "$gpu" ]; do
  for g in 3 2 1 0; do
    if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 2000 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
      gpu=$g; break
    fi
  done
  [ -z "$gpu" ] && sleep 30
done
trap 'rm -rf $L/gpu$gpu.lock' EXIT
printf "agent D148\nrun d151/suite (repo_d151 full pytest)\npid %s\nstart %s\n" "$$" "$(date '+%Y-%m-%d %H:%M %Z')" > $L/gpu$gpu.lock/info
export PHYSMORPH_RUN_REPO=$B/repo_d151
source $B/repo_d151/scripts/ops/hyde06_env.sh > /dev/null 2>&1
cd $B/repo_d151
echo "suite on gpu $gpu from $(date '+%H:%M')" > $log
CUDA_VISIBLE_DEVICES=$gpu $PY -m pytest -q >> $log 2>&1
echo "exit $? at $(date '+%H:%M')" >> $log
