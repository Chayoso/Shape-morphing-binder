#!/bin/bash
# d148_proto.sh [suite] -- D148 on one card held through gpu_locks (agent D148): the CUDA tests of tests/test_gap_split.py
# on repo_d148 (the chain stops if they fail), then the 40k go / no-go: the dragon at 40k, seed 97, A (--gap_split on),
# D1 and D2 (the frozen default twice: the atomics' spread), each with D140's dump of windows 1-9 and the side render;
# then d148_proto_read.py. With "suite": the full suite on repo_d148 instead (its exit status in suite_d148.log).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/d148; D=$B/output/gpu/d148; L=$B/gpu_locks; mkdir -p $D
source $B/tmp/d146/cache_env.sh
mode=${1:-proto}; MAX=${D148_MAX:-2}
gpu=""
while [ -z "$gpu" ]; do
  held=$(grep -l "^agent D148" $L/gpu*.lock/info 2>/dev/null | wc -l)
  if [ "$held" -lt "$MAX" ]; then
    for g in 0 1 2 3; do
      if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 2000 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        gpu=$g; break
      fi
    done
  fi
  [ -z "$gpu" ] && sleep 30
done
trap 'rm -rf $L/gpu$gpu.lock' EXIT
printf "agent D148\nrun d148/%s\npid %s\nstart %s\n" "$mode" "$$" "$(date '+%Y-%m-%d %H:%M %Z')" > $L/gpu$gpu.lock/info
echo "start $mode gpu $gpu $(date '+%m-%d %H:%M')" >> $D/status.txt
if [ "$mode" = suite ]; then
  ( export PHYSMORPH_RUN_REPO=$B/repo_d148; source $B/repo_d148/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_d148
    CUDA_VISIBLE_DEVICES=$gpu $PY -m pytest -q -p no:cacheprovider tests > $T/suite_d148.log 2>&1
    echo "EXIT $?" >> $T/suite_d148.log )
  echo "suite $(tail -1 $T/suite_d148.log) $(date '+%m-%d %H:%M')" >> $D/status.txt
  exit 0
fi
( export PHYSMORPH_RUN_REPO=$B/repo_d148; source $B/repo_d148/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_d148
  CUDA_VISIBLE_DEVICES=$gpu $PY -m pytest -q -p no:cacheprovider tests/test_gap_split.py > $T/gaptests_cuda.log 2>&1
  echo "EXIT $?" >> $T/gaptests_cuda.log )
echo "gap tests $(tail -1 $T/gaptests_cuda.log) $(date '+%m-%d %H:%M')" >> $D/status.txt
[ "$(tail -1 $T/gaptests_cuda.log)" = "EXIT 0" ] || { echo "proto stopped: the CUDA tests failed" >> $D/status.txt; exit 1; }
D148_GPU=$gpu $T/d148_run.sh dragon 40000 A 2 --gap_split on
D148_GPU=$gpu $T/d148_run.sh dragon 40000 D1 2
D148_GPU=$gpu $T/d148_run.sh dragon 40000 D2 2
( source $B/repo_d148/scripts/ops/hyde06_env.sh > /dev/null
  $PY $T/d148_proto_read.py $D dragon40k D1,D2 A > $D/proto_read.log 2>&1 )
echo "proto done $(date '+%m-%d %H:%M')" >> $D/status.txt
