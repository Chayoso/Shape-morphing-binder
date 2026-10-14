#!/bin/bash
# d148_pregate.sh -- the 300k gate's start (pre-registered in D148): on the first card D148 gets (gpu_locks, at most
# D148_MAX = 3 cards), the CUDA tests of tests/test_gap_split.py on repo_d148b (the split G2P adjoint in cell order);
# if they pass the gate runs on its hook copy repo_d148g, else on repo_d148g with WARP_ADJOINT off (the per-particle
# adjoint, already tested). Then the waiters GT, BA, BT, GA99, and B's 40k information run last; the dragon render
# arm GA on this card.
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/d148; D=$B/output/gpu/d148; L=$B/gpu_locks
source $B/tmp/d146/cache_env.sh
export D148_MAX=3
gpu=""
while [ -z "$gpu" ]; do
  held=$(grep -l "^agent D148" $L/gpu*.lock/info 2>/dev/null | wc -l)
  if [ "$held" -lt "$D148_MAX" ]; then
    for g in 0 1 2 3; do
      if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 2000 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        gpu=$g; break
      fi
    done
  fi
  [ -z "$gpu" ] && sleep 20
done
trap 'rm -rf $L/gpu$gpu.lock' EXIT
printf "agent D148\nrun d148/dragon300k_GA\npid %s\nstart %s\n" "$$" "$(date '+%Y-%m-%d %H:%M %Z')" > $L/gpu$gpu.lock/info
echo "start pregate gpu $gpu $(date '+%m-%d %H:%M')" >> $D/status.txt
( export PHYSMORPH_RUN_REPO=$B/repo_d148b; source $B/repo_d148b/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_d148b
  CUDA_VISIBLE_DEVICES=$gpu $PY -m pytest -q -p no:cacheprovider tests/test_gap_split.py > $T/gaptests_cuda_b.log 2>&1
  echo "EXIT $?" >> $T/gaptests_cuda_b.log )
if [ "$(tail -1 $T/gaptests_cuda_b.log)" != "EXIT 0" ]; then
  sed -i 's/^WARP_ADJOINT = True /WARP_ADJOINT = False/' $B/repo_d148g/physmorph/mpm/gap_kernels.py
  echo "pregate: the cell-order adjoint's tests failed; the gate runs with WARP_ADJOINT False" >> $D/status.txt
fi
echo "pregate tests $(tail -1 $T/gaptests_cuda_b.log); WARP_ADJOINT: $(grep '^WARP_ADJOINT' $B/repo_d148g/physmorph/mpm/gap_kernels.py) $(date '+%m-%d %H:%M')" >> $D/status.txt
export D148_REPO=repo_d148g
cd $T
for spec in "dragon GT 2 --render_weight_scale 0" "bunny BA 1" "bunny BT 0 --render_weight_scale 0" "dragon GA99 2 --seed 99"; do
  set -- $spec
  m=$1; tag=$2; arm=$3; shift 3
  setsid nohup ./d148_run.sh $m 300000 $tag $arm --gap_split on "$@" > $T/q_$tag.log 2>&1 < /dev/null &
  sleep 30
done
D148_GPU=$gpu ./d148_run.sh dragon 300000 GA 1 --gap_split on
D148_REPO=repo_d148f setsid nohup ./d148_run.sh dragon 40000 B 2 --gap_split interface > $T/q_B40k.log 2>&1 < /dev/null &
echo "pregate done (GA) $(date '+%m-%d %H:%M')" >> $D/status.txt
