#!/bin/bash
# chain_zdumpC.sh -- D155c: after part C, one card (lock protocol, agent D155): the full suite again on repo_d155c with the
# test fix (tests only; code ef3071c), then d155c_zdump.py on the five meshes (the moving target's own points, seconds
# each). Replaces chain_suiteC.sh.
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; T=$B/tmp; G=$B/output/gpu; W=$T/d155; O=$G/d155/gate
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir; mkdir -p $TMPDIR
S=$O/status.txt
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
true
declare -A idle
GPU=""
while [ -z "$GPU" ]; do
  for g in 0 1 3; do
    if [ -e $L/gpu$g.lock ]; then idle[$g]=0; continue; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
    napps=0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g); do [ "$(ps -o user= -p $p 2>/dev/null | tr -d " ")" = chayo ] || napps=$((napps + 1)); done
    if [ "$used" -lt 2000 ] && [ "$util" -lt 5 ] && [ "$napps" -eq 0 ]; then idle[$g]=$(( ${idle[$g]:-0} + 1 )); else idle[$g]=0; fi
    if [ "${idle[$g]}" -ge 10 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
      printf "agent D155\nrun d155/suite3 (the full suite after a second test fix; ~5 min)\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
      GPU=$g; break
    fi
  done
  [ -z "$GPU" ] && sleep 60
done
trap "rm -rf $L/gpu$GPU.lock" EXIT
trap "exit 1" INT TERM HUP
echo "suite3 claimed gpu $GPU $(date '+%m-%d %H:%M:%S')" >> $S
export CUDA_VISIBLE_DEVICES=$GPU
export PHYSMORPH_RUN_REPO=$B/repo_d155c PHYSMORPH_OUT=$G
source $B/repo_d155c/scripts/ops/hyde06_env.sh > /dev/null
export TMPDIR=$W/tmpdir
cd $B/repo_d155c
$PY -m pytest -q tests -p no:cacheprovider > $O/pytest_full_c3.log 2>&1 < /dev/null
echo "full suite (c, test fix 2) exit $? ($(tail -1 $O/pytest_full_c3.log)) $(date '+%m-%d %H:%M:%S')" >> $S
for m in; do
  $PY $W/d155c_zdump.py $m 40000 > $O/zdump_$m.log 2>&1 < /dev/null
  echo "zdump $m rc $? $(date '+%m-%d %H:%M:%S')" >> $S
done
echo "suite3 done $(date '+%m-%d %H:%M:%S')" >> $S
