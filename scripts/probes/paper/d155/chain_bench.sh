#!/bin/bash
# chain_bench.sh -- D155b: after part B, one card (lock protocol, agent D155) for d155_build_bench.py on bunny and dragon 40k
# (where a pack on the way spends its build); a few minutes.
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; T=$B/tmp; G=$B/output/gpu; W=$T/d155; O=$G/d155/gate
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir; mkdir -p $TMPDIR $O
S=$O/status.txt
part=bench
until grep -q "part B done" $S 2>/dev/null; do sleep 60; done
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
declare -A idle
GPU=""
echo "part $part ($$) waiting for a card idle 10 min $(date '+%m-%d %H:%M:%S')" >> $S
while [ -z "$GPU" ]; do
  for g in 0 1 3; do
    if [ -e $L/gpu$g.lock ]; then idle[$g]=0; continue; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
    napps=0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g); do [ "$(ps -o user= -p $p 2>/dev/null | tr -d " ")" = chayo ] || napps=$((napps + 1)); done
    if [ "$used" -lt 2000 ] && [ "$util" -lt 5 ] && [ "$napps" -eq 0 ]; then idle[$g]=$(( ${idle[$g]:-0} + 1 )); else idle[$g]=0; fi
    if [ "${idle[$g]}" -ge 10 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
      printf "agent D155\nrun d155/gate part $part (option A's 40k gate: 40k runs of 1-3 min with their readers)\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
      GPU=$g; break
    fi
  done
  [ -z "$GPU" ] && sleep 60
done
trap "rm -rf $L/gpu$GPU.lock" EXIT
trap "exit 1" INT TERM HUP
echo "part $part claimed gpu $GPU $(date '+%m-%d %H:%M:%S')" >> $S
export CUDA_VISIBLE_DEVICES=$GPU

(
  export PHYSMORPH_RUN_REPO=$B/repo_d155b PHYSMORPH_OUT=$G
  source $B/repo_d155b/scripts/ops/hyde06_env.sh > /dev/null
  export TMPDIR=$W/tmpdir
  cd $B/repo_d155b
  for m in bunny dragon; do
    $PY $W/d155_build_bench.py $m 40000 > $O/build_bench_$m.log 2>&1 < /dev/null
    echo "build bench $m rc $? $(date '+%m-%d %H:%M:%S')" >> $S
  done
)
echo "bench done $(date '+%m-%d %H:%M:%S')" >> $S
