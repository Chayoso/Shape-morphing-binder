#!/bin/bash
# d152_gpu.sh -- D152's card readings (measurement only): the render term's own floor on the tag (d152_own_floor.py,
# repo_t) and the yardstick's floor (d152_yard_floor.py, repo_r77). One card at most, through the lock protocol: a card
# with no lock, no other user's compute process, < 2000 MiB and < 5 % use at ten one-minute samples in a row (idle >= 10 min) is
# claimed by mkdir with an info file, released by the trap. GPU 2 is never used (another user's display session).
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; W=$B/tmp/d152; O=$B/output/gpu/d152
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir; mkdir -p $TMPDIR $O
S=$W/status.txt
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
declare -A idle
G=""
echo "job $$ waiting for a card idle 10 min $(date '+%m-%d %H:%M:%S')" >> $S
while [ -z "$G" ]; do
  for g in 0 3 1; do
    if [ -e $L/gpu$g.lock ]; then idle[$g]=0; continue; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
    napps=0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g); do [ "$(ps -o user= -p $p 2>/dev/null | tr -d " ")" = chayo ] || napps=$((napps + 1)); done
    if [ "$used" -lt 2000 ] && [ "$util" -lt 5 ] && [ "$napps" -eq 0 ]; then idle[$g]=$(( ${idle[$g]:-0} + 1 )); else idle[$g]=0; fi
    if [ "${idle[$g]}" -ge 10 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
      printf "agent D152\nrun d152/floors (render-term and yardstick floors; measurement only, about 1 h)\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
      G=$g; break
    fi
  done
  [ -z "$G" ] && sleep 60
done
trap "rm -rf $L/gpu$G.lock" EXIT
trap "exit 1" INT TERM HUP
echo "claimed gpu $G $(date '+%m-%d %H:%M:%S')" >> $S
export CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null
export TMPDIR=$W/tmpdir
own() {
  cd $B/repo_t
  nice -n 5 $PY $W/d152_own_floor.py $O/own_floor.jsonl $1 $2 > $W/own_floor_$1_$2.log 2>&1 < /dev/null
  echo "own $1 $2 rc $? $(date '+%m-%d %H:%M:%S')" >> $S
}
own bunny 97
cd $B/repo_r77
nice -n 5 $PY $W/d152_yard_floor.py $O/yard_floor.jsonl bunny:97 dragon:97 bunny:99 dragon:99 bunny:101 dragon:101 \
  bunny:103 dragon:103 bunny:105 dragon:105 teapot:97 teapot:99 teapot:101 bob:97 bob:99 bob:101 armadilo:97 \
  armadilo:99 armadilo:101 homer:97 homer:99 homer:101 > $W/yard_floor.log 2>&1 < /dev/null
echo "yard rc $? $(date '+%m-%d %H:%M:%S')" >> $S
for ms in "dragon 97" "bunny 99" "dragon 99" "bunny 101" "dragon 101" "teapot 97" "bob 97" "armadilo 97" "homer 97" \
          "teapot 99" "bob 99" "armadilo 99" "homer 99" "teapot 101" "bob 101" "armadilo 101" "homer 101"; do
  own $ms
done
echo "done $(date '+%m-%d %H:%M:%S')" >> $S
