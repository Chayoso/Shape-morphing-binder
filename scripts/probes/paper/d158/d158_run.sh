#!/bin/bash
# d158_run.sh -- D158 (agent D158, a measurement): D115's fit with the decomposition (tmp/d158/d158_fit.py) on B1's render
# arms, bunny and dragon 300k L97. One card by the lock protocol (idle 10 minutes: < 2000 MiB, < 5 % use, no other
# user's process, no lock; never GPU 2; released by the trap). A smoke test first (bunny, kept frames 0-12, a dump at
# frame 10 with a 5-step static fit); only if it ends rc 0, both meshes in parallel on the same card.
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; W=$B/tmp/d158; G=$B/output/gpu; O=$G/d158
mkdir -p $O $W/tmpdir
S=$O/status.txt
export PHYSMORPH_RUN_REPO=$B/repo_freeze1015 PHYSMORPH_OUT=$W/tmpdir/pm_out
source $B/repo_freeze1015/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir D158_REPO=$B/repo_freeze1015
declare -A idle
GPU=""
echo "d158 ($$) waiting for a card idle 10 min $(date '+%m-%d %H:%M:%S')" >> $S
while [ -z "$GPU" ]; do
  for g in 0 3 1; do
    if [ -e $L/gpu$g.lock ]; then idle[$g]=0; continue; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
    napps=0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g); do [ "$(ps -o user= -p $p 2>/dev/null | tr -d " ")" = chayo ] || napps=$((napps + 1)); done
    if [ "$used" -lt 2000 ] && [ "$util" -lt 5 ] && [ "$napps" -eq 0 ]; then idle[$g]=$(( ${idle[$g]:-0} + 1 )); else idle[$g]=0; fi
    if [ "${idle[$g]}" -ge 10 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
      printf "agent D158\nrun d158: D115's display-layer fit + its decomposition on B1 bunny/dragon 300k L97 (measurement, ~40 min)\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
      GPU=$g; break
    fi
  done
  [ -z "$GPU" ] && sleep 60
done
trap "rm -rf $L/gpu$GPU.lock" EXIT
trap "exit 1" INT TERM HUP
echo "d158 claimed gpu $GPU $(date '+%m-%d %H:%M:%S')" >> $S
export CUDA_VISIBLE_DEVICES=$GPU
dense() { echo "$G/d111/$1_dense15.npz,$G/d111/$1_dense15_s9.npz,$G/d111/$1_dense15_s10.npz,$G/d111/$1_dense15_s11.npz"; }
fit() {   # fit MESH OUTDIR OPTS...
  local mesh=$1 od=$2; shift 2
  t0=$(date +%s)
  $PY $W/d158_fit.py $G/d145t_b1/${mesh}300k_L97_frames12.npz $(dense $mesh) $od +mesh=$mesh +seed=97 +fit=1 +steps=15 \
    +carry=h +decay=1 +video=0 "$@" > $od.log 2>&1 < /dev/null
  rc=$?
  echo "$(basename $od) rc $rc wall $(( $(date +%s) - t0 )) s $(date '+%m-%d %H:%M:%S')" >> $S
  return $rc
}
rm -rf $O/smoke_bunny
if ! fit bunny $O/smoke_bunny +frames=0-12 +dump=10 +sfit=10 +sfit_steps=5; then
  echo "d158 smoke failed: card released $(date '+%m-%d %H:%M:%S')" >> $S
  exit 1
fi
rm -rf $O/bunny300k_L97 $O/dragon300k_L97
fit bunny $O/bunny300k_L97 &
sleep 20
fit dragon $O/dragon300k_L97 &
wait
echo "d158 done, card released $(date '+%m-%d %H:%M:%S')" >> $S
