#!/bin/bash
# rgv_job.sh MESH NAME [SEED] -- the render-gradient videos' data run (RGV) on hyde06: waits for a free card (the agents'
# protocol: no gpu_locks/gpuN.lock and < 2000 MiB in use, claimed by mkdir with agent / run / pid / start inside,
# released by a trap; RGV holds at most one card before 19:00 and two after), then on it: (1) once, the measurement
# copy's hook test (tests/test_dump_hooks.py of repo_rgv: the hooks change nothing bit for bit, the control records
# are written and vanish with the render weight at zero); (2) the 300k run of the deliverable code (repo_rgv =
# repo_d145g + rgv_patch.py) with the gradient dump and the lambda = 0 counterfactual at EVERY window. The card is
# released when the run ends. Output: output/gpu/rgv/NAME{.log,.json,_frames12.npz,_dump/,_cf/}.
MESH=$1; NAME=$2; SEED=${3:-97}
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; R=$B/repo_rgv; O=$B/output/gpu/rgv
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
export PHYSMORPH_RUN_REPO=$R PHYSMORPH_OUT=$O OMP_NUM_THREADS=8
source $R/scripts/ops/hyde06_env.sh > /dev/null
cd $R
if [ ! -e $O/hooktest.passed ]; then
  CUDA_VISIBLE_DEVICES=$g $PY -m pytest tests/test_dump_hooks.py tests/test_frozen_recipe.py -q -p no:cacheprovider \
    > $O/hooktest.log 2>&1 < /dev/null
  rc=$?
  echo "hook test rc $rc $(date '+%m-%d %H:%M:%S')" >> $S
  [ $rc = 0 ] || { echo "$NAME NOT RUN: the hook test failed" >> $S; exit 1; }
  touch $O/hooktest.passed
fi
CFW=$(seq -s, 0 399)
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)"; sleep 10; done ) > $O/$NAME.mem &
mem=$!
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=$g $PY scripts/pipeline_run.py --tgt assets/$MESH.obj --n 300000 --seed $SEED --out $O/$NAME \
  --grad_dump $O/${NAME}_dump --dump_windows 0,+1 --cf_dir $O/${NAME}_cf --cf_windows $CFW > $O/$NAME.log 2>&1 < /dev/null
rc=$?
kill $mem
echo $rc > $O/rc_$NAME
echo "$NAME sim rc $rc wall $(( $(date +%s) - t0 )) s peak_mib $(sort -k2 -n $O/$NAME.mem | tail -1 | cut -d' ' -f2) $(date '+%m-%d %H:%M:%S')" >> $S
du -sh $O/${NAME}_dump $O/${NAME}_cf >> $S
