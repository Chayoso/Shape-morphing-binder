#!/bin/bash
# D147 chain 10: the 300k scene (2 x 150k) from 19:00 on the leak-fixed code: the picked variant (seed 97, render on),
# its render-off twin and seed 99, side by side as cards come free (D148 first, job.sh), then the 4K render of the
# seed-97 run and the readings (timeline, angular momentum's split).
#   bash scripts/probes/d147/chain10.sh REPO SPEC        (SPEC: a file name in specs/ without .json, e.g. scene_long)
REPO=$1; SPEC=$2
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
log() { echo "$(date '+%F %T') $*" >> $O/chain10.log; }
claim_and_run() {
  local tag=$1 cmd=$2
  while true; do
    for g in 3 2 1 0; do
      [ -d $L/gpu$g.lock ] && continue
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$used" -gt 2000 ] && continue
      if bash $S/job.sh $g "$REPO" "$tag" "$cmd"; then
        rc=$(grep "^${tag}_end " $O/jobs.log | tail -1 | awk '{print $NF}'); log "$tag gpu$g exit $rc"; return $rc
      fi
    done
    sleep 30
  done
}
R=$O/runs; SP=scripts/probes/d147/specs; B=scene300k_$SPEC
run() { echo "\$PY scripts/scene_run.py --n 150000 --seed $1 --scene $SP/$SPEC.json --flags=\"--assim 0\" --save_stride 12 $3 --out $R/$2 > $R/$2.log 2>&1; echo $2 exit \$?"; }
log "chain10 start, repo $REPO, spec $SPEC; waiting for 19:00"
until [ "$(date +%H%M)" -ge 1900 ]; do sleep 60; done
claim_and_run P300 "$(run 97 ${B}_s97)" &
# the others once the first has passed its second window (memory: the first 2 x 150k scene on this card)
until grep -q "window 2/" $R/${B}_s97.log 2>/dev/null || grep -q "^P300_end " $O/jobs.log; do sleep 30; done
claim_and_run P300_twin "$(run 97 ${B}_s97_off '--render_weight_scale 0')" &
sleep 90
claim_and_run P300_s99 "$(run 99 ${B}_s99)" &
wait
claim_and_run RENDER300 "bash $S/render_scene_job.sh $R ${B}_s97"
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
  nice $PY $S/read40k.py $R ${B}_s97,${B}_s97_off,${B}_s99 > $R/read_300k.txt 2>&1 )
log "chain10 end (read_300k.txt)"
