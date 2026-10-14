#!/bin/bash
# D147 chain 8 (after chain 7): the 4K render of the 40k scene LB_v2 (render_scene.py, a test of the renderer and the
# floor), then (if the coordinator asks) the long flight with larger offsets at contact, --assim 0 and the floor.
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
log() { echo "$(date '+%F %T') $*" >> $O/chain8.log; }
claim_and_run() {
  local tag=$1 cmd=$2
  while true; do
    for g in 3 1 0 2; do
      [ -d /data/relcfd/chayo/physmorph_v2/gpu_locks/gpu$g.lock ] && continue
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$used" -gt 30000 ] && continue
      if bash $S/job.sh $g "$REPO" "$tag" "$cmd"; then
        rc=$(grep "^${tag}_end " $O/jobs.log | tail -1 | awk '{print $NF}'); log "$tag gpu$g exit $rc"; return $rc
      fi
    done
    sleep 15
  done
}
log "chain8 start, repo $REPO; waiting for chain7"
until grep -q "chain7 end" $O/chain7.log 2>/dev/null; do sleep 30; done
R=$O/runs; SP=scripts/probes/d147/specs
claim_and_run RENDER_LB_v2 "bash $S/render_scene_job.sh $R scene40k_LB_v2"
if [ -n "$D147_OFFSETS" ]; then
  claim_and_run S3 "for o in o75 o100; do \$PY scripts/scene_run.py --n 40000 --seed 97 --scene $SP/scene_long_\$o.json --flags=\"--assim 0\" --out $R/scene40k_LB_\${o}_v2 > $R/scene40k_LB_\${o}_v2.log 2>&1; echo scene40k_LB_\${o}_v2 exit \$?; done"
  export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
  ( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
    nice $PY $S/read40k.py $R scene40k_LB_v2,scene40k_LB_o75_v2,scene40k_LB_o100_v2 > $R/read_s3.txt 2>&1 )
fi
log "chain8 end"
