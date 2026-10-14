#!/bin/bash
# D147 chain 3: after chain 2, the three 40k timelines again with the floor's position constraint, the domain from the
# extents and --assim 0 (S2), then the readings of scene_na and S2 by the selection rule (CPU).
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
log() { echo "$(date '+%F %T') $*" >> $O/chain3.log; }
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
log "chain3 start; waiting for chain2"
until grep -q "chain2 end" $O/chain2.log 2>/dev/null; do sleep 30; done
R=$O/runs; SP=scripts/probes/d147/specs
claim_and_run S2 "for v in \"L:\" \"LB:--scene $SP/scene_long.json\" \"LS:--young 9.8e5\"; do t=\${v%%:*}; \$PY scripts/scene_run.py --n 40000 --seed 97 \${v#*:} --flags=\"--assim 0\" --out $R/scene40k_\${t}_v2 > $R/scene40k_\${t}_v2.log 2>&1; echo scene40k_\${t}_v2 exit \$?; done"
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
  nice $PY $S/read40k.py $R scene40k_L_na,scene40k_LB_na,scene40k_LS_na > $R/read_na.txt 2>&1
  nice $PY $S/read40k.py $R scene40k_L_v2,scene40k_LB_v2,scene40k_LS_v2 > $R/read_v2.txt 2>&1 )
log "chain3 end"
