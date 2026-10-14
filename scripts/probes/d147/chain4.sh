#!/bin/bash
# D147 chain 4: after chain 3, the scene tests (with the new assimilation's) and the suite, then V1 and the three 40k
# timelines with the assimilation reading the controls' deformation without the loads (A1), read by the rule.
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
log() { echo "$(date '+%F %T') $*" >> $O/chain4.log; }
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
log "chain4 start; waiting for chain3"
until grep -q "chain3 end" $O/chain3.log 2>/dev/null; do sleep 30; done
R=$O/runs; SP=scripts/probes/d147/specs
claim_and_run T6_scene_tests '$PY -m pytest tests/test_scene_frame.py tests/test_scene_bodies.py tests/test_scene_assim.py -q -p no:cacheprovider' \
  || { log "scene tests FAILED: stop"; exit 1; }
claim_and_run T7_suite '$PY -m pytest tests -q -p no:cacheprovider' || { log "suite FAILED: stop"; exit 1; }
claim_and_run A1 "\$PY scripts/scene_run.py --n 40000 --seed 97 --scene $SP/v1_g005.json --out $R/v1_af > $R/v1_af.log 2>&1; echo v1_af exit \$?; for v in \"L:\" \"LB:--scene $SP/scene_long.json\" \"LS:--young 9.8e5\"; do t=\${v%%:*}; \$PY scripts/scene_run.py --n 40000 --seed 97 \${v#*:} --out $R/scene40k_\${t}_af > $R/scene40k_\${t}_af.log 2>&1; echo scene40k_\${t}_af exit \$?; done"
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
  nice $PY $S/read40k.py $R scene40k_L_af,scene40k_LB_af,scene40k_LS_af > $R/read_af.txt 2>&1 )
log "chain4 end"
