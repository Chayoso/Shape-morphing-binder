#!/bin/bash
# D147 chain 2 (after scene_na): the floor's position constraint tested and the suite run, then V1 again with it and
# without the assimilation (V1e), then V0's third seed. One card at a time, through the locks.
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
log() { echo "$(date '+%F %T') $*" >> $O/chain2.log; }
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
log "chain2 start, repo $REPO; waiting for scene_na"
until grep -q scene_na_end $O/runs/scene_na.log 2>/dev/null; do sleep 30; done
claim_and_run T4_scene_tests '$PY -m pytest tests/test_scene_frame.py tests/test_scene_bodies.py -q -p no:cacheprovider' \
  || { log "scene tests FAILED: stop"; exit 1; }
claim_and_run T5_suite '$PY -m pytest tests -q -p no:cacheprovider' || { log "suite FAILED: stop"; exit 1; }
R=$O/runs; SP=scripts/probes/d147/specs
claim_and_run V1e "for g in 005 010; do \$PY scripts/scene_run.py --n 40000 --seed 97 --scene $SP/v1_g\$g.json --flags=\"--assim 0\" --out $R/v1e_g\$g > $R/v1e_g\$g.log 2>&1; echo v1e_g\$g exit \$?; done" 
claim_and_run S2 "for v in \"L:\" \"LB:--scene $SP/scene_long.json\" \"LS:--young 9.8e5\"; do t=\${v%%:*}; \$PY scripts/scene_run.py --n 40000 --seed 97 \${v#*:} --flags=\"--assim 0\" --out $R/scene40k_\${t}_v2 > $R/scene40k_\${t}_v2.log 2>&1; echo scene40k_\${t}_v2 exit \$?; done"
claim_and_run V0c "for m in bunny dragon; do for a in fixed body; do \$PY scripts/scene_run.py --n 40000 --seed 101 --tgt assets/\$m.obj --scene $SP/v0_\${m}_\$a.json --out $R/v0_\${m}_\${a}_s101 > $R/v0_\${m}_\${a}_s101.log 2>&1; echo v0_\${m}_\${a}_s101 exit \$?; done; done"
log "chain2 end"
