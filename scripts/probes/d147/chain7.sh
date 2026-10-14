#!/bin/bash
# D147 chain 7 (replaces chains 5-6; the 300k code: D148's leak fix, the weak views): the scene tests with the memory
# test, the suite, V1e, S2, A1 (the new assimilation), the readings, V0c, the 2x20k memory probe with gc off.
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
log() { echo "$(date '+%F %T') $*" >> $O/chain7.log; }
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
log "chain7 start, repo $REPO"
R=$O/runs; SP=scripts/probes/d147/specs
claim_and_run T12_scene_tests '$PY -m pytest tests/test_scene_frame.py tests/test_scene_bodies.py tests/test_scene_assim.py -q -p no:cacheprovider' \
  || { log "scene tests FAILED: stop"; exit 1; }
claim_and_run T13_suite '$PY -m pytest tests -q -p no:cacheprovider' || { log "suite FAILED: stop"; exit 1; }
claim_and_run V1e "for g in 005 010; do \$PY scripts/scene_run.py --n 40000 --seed 97 --scene $SP/v1_g\$g.json --flags=\"--assim 0\" --out $R/v1e_g\$g > $R/v1e_g\$g.log 2>&1; echo v1e_g\$g exit \$?; done"
claim_and_run S2 "for v in \"L:\" \"LB:--scene $SP/scene_long.json\" \"LS:--young 9.8e5\"; do t=\${v%%:*}; \$PY scripts/scene_run.py --n 40000 --seed 97 \${v#*:} --flags=\"--assim 0\" --out $R/scene40k_\${t}_v2 > $R/scene40k_\${t}_v2.log 2>&1; echo scene40k_\${t}_v2 exit \$?; done"
claim_and_run A1 "\$PY scripts/scene_run.py --n 40000 --seed 97 --scene $SP/v1_g005.json --out $R/v1_af > $R/v1_af.log 2>&1; echo v1_af exit \$?; for v in \"L:\" \"LB:--scene $SP/scene_long.json\" \"LS:--young 9.8e5\"; do t=\${v%%:*}; \$PY scripts/scene_run.py --n 40000 --seed 97 \${v#*:} --out $R/scene40k_\${t}_af > $R/scene40k_\${t}_af.log 2>&1; echo scene40k_\${t}_af exit \$?; done"
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
  nice $PY $S/read40k.py $R scene40k_L_na,scene40k_LB_na,scene40k_LS_na > $R/read_na.txt 2>&1
  nice $PY $S/read40k.py $R scene40k_L_v2,scene40k_LB_v2,scene40k_LS_v2 > $R/read_v2.txt 2>&1
  nice $PY $S/read40k.py $R scene40k_L_af,scene40k_LB_af,scene40k_LS_af > $R/read_af.txt 2>&1 )
log "readings done (read_na, read_v2, read_af)"
claim_and_run V0c "for m in bunny dragon; do for a in fixed body; do \$PY scripts/scene_run.py --n 40000 --seed 101 --tgt assets/\$m.obj --scene $SP/v0_\${m}_\$a.json --out $R/v0_\${m}_\${a}_s101 > $R/v0_\${m}_\${a}_s101.log 2>&1; echo v0_\${m}_\${a}_s101 exit \$?; done; done"
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
  nice $PY $S/read40k.py $R scene40k_L_af,scene40k_LB_af,scene40k_LS_af > $R/read_af.txt 2>&1 )
claim_and_run MEM20k "\$PY scripts/scene_run.py --n 20000 --seed 97 --windows 6 --gc_off --out $R/mem20k > $R/mem20k.log 2>&1; echo mem20k exit \$?; grep -o 'gpu_free_mib[^,]*' $R/mem20k.json | head -8"
log "chain7 end"
