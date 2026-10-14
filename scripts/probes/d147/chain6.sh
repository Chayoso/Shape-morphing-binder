#!/bin/bash
# D147 chain 6 (after chain 5): on the code with D148's leak fix and the scene's weak views (the 300k code): the scene
# tests with the memory test, the suite, and the device memory over 6 windows of a 2x20k scene with gc off.
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
log() { echo "$(date '+%F %T') $*" >> $O/chain6.log; }
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
log "chain6 start, repo $REPO; waiting for chain5"
until grep -q "chain5 end" $O/chain5.log 2>/dev/null; do sleep 30; done
R=$O/runs
claim_and_run T10_scene_tests '$PY -m pytest tests/test_scene_frame.py tests/test_scene_bodies.py tests/test_scene_assim.py -q -p no:cacheprovider' \
  || { log "scene tests FAILED: stop"; exit 1; }
claim_and_run T11_suite '$PY -m pytest tests -q -p no:cacheprovider' || { log "suite FAILED: stop"; exit 1; }
claim_and_run MEM20k "\$PY scripts/scene_run.py --n 20000 --seed 97 --windows 6 --gc_off --out $R/mem20k > $R/mem20k.log 2>&1; echo mem20k exit \$?; grep -o 'gpu_free_mib[^,]*' $R/mem20k.json | head -8"
log "chain6 end"
