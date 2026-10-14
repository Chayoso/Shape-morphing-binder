#!/bin/bash
# D147's chain on hyde06, from the deployed repository REPO (started by resume.sh): each GPU stage claims a free card
# through the lock directory (gpu_locks/gpuN.lock; a card with no lock and under 30 GB in use), runs, releases it, and
# the next stage starts only if this one passed (its exit status read from jobs.log):
#   0  the floor drops' readings (P8, P9; CPU)
#   1  the scene tests            (tests/test_scene_frame.py, tests/test_scene_bodies.py)
#   2  the full suite             (the single-body path bit for bit: test_frozen_recipe, test_runtime_exact, ...)
#   3  the prototype re-runs      (P1 with the sample's fill volume, P7 from the first contact)
#   4  the 40k batch              (V0, V1, the scene's render arm and twin: d147_batch40k.sh)
#   5  the readings               (read40k.py)
# Logs: output/gpu/d147/{jobs.log, TAG.log, chain.log}.
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
mkdir -p $O/runs
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }

claim_and_run() {                       # TAG "command": wait for a free card, run there, return the command's status
  local tag=$1 cmd=$2
  while true; do
    for g in 3 1 0 2; do
      [ -d /data/relcfd/chayo/physmorph_v2/gpu_locks/gpu$g.lock ] && continue
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$used" -gt 30000 ] && continue
      if bash $S/job.sh $g "$REPO" "$tag" "$cmd"; then
        rc=$(grep "^${tag}_end " $O/jobs.log | tail -1 | awk '{print $NF}')
        log "$tag gpu$g exit $rc"
        return $rc
      fi
    done
    sleep 15
  done
}

log "chain start, repo $REPO"
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; CUDA_VISIBLE_DEVICES="" nice $PY $S/read_drops.py $O > $O/P89_read.txt 2>&1 )
log "stage 0 done (P89_read.txt)"
claim_and_run T2_scene_tests '$PY -m pytest tests/test_scene_frame.py tests/test_scene_bodies.py -q -p no:cacheprovider' \
  || { log "stage 1 FAILED: stop"; exit 1; }
claim_and_run T3_suite '$PY -m pytest tests -q -p no:cacheprovider' || { log "stage 2 FAILED: stop"; exit 1; }
claim_and_run P1b_P7b 'for s in "P1b_multi:--mode multi" "P7b_grad:--mode multi --gradcheck"; do t=${s%%:*}; $PY scripts/probes/d147/contact_proto.py ${s#*:} --out '$O'/$t > '$O'/$t.out 2>&1; echo "$t exit $?"; done'
claim_and_run B40k "bash $S/batch40k.sh"
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; CUDA_VISIBLE_DEVICES="" nice $PY $S/read40k.py $O/runs > $O/runs/read40k.txt 2>&1 )
log "chain end"
