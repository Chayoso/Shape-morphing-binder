#!/bin/bash
# D147 chain 12: the scene tests and the suite on the rebased branch (before settled-base is fast-forwarded to it).
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
log() { echo "$(date '+%F %T') $*" >> $O/chain12.log; }
claim_and_run() {
  local tag=$1 cmd=$2
  while true; do
    hm=$(date +%H%M)
    if [ "$hm" -ge 1500 ] && [ "$hm" -lt 1900 ] && grep -qs "agent D147" $L/gpu*.lock/info; then sleep 30; continue; fi
    for g in 3 2 1 0; do
      [ -d $L/gpu$g.lock ] && continue
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$used" -gt 30000 ] && continue
      if bash $S/job.sh $g "$REPO" "$tag" "$cmd"; then
        rc=$(grep "^${tag}_end " $O/jobs.log | tail -1 | awk '{print $NF}'); log "$tag gpu$g exit $rc"; return $rc
      fi
    done
    sleep 20
  done
}
log "chain12 start, repo $REPO"
claim_and_run T14_scene_tests '$PY -m pytest tests/test_scene_frame.py tests/test_scene_bodies.py tests/test_scene_assim.py -q -p no:cacheprovider' \
  && claim_and_run T15_suite '$PY -m pytest tests -q -p no:cacheprovider'
log "chain12 end"
