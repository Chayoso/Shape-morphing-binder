#!/bin/bash
# D147 chain 11: the renderer's test with the camera that follows the pair, on the 40k pick (o100) and LB_v2.
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
log() { echo "$(date '+%F %T') $*" >> $O/chain11.log; }
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
R=$O/runs
log "chain11 start, repo $REPO"
claim_and_run RENDER_track "bash $S/render_scene_job.sh $R scene40k_LB_o100_v2; bash $S/render_scene_job.sh $R scene40k_LB_v2"
log "chain11 end"
