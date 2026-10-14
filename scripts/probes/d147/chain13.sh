#!/bin/bash
# D147 chain 13: the 40k render-off twins of o100 (seed 97 twice, the same input as o100_v2 and o100_v2b), read with them.
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
log() { echo "$(date '+%F %T') $*" >> $O/chain13.log; }
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
log "chain13 start, repo $REPO"
SP=scripts/probes/d147/specs
claim_and_run S5 "for t in a b; do \$PY scripts/scene_run.py --n 40000 --seed 97 --scene $SP/scene_long_o100.json --render_weight_scale 0 --flags=\"--assim 0\" --out $R/scene40k_LB_o100_off\$t > $R/scene40k_LB_o100_off\$t.log 2>&1; echo scene40k_LB_o100_off\$t exit \$?; done"
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
  nice $PY $S/read40k.py $R scene40k_LB_o100_v2,scene40k_LB_o100_v2b,scene40k_LB_o100_offa,scene40k_LB_o100_offb > $R/read_s5.txt 2>&1 )
log "chain13 end (read_s5.txt)"
