#!/bin/bash
# D147 chain 18: what tears the droplet off after the landing (throwY40k_a, 12.5-15.9 s, body 1's face): the same input
# with u off, the render-off twin and D146b (relax_driven + u_adam_carry; the branch merged with refreeze-prep), each
# read with apart.py over 9-16 s, the progress and the judge.
#   bash scripts/probes/d147/chain18.sh REPO
REPO=$1; SPEC=scene_throw
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
log() { echo "$(date '+%F %T') $*" >> $O/chain18.log; }
claim_and_run() {                      # one D147 card before 19:00 (the coordinator), any free card after
  local tag=$1 cmd=$2
  while true; do
    if [ "$(date +%H%M)" -lt 1900 ] && grep -qs "agent D147" $L/gpu*.lock/info; then sleep 30; continue; fi
    for g in 3 0 2 1; do
      [ -d $L/gpu$g.lock ] && continue
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$used" -gt 2000 ] && continue
      if bash $S/job.sh $g "$REPO" "$tag" "$cmd"; then
        rc=$(grep "^${tag}_end " $O/jobs.log | tail -1 | awk '{print $NF}'); log "$tag gpu$g exit $rc"; return $rc
      fi
    done
    sleep 30
  done
}
R=$O/runs; SP=scripts/probes/d147/specs; B=scene300k_$SPEC
log "chain18 start, repo $REPO"
claim_and_run T19_scene_tests '$PY -m pytest tests/test_scene_schedule.py tests/test_scene_bodies.py -q -p no:cacheprovider' \
  || { log "scene tests FAILED: stop"; exit 1; }
one() { echo "\$PY scripts/scene_run.py --n 40000 --seed 97 --scene $SP/scene_throw.json $2 --save_stride 12 --out $R/throwY40k_$1 > $R/throwY40k_$1.log 2>&1; echo throwY40k_$1 exit \$?"; }
claim_and_run A1 "$(one uoff '--flags="--assim 0 --u_off"')" &
sleep 60
claim_and_run A2 "$(one off '--render_weight_scale 0 --flags="--assim 0"')" &
sleep 60
claim_and_run A3 "$(one d146b '--flags="--assim 0 --relax_driven --u_adam_carry"')" &
wait
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
T=throwY40k_uoff,throwY40k_off,throwY40k_d146b
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
  for t in ${T//,/ }; do nice $PY $S/apart.py $R $t 9.0 16.0 > $R/apart_$t.txt 2>&1; done
  nice $PY $S/read40k.py $R $T > $R/read_ablate.txt 2>&1
  nice $PY $S/progress.py $R $T 0.15 > $R/progress_ablate.txt 2>&1
  nice $PY $S/judge.py $R/read_ablate.txt $R/progress_ablate.txt $T > $R/judge_ablate.txt 2>&1 )
log "chain18 end (apart_throwY40k_{uoff,off,d146b}.txt, judge_ablate.txt)"
