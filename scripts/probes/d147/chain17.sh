#!/bin/bash
# D147 chain 17 (the continuous target, pre-registered): the scene tests, the 40k throw on the target-side
# interpolation at seed 97 and its same-input repeat (throwY40k_a, _b), the readings and the judge (judge.py: the
# rule with the snap item), the static 4K render of _a; on PASS chain 16 runs the 300k (s97, twin, s99, gated renders).
#   bash scripts/probes/d147/chain17.sh REPO
REPO=$1; SPEC=scene_throw
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
log() { echo "$(date '+%F %T') $*" >> $O/chain17.log; }
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
log "chain17 start, repo $REPO"
claim_and_run T18_scene_tests '$PY -m pytest tests/test_scene_schedule.py tests/test_scene_bodies.py tests/test_scene_assim.py tests/test_scene_frame.py -q -p no:cacheprovider' \
  || { log "scene tests FAILED: stop"; exit 1; }
claim_and_run S7 "for t in a b; do \$PY scripts/scene_run.py --n 40000 --seed 97 --scene $SP/scene_throw.json --flags=\"--assim 0\" --save_stride 12 --out $R/throwY40k_\$t > $R/throwY40k_\$t.log 2>&1; echo throwY40k_\$t exit \$?; done"
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
  nice $PY $S/read40k.py $R throwY40k_a,throwY40k_b > $R/read_throwY40k.txt 2>&1
  nice $PY $S/progress.py $R throwY40k_a,throwY40k_b 0.15 > $R/progress_throwY40k.txt 2>&1
  nice $PY $S/cut.py $R throwY40k_a,throwY40k_b > $R/cut_throwY40k.txt 2>&1
  nice $PY $S/judge.py $R/read_throwY40k.txt $R/progress_throwY40k.txt throwY40k_a,throwY40k_b > $R/judge_throwY40k.txt 2>&1 )
log "S7 read: $(tail -1 $R/judge_throwY40k.txt) (judge_throwY40k.txt)"
claim_and_run RENDER_Y40k "NAME=throwY40k_a_static bash $S/render_scene_job.sh $R throwY40k_a +stride=1"
if [ "$(tail -1 $R/judge_throwY40k.txt)" = "PASS" ]; then
  log "the 40k passes: chain 16 (the 300k) follows"
  SKIP40K=1 exec bash $S/chain16.sh $REPO
fi
log "the 40k fails the rule: no 300k; report"
