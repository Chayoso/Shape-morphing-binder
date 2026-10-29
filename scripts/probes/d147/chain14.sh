#!/bin/bash
# D147 chain 14 (the redesign: farther apart, the bodies meet mid-morph): the scene tests with the schedule, the 40k
# throw (specs/scene_throw.json) at seed 97 and its same-input repeat, the readings (timeline, progress), the 4K render.
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
log() { echo "$(date '+%F %T') $*" >> $O/chain14.log; }
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
log "chain14 start, repo $REPO"
SP=scripts/probes/d147/specs
claim_and_run T16_scene_tests '$PY -m pytest tests/test_scene_schedule.py tests/test_scene_frame.py tests/test_scene_bodies.py tests/test_scene_assim.py -q -p no:cacheprovider' \
  || { log "scene tests FAILED: stop"; exit 1; }
claim_and_run S6 "for t in a b; do \$PY scripts/scene_run.py --n 40000 --seed 97 --scene $SP/scene_throw.json --flags=\"--assim 0\" --save_stride 12 --out $R/throw40k_\$t > $R/throw40k_\$t.log 2>&1; echo throw40k_\$t exit \$?; done"
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
  nice $PY $S/read40k.py $R throw40k_a,throw40k_b > $R/read_throw40k.txt 2>&1 )
log "S6 timeline read (read_throw40k.txt)"
claim_and_run P6 "\$PY $S/progress.py $R throw40k_a,throw40k_b > $R/progress_throw40k.txt 2>&1; bash $S/render_scene_job.sh $R throw40k_a +stride=1"
log "chain14 end (progress_throw40k.txt, throw40k_a.mp4)"
