#!/bin/bash
# D147 chain 20: the fixes' tests and 40k gate (pre-registered in docs/experiments.md, D147, 2026-10-16, before its runs).
# On ONE card idle for 10 minutes (tmp/d147/claim.sh): the scene tests and the full suite (pytest exit statuses; a
# failure stops the chain), then fix40k_s97 (render arm), fix40k_s97_off (its render-off twin), fix40k_s99 on
# specs/scene_throw.json (2 x 40k, --assim 0, every 12th step kept), then the readings: cut.py per run (and the
# references'), read40k.py's timeline, progress.py,
# judge.py, head.py (runs and references), apart.py over the whole run, and gate.py -> runs/gate_40k.txt.
#   nohup setsid bash scripts/probes/d147/chain20.sh REPO > /dev/null 2>&1 < /dev/null &
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d147; R=$O/runs; LOG=$O/chain20.log
S=$REPO/scripts/probes/d147; SPEC=scripts/probes/d147/specs/scene_throw.json
T=fix40k_s97,fix40k_s97_off,fix40k_s99
REFS=throwY40k_a,throwY40k_b                        # the D135 recipe at 40k on the same spec (seed 97, 2026-10-14)
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $B/tmp/d147/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null
log "chain20 start ($(cat $REPO/DEPLOYED_FROM))"
claim_card "chain20 fix tests + 40k gate (fix40k s97, twin, s99)"
log "claimed gpu$CARD"
$PY -m pytest -q -p no:cacheprovider tests/test_scene_fixes.py tests/test_scene_bodies.py tests/test_scene_schedule.py \
  tests/test_scene_assim.py tests/test_scene_frame.py > $O/T21_scene_tests.log 2>&1
rc=$?; log "scene tests exit $rc: $(tail -1 $O/T21_scene_tests.log)"
[ $rc -eq 0 ] || { log "the scene tests failed: no gate run"; exit 1; }
$PY -m pytest -q -p no:cacheprovider tests > $O/T21_suite.log 2>&1
rc=$?; log "suite exit $rc: $(tail -1 $O/T21_suite.log)"
[ $rc -eq 0 ] || { log "the suite failed: no gate run"; exit 1; }
for spec in "fix40k_s97 97" "fix40k_s97_off 97 --render_weight_scale 0" "fix40k_s99 99"; do
  set -- $spec
  tag=$1 seed=$2; shift 2
  log "$tag start"
  $PY scripts/scene_run.py --n 40000 --seed $seed --scene $SPEC --flags="--assim 0" --save_stride 12 "$@" \
    --out $R/$tag > $R/$tag.log 2>&1
  log "$tag exit $?: $(tail -1 $R/$tag.log | cut -c1-200)"
done
$PY $S/progress.py $R $T 0.15 > $R/progress_gate.txt 2>&1
log "progress exit $?"
release_card
( export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
  for t in ${T//,/ } ${REFS//,/ }; do $PY $S/cut.py $R $t > $R/cut_$t.txt 2>&1; done
  $PY $S/read40k.py $R $T,$REFS > $R/read_gate.txt 2>&1
  $PY $S/judge.py $R/read_gate.txt $R/progress_gate.txt $T > $R/judge_gate.txt 2>&1
  log "judge: $(tail -1 $R/judge_gate.txt)"
  $PY $S/head.py $R $T,$REFS > $R/head_gate.txt 2>&1
  log "head exit $?"
  for t in ${T//,/ }; do $PY $S/apart.py $R $t 0 16 > $R/apart_$t.txt 2>&1 & done
  wait
  $PY $S/gate.py $R $T $REFS > $R/gate_40k.txt 2>&1
  log "gate: $(tail -1 $R/gate_40k.txt)" )
log "chain20 end"
