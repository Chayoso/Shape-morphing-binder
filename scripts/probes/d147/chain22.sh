#!/bin/bash
# D147 chain 22: the ear-tip dump read and the damping's speed-up checked. After chain 21 (the forensic dump runs) ends,
# on one card idle 10 min: dump_forensic.py on chain 21's two dump runs (the layer's normals need the card);
# the scene tests and the full suite on repo_d147fast (exit statuses), then the equality runs at 40k, 6 windows each,
# seed 97: eqA and eqB on the old damping (repo_d147fix, atomic sums), eqN on the new (repo_d147fast), and
# drag_equal.py -> runs/drag_equal.txt.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d147; R=$O/runs; NEW=$B/repo_d147fast; OLD=$B/repo_d147fix
LOG=$O/chain22.log; SPEC=scripts/probes/d147/specs/scene_throw.json
log() { echo "$(date "+%F %T") $*" >> $LOG; }
source $B/tmp/d147/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
log "chain22 start, waiting for chain21's end"
until grep -q "chain21 end\|scene tests exit [1-9]" $O/chain21.log 2>/dev/null; do sleep 60; done
claim_card "chain22 damping speed-up: tests, suite, equality runs"
log "claimed gpu$CARD"
( export PHYSMORPH_RUN_REPO=$NEW PHYSMORPH_OUT=$O; source $NEW/scripts/ops/hyde06_env.sh > /dev/null
  for t in dump40k_s97 dump40k_s97_off; do
    $PY $NEW/scripts/probes/d147/dump_forensic.py $R $t 1 auto > $R/dumpfx_$t.txt 2>&1
    echo "dump forensic $t exit $?: $(head -1 $R/dumpfx_$t.txt | cut -c1-160)" >> $LOG
  done )
run_in() {                               # REPO TAG: a 6-window 40k scene run from that repository
  ( export PHYSMORPH_RUN_REPO=$1 PHYSMORPH_OUT=$O; source $1/scripts/ops/hyde06_env.sh > /dev/null
    $PY scripts/scene_run.py --n 40000 --seed 97 --scene $SPEC --flags="--assim 0" --save_stride 1 --windows 6 \
      --out $R/$2 > $R/$2.log 2>&1; echo "$2 exit $?" )
}
( export PHYSMORPH_RUN_REPO=$NEW PHYSMORPH_OUT=$O; source $NEW/scripts/ops/hyde06_env.sh > /dev/null
  $PY -m pytest -q -p no:cacheprovider tests/test_scene_fixes.py tests/test_scene_bodies.py tests/test_scene_schedule.py \
    tests/test_scene_assim.py tests/test_scene_frame.py > $O/T23_scene_tests.log 2>&1
  echo "scene tests exit $?: $(tail -1 $O/T23_scene_tests.log)" >> $LOG
  $PY -m pytest -q -p no:cacheprovider tests > $O/T23_suite.log 2>&1
  echo "suite exit $?: $(tail -1 $O/T23_suite.log)" >> $LOG )
log "$(run_in $OLD eq40k_oldA)"
log "$(run_in $OLD eq40k_oldB)"
log "$(run_in $NEW eq40k_new)"
release_card
( source $NEW/scripts/ops/hyde06_env.sh > /dev/null; CUDA_VISIBLE_DEVICES="" $PY $NEW/scripts/probes/d147/drag_equal.py $R \
    eq40k_oldA,eq40k_oldB,eq40k_new > $R/drag_equal.txt 2>&1; echo "drag_equal exit $?" >> $LOG )
log "chain22 end"
