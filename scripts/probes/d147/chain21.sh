#!/bin/bash
# D147 chain 21: the ear-tip forensic dump runs (repo_d147dump = the fixes + the dump hook): on one card idle 10 min,
# the scene tests, then dump40k_s97 and dump40k_s97_off (the gate's inputs) with --dump 10.0,12.5,1 (body 1, the
# windows starting 10.0-12.5 s), then apart.py over 9-14 s on each.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d147; R=$O/runs; REPO=$B/repo_d147dump; LOG=$O/chain21.log
S=$REPO/scripts/probes/d147; SPEC=scripts/probes/d147/specs/scene_throw.json
log() { echo "$(date "+%F %T") $*" >> $LOG; }
source $B/tmp/d147/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null
log "chain21 start ($(cat $REPO/DEPLOYED_FROM))"
claim_card "chain21 ear-tip forensic dump runs (40k s97, twin)"
log "claimed gpu$CARD"
$PY -m pytest -q -p no:cacheprovider tests/test_scene_fixes.py tests/test_scene_bodies.py tests/test_scene_schedule.py > $O/T22_scene_tests.log 2>&1
rc=$?; log "scene tests exit $rc: $(tail -1 $O/T22_scene_tests.log)"
[ $rc -eq 0 ] || exit 1
for spec in "dump40k_s97 97" "dump40k_s97_off 97 --render_weight_scale 0"; do
  set -- $spec; tag=$1 seed=$2; shift 2
  log "$tag start"
  $PY scripts/scene_run.py --n 40000 --seed $seed --scene $SPEC --flags="--assim 0" --save_stride 12 --dump 10.0,12.5,1 "$@" \
    --out $R/$tag > $R/$tag.log 2>&1
  log "$tag exit $?: $(tail -1 $R/$tag.log | cut -c1-160); dump $(ls $R/${tag}_dump 2>/dev/null | wc -l) windows, $(du -sh $R/${tag}_dump 2>/dev/null | cut -f1)"
  ( CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice $PY $S/apart.py $R $tag 9.0 14.0 > $R/apart_$tag.txt 2>&1; log "apart $tag: $(tail -1 $R/apart_$tag.txt)" ) &
done
release_card
wait
log "chain21 end"
