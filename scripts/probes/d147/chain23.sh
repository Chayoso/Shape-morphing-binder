#!/bin/bash
# D147 chain 23: the stiff-material 40k gate (pre-registered in docs/experiments.md, D147, 2026-10-17, before its runs).
# The user's decision 2026-10-17 ("재료 단단하게"): B8's stiff material, --young 9.8e5 (E x 7), with the scene's motion
# kept in world units: specs/scene_throw_E7.json re-expresses the scene's two dimensionless groups for E x 7
# (G / 7, v_rel/c / sqrt 7), so gravity, the throw and the timings are scene_throw's. repo_d147fast (8f2af48: the four
# fixes and the damping's speed-up). One card idle 3 min: stiff40k_s97, its render-off twin, stiff40k_s99; progress on
# the card; then on the CPU the cuts, the timeline (timeline.py), the judge, head.py (the runs, the D135 references),
# apart.py over the whole run, gate.py (suffix stiff) and stiff_extra.py against the previous gate's fix40k runs.
#   nohup setsid bash scripts/probes/d147/chain23.sh REPO > /dev/null 2>&1 < /dev/null &
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d147; R=$O/runs; LOG=$O/chain23.log
S=$REPO/scripts/probes/d147; SPEC=scripts/probes/d147/specs/scene_throw_E7.json
T=stiff40k_s97,stiff40k_s97_off,stiff40k_s99
HREFS=throwY40k_a,throwY40k_b                       # the head bar's references (the previous gate's)
FREFS=fix40k_s97,fix40k_s97_off,fix40k_s99          # the previous gate's runs (the frozen material with the fixes)
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $B/tmp/d147/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null
log "chain23 start ($(cat $REPO/DEPLOYED_FROM | tr '\n' ' '))"
CLAIM_IDLE=180 claim_card "chain23 stiff-material 40k gate (s97, twin, s99)"
log "claimed gpu$CARD"
for spec in "stiff40k_s97 97" "stiff40k_s97_off 97 --render_weight_scale 0" "stiff40k_s99 99"; do
  set -- $spec
  tag=$1 seed=$2; shift 2
  log "$tag start"
  $PY scripts/scene_run.py --n 40000 --seed $seed --scene $SPEC --young 9.8e5 --flags="--assim 0" --save_stride 12 "$@" \
    --out $R/$tag > $R/$tag.log 2>&1
  log "$tag exit $?: $(tail -1 $R/$tag.log | cut -c1-200)"
done
$PY $S/progress.py $R $T 0.15 > $R/progress_stiff.txt 2>&1
log "progress exit $?"
release_card
( export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
  for t in ${T//,/ }; do $PY $S/cut.py $R $t > $R/cut_$t.txt 2>&1; done
  $PY $S/timeline.py $R $T > $R/read_stiff.txt 2>&1
  $PY $S/judge.py $R/read_stiff.txt $R/progress_stiff.txt $T > $R/judge_stiff.txt 2>&1
  log "judge: $(tail -1 $R/judge_stiff.txt)"
  $PY $S/head.py $R $T,$HREFS > $R/head_stiff.txt 2>&1
  log "head exit $?"
  for t in ${T//,/ }; do $PY $S/apart.py $R $t 0 16 > $R/apart_$t.txt 2>&1 & done
  wait
  $PY $S/gate.py $R $T $HREFS stiff > $R/gate_stiff40k.txt 2>&1
  log "gate: $(tail -1 $R/gate_stiff40k.txt)"
  $PY $S/stiff_extra.py $R $T $FREFS > $R/extra_stiff40k.txt 2>&1
  log "extra exit $?" )
log "chain23 end"
