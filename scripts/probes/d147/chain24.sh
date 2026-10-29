#!/bin/bash
# D147 chain 24: the stiff 300k (pre-registered in docs/experiments.md, D147, 2026-10-17 13:55, before its runs).
# repo_d147fast (8f2af48 + the amended cut): stiff300k_s97, its render-off twin, stiff300k_s99 on scene_throw_E7.json
# with --young 9.8e5, 2 x 150k, on ONE card idle 3 min; progress on the card; the card released; on the CPU the cuts,
# the timeline, the judge, head.py (the runs and the 300k interim), apart.py over the whole run, clean.py, gate.py
# (suffix stiff300k), stiff_extra.py against the final300k runs, table.py; then a card claimed again for the renders:
# per run the cut and the contact close-up at 4K and 1080p and the whole run at 1080p (still camera, studio look).
#   nohup setsid bash scripts/probes/d147/chain24.sh REPO > /dev/null 2>&1 < /dev/null &
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d147; R=$O/runs; LOG=$O/chain24.log
S=$REPO/scripts/probes/d147; SPEC=scripts/probes/d147/specs/scene_throw_E7.json
T=stiff300k_s97,stiff300k_s97_off,stiff300k_s99
HREF=scene300k_scene_throw_s97                      # the head bar: the 300k interim (the D135 scene)
FREFS=final300k_s97,final300k_s97_off,final300k_s99 # the frozen material's final 300k runs
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $B/tmp/d147/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null
log "chain24 start ($(cat $REPO/DEPLOYED_FROM | tr '\n' ' '))"
CLAIM_IDLE=180 claim_card "chain24 stiff 300k runs (s97, twin, s99)"
log "claimed gpu$CARD"
for spec in "stiff300k_s97 97" "stiff300k_s97_off 97 --render_weight_scale 0" "stiff300k_s99 99"; do
  set -- $spec
  tag=$1 seed=$2; shift 2
  log "$tag start"
  $PY scripts/scene_run.py --n 150000 --seed $seed --scene $SPEC --young 9.8e5 --flags="--assim 0" --save_stride 12 "$@" \
    --out $R/$tag > $R/$tag.log 2>&1
  log "$tag exit $?: $(tail -1 $R/$tag.log | cut -c1-200)"
done
$PY $S/progress.py $R $T 0.15 > $R/progress_stiff300k.txt 2>&1
log "progress exit $?"
release_card
log "card released for the readings"
( export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
  for t in ${T//,/ } $HREF; do $PY $S/cut.py $R $t > $R/cut_$t.txt 2>&1; done
  $PY $S/timeline.py $R $T > $R/read_stiff300k.txt 2>&1
  $PY $S/judge.py $R/read_stiff300k.txt $R/progress_stiff300k.txt $T > $R/judge_stiff300k.txt 2>&1
  log "judge: $(tail -1 $R/judge_stiff300k.txt)"
  $PY $S/head.py $R $T,$HREF > $R/head_stiff300k.txt 2>&1
  log "head exit $?"
  for t in ${T//,/ }; do $PY $S/apart.py $R $t 0 16 > $R/apart_$t.txt 2>&1 & done
  wait
  : > $R/clean_stiff300k.txt; : > $R/cut_stiff300k.txt
  for t in ${T//,/ }; do $PY $S/clean.py $R $R/cut_$t.txt $t >> $R/clean_stiff300k.txt 2>&1; cat $R/cut_$t.txt >> $R/cut_stiff300k.txt; done
  $PY $S/gate.py $R $T $HREF stiff300k > $R/gate_stiff300k.txt 2>&1
  log "gate: $(tail -1 $R/gate_stiff300k.txt)"
  $PY $S/stiff_extra.py $R $T $FREFS > $R/extra_stiff300k.txt 2>&1
  log "extra exit $?"
  $PY $S/table.py $R $T $R/read_stiff300k.txt $R/progress_stiff300k.txt $R/judge_stiff300k.txt $R/cut_stiff300k.txt \
    $R/clean_stiff300k.txt > $R/table_stiff300k.txt 2>&1
  log "table exit $?" )
CLAIM_IDLE=180 claim_card "chain24 stiff 300k renders"
log "claimed gpu$CARD for the renders"
render() {                                                      # TAG NAME K0 K1 RES [+camera=...]
  local tag=$1 name=$2 k0=$3 k1=$4 res=$5; shift 5
  NAME=$name bash $S/render_scene_job.sh $R $tag +stride=1 +frames=$k0-$k1 +res=$res "$@" >> $LOG 2>&1
  log "render $name exit $? ($(ls $R/render_$name/frames 2>/dev/null | wc -l) frames)"
  rm -rf $R/render_$name/frames
}
for t in ${T//,/ }; do
  [ -f $R/$t.npz ] || continue
  read CUT K0 K1 <<< "$($PY $S/spans.py $R $t $R/read_stiff300k.txt $R/cut_$t.txt)"
  log "$t spans: cut kept frame $CUT, contact close-up $K0-$K1"
  [ -n "$K1" ] || continue
  render $t ${t}_4k_cut 0 $CUT 2160
  render $t ${t}_4k_contact $K0 $K1 2160 +camera=contact
  render $t ${t}_1080p_cut 0 $CUT 1080
  render $t ${t}_1080p_contact $K0 $K1 1080 +camera=contact
  render $t ${t}_1080p_full 0 100000 1080
done
release_card
log "chain24 end"
