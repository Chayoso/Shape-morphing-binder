#!/bin/bash
# B9b: the controls' share of the assimilation (SceneSpec.assim_share), the 40k gate (pre-registered in
# docs/experiments.md, B9b, "The controls' share of the assimilation", before its runs). An experiment, not adopted.
#   setsid -f bash scripts/probes/b9b/chainshare.sh REPO > /dev/null 2>&1 < /dev/null
# ONE card idle 10 min (lock protocol, agent B9b): the full suite (the frozen path's pinned references; the scene's tests
# with the field off and the new ones), exit status recorded, non-zero stops the chain; the no-gravity runs (standing
# frame, g 0: bunny ZA97 / ZB97 the current rule twice, ZS97 the share; dragon ZA97); the eight gravity runs on
# specs/b9s_MESH.json (option 1 + assim_share; L97 / L99 / L101 / P97); the GPU readings (tracks, the lifted independent
# sample, silhouette IoU and chamfer, momentum, the 40k yardstick, also for option 1's runs in output/gpu/b9b/world).
# The card released: apart.py (niced, three at once), read_share.py (the gate). A second claim for the 40k side renders.
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/b9b; R=$O/share; WD=$O/world; LOG=$O/chainshare.log; T=$B/tmp
S=$REPO/scripts/probes/b9b; SP=scripts/probes/b9b/specs
mkdir -p $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $S/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null          # $PY; cwd = $REPO
echo $$ > $O/chainshare.pid
log "chainshare start (pid $$; $(tail -1 $REPO/DEPLOYED_FROM))"
# the coordinator (2026-10-21): SHOWCASE, D145r and the D155 dragon re-run come first after 17:00; this chain starts its
# 10-minute idle wait only at 17:05, so any of them waiting for a card takes it first
while [ "$(date +%H%M)" -lt 1705 ] && [ "$(date +%F)" = "2026-10-21" ]; do sleep 60; done
log "claim wait starts"
claim_card "b9b/chainshare: full suite, 12 one-body 40k runs (the controls' share of the assimilation), readings (~70 min)"
log "claimed gpu$CARD"
$PY -m pytest -q -p no:cacheprovider tests > $R/suite.txt 2>&1
rc=$?
log "full suite exit $rc: $(tail -1 $R/suite.txt)"
[ $rc = 0 ] || { log "STOP: the full suite did not pass (exit $rc)"; exit 1; }
run() {                                         # TAG SPEC MESH SEED [flags]
  local tag=$1 spec=$2 mesh=$3 seed=$4; shift 4
  $PY scripts/scene_run.py --scene $SP/$spec.json --tgt assets/$mesh.obj --n 40000 --seed $seed --save_stride 12 "$@" \
    --out $R/$tag > $R/$tag.log 2>&1
  log "$tag exit $?: $(tail -2 $R/$tag.log | head -1 | cut -c1-220)"
}
run bunny40k_ZA97 b9z_bunny bunny 97
run bunny40k_ZB97 b9z_bunny bunny 97
run bunny40k_ZS97 b9zs_bunny bunny 97
run dragon40k_ZA97 b9z_dragon dragon 97
for mesh in bunny dragon; do
  for a in "L97 97" "L99 99" "L101 101" "P97 97 --render_weight_scale 0"; do
    set -- $a; arm=$1 seed=$2; shift 2
    run ${mesh}40k_$arm b9s_$mesh $mesh $seed "$@"
  done
done
times() { [ $1 = bunny ] && echo 5.33333 || echo 8.16667; }
: > $R/ind40.log
yardstick() {                                   # DIR TAG IND
  local d=$1 tag=$2 ind=$3
  $PY $S/endarch.py $d $tag $d/${tag}_endobj.npz > /dev/null 2>&1
  ( export PHYSMORPH_RUN_REPO=$B/repo_r77; source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_r77
    $PY scripts/probes/settled/render_terms_probe.py $d/${tag}_endobj.npz ref=$ind > $d/t3end_$tag.log 2>&1 < /dev/null )
  rm -f $d/${tag}_endobj.npz
}
for tag in bunny40k_ZA97 bunny40k_ZB97 bunny40k_ZS97 dragon40k_ZA97 bunny40k_L97 bunny40k_L99 bunny40k_L101 bunny40k_P97 \
           dragon40k_L97 dragon40k_L99 dragon40k_L101 dragon40k_P97; do
  [ -f $R/$tag.npz ] || continue
  mesh=${tag%%40k_*}
  OMP_NUM_THREADS=4 nice -n 10 $PY $S/tobody.py $R $tag $(times $mesh) --shape_every 20 > $R/tobody_$tag.txt 2>&1
  at=$(ls $R/${tag}_at*_body.npz 2>/dev/null | head -1)
  ( export PHYSMORPH_RUN_REPO=$B/repo_r77; source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_r77
    $PY $T/d145/ind_shift.py $G/d92/ind_$mesh.npz $R/${tag}_body12.npz $R/ind_floor_$tag.npz > $R/ind_shift_$tag.txt 2>&1
    $PY $T/gallery_ind.py $R/ind40.log "$tag=$R/${tag}_body12.npz,$R/ind_floor_$tag.npz" \
      "$(basename $at .npz)=$at,$R/ind_floor_$tag.npz" > $R/ind40_$tag.txt 2>&1 < /dev/null )
  $T/mom.sh $CARD $R/m_$tag.log "$mesh $tag=$R/${tag}_frames12.npz"
  yardstick $R $tag $R/ind_floor_$tag.npz
  log "$tag read: $(tail -1 $R/tobody_$tag.txt | cut -c1-240)"
done
for tag in bunny40k_L97 bunny40k_L99 bunny40k_L101 bunny40k_P97 dragon40k_L97 dragon40k_L99 dragon40k_L101 dragon40k_P97; do
  [ -f $WD/$tag.npz ] && yardstick $WD $tag $WD/ind_floor_$tag.npz
done
log "option 1's 40k yardstick read: $(ls $WD/t3end_*40k_*.log 2>/dev/null | wc -l) runs"
release_card
log "card released for the CPU readings"
( export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
  n=0
  for tag in bunny40k_ZA97 bunny40k_ZB97 bunny40k_ZS97 dragon40k_ZA97 bunny40k_L97 bunny40k_L99 bunny40k_L101 bunny40k_P97 \
             dragon40k_L97 dragon40k_L99 dragon40k_L101 dragon40k_P97; do
    [ -f $R/$tag.npz ] || continue
    nice -n 15 $PY $REPO/scripts/probes/d147/apart.py $R $tag > $R/apart_$tag.txt 2>&1 &
    n=$((n + 1)); [ $((n % 3)) = 0 ] && wait
  done
  wait
  nice -n 15 $PY $S/read_share.py $R $WD $R/read_share > $R/read_share.txt 2>&1
  log "read_share exit $?: $(grep '^GATE' $R/read_share.md)" )
for f in $R/*40k_*_frames12.npz $R/*40k_*_body12.npz $R/*40k_*_at*_body.npz; do
  [ -f $f ] || continue
  sz=$(du -m $f | cut -f1); rm -f $f
  echo "$(date '+%Y-%m-%d %H:%M') b9b: removed $f ($sz MB; B9b assim_share 40k reading copy, read; the run's own kept frames stay in its .npz)" >> $G/cleanup.log
done
claim_card "b9b/chainshare renders: bunny40k_L97 and dragon40k_L97 (the share) side views at 1080p (~5 min)"
log "claimed gpu$CARD for the renders"
for tag in bunny40k_L97 dragon40k_L97; do
  [ -f $R/$tag.npz ] || continue
  NAME=${tag}_side1080 bash $REPO/scripts/probes/d147/render_scene_job.sh $R $tag +stride=1 +res=1080 +az=90 +el=18 >> $LOG 2>&1
  log "render ${tag}_side1080 exit $? ($(ls $R/render_${tag}_side1080/frames 2>/dev/null | wc -l) frames)"
  [ -f $R/${tag}_side1080.mp4 ] && rm -rf $R/render_${tag}_side1080/frames
done
release_card
log "chainshare end"
