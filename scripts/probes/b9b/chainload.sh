#!/bin/bash
# B9b: the loads' own effect removed from the assimilation (SceneSpec.assim_load) with the recipe's drag in the standing
# frame, the 40k gate (pre-registered in docs/experiments.md, B9b, "The loads' own effect", before its runs). Experiment.
#   setsid -f bash scripts/probes/b9b/chainload.sh REPO > /dev/null 2>&1 < /dev/null
# ONE card idle 10 min, and none while a render experiment's chain (D159 / D160 / D161) waits (claim.sh YIELD_RENDER):
# the full suite (exit status recorded, non-zero stops); load_equal.py (the window-level frozen check); the no-gravity
# runs (frame fixed, no floor, drag 0.9: bunny ZA / ZB / ZC the current rule, ZL assim_load; dragon ZA); the definition
# (1)+(2) on specs/b9d_MESH.json (L97 / L99 / L101 / P97); (1) alone on specs/b9l_MESH.json (A97 / A99 / A101); the GPU
# readings as the share gate's. The card released: apart.py (niced, three at once), read_load.py. A second claim for the
# definition's 40k side renders.
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/b9b; R=$O/load; WD=$O/world; LOG=$O/chainload.log; T=$B/tmp
S=$REPO/scripts/probes/b9b; SP=scripts/probes/b9b/specs
mkdir -p $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $S/claim.sh
export YIELD_RENDER=1
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null          # $PY; cwd = $REPO
echo $$ > $O/chainload.pid
log "chainload start (pid $$; $(tail -1 $REPO/DEPLOYED_FROM)); claims yield to D159 / D160 / D161"
claim_card "b9b/chainload: full suite, the frozen checks, 19 one-body 40k runs (assim_load +- drag), readings (~100 min)"
log "claimed gpu$CARD"
$PY -m pytest -q -p no:cacheprovider tests > $R/suite.txt 2>&1
rc=$?
log "full suite exit $rc: $(tail -1 $R/suite.txt)"
[ $rc = 0 ] || { log "STOP: the full suite did not pass (exit $rc)"; exit 1; }
$PY $S/load_equal.py $R/load_equal.json 40000 97 bunny > $R/load_equal.txt 2>&1
log "load_equal exit $?: $(grep -h 'LOAD-EQUAL' $R/load_equal.txt)"
run() {                                         # TAG SPEC MESH SEED [flags]
  local tag=$1 spec=$2 mesh=$3 seed=$4; shift 4
  $PY scripts/scene_run.py --scene $SP/$spec.json --tgt assets/$mesh.obj --n 40000 --seed $seed --save_stride 12 "$@" \
    --out $R/$tag > $R/$tag.log 2>&1
  log "$tag exit $?: $(tail -2 $R/$tag.log | head -1 | cut -c1-220)"
}
run bunny40k_ZA97 b9n_bunny bunny 97
run bunny40k_ZB97 b9n_bunny bunny 97
run bunny40k_ZC97 b9n_bunny bunny 97
run bunny40k_ZL97 b9nl_bunny bunny 97
run dragon40k_ZA97 b9n_dragon dragon 97
for mesh in bunny dragon; do
  for a in "L97 97" "L99 99" "L101 101" "P97 97 --render_weight_scale 0"; do
    set -- $a; arm=$1 seed=$2; shift 2
    run ${mesh}40k_$arm b9d_$mesh $mesh $seed "$@"
  done
  for a in "A97 97" "A99 99" "A101 101"; do
    set -- $a; arm=$1 seed=$2
    run ${mesh}40k_$arm b9l_$mesh $mesh $seed
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
GRAV="L97 L99 L101 P97 A97 A99 A101"
for mesh in bunny dragon; do
  tags="${mesh}40k_ZA97"; [ $mesh = bunny ] && tags="$tags bunny40k_ZB97 bunny40k_ZC97 bunny40k_ZL97"
  for t in $tags; do
    [ -f $R/$t.npz ] && OMP_NUM_THREADS=4 nice -n 10 $PY $S/tobody.py $R $t --shape_every 20 > $R/tobody_$t.txt 2>&1
    rm -f $R/${t}_body12.npz $R/${t}_frames12.npz
  done
  for arm in $GRAV; do
    tag=${mesh}40k_$arm
    [ -f $R/$tag.npz ] || continue
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
done
release_card
log "card released for the CPU readings"
( export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
  n=0
  for mesh in bunny dragon; do
    for arm in $GRAV; do
      tag=${mesh}40k_$arm
      [ -f $R/$tag.npz ] || continue
      nice -n 15 $PY $REPO/scripts/probes/d147/apart.py $R $tag > $R/apart_$tag.txt 2>&1 &
      n=$((n + 1)); [ $((n % 3)) = 0 ] && wait
    done
  done
  wait
  nice -n 15 $PY $S/read_load.py $R $WD $R/read_load > $R/read_load.txt 2>&1
  log "read_load exit $?: $(grep '^GATE' $R/read_load.md)" )
for f in $R/*40k_*_frames12.npz $R/*40k_*_body12.npz $R/*40k_*_at*_body.npz; do
  [ -f $f ] || continue
  sz=$(du -m $f | cut -f1); rm -f $f
  echo "$(date '+%Y-%m-%d %H:%M') b9b: removed $f ($sz MB; B9b assim_load 40k reading copy, read; the run's own kept frames stay in its .npz)" >> $G/cleanup.log
done
claim_card "b9b/chainload renders: bunny40k_L97 and dragon40k_L97 (assim_load + drag) side views at 1080p (~3 min)"
log "claimed gpu$CARD for the renders"
for tag in bunny40k_L97 dragon40k_L97; do
  [ -f $R/$tag.npz ] || continue
  NAME=${tag}_side1080 bash $REPO/scripts/probes/d147/render_scene_job.sh $R $tag +stride=1 +res=1080 +az=90 +el=18 >> $LOG 2>&1
  log "render ${tag}_side1080 exit $? ($(ls $R/render_${tag}_side1080/frames 2>/dev/null | wc -l) frames)"
  [ -f $R/${tag}_side1080.mp4 ] && rm -rf $R/render_${tag}_side1080/frames
done
release_card
log "chainload end"
