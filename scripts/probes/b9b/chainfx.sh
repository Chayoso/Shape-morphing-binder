#!/bin/bash
# B9b forensic after the 40k gate's FAIL (pre-registered in docs/experiments.md, B9b, before its runs):
#   setsid -f bash scripts/probes/b9b/chainfx.sh REPO > /dev/null 2>&1 < /dev/null
# On ONE card idle 10 min (agent B9b): what tilts the body. Z1 fx_sphere0: the source sphere on the floor under gravity
# with no controls (zero_scene.py: every window commits the zero control); Z4 fx_bunny0 and Z2 fx_dragon0: the target's
# own shape as the source (the morph's end from the start), standing on the floor, no controls; Z3 fx_dragonC: the same
# dragon with the optimiser (the scene's default, the target = the source). Then the 40k side renders of bunny40k_L97 and
# dragon40k_L97 at 1080p (the 300k ones were not run). Card released; tobody.py tracks, fx_read.py.
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/b9b; R=$O/runs; LOG=$O/chainfx.log
S=$REPO/scripts/probes/b9b; SP=scripts/probes/b9b/specs
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $S/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null          # $PY; cwd = $REPO
echo $$ > $O/chainfx.pid
log "chainfx start (pid $$; $(tail -1 $REPO/DEPLOYED_FROM))"
claim_card "b9b/chainfx: 4 short 40k forensic runs and two 1080p side renders (~35 min)"
log "claimed gpu$CARD"
run() {                                         # TAG SCRIPT SPEC MESH_SRC MESH_TGT
  local tag=$1 script=$2 spec=$3 src=$4 tgt=$5
  $PY $script --scene $SP/$spec --src assets/$src.obj --tgt assets/$tgt.obj --n 40000 --seed 97 --save_stride 12 \
    --out $R/$tag > $R/$tag.log 2>&1
  log "$tag exit $?: $(tail -2 $R/$tag.log | head -1 | cut -c1-200)"
}
run fx_sphere0 $S/zero_scene.py b9b_bunny.json isosphere bunny
run fx_bunny0 $S/zero_scene.py b9b_bunny.json bunny bunny
run fx_dragon0 $S/zero_scene.py b9b_dragon.json dragon dragon
run fx_dragonC scripts/scene_run.py b9b_dragon.json dragon dragon
for tag in bunny40k_L97 dragon40k_L97; do
  NAME=${tag}_side1080 bash $REPO/scripts/probes/d147/render_scene_job.sh $R $tag +stride=1 +res=1080 +az=90 +el=18 >> $LOG 2>&1
  log "render ${tag}_side1080 exit $? ($(ls $R/render_${tag}_side1080/frames 2>/dev/null | wc -l) frames)"
  [ -f $R/${tag}_side1080.mp4 ] && rm -rf $R/render_${tag}_side1080/frames
done
release_card
log "card released"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
for tag in fx_sphere0 fx_bunny0 fx_dragon0 fx_dragonC; do
  [ -f $R/$tag.npz ] && $PY $S/tobody.py $R $tag > $R/tobody_$tag.txt 2>&1
  rm -f $R/${tag}_body12.npz $R/${tag}_frames12.npz
done
$PY $S/fx_read.py $R fx_sphere0 fx_bunny0 fx_dragon0 fx_dragonC bunny40k_L97 bunny40k_L99 bunny40k_L101 bunny40k_P97 \
  dragon40k_L97 dragon40k_L99 dragon40k_L101 dragon40k_P97 > $R/fx_read.txt 2>&1
log "fx_read exit $?"
log "chainfx end"
