#!/bin/bash
# B9b option 1, the 300k stage (pre-registered in docs/experiments.md, B9b, option 1, 300k, before its runs):
#   setsid -f bash scripts/probes/b9b/chain300kw.sh REPO > /dev/null 2>&1 < /dev/null
# On ONE card idle 10 min (lock protocol, agent B9b): the eight 300k runs (bunny / dragon x L97 L99 L101 P97 on
# specs/b9w_MESH.json, every 12th frame kept); the GPU readings as B9's read300.sh, in the objective's frame (B9's world)
# against the independent 300k sample lifted as the run's target (ind_shift.py): the end and B9's end times
# (endgen145t.py), the yardstick (render_terms_probe.py) and the display (surface_layer_probe.py) at the last kept frame,
# momentum_probe.py; tobody.py's tracks (the corrected shape measure every 20th kept frame and the last). The card
# released: apart.py (niced, at most three at once), read_b9b.py 300k. A card claimed again for the side renders at 1080p
# (bunny300k_L97, dragon300k_L97: the studio look, the floor, a still camera, az 90 el 18).
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/b9b; R=$O/world; LOG=$O/chain300kw.log; T=$B/tmp
S=$REPO/scripts/probes/b9b; SP=scripts/probes/b9b/specs
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $S/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null          # $PY; cwd = $REPO
echo $$ > $O/chain300kw.pid
log "chain300kw start (pid $$; $(tail -1 $REPO/DEPLOYED_FROM))"
claim_card "b9b/chain300kw: 8 one-body 300k runs (option 1, ~2.5 h) and their GPU readings"
log "claimed gpu$CARD"
for mesh in bunny dragon; do
  for a in "L97 97" "L99 99" "L101 101" "P97 97 --render_weight_scale 0"; do
    set -- $a; arm=$1 seed=$2; shift 2; tag=${mesh}300k_$arm
    $PY scripts/scene_run.py --scene $SP/b9w_$mesh.json --tgt assets/$mesh.obj --n 300000 --seed $seed --save_stride 12 \
      "$@" --out $R/$tag > $R/$tag.log 2>&1
    log "$tag exit $?: $(tail -2 $R/$tag.log | head -1 | cut -c1-220)"
  done
done
times() { [ $1 = bunny ] && echo 6 1.16667 || echo 9.5 8 7.83333 7.5; }      # B9's 300k ends (frames 1440 / 280; 2280 1920 1880 1800)
box() { [ $1 = bunny ] && echo 1900,100,2800,700 || echo 1900,60,2500,560; }    # read300.sh's crops
rm -f $R/endgen.jsonl
for mesh in bunny dragon; do
  for arm in L97 L99 L101 P97; do
    tag=${mesh}300k_$arm
    [ -f $R/$tag.npz ] || continue
    OMP_NUM_THREADS=4 nice -n 10 $PY $S/tobody.py $R $tag $(times $mesh) --shape_every 100000 > $R/tobody_$tag.txt 2>&1
    IND=$R/ind_floor_$tag.npz
    ( export PHYSMORPH_RUN_REPO=$B/repo_r77; source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_r77
      $PY $T/d145/ind_shift.py $G/d90/${mesh}_ind300k.npz $R/${tag}_body12.npz $IND > $R/ind_shift_$tag.txt 2>&1 )
    specs="$tag=$R/${tag}_body12.npz,$IND"
    for f in $R/${tag}_at*_body.npz; do [ -f $f ] && specs="$specs $(basename $f .npz)=$f,$IND"; done
    ( export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_t
      $PY $T/d145r/endgen145t.py $R/endgen.jsonl $specs > $R/endgen_$tag.txt 2>&1 < /dev/null )
    ( export PHYSMORPH_RUN_REPO=$B/repo_r77; source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_r77
      F=$R/${tag}_body12.npz
      last=$($PY $T/lastraw.py $F)
      $PY $T/d145/endframe.py $F $R/${tag}_end1.npz
      $PY scripts/probes/settled/render_terms_probe.py $R/${tag}_end1.npz ref=$IND > $R/t3end_$tag.log 2>&1 < /dev/null
      $PY scripts/probes/settled/surface_layer_probe.py $F $R/e3end_$tag $(box $mesh) measure ref=$IND target own $last \
        > $R/e3end_$tag.log 2>&1 < /dev/null )
    $T/mom.sh $CARD $R/m_$tag.log "$mesh $arm=$R/${tag}_frames12.npz"
    log "$tag read: $(tail -1 $R/tobody_$tag.txt | cut -c1-260)"
  done
done
release_card
log "card released for the CPU readings"
( export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
  n=0
  for mesh in bunny dragon; do
    for arm in L97 L99 L101 P97; do
      tag=${mesh}300k_$arm
      [ -f $R/$tag.npz ] || continue
      ( nice -n 15 $PY $REPO/scripts/probes/d147/apart.py $R $tag > $R/apart_$tag.txt 2>&1
        nice -n 15 $PY $S/tobody.py $R $tag $(times $mesh) --shape_every 20 --track_only > $R/tobody2_$tag.txt 2>&1 ) &
      n=$((n + 1)); [ $((n % 3)) = 0 ] && wait
    done
  done
  wait
  nice -n 15 $PY $S/read_b9b.py 300k $R $R/read_300k "option 1: the scene path with the world-fixed standing target" > $R/read_300k.txt 2>&1
  log "read_300k exit $?" )
for f in $R/*300k_*_frames12.npz $R/*300k_*_body12.npz $R/*300k_*_at*_body.npz $R/*300k_*_end1.npz; do
  [ -f $f ] || continue
  sz=$(du -m $f | cut -f1); rm -f $f
  echo "$(date '+%Y-%m-%d %H:%M') b9b: removed $f ($sz MB; B9b option 1 300k reading copy, read; the run's own kept frames stay in its .npz)" >> $G/cleanup.log
done
claim_card "b9b/chain300kw renders: bunny300k_L97 and dragon300k_L97 side views at 1080p (~10 min)"
log "claimed gpu$CARD for the renders"
for tag in bunny300k_L97 dragon300k_L97; do
  [ -f $R/$tag.npz ] || continue
  NAME=${tag}_side1080 bash $REPO/scripts/probes/d147/render_scene_job.sh $R $tag +stride=1 +res=1080 +az=90 +el=18 >> $LOG 2>&1
  log "render ${tag}_side1080 exit $? ($(ls $R/render_${tag}_side1080/frames 2>/dev/null | wc -l) frames)"
  [ -f $R/${tag}_side1080.mp4 ] && rm -rf $R/render_${tag}_side1080/frames
done
release_card
log "chain300kw end"
