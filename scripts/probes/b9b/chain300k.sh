#!/bin/bash
# B9b chain 2, the 300k stage (pre-registered in docs/experiments.md, D145, "B9b"; started by chain40k.sh on its gate's
# PASS, or by hand):  nohup setsid bash scripts/probes/b9b/chain300k.sh REPO > /dev/null 2>&1 < /dev/null &
# On ONE card idle 10 min (lock protocol, agent B9b): the eight 300k runs (bunny / dragon x L97 L99 L101 P97, every 12th
# frame kept); the GPU readings as B9's read300.sh, in the body frame against the independent 300k sample: the end and
# B9's end times (endgen145t.py), the yardstick (render_terms_probe.py) and the display (surface_layer_probe.py) at the
# last kept frame, momentum_probe.py on the world frames; B9's G2L97 ends in their own Buckens frame (b9body.py +
# endgen145t.py). The card released: apart.py, read_b9b.py 300k. A card claimed again for the two side renders at 1080p
# (bunny300k_L97, dragon300k_L97; the studio look, the floor, a still camera, az 90 el 18 as the deliverables' side).
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/b9b; R=$O/runs; LOG=$O/chain300k.log; T=$B/tmp
S=$REPO/scripts/probes/b9b; SP=scripts/probes/b9b/specs
mkdir -p $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $S/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null          # $PY; cwd = $REPO
echo $$ > $O/chain300k.pid
log "chain300k start (pid $$; $(head -1 $REPO/DEPLOYED_FROM))"
claim_card "b9b/chain300k: 8 one-body 300k gravity runs (~2 h) and their GPU readings"
log "claimed gpu$CARD"
for mesh in bunny dragon; do
  for a in "L97 97" "L99 99" "L101 101" "P97 97 --render_weight_scale 0"; do
    set -- $a; arm=$1 seed=$2; shift 2; tag=${mesh}300k_$arm
    $PY scripts/scene_run.py --scene $SP/b9b_$mesh.json --tgt assets/$mesh.obj --n 300000 --seed $seed --save_stride 12 \
      "$@" --out $R/$tag > $R/$tag.log 2>&1
    log "$tag exit $?: $(tail -2 $R/$tag.log | head -1 | cut -c1-220)"
  done
done
times() { [ $1 = bunny ] && echo 6 1.16667 || echo 9.5 8 7.83333 7.5; }      # B9's 300k ends (frames 1440 / 280; 2280 1920 1880 1800)
box() { [ $1 = bunny ] && echo 1900,100,2800,700 || echo 1900,60,2500,560; }    # read300.sh's crops
rm -f $R/endgen.jsonl
for mesh in bunny dragon; do
  IND=$G/d90/${mesh}_ind300k.npz
  for arm in L97 L99 L101 P97; do
    tag=${mesh}300k_$arm
    [ -f $R/$tag.npz ] || continue
    $PY $S/tobody.py $R $tag $(times $mesh) > $R/tobody_$tag.txt 2>&1
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
    log "$tag read: $(tail -1 $R/tobody_$tag.txt | cut -c1-200)"
  done
  # B9's own G2L97 (frames12 kept) read as the scene reads its bodies: its end in its Buckens frame
  b=$G/d145t_b9/${mesh}300k_G2L97
  $PY $S/b9body.py ${b}_frames12.npz $b.json $R/b9_${mesh}300k_G2L97_body.npz > $R/b9body_$mesh.txt 2>&1
  ( export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_t
    $PY $T/d145r/endgen145t.py $R/endgen.jsonl "b9_${mesh}300k_G2L97_body=$R/b9_${mesh}300k_G2L97_body.npz,$IND" \
      > $R/endgen_b9_$mesh.txt 2>&1 < /dev/null )
  log "B9 ${mesh} G2L97 in its body frame: $(tail -1 $R/b9body_$mesh.txt | cut -c1-200)"
done
release_card
log "card released for the CPU readings"
( export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
  for mesh in bunny dragon; do
    for arm in L97 L99 L101 P97; do
      tag=${mesh}300k_$arm
      [ -f $R/$tag.npz ] && $PY $REPO/scripts/probes/d147/apart.py $R $tag > $R/apart_$tag.txt 2>&1 &
    done
  done
  wait
  $PY $S/read_b9b.py 300k $R $R/read_300k > $R/read_300k.txt 2>&1
  log "read_300k exit $?" )
for f in $R/*300k_*_frames12.npz $R/*300k_*_body12.npz $R/*300k_*_at*_body.npz $R/*300k_*_end1.npz $R/b9_*_body.npz; do
  [ -f $f ] || continue
  sz=$(du -m $f | cut -f1); rm -f $f
  echo "$(date '+%Y-%m-%d %H:%M') b9b: removed $f ($sz MB; B9b 300k reading copy, read; the run's own kept frames stay in its .npz)" >> $G/cleanup.log
done
claim_card "b9b/chain300k renders: bunny300k_L97 and dragon300k_L97 side views at 1080p (~20 min)"
log "claimed gpu$CARD for the renders"
for tag in bunny300k_L97 dragon300k_L97; do
  [ -f $R/$tag.npz ] || continue
  NAME=${tag}_side1080 bash $REPO/scripts/probes/d147/render_scene_job.sh $R $tag +stride=1 +res=1080 +az=90 +el=18 >> $LOG 2>&1
  log "render ${tag}_side1080 exit $? ($(ls $R/render_${tag}_side1080/frames 2>/dev/null | wc -l) frames)"
  [ -f $R/${tag}_side1080.mp4 ] && rm -rf $R/render_${tag}_side1080/frames
done
release_card
log "chain300k end"
