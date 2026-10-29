#!/bin/bash
# post172.sh MODE JOB -- D172: what a job's read arms (run by tmp/d172/after.sh, detached, CPU):
#   gal: an arm holding 6/6 at seed 97 gets seed 99 (job 191); scales40: the best arm keeping the scales at both seeds goes to
#   300k with its twin (job 190); combined: the deliverable stills (grey studio look on the CPU, the paint on a card, job 192);
#   plq / scales: raking close-ups (84 deg) for content-passing runs, twin | full videos for PASS runs at 300k (d170_render.py).
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d172; O=$B/output/gpu/d172; G=$B/output/gpu; Q=$B/tmp/gpu1_queue
MODE=$1; JOB=$2; D=$O/auto/$JOB; DL=$O/deliver
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=4 D157_THREADS=8 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d172
log() { echo "$(date '+%m-%d %H:%M:%S') [post $JOB] $*" >> $O/status.txt; }
queue() { cp $1 $Q/$(basename $1).tmp && mv $Q/$(basename $1).tmp $Q/$(basename $1); }
cd $B/repo_d172
if [ "$MODE" = gal ] && [ ! -e $W/191_d172_gal99.sh ]; then
  arms=$($PY $W/pick.py gal $D/verdict.json)
  if [ -n "$arms" ]; then
    $PY $W/pick.py job191 "$arms" > $W/191_d172_gal99.sh
    queue $W/191_d172_gal99.sh; log "gallery: $arms hold 6/6 at seed 97: seed 99 queued (job 191)"
    setsid nohup bash $W/after.sh gal 186_gal 191_d172_gal99 > $D/after99.out 2>&1 < /dev/null &
  else log "gallery: no arm holds 6/6 at seed 97: seed 99 not run"; fi
fi
if [ "$MODE" = scales40 ]; then
  arm=$($PY $W/pick.py scales $D/verdict.json)
  [ -n "$arm" ] || { arm=Hu; log "scales 40k: no arm keeps >= 0.40 above the twin + 0.10 at both seeds; 300k runs Hu anyway (the measurement 40k cannot give)"; }
  $PY $W/pick.py job190 "$arm" "$(cat $O/assets/variant.txt)" > $W/190_d172_scales300.sh
  queue $W/190_d172_scales300.sh; log "scales: 300k queued with $arm (job 190)"
  mkdir -p $O/auto/190_scales300
  D172_SCL300_ARMS="T $arm" setsid nohup bash $W/after.sh scales300 190_scales300 190_d172_scales300 > $O/auto/190_scales300/after.out 2>&1 < /dev/null &
fi
if [ "$MODE" = combined ]; then
  mkdir -p $DL/combined
  for a in T Rx Hu Hf; do
    p=$O/runs/cmb_${a}_97
    [ -e ${p}_frames12.npz ] || continue
    nice -n 10 $PY $B/tmp/d170/d170_render.py still $DL/combined/${a}_97_end_grey.png $G/d161/assets/d161smooth_bunny.obj 40000 \
      ${p}_frames12.npz -1 1.5708 0.10 1920 1080 72 >> $DL/combined/stills.log 2>&1
    nice -n 10 $PY $B/tmp/d170/d170_render.py still $DL/combined/${a}_97_end_grey_rake84_zoom2.png $G/d161/assets/d161smooth_bunny.obj 40000 \
      ${p}_frames12.npz -1 1.5708 0.10 1920 1080 84 2.0 >> $DL/combined/stills.log 2>&1
  done
  log "combined: grey stills written ($(ls $DL/combined/*.png 2>/dev/null | wc -l))"
  queue $W/192_d172_combined_paint.sh; log "combined: the painted end frames queued (job 192)"
fi
if [ "$MODE" = plq ] || [ "$MODE" = scales300 ] || [ "$MODE" = scales40 ]; then
  $PY $W/pick.py list $D/verdict.json > $D/post_list.txt
  V=$(cat $O/assets/variant.txt 2>/dev/null)
  while read k st ok full twin; do
    [ "$ok" = 1 ] || [ "$st" = PASS ] || [ "$MODE" != plq ] || continue
    stop=$($PY $W/pick.py stop $full.json)
    out=$DL/$k; mkdir -p $out
    case $MODE in
      plq) base=$G/d163/assets/d163plaque.obj; rel=$G/d163/assets/d163R300plaque.obj; n=300000; look=plaque ;;
      scales40) base=$G/d170/assets/d170smooth_dragon.obj; rel=$O/assets/d172S${V}40_dragon.obj; n=40000; look=auto ;;
      scales300) base=$G/d170/assets/d170smooth_dragon.obj; rel=$O/assets/d172S${V}300_dragon.obj; n=300000; look=auto ;;
    esac
    log "$k: close-ups (rake 84)"
    nice -n 10 $PY $B/tmp/d170/d170_render.py close $out/close $base $n $rel ${twin}_frames12.npz ${full}_frames12.npz $stop $look auto 1.0 84 \
      > $out/close.log 2>&1
    log "$k: close-ups exit $?"
    if [ "$st" = PASS ] && [ $n = 300000 ]; then
      nice -n 10 $PY $B/tmp/d170/d170_render.py sbs $out/frames $base $n ${twin}_frames12.npz ${full}_frames12.npz $stop 3840 2160 1.5708 0.12 160 24 72 \
        > $out/sbs.log 2>&1
      ffmpeg -loglevel error -y -framerate 20 -i $out/frames/f%05d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p $out/${k}_twin_full_4k.mp4 < /dev/null
      ffmpeg -loglevel error -y -framerate 20 -i $out/frames/f%05d.jpg -vf scale=1920:1080:flags=lanczos -c:v libx264 -crf 18 \
        -pix_fmt yuv420p $out/${k}_twin_full_1080p.mp4 < /dev/null
      if [ -s $out/${k}_twin_full_1080p.mp4 ]; then
        sz=$(du -sm $out/frames | cut -f1); rm -rf $out/frames
        echo "$(date '+%Y-%m-%d %H:%M') D172: removed $out/frames ($sz MB; the frames of the $k video, both videos written)" >> $G/cleanup.log
      fi
      log "$k: video done"
    fi
  done < $D/post_list.txt
fi
log "post done"
