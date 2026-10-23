#!/bin/bash
# D165 job c (gpu1 queue, 2026-10-22): the stamp test on the kept end states (job b): (a) H, the full arm, (b) the twin
# (its post-hoc decal is display only and takes no part); then the studio videos (scripts/render_scene.py, the still
# camera, the floor) at 1080p and 4K per run, flipped left-right at encoding (the studio raster's mirror), side by side.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d165; W=$B/tmp/d165
mkdir -p $O/stamp $O/video $W/tmpdir
source $B/tmp/d146/cache_env.sh
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%F %T') [c] $*" >> $O/status.txt; }
log "start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
( export PHYSMORPH_RUN_REPO=$B/repo_d165 PHYSMORPH_OUT=$B/output/gpu; source $B/repo_d165/scripts/ops/hyde06_env.sh > /dev/null 2>&1
  export TMPDIR=$W/tmpdir; cd $B/repo_d165
  for arm in H T; do
    st=$O/runs300/plq300k_${arm}_97_end.npz
    [ -f $st ] || { log "no end state $st"; continue; }
    $PY scripts/probes/d165/stamp.py $st $O/stamp/e_$arm > $O/stamp/e_$arm.log 2>&1
    log "stamp e_$arm exit $?: $(tail -1 $O/stamp/e_$arm.log | cut -c1-300)"
  done
  for arm in H T k_H k_T; do
    f=$O/stamp/e_$arm; case $arm in k_*) f=$O/stamp/$arm;; esac
    [ -f ${f}_stamp.npz ] || continue
    for res in 1080 2160; do
      d=$O/video/render_${arm}_$res
      $PY scripts/render_scene.py ${f}_stamp.npz $d +stride=1 +res=$res +az=25 +el=28 +field=zbs > $d.log 2>&1
      log "render $arm $res exit $? ($(ls $d/frames 2>/dev/null | wc -l) frames)"
      ffmpeg -y -loglevel error -framerate 20 -pattern_type glob -i "$d/frames/*.jpg" -vf hflip -c:v libx264 -crf 18 \
        -pix_fmt yuv420p $O/video/stamp_${arm}_$res.mp4 < /dev/null && rm -rf $d/frames
    done
  done
  for res in 1080 2160; do
    [ -f $O/video/stamp_H_$res.mp4 ] && [ -f $O/video/stamp_T_$res.mp4 ] && ffmpeg -y -loglevel error \
      -i $O/video/stamp_H_$res.mp4 -i $O/video/stamp_T_$res.mp4 -filter_complex hstack -c:v libx264 -crf 18 \
      -pix_fmt yuv420p $O/video/stamp_full_vs_twin_$res.mp4 < /dev/null && log "side by side $res"
  done )
touch $O/STAMP_DONE
log "end"
