#!/bin/bash
# 80_ri_panel3_video.sh -- gpu1 queue job (agent RI, render influence; after the held-lambda RGV job and D164's gates):
# panel 3 "Where does the render change the shape?" as RGV-look videos, bunny and dragon 300k seed 97 (ri_video.py,
# pass 2; pass 1 = ri_video_values.py, done on the CPU), 1080p front and side, the 2 x 2 (plain full | plain twin /
# map front | map side) at full and half size, then mp4s at 20 fps (kept frames, real time). The body images are
# flipped left-right before the text inside ri_video.py (the studio camera mirrors). Afterwards ri_video_after.sh
# (CPU, already waiting) draws the hero still sheet and the combined figure.
B=/data/relcfd/chayo/physmorph_v2
O=$B/output/gpu/render_influence
V=$O/video
ST=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $ST; }
source $B/repo_rgvtag/scripts/ops/hyde06_env.sh > /dev/null
source $B/tmp/d146/cache_env.sh
export PYTHONPATH=$GPU_DEPS:$B/repo_rgvtag OMP_NUM_THREADS=8
cd $B/repo_rgvtag
log "RI pass 2 start (gpu1 queue, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
for M in bunny dragon; do
  grep -q "video values $M rc 0" $ST || { log "RI pass 1 $M missing; skipped"; continue; }
  # one fixed colour scale for the run: the 95th percentile over the kept frames of the per-frame 99th percentile of
  # |d| on the outer layer, rounded up to 0.25 / 0.5 / 0.75 / 1 / 1.5 / 2 / 3 pitches
  [ -s $V/dmax_$M ] || CUDA_VISIBLE_DEVICES="" $PY -c "
import json, numpy as np
r = json.load(open('$V/vals_$M.json'))['rows']
q = float(np.percentile([x['p99_ds'] for x in r], 95))
print(next(v for v in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5) if v >= q))" > $V/dmax_$M
  DM=$(cat $V/dmax_$M)
  $PY $B/tmp/render_influence/ri_video.py $M $V $V/frames +res=1080 +views=35,90 +dmax=$DM +grid=1 > $V/render_$M.log 2>&1
  rc=$?
  log "RI render $M dmax $DM rc $rc"
  [ $rc = 0 ] || continue
  for kd in map_35 map_90 grid gridhalf; do
    case $kd in map_35) nm=E_where_render_changes_shape_front_1080p ;; map_90) nm=E_where_render_changes_shape_side_1080p ;;
      grid) nm=E_2x2_front_2160p ;; gridhalf) nm=E_2x2_half_front_1080p ;; esac
    ffmpeg -loglevel error -y -framerate 20 -pattern_type glob -i "$V/frames/$M/$kd/*.jpg" -c:v libx264 -crf 18 \
      -pix_fmt yuv420p $V/${M}300k_$nm.mp4 < /dev/null
    log "RI ffmpeg $M $kd rc $? $(ls $V/frames/$M/$kd | wc -l) frames"
  done
  ffmpeg -loglevel error -y -framerate 10 -pattern_type glob -i "$V/frames/$M/gridhalf/*.jpg" -c:v libx264 -crf 18 \
    -pix_fmt yuv420p $V/${M}300k_E_2x2_halfspeed_front_1080p.mp4 < /dev/null
  log "RI ffmpeg $M gridhalf halfspeed rc $?"
done
log "RI pass 2 done"
