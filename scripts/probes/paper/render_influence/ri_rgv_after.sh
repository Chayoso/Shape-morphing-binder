#!/bin/bash
# ri_rgv_after.sh -- after gpu1 queue job 01 (CPU): the "tag | render weight held" stills and side-by-side video.
B=/data/relcfd/chayo/physmorph_v2
D=$B/output/gpu/rgv_held
S=$D/status.txt
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
until grep -q "job 01 done" $S 2>/dev/null || grep -q "end 01_ri_rgv_held_bunny" $B/tmp/gpu1_queue/runner.log 2>/dev/null; do sleep 30; done
grep -q "job 01 done" $S || { echo "$(date '+%m-%d %H:%M:%S') compare: job 01 ended without its done line; not run" >> $S; exit 1; }
nice -n 10 $PY $B/tmp/render_influence/ri_rgv_compare.py $D > $D/compare.log 2>&1
echo "$(date '+%m-%d %H:%M:%S') compare rc $?" >> $S
ffmpeg -loglevel error -y -framerate 20 -pattern_type glob -i "$D/compare/sbs_frames/*.jpg" -c:v libx264 -crf 18 \
  -pix_fmt yuv420p $D/bunny300k_tag_vs_held_AB2CD_2x2_front_1080p.mp4 < /dev/null
echo "$(date '+%m-%d %H:%M:%S') sbs video rc $?" >> $S
ffmpeg -loglevel error -y -framerate 10 -pattern_type glob -i "$D/compare/sbs_frames/*.jpg" -c:v libx264 -crf 18 \
  -pix_fmt yuv420p $D/bunny300k_tag_vs_held_AB2CD_2x2_halfspeed_front_1080p.mp4 < /dev/null
echo "$(date '+%m-%d %H:%M:%S') sbs halfspeed video rc $?" >> $S
