#!/bin/bash
# d163_video.sh TAG BASE N RELIEF TOP_PREFIX BOTTOM_PREFIX "TOP_LABEL" "BOTTOM_LABEL" [WORKERS] -- D163's deliverable
# renders on hyde06, CPU only (nice 10, no card): the stacked video frames at 3840 x 2160 (d163_render.py video: the full
# arm above its twin over every kept frame, 20 frames held, one turn of both end states), the 4K and 1080p videos (20 fps,
# libx264 crf 18; no hflip: the renderer's camera is the true one, the words read correctly), the close-ups of the words
# at the end, the contact sheet; the frames folder removed once both videos exist (cleanup.log).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d163; W=$B/tmp/d163; V=$O/video/$1
TAG=$1; BASE=$2; N=$3; RELIEF=$4; TOP=$5; BOT=$6; TL=$7; BL=$8; WK=${9:-8}
mkdir -p $V
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=8 D157_THREADS=8 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d163
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') [video $TAG] $*" >> $O/status.txt; }
cd $B/repo_d163
log "start"
s=$(date +%s)
nice -n 10 $PY $W/d163_render.py video $V/frames $BASE $N ${TOP}_frames12.npz ${BOT}_frames12.npz "$TL" "$BL" 3840 2160 0.30 0.18 160 $WK > $V/video.log 2>&1
log "frames exit $? ($(( $(date +%s) - s )) s)"
ffmpeg -loglevel error -y -framerate 20 -i $V/frames/f%05d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p $V/${TAG}_4k.mp4 < /dev/null
ffmpeg -loglevel error -y -framerate 20 -i $V/frames/f%05d.jpg -vf scale=1920:1080:flags=lanczos -c:v libx264 -crf 18 \
  -pix_fmt yuv420p $V/${TAG}_1080p.mp4 < /dev/null
log "videos $(ls -la $V/${TAG}_4k.mp4 $V/${TAG}_1080p.mp4 2>&1 | awk '{print $5}' | tr '\n' ' ')"
s=$(date +%s)
nice -n 10 $PY $W/d163_render.py closeup $V/closeup $BASE $N $RELIEF ${TOP}_frames12.npz:"$TL" ${BOT}_frames12.npz:"$BL" 1920 1080 > $V/closeup.log 2>&1
log "closeup exit $? ($(( $(date +%s) - s )) s)"
s=$(date +%s)
nice -n 10 $PY $W/d163_render.py sheet $V/${TAG}_sheet.jpg $BASE $N $RELIEF 0.5,1.0,1.5,2.5,4.0 ${TOP}_frames12.npz:"$TL" ${BOT}_frames12.npz:"$BL" > $V/sheet.log 2>&1
log "sheet exit $? ($(( $(date +%s) - s )) s)"
mkdir -p $V/stills
for k in 0 20 40 60 100; do f=$V/frames/$(printf f%05d $k).jpg; [ -e $f ] && cp $f $V/stills/; done
last=$(ls $V/frames | grep -c jpg); cp $V/frames/$(printf f%05d $((last - 181))).jpg $V/stills/end.jpg 2>/dev/null
cp $V/frames/$(printf f%05d $((last - 80))).jpg $V/stills/end_back.jpg 2>/dev/null
if [ -s $V/${TAG}_4k.mp4 ] && [ -s $V/${TAG}_1080p.mp4 ]; then
  sz=$(du -sm $V/frames | cut -f1); rm -rf $V/frames
  echo "$(date '+%Y-%m-%d %H:%M') D163: removed $V/frames ($sz MB; the video frames of $TAG, both videos written, stills in $V/stills)" >> $G/cleanup.log
fi
log "done"
