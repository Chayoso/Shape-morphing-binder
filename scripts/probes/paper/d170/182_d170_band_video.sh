#!/bin/bash
# 182_d170_band_video.sh -- queue job (agent D170, queued by post.sh only if the 300k band passes): the colour deliverable,
# twin | full | target paint, front and back, every kept frame through the settle (tmp/d170/d170_studio.py = D160's studio
# renderer with each StudioRaster tile flipped back left-right); output/gpu/d170/deliver/band/.
J=182_band_video; TITLE="D170: the 300k band video"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
D=$O/deliver/band; mkdir -p $D; log "start"
( envr repo_d166; $PY $W/d170_studio.py $D band $O/runs/band300k_Rx_97_frames12.npz $G/d162f/runs/band300k_T_97_frames12.npz \
    $G/d162f/assets/ind_bunny300k.npz +stride=1 +w=960 +h=540 > $D/studio.log 2>&1 )
log "studio exit $?"
if [ -s $D/side_by_side.mp4 ]; then
  sz=$(du -sm $D/frames | cut -f1); rm -rf $D/frames
  echo "$(date '+%Y-%m-%d %H:%M') D170: removed $D/frames ($sz MB; the band video's frames, side_by_side.mp4 and end.png written)" >> $G/cleanup.log
fi
log "done"
