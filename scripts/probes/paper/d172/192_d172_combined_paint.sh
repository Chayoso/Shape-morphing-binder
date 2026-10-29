#!/bin/bash
# 192_d172_combined_paint.sh -- queue job (agent D172, queued by post172.sh after the combined read): the painted end frames of
# the combined scene (tmp/d170/d170_studio.py +endonly=1: D160's studio renderer, each StudioRaster tile flipped back;
# twin | full | target paint), beside the grey stills of the same frames (deliver/combined/*_grey*.png).
J=192_combined_paint; TITLE="D172: combined scene, painted end frames"
source /data/relcfd/chayo/physmorph_v2/tmp/d172/job_common.sh
log "start"
for a in Rx Hu Hf; do
  [ -e $O/runs/cmb_${a}_97_frames12.npz ] || continue
  D=$O/deliver/combined/paint_$a; mkdir -p $D
  ( envr repo_d172; $PY $B/tmp/d170/d170_studio.py $D band $O/runs/cmb_${a}_97_frames12.npz $O/runs/cmb_T_97_frames12.npz \
      $O/runs/cmb_T_97_frames12.npz +endonly=1 +w=1280 +h=720 > $D/studio.log 2>&1 )
  log "paint $a exit $?"
done
log "done"
