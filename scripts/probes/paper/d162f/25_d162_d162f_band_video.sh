#!/bin/bash
# 25_d162_d162f_band_video.sh -- gpu1 queue job (agent D162): D162f (c)'s studio side-by-side (d160_studio.py: the full
# arm | its twin | the target paint, front and back) of the 300k band runs of job 24; the target paint column on an
# independent 300k sample (seed 99, drawn here in the target's frame).
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162f; mkdir -p $O/after
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
grep -q "R done (d162f band)" $O/status.txt || { log "job 25: the band runs are not done; stop"; exit 1; }
IND=$O/assets/ind_bunny300k.npz
if [ ! -e $IND ]; then
  ( envr repo_d160s; $PY - << 'EOP' > $O/after/ind.log 2>&1
import numpy as np
from physmorph.sampling.mesh import draws_in_frame, load_normalized
_, v_src = load_normalized("assets/isosphere.obj", 300000, 97, return_volume=True, sample="stratified", fill={})
frame = {}
load_normalized("assets/bunny.obj", 300000, 98, match_volume=v_src, sample="stratified", frame=frame, fill={})
x = draws_in_frame("assets/bunny.obj", frame, 300000, [99])[0]
np.savez("/data/relcfd/chayo/physmorph_v2/output/gpu/d162f/assets/ind_bunny300k.npz", tgt=x)
print("ind", x.shape)
EOP
  )
fi
log "job 25: band video start"
( envr repo_d160s; $PY $B/tmp/d160/d160_studio.py $O/after/vid_band300k band $O/runs/band300k_D_97_frames12.npz \
    $O/runs/band300k_T_97_frames12.npz $IND +w=1280 +h=720 > $O/after/vid_band300k.log 2>&1 < /dev/null; echo $? > $O/after/vid_band300k.rc )
log "job 25: band video exit $(cat $O/after/vid_band300k.rc)"
