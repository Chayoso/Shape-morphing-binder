#!/bin/bash
# 174_d170_gal.sh -- queue job (agent D170): Rx on the 6-mesh 40k gallery, seeds 97 / 99. Baselines reused: the tag (rule)
# d167/runs/gal_*_R_97 (repo_d166 defaults) and d155/gate | d154/g40runs *40k_V97 / V99; the twin *40k_W97 / W99.
J=174_gal; TITLE="D170 gallery: Rx on 6 meshes at 40k, seeds 97 / 99"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
log "start"; PAR=3
for s in 97 99; do for m in bunny dragon armadilo bob homer teapot; do
  runp gal_${m}_Rx_$s --tgt assets/$m.obj --n 40000 --seed $s $RX; done; done
wait; log "done"
