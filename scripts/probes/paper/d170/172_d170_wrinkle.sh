#!/bin/bash
# 172_d170_wrinkle.sh -- queue job (agent D170): the fine relief of the meshes themselves (D159's pictures, --render_target
# mesh_shell: the bunny's skin wrinkles, the dragon's scales), Rx, 300k seed 97 (D159 ran 40k M97 on the gallery already).
# Twins reused: d167/runs/{bunny,dragon}300k_T_97 (lambda = 0: the pictures are unread).
J=172_wrinkle; TITLE="D170: 300k mesh-relief pictures (bunny wrinkles, dragon scales), Rx vs twin"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
log "start"; PAR=2
runp wrkb300k_Rx_97 --tgt assets/bunny.obj --render_target mesh_shell --n 300000 --seed 97 $RX
runp wrkd300k_Rx_97 --tgt assets/dragon.obj --render_target mesh_shell --n 300000 --seed 97 $RX
wait; log "done"
