#!/bin/bash
# 218_d173_tex300n.sh -- queue job (agent D173): the duck texture at 300k with the nearest-particle colour in the loss (Rxn), seed 97.
J=218_tex300n; TITLE="D173 texture 300k, nearest colour"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
TX="--appearance texture --texture $A/d173bob_tex.obj,$A/d173bob_diffuse.png"
gate_t; log "start"
run1 tex300k_Rxn_97 --tgt assets/bob.obj $TX --n 300000 --seed 97 $RX --colour_kernel nearest
log "done"
