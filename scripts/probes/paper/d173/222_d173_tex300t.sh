#!/bin/bash
# 222_d173_tex300t.sh -- queue job (agent D173): Bob at 300k, Rx rerun on the code that records the block shares (job 202's
# tex300k_Rx_97 predates it), seed 97: the shares beside job 218's Rx + nearest colour.
J=222_tex300t; TITLE="D173 texture 300k Rx rerun with the block records"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
TX="--appearance texture --texture $A/d173bob_tex.obj,$A/d173bob_diffuse.png"
gate_t; log "start"
run1 tex300k_Rxt_97 --tgt assets/bob.obj $TX --n 300000 --seed 97 $RX
log "done"
