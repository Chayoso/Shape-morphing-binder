#!/bin/bash
# 209_d173_colour_retry.sh -- queue job (agent D173): the two job-205 runs that died of CUDA OOM beside three others on one card
# (band300k_Rxn_97, tex40k_Rxn_97), again one at a time.
J=209_colour_retry; TITLE="D173 colour nearest, the OOM retries"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
TX="--appearance texture --texture $A/d173bob_tex.obj,$A/d173bob_diffuse.png"; NC="--colour_kernel nearest"
for n in band300k_Rxn_97 tex40k_Rxn_97; do [ -e $O/runs/$n.rc ] && [ "$(cat $O/runs/$n.rc)" != 0 ] && { mv $O/runs/$n.rc $O/runs/$n.rc_oom1; mv $O/runs/$n.log $O/runs/$n.log_oom1; }; done
log "start"
run1 band300k_Rxn_97 --tgt assets/bunny.obj --appearance band --n 300000 --seed 97 $RX $NC
run1 tex40k_Rxn_97 --tgt assets/bob.obj $TX --n 40000 --seed 97 $RX $NC
log "done"
