#!/bin/bash
# 179_d170_nostop.sh -- queue job (agent D170): the diagnostic after Rx failed the 40k gallery (0/6) and the 300k shapes (0/2):
# is the shape loss the early stop? Rx's held weight stops the runs by the frozen reject rule (3 rejected candidates in a row)
# before arrival (gallery dragon window 41 vs the rule's 69; plaque 300k window 21 vs the twin's 112). RxNS = Rx with
# --reject_stop 0 (the existing switch: runner.py:274 skips the rule at 0; patience and the window budget still end the run).
# 6-mesh gallery 40k seed 97, and the 300k plaque, repo_d166.
J=179_nostop; TITLE="D170 diagnostic: Rx without the reject stop"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
log "start"; PAR=4
runp plq300k_RxNS_97 --tgt $G/d163/assets/d163plaque.obj --render_tgt $G/d163/assets/d163R300plaque.obj --n 300000 --seed 97 $RX --reject_stop 0
for m in dragon armadilo homer bunny bob teapot; do runp gal_${m}_RxNS_97 --tgt assets/$m.obj --n 40000 --seed 97 $RX --reject_stop 0; done
wait; log "done"
