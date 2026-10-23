#!/bin/bash
# 53_d162_500k_rip.sh -- 500k deliverable on a free card: H+C (held weight, placement at the cell; D162g carved the most)
# and its twin (dF_c + layer u). Output output/gpu/d162k/.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162k; mkdir -p $O/runs
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
A1=$G/d161/assets; A3=$G/d163/assets; AF=$G/d162f/assets
run rip500k_HC_97 repo_d162zs --tgt $A1/d161smooth_bunny.obj --render_tgt $AF/d162fR300ripple_bunny.obj --n 500000 --seed 97 --lambda_ema 0 --placement_at_cell
run rip500k_T_97 repo_d162zs --tgt $A1/d161smooth_bunny.obj --render_tgt $AF/d162fR300ripple_bunny.obj --n 500000 --seed 97 --render_weight_scale 0
log "R done (rip 500k)"
