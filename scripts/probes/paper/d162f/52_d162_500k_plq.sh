#!/bin/bash
# 52_d162_500k_plq.sh -- 500k deliverable on a free card: H+C (held weight, placement at the cell; D162g carved the most)
# and its twin (dF_c + layer u). Output output/gpu/d162k/.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162k; mkdir -p $O/runs
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
A1=$G/d161/assets; A3=$G/d163/assets; AF=$G/d162f/assets
run plq500k_HC_97 repo_d162zs --tgt $A3/d163plaque.obj --render_tgt $A3/d163R300plaque.obj --n 500000 --seed 97 --lambda_ema 0 --placement_at_cell
run plq500k_T_97 repo_d162zs --tgt $A3/d163plaque.obj --render_tgt $A3/d163R300plaque.obj --n 500000 --seed 97 --render_weight_scale 0
log "R done (plq 500k)"
