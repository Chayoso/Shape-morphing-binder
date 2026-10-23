#!/bin/bash
# 55_d162_300k_ripple_band.sh -- jobs 23 / 24 again (they died of CUDA OOM on the shared gpu1) on a free card: the 300k
# ripples with H+C and its twin, the 300k band with D160d and its twin. Output output/gpu/d162f/.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162f; mkdir -p $O/runs
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
A1=$G/d161/assets; A=$O/assets
run rip300k_T_97 repo_d162zs --tgt $A1/d161smooth_bunny.obj --render_tgt $A/d162fR300ripple_bunny.obj --n 300000 --seed 97 --render_weight_scale 0
run rip300k_HC_97 repo_d162zs --tgt $A1/d161smooth_bunny.obj --render_tgt $A/d162fR300ripple_bunny.obj --n 300000 --seed 97 --lambda_ema 0 --placement_at_cell
run band300k_T_97 repo_d160s --tgt assets/bunny.obj --appearance band --n 300000 --seed 97 --render_weight_scale 0
run band300k_D_97 repo_d160s --tgt assets/bunny.obj --appearance band --n 300000 --seed 97 --appearance_weight held
log "R done (300k ripple band)"
