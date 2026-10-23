#!/bin/bash
# 54_d162_500k_band.sh -- 500k colour band on a free card: D160d (the colour arm) and its twin. Output output/gpu/d162k/.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162k; mkdir -p $O/runs
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
run band500k_D_97 repo_d160s --tgt assets/bunny.obj --appearance band --n 500000 --seed 97 --appearance_weight held
run band500k_T_97 repo_d160s --tgt assets/bunny.obj --appearance band --n 500000 --seed 97 --render_weight_scale 0
log "R done (band 500k)"
