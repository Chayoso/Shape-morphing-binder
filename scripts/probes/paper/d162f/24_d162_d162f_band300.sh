#!/bin/bash
# 24_d162_d162f_band300.sh -- gpu1 queue job (agent D162): D162f (c), colour from pictures at 300k: the bunny band scene
# with D160d's arm (the colour term at its first weight, the geometry render terms on the rule; wip/d160 963cd2b,
# repo_d160s) and its twin (dF_c + layer u, the same paint), seed 97.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162f; mkdir -p $O/runs
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
log "job 24 start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
run band300k_T_97 repo_d160s --tgt assets/bunny.obj --appearance band --n 300000 --seed 97 --render_weight_scale 0
run band300k_D_97 repo_d160s --tgt assets/bunny.obj --appearance band --n 300000 --seed 97 --appearance_weight held
log "R done (d162f band)"
