#!/bin/bash
# 23_d162_d162f_ripple300.sh -- gpu1 queue job (agent D162): D162f (b), surface ripples from pictures at 300k: the
# smoothed bunny whose render target carries concentric ridges on its flank (4 pitches wide, period 8, 1.5 pitches high;
# d161_mesh.py, output/gpu/d162f/assets/ripple300.json), the lambda-held full arm and its twin (dF_c + layer u), seed 97.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162f; mkdir -p $O/runs
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
A1=$G/d161/assets; A=$O/assets
log "job 23 start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
run rip300k_T_97 repo_d162ds --tgt $A1/d161smooth_bunny.obj --render_tgt $A/d162fR300ripple_bunny.obj --n 300000 --seed 97 --render_weight_scale 0
run rip300k_H_97 repo_d162ds --tgt $A1/d161smooth_bunny.obj --render_tgt $A/d162fR300ripple_bunny.obj --n 300000 --seed 97 --lambda_ema 0
log "R done (d162f ripple)"
