#!/bin/bash
# 183_d170_dragon300_r15.sh -- queue job (agent D170): job 181's H0mr15 again, alone on a card (it died of CUDA OOM beside
# H0m: the radius-1.5 exterior's lattice is 0.46 pitch, twice as many discs per area): the dragon render-only relief at 300k,
# held weight, relaxation off, mesh-shell pictures, exterior radius 1.5 pitches; repo_d170.
J=183_dragon300_r15; TITLE="D170: dragon render-only relief at 300k, exterior radius 1.5"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
REPO=repo_d170; PAR=1; OFF="--settle_windows 30 --no-settle_relax --no-layer_relax"
log "start"
run1 dsm300k_H0mr15_97 --tgt $A/d170smooth_dragon.obj --render_tgt assets/dragon.obj --n 300000 --seed 97 --lambda_ema 0 $OFF \
  --render_target mesh_shell --exterior_radius 1.5
log "done"
