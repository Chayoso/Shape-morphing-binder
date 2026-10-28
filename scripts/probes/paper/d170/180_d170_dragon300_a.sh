#!/bin/bash
# 180_d170_dragon300_a.sh -- queue job (agent D170): the dragon with render-only relief at 300k, seed 97 (3D target the smoothed
# dragon, pictures the detailed one), repo_d170: the blocker test the size probe calls for (the scales: rms 0.09 pitch,
# half-correlation ~1.1 pitch at 300k; the render's body surface is a radius-3-pitch field). Arms: T:$TW Rxm:$RX --render_target mesh_shell.
J=180_dragon300_a; TITLE="D170: dragon render-only relief at 300k (a)"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
REPO=repo_d170; PAR=2; OFF="--settle_windows 30 --no-settle_relax --no-layer_relax"
DR="--tgt $A/d170smooth_dragon.obj --render_tgt assets/dragon.obj --n 300000 --seed 97"
log "start"
runp dsm300k_T_97 $DR $TW
runp dsm300k_Rxm_97 $DR $RX --render_target mesh_shell
wait; log "done"
