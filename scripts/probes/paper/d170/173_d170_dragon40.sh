#!/bin/bash
# 173_d170_dragon40.sh -- queue job (agent D170): the dragon with render-only relief at 40k, seeds 97 / 99: the 3D target the
# Taubin-smoothed dragon (output/gpu/d170/assets/d170smooth_dragon.obj), the render's pictures the detailed dragon
# (assets/dragon.obj, scales and ridges): Rx with the pictures drawn from samples of it (the frozen path) and Rxm with them
# drawn from its mesh shell (--render_target mesh_shell, D159; prepare.py:114 takes the --render_tgt mesh), and the twin.
# dsm40k_T_97 / dsm40k_Rxm_97 run detached on card 0 (tmp/d170/dragon40_now.sh); this job waits for them.
J=173_dragon40; TITLE="D170: dragon render-only relief at 40k, Rx / Rxm vs twin"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
log "start"; PAR=4
DR="--tgt $A/d170smooth_dragon.obj --render_tgt assets/dragon.obj --n 40000"
runp dsm40k_T_99 $DR --seed 99 $TW
runp dsm40k_Rx_97 $DR --seed 97 $RX
runp dsm40k_Rx_99 $DR --seed 99 $RX
runp dsm40k_Rxm_99 $DR --seed 99 $RX --render_target mesh_shell
wait
for k in $(seq 1 360); do [ -e $O/runs/dsm40k_T_97.rc ] && [ -e $O/runs/dsm40k_Rxm_97.rc ] && break; sleep 20; done
log "done"
