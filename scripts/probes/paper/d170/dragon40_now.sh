#!/bin/bash
# dragon40_now.sh -- D170: the two first dragon 40k runs at once on card 0 (beside job 170's two 300k runs, ~34 GB of 49 GB;
# our runner holds gpu0's lock), so the top item does not wait for the queue.
J=173_dragon40; TITLE="D170: dragon render-only relief at 40k"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
export CUDA_VISIBLE_DEVICES=0
DR="--tgt $A/d170smooth_dragon.obj --render_tgt assets/dragon.obj --n 40000"
run1 dsm40k_T_97 $DR --seed 97 $TW &
sleep 3
run1 dsm40k_Rxm_97 $DR --seed 97 $RX --render_target mesh_shell &
wait
