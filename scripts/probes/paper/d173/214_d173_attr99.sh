#!/bin/bash
# 214_d173_attr99.sh -- queue job (agent D173): the attribution proof on the stylized scales E at 300k, seed 99: the twin, Hu with the
# scaled pictures (Hs), Hu with the smooth pictures (Hm); after the gate t.
J=214_attr99; TITLE="D173 attribution, seed 99"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
gate_t; log "start"; PAR=2
runp atr300k_T_99 $SM300 --seed 99 --render_tgt $G/d172/assets/d172SE300_dragon.obj $TW
runp atr300k_Hs_99 $SM300 --seed 99 --render_tgt $G/d172/assets/d172SE300_dragon.obj $HU
wait
runp atr300k_Hm_99 $SM300 --seed 99 --render_tgt $G/d170/assets/d170smooth_dragon.obj $HU
wait; log "done"
