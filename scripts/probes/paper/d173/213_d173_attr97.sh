#!/bin/bash
# 213_d173_attr97.sh -- queue job (agent D173): after the gate t (job 209_d173_gate_t: defaults, band, Hu bit-SAME; the block records),
# the attribution proof on the stylized scales E at 300k, seed 97 (the 3D target the smoothed dragon): Hu with pictures of the
# scaled dragon (Hs) and with pictures of the SMOOTH dragon (Hm); the twin is job 190's scl300k_T_97.
J=213_attr97; TITLE="D173 attribution, seed 97"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
gate_t; log "start"; PAR=2
runp atr300k_Hs_97 $SM300 --seed 97 --render_tgt $G/d172/assets/d172SE300_dragon.obj $HU
runp atr300k_Hm_97 $SM300 --seed 97 --render_tgt $G/d170/assets/d170smooth_dragon.obj $HU
wait; log "done"
