#!/bin/bash
# 217_d173_magC99.sh -- queue job (agent D173): option C, the dragon's own scales magnified K times (the smallest K whose
# perfect 300k sample reads corr >= 0.25: output/gpu/d173/assets/magC/variant.txt = "TOKEN K", build_magC2.sh): Hu with the
# magnified-scale pictures, seed 99, the 3D target the smoothed dragon (the twin and the smooth-picture arm: the stylized test's runs).
J=217_magC99; TITLE="D173 option C, seed 99"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
gate_t; for k in $(seq 1 360); do [ -e $A/magC/variant.txt ] && break; sleep 20; done
read KT K SB < $A/magC/variant.txt; [ -n "$KT" ] || { log "no K passes the probe: stop"; exit 1; }
log "start (K $K)"
run1 mag300k_Hk_99 --tgt $G/d170/assets/d170smooth_dragon.obj --n 300000 --seed 99 --render_tgt $A/magC/d173M${KT}_300_dragon.obj $HU
log "done"
