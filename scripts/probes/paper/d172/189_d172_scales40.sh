#!/bin/bash
# 189_d172_scales40.sh -- queue job (agent D172): D, the stylized dragon scales at 40k (output/gpu/d172/assets, built by
# tmp/d172/build_scales.sh; the variant whose perfect 300k sample reads corr >= 0.25, A before B): the 3D target the smoothed
# dragon, the pictures the scale relief; the twin, Rx, Hu, Hf, seeds 97 / 99.
J=189_scales40; TITLE="D172: stylized dragon scales, 40k"
source /data/relcfd/chayo/physmorph_v2/tmp/d172/job_common.sh
gate_wait
for k in $(seq 1 360); do [ -e $A/variant.txt ] && break; sleep 20; done
[ -s $A/variant.txt ] || { log "no representable scale variant (see size_probe): stop"; exit 1; }
V=$(cat $A/variant.txt); log "start (variant $V)"; PAR=4
S="--tgt $G/d170/assets/d170smooth_dragon.obj --render_tgt $A/d172S${V}40_dragon.obj --n 40000"
for s in 97 99; do
  runp scl40k_T_$s $S --seed $s $TW
  runp scl40k_Rx_$s $S --seed $s $RX
  runp scl40k_Hu_$s $S --seed $s $HU
  runp scl40k_Hf_$s $S --seed $s $HF
done
wait; log "done"
