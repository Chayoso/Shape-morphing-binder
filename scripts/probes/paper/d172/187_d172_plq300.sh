#!/bin/bash
# 187_d172_plq300.sh -- queue job (agent D172): Hu / Hf on the 300k SIGGRAPH plaque, seed 97 (twin d163/runs300/plq300k_T_97).
J=187_plq300; TITLE="D172: Hu / Hf on the 300k plaque"
source /data/relcfd/chayo/physmorph_v2/tmp/d172/job_common.sh
gate_wait; log "start"; PAR=2
P="--tgt $G/d163/assets/d163plaque.obj --render_tgt $G/d163/assets/d163R300plaque.obj --n 300000 --seed 97"
runp plq300k_Hu_97 $P $HU
runp plq300k_Hf_97 $P $HF
wait; log "done"
