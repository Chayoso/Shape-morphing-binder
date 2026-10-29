#!/bin/bash
# 208_d174_ab300.sh -- queue job (agent D174): the runtime A/B at 300k, the stylized dragon scales (D172's scl300k, variant E,
# seed 97): Hu2 and Hu4 against D172's Hu and twin (output/gpu/d172/runs/scl300k_{Hu,T}_97), PAR 2; after gate ab.
J=208_ab300
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
gate_wait ab
log "start (card $CUDA_VISIBLE_DEVICES)"; PAR=2
runp repo_d174ab scl300k_Hu2_97 $F_scl $HU --render_every 2
runp repo_d174ab scl300k_Hu4_97 $F_scl $HU --render_every 4
wait; log "done"
