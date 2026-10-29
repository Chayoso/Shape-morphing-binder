#!/bin/bash
# 206_d174_mem300.sh -- queue job (agent D174, memory track): where the GPU memory of a 300k Hu run goes (tmp/d174/d174_mem.py: phases
# of two windows and one settle window, every device buffer by owner), the stylized dragon scales scene (D172's 300k arguments
# at N = 300000), alone on this card.
J=mem300
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr repo_d174; $PY $W/d174_mem.py $O/auto/mem_scl300k.json --tgt $G/d170/assets/d170smooth_dragon.obj --render_tgt $A2/d172SE300_dragon.obj     --n 300000 --seed 97 $HU --animations 2 --settle_windows 1 --out $O/runs/M_scl300k > $O/auto/mem_scl300k.log 2>&1 )
rm -f $O/runs/M_scl300k_frames12.npz
log "done: $(grep -c "\[mem\]" $O/auto/mem_scl300k.log) phases recorded"
