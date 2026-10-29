#!/bin/bash
# 206_d174_gate_m1.sh -- queue job (agent D174, memory track): the V3 gate of memory batch m1 on repo_d174m (the driven steps'
# control buffers only, no kept velocity stack, the domain test frame by frame): G1-G4 bit-SAME; then the memory probes at
# 300k and 500k on repo_d174m (the same scene and flags as 206_d174_mem300 / 500 on repo_d174).
J=206_gate_m1
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"
NOLOCK=1 bash $W/d174_gate.sh m1 repo_d174m
log "gate: $(tail -1 $O/gates/m1.md)"
[ -e $O/gates/m1.ok ] || exit 1
for n in 300000 500000; do k=$((n/1000))
  ( envr repo_d174m; $PY $W/d174_mem.py $O/auto/mem_scl${k}k_m1.json --tgt $G/d170/assets/d170smooth_dragon.obj --render_tgt $A2/d172SE300_dragon.obj \
      --n $n --seed 97 $HU --animations 2 --settle_windows 1 --out $O/runs/M_scl${k}k_m1 > $O/auto/mem_scl${k}k_m1.log 2>&1 )
  rm -f $O/runs/M_scl${k}k_m1_frames12.npz
done
log "done"
