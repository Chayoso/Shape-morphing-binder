#!/bin/bash
# 206_d174_gate_m2.sh -- queue job (agent D174, memory track): memory batch m2 on repo_d174m2 (the exterior lattice's ring
# of nodes a chunk of cells at a time): the multi-chunk path against one chunk on 300k / 500k bodies (check_m2.py), the V3
# gate G1-G4, then the memory probes at 300k / 500k.
J=206_gate_m2
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr repo_d174m2; $PY $W/check_m2.py > $O/gates/m2_chunks.txt 2>&1 )
log "chunks: $(tail -1 $O/gates/m2_chunks.txt)"
NOLOCK=1 bash $W/d174_gate.sh m2 repo_d174m2
log "gate: $(tail -1 $O/gates/m2.md)"
[ -e $O/gates/m2.ok ] || exit 1
for n in 300000 500000; do k=$((n/1000))
  ( envr repo_d174m2; $PY $W/d174_mem.py $O/auto/mem_scl${k}k_m2.json --tgt $G/d170/assets/d170smooth_dragon.obj --render_tgt $A2/d172SE300_dragon.obj \
      --n $n --seed 97 $HU --animations 2 --settle_windows 1 --out $O/runs/M_scl${k}k_m2 > $O/auto/mem_scl${k}k_m2.log 2>&1 )
  rm -f $O/runs/M_scl${k}k_m2_frames12.npz
done
log "done"
