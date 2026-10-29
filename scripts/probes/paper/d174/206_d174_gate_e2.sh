#!/bin/bash
# 206_d174_gate_e2.sh -- queue job (agent D174): the V3 gate of exact batch 2 on repo_d174 (batch 1 + the shared-memory
# Sinkhorn log-sum-exp + the u gate's map on the layer's rows + the A/B switches off): G1-G4 bit-SAME (tmp/d174/d174_gate.sh).
J=206_gate_e2
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"
NOLOCK=1 bash $W/d174_gate.sh e2 repo_d174
log "done: $(tail -1 $O/gates/e2.md)"
