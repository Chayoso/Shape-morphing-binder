#!/bin/bash
# 205_d174_gate_t10.sh -- queue job (agent D174, stop A/B): the V3 gate on repo_d174t10 (tol 0.10): G1-G4 against the references
# (informational: the tolerance acts once a window's merit gain falls below it; SAME means the 4 gate windows never do).
J=gate_t10
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
NOLOCK=1 bash $W/d174_gate.sh t10 repo_d174t10
log "gate t10: $(tail -1 $O/gates/t10.md)"
