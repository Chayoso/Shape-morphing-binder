#!/bin/bash
# 195_d174_gate_s.sh -- queue job (agent D174): the V3 gate of --stop_tol on repo_d174s (default unset = tol): G1-G4 bit-SAME.
J=gate_s
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
NOLOCK=1 bash $W/d174_gate.sh s repo_d174s
log "gate s: $(tail -1 $O/gates/s.md)"
