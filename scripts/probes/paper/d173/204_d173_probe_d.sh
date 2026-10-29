#!/bin/bash
# 204_d173_probe_d.sh -- queue job (agent D173): 300k interior with the outer band densified so the SURFACE spacing falls by sqrt2 / 2 / 2.83 (band volume density x2.83 / x8 / x22.6), the field radius 2 and 3 surface spacings, the pixel ~0.8 surface spacing: the real scales' representability probe (tmp/d173/d173_probe.py, no run),
# configurations tmp/d173/probe_d.json -> output/gpu/d173/probe/probe_d.jsonl.
J=204_probe_d; TITLE="D173 probe probe_d"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
mkdir -p $O/probe; log "start"
( envr $REPO; export D157_THREADS=16 OMP_NUM_THREADS=16; $PY $W/d173_probe.py $O/probe/probe_d.jsonl $W/probe_d.json > $O/probe/probe_d.log 2>&1 )
log "done (exit $?)"
