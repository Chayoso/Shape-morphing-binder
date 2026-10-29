#!/bin/bash
# 203_d173_probe_c.sh -- queue job (agent D173): where the real scales first pass (above 500k, and combined levers): the real scales' representability probe (tmp/d173/d173_probe.py, no run),
# configurations tmp/d173/probe_c.json -> output/gpu/d173/probe/probe_c.jsonl.
J=203_probe_c; TITLE="D173 probe probe_c"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
mkdir -p $O/probe; log "start"
( envr $REPO; export D157_THREADS=16 OMP_NUM_THREADS=16; $PY $W/d173_probe.py $O/probe/probe_c.jsonl $W/probe_c.json > $O/probe/probe_c.log 2>&1 )
log "done (exit $?)"
