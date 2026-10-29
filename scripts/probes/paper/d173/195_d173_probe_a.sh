#!/bin/bash
# 195_d173_probe_a.sh -- queue job (agent D173): the real scales' representability probe (tmp/d173/d173_probe.py, no run),
# configurations tmp/d173/probe_a.json -> output/gpu/d173/probe/probe_a.jsonl.
J=195_probe_a; TITLE="D173 probe probe_a"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
mkdir -p $O/probe; log "start"
( envr $REPO; export D157_THREADS=16 OMP_NUM_THREADS=16; $PY $W/d173_probe.py $O/probe/probe_a.jsonl $W/probe_a.json > $O/probe/probe_a.log 2>&1 )
log "done (exit $?)"
