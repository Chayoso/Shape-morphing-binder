#!/bin/bash
# 197_d173_probe_b.sh -- queue job (agent D173): the real scales' representability probe (tmp/d173/d173_probe.py, no run),
# configurations tmp/d173/probe_b.json -> output/gpu/d173/probe/probe_b.jsonl.
J=197_probe_b; TITLE="D173 probe probe_b"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
mkdir -p $O/probe; log "start"
( envr $REPO; export D157_THREADS=16 OMP_NUM_THREADS=16; $PY $W/d173_probe.py $O/probe/probe_b.jsonl $W/probe_b.json > $O/probe/probe_b.log 2>&1 )
log "done (exit $?)"
