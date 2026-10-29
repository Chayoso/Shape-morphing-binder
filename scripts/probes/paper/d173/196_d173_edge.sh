#!/bin/bash
# 196_d173_edge.sh -- queue job (agent D173): the colour edge widths (tmp/d173/d173_edge.py on tmp/d173/edge_jobs.txt).
J=196_edge; TITLE="D173 colour edge"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
mkdir -p $O/edge; log "start"
( envr $REPO; $PY $W/d173_edge.py $O/edge/edge.jsonl $W/edge_jobs.txt > $O/edge/edge.log 2>&1 )
log "done (exit $?)"
