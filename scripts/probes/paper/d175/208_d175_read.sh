#!/bin/bash
# 208_d175_read.sh -- queue job (agent D175): the reader of job 208's runs on a card (the exterior operators are CUDA),
# -> output/gpu/d175/auto/208_kf40 and output/gpu/D175_SUMMARY.md. (Job 208's own reader stage stopped at an import order.)
J=208_kf40; TITLE="D175 keyframe E reader"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
log "reader start (card $CUDA_VISIBLE_DEVICES)"
( envr $REPO; $PY $W/d175_read.py $O/auto/$J $O/runs $A --summary $G/D175_SUMMARY.md > $O/auto/$J/read.log 2>&1 < /dev/null )
log "reader exit $? -> $(grep '^VERDICT' $O/auto/$J/summary.md 2>/dev/null)"
