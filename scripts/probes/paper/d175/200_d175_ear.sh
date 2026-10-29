#!/bin/bash
# 200_d175_ear.sh -- queue job (agent D175): the forensic of W97's ear-tip piece (D175d's P3 miss), no definition change:
# ear_read.py on a card -> output/gpu/d175/auto/200_ear/ear.md
J=200_ear; TITLE="D175 forensic: W97's ear-tip piece"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
RO=$O/auto/$J; mkdir -p $RO
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr $REPO; $PY $W/ear_read.py $RO/ear.md $O/runs > $RO/ear.log 2>&1 < /dev/null )
log "ear exit $?"
