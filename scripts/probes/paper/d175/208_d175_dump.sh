#!/bin/bash
# 208_d175_dump.sh -- queue job (agent D175): the next measurement after 208's FAIL. S97's six segment windows rerun with
# --term_dump (each term's position gradient at every committed state, a record only), then dump_read.py (CPU): the share
# of the render's pull by depth below the body's hull and its alignment with the bulk stretch the keyframe asks for.
J=208_dump; TITLE="D175 forensic: the render's pull in the segment"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
D=$O/forensic/S97_dump; mkdir -p $D
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr $REPO; $PY scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 $HU --settle_windows 0 --animations 6 \
    --keyframe $A/kf_s97.npz --keyframe_time 1.0 --term_dump $D --out $O/forensic/S97_6w > $O/forensic/S97_6w.log 2>&1 < /dev/null
  export CUDA_VISIBLE_DEVICES=; $PY $W/dump_read.py $D $A/kf_s97.npz $O/auto/208_kf40/forensic_pull.md > $O/forensic/dump_read.log 2>&1 )
log "dump exit $? -> auto/208_kf40/forensic_pull.md"
