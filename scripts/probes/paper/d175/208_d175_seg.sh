#!/bin/bash
# 208_d175_seg.sh -- queue job (agent D175): S97 rerun (40k, the 208 flags, the six segment windows) with every segment
# window's full start state saved and read at its start (seg_read.py: the one-step reading per render term, the pole
# pictures). Measurement only. -> output/gpu/d175/forensic/seg/{seg_read.md, rows.json, state_w*.npz}
J=208_seg; TITLE="D175 forensic: the one-step reading at segment windows 1-6"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
RD=$O/forensic/seg; mkdir -p $RD
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr $REPO; $PY $W/seg_read.py $RD $A/kf_s97.npz -- --tgt assets/bunny.obj --n 40000 --seed 97 $HU --settle_windows 0 \
    --animations 6 --keyframe $A/kf_s97.npz --keyframe_time 1.0 --out $O/forensic/S97_seg > $RD/seg.log 2>&1 < /dev/null )
log "seg exit $? -> $RD/seg_read.md"
