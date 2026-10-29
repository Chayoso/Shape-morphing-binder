#!/bin/bash
# 206_d174_det300.sh -- queue job (agent D174, memory track): memory batch m1 bit-SAME at 300k too (deterministic mode,
# tmp/rt/det_run.py): the stylized dragon scales at 300k, Hu's flags, 2 windows + 1 settle window, repo_d174 against
# repo_d174m (frames12 + JSON by det_cmp12.py). After gate m1.
J=206_det300
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
gate_wait m1
log "start (card $CUDA_VISIBLE_DEVICES)"
D=$O/gates; DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py $F_scl $HU --animations 2 --settle_windows 1"
( envr repo_d174; $PY $DR --out $D/d300_base > $D/d300_base.log 2>&1 < /dev/null )
( envr repo_d174m; $PY $DR --out $D/d300_m1 > $D/d300_m1.log 2>&1 < /dev/null
  $PY $B/tmp/d145/det_cmp12.py $D/d300_base $D/d300_m1 > $D/d300_cmp.txt 2>&1 )
log "300k deterministic m1 vs base: $(tail -1 $D/d300_cmp.txt)"
rm -f $D/d300_base_frames12.npz $D/d300_m1_frames12.npz
