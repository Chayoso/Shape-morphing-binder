#!/bin/bash
# D165 job a (gpu1 queue, 2026-10-22): (1) D163's relief mesh sampled at 1.5M in the runs' frame (the decal's pictures);
# (2) the post-hoc decal stills: scripts/render_deliverable.py +fit=1 (D115's display-only layer) on the twin's last 8
# kept frames, fitted to the relief's pictures, and the base display (+fit=0) of the twin's and H's ends, front word,
# two zooms; (3) the stamp on the kept frames' end positions (F = Fp = I, the end shape as the rest shape) of H and the
# twin: the code path's first run and the rest-shape-from-the-end variant.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d165; W=$B/tmp/d165; D3=$B/output/gpu/d163
mkdir -p $O/decal $O/stamp $W/tmpdir
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%F %T') [a] $*" >> $O/status.txt; }
log "start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
T=$D3/runs300/plq300k_T_97_frames12.npz; H=$D3/runs300/plq300k_H_97_frames12.npz
( export PHYSMORPH_RUN_REPO=$B/repo_d163 PHYSMORPH_OUT=$B/output/gpu; source $B/repo_d163/scripts/ops/hyde06_env.sh > /dev/null 2>&1
  export TMPDIR=$W/tmpdir; cd $B/repo_d163
  [ -f $O/relief_dense15.npz ] || $PY $W/d165_dense.py $D3/assets/d163plaque.obj $D3/assets/d163R300plaque.obj 1500000 $O/relief_dense15.npz > $O/dense.log 2>&1
  log "dense: $(tail -2 $O/dense.log | tr '\n' ' ' | cut -c1-250)"
  KT=$($PY -c "import numpy as np; print(len(np.load('$T')['raws']))"); KH=$($PY -c "import numpy as np; print(len(np.load('$H')['raws']))")
  log "kept frames: twin $KT, H $KH"
  for d in 0.35 0.55; do
    $PY scripts/render_deliverable.py $T $O/relief_dense15.npz $O/decal/T_decal_d$d +field=zbs +az=0 +el=8 +dist=$d \
      +frames=$((KT-8))-$((KT-1)) +fit=1 +steps=15 +carry=h +decay=1 > $O/decal/T_decal_d$d.log 2>&1
    log "decal twin d$d exit $? ($(tail -1 $O/decal/T_decal_d$d/rows.txt 2>/dev/null | cut -c1-200))"
    $PY scripts/render_deliverable.py $T $O/relief_dense15.npz $O/decal/T_base_d$d +field=zbs +az=0 +el=8 +dist=$d \
      +frames=$((KT-1))-$((KT-1)) +fit=0 > $O/decal/T_base_d$d.log 2>&1
    log "base twin d$d exit $?"
    $PY scripts/render_deliverable.py $H $O/relief_dense15.npz $O/decal/H_base_d$d +field=zbs +az=0 +el=8 +dist=$d \
      +frames=$((KH-1))-$((KH-1)) +fit=0 > $O/decal/H_base_d$d.log 2>&1
    log "base H d$d exit $?"
  done )
( export PHYSMORPH_RUN_REPO=$B/repo_d165 PHYSMORPH_OUT=$B/output/gpu; source $B/repo_d165/scripts/ops/hyde06_env.sh > /dev/null 2>&1
  export TMPDIR=$W/tmpdir; cd $B/repo_d165
  for arm in H T; do
    $PY scripts/probes/d165/stamp.py $D3/runs300/plq300k_${arm}_97_frames12.npz $O/stamp/k_$arm > $O/stamp/k_$arm.log 2>&1
    log "stamp k_$arm exit $?: $(tail -1 $O/stamp/k_$arm.log | cut -c1-300)"
  done )
log "end"
