#!/bin/bash
# read300_full.sh GPU DIR NAME MESH KEEP -- D145: the every-frame readings of a 300k run (the yardstick t3_NAME.log and the
# display e3_NAME.log on every kept frame, as runeval3.sh), split from read300.sh so that the simulations go first while
# D145 holds one card (2026-10-14); then the kept frames removed unless KEEP = 1 (logged).
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp; G=$B/output/gpu
gpu=$1; dir=$2; name=$3; mesh=$4; keep=$5
D=$G/$dir; S=$D/status.txt; F=$D/${name}_frames12.npz; IND=$G/d90/${mesh}_ind300k.npz
case $mesh in bunny) box=1900,100,2800,700;; dragon) box=1900,60,2500,560;; *) box=0,0,3840,2160;; esac
unset PHYSMORPH_RUN_REPO
source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null
cd $B/repo_r77
CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/settled/render_terms_probe.py $F ref=$IND > $D/t3_$name.log 2>&1 < /dev/null
CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/settled/surface_layer_probe.py $F $D/e3_$name $box measure ref=$IND target own $($PY -c "import numpy as np; print(' '.join(str(int(r)) for r in np.load('$F')['raws']))") > $D/e3_$name.log 2>&1 < /dev/null
echo "read gpu full $name $(date '+%m-%d %H:%M')" >> $S
if [ "$keep" = 0 ]; then sz=$(du -m $F | cut -f1); rm -f $F; echo "$(date '+%Y-%m-%d %H:%M') $dir: removed $F ($sz MB; D145 run $name, every frame read)" >> $G/cleanup.log; fi
