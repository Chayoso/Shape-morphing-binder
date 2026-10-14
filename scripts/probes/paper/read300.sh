#!/bin/bash
# read300.sh GPU DIR NAME MESH MODE [KEEP] -- D145's readings of one 300k run (NAME = MESH300k_ARM, kept frames in
# output/gpu/DIR/NAME_frames12.npz) against the independent 300k sample d90/MESH_ind300k.npz (MODE suffix f: the sample
# lifted onto the floor as the run's target was, tmp/d145/ind_shift.py).
#   GPU part (here, on the caller's card): the end frame kept as NAME_end1.npz; the yardstick (render_terms_probe,
#     t3end_NAME.log) and the display (surface_layer_probe, e3end_NAME.log) at the last kept frame; the arrangement at
#     the end (arrangement.py, a_NAME.log); momentum on every kept frame (mom.sh, m_NAME.log); MODE full300 also the
#     yardstick and the display on every kept frame (t3_NAME.log, e3_NAME.log) as runeval3.sh.
#   CPU part (read300_cpu.sh, started detached so that the card's lock can be released): apart particles over the first
#     35 % of the kept frames (d120_apart.py), the tail pulse (d116_pulse.py) for bunny and dragon, then the kept frames
#     removed unless KEEP = 1 (default 1).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp; G=$B/output/gpu
gpu=$1; dir=$2; name=$3; mesh=$4; mode=$5; keep=${6:-1}
D=$G/$dir; S=$D/status.txt; F=$D/${name}_frames12.npz; IND=$G/d90/${mesh}_ind300k.npz
source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null
case $mode in *f) $PY $T/d145/ind_shift.py $IND $F $D/ind_floor_$name.npz > $D/ind_shift_$name.log 2>&1; IND=$D/ind_floor_$name.npz; mode=${mode%f};; esac
case $mesh in bunny) box=1900,100,2800,700;; dragon) box=1900,60,2500,560;; *) box=0,0,3840,2160;; esac
cd $B/repo_r77
last=$($PY $T/lastraw.py $F)
$PY $T/d145/endframe.py $F $D/${name}_end1.npz
CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/settled/render_terms_probe.py $D/${name}_end1.npz ref=$IND > $D/t3end_$name.log 2>&1 < /dev/null
CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/settled/surface_layer_probe.py $F $D/e3end_$name $box measure ref=$IND target own $last > $D/e3end_$name.log 2>&1 < /dev/null
CUDA_VISIBLE_DEVICES=$gpu $PY $T/arrangement.py $F $last > $D/a_$name.log 2>&1 < /dev/null
$T/mom.sh $gpu $D/m_$name.log "$mesh ${name#${mesh}300k_}=$F"
echo "read gpu end $name $(date '+%m-%d %H:%M')" >> $S
if [ "$mode" = full300 ]; then
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/settled/render_terms_probe.py $F ref=$IND > $D/t3_$name.log 2>&1 < /dev/null
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/settled/surface_layer_probe.py $F $D/e3_$name $box measure ref=$IND target own \
    $($PY -c "import numpy as np; print(' '.join(str(int(r)) for r in np.load('$F')['raws']))") > $D/e3_$name.log 2>&1 < /dev/null
  echo "read gpu full $name $(date '+%m-%d %H:%M')" >> $S
fi
setsid nohup bash $T/d145/read300_cpu.sh $dir $name $mesh $keep > /dev/null 2>&1 < /dev/null &
