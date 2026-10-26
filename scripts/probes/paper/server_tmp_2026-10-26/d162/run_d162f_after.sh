#!/bin/bash
# d162f_after.sh -- D162e / D162f readers on hyde06's CPU (detached, from a copy): (1) once job 22 is done, D161's
# reader on the u-off held arm Hu against D161's twins; (2) once job 23 is done, D161's reader on the 300k ripples (H vs
# T) and the studio side-by-side videos (d161_render.py pair: whole view and a raking close-up on the flank); (3) once
# job 24 is done, D160's reader on the 300k band (D vs T). The colour video is gpu1 job 25 (d160_studio.py).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; W=$B/tmp/d162; T1=$B/tmp/d161; A1=$G/d161/assets; R1=$G/d161/runs40
E=$G/d162e; F=$G/d162f; A=$F/assets
mkdir -p $E/after $F/after
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d162d:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
cd $B/repo_d162d
# (1) D162e
until grep -q "R done (d162e)" $E/status.txt 2>/dev/null; do sleep 60; done
echo "$(date '+%m-%d %H:%M:%S') [after] D162e start" >> $E/status.txt
for s in coin bunny; do
  if [ $s = coin ]; then sm=$A1/d161coin.obj; rl=$A1/d161L40coin.obj; else sm=$A1/d161smooth_bunny.obj; rl=$A1/d161L40_bunny.obj; fi
  cat > $E/after/scene_${s}_Hu.json << EOS
{"scene": "${s}40k_Hu", "N": 40000, "smooth": "$sm", "relief": "$rl",
 "runs": {"full97": "$E/runs/${s}40k_Hu_97", "twin97": "$R1/${s}40k_twin_97", "full99": "$E/runs/${s}40k_Hu_99", "twin99": "$R1/${s}40k_twin_99"}}
EOS
  ( nice -n 10 $PY $T1/d161_read.py $E/after/read_${s}_Hu.json $E/after/scene_${s}_Hu.json > $E/after/read_${s}_Hu.log 2>&1
    $PY $T1/d161_summary.py $E/after/summary_${s}_Hu.md $E/after/read_${s}_Hu.json > /dev/null 2>&1
    echo "$(date '+%m-%d %H:%M:%S') [after] D162e $s read" >> $E/status.txt ) &
done
wait
# (2) D162f ripples
until grep -q "R done (d162f ripple)" $F/status.txt 2>/dev/null; do sleep 60; done
echo "$(date '+%m-%d %H:%M:%S') [after] ripples start" >> $F/status.txt
cat > $F/after/scene_ripple.json << EOS
{"scene": "rip300k", "N": 300000, "smooth": "$A1/d161smooth_bunny.obj", "relief": "$A/d162fR300ripple_bunny.obj",
 "runs": {"full97": "$F/runs/rip300k_H_97", "twin97": "$F/runs/rip300k_T_97"}}
EOS
nice -n 10 $PY $T1/d161_read.py $F/after/read_ripple.json $F/after/scene_ripple.json > $F/after/read_ripple.log 2>&1
echo "$(date '+%m-%d %H:%M:%S') [after] ripples read exit $?" >> $F/status.txt
n=$($PY -c "import numpy as np; print(len(np.load('$F/runs/rip300k_H_97_frames12.npz')['raws']))")
ev=$(( n / 60 > 1 ? n / 60 : 1 ))
( nice -n 10 $PY $T1/d161_render.py pair $F/after/vid_ripple_whole $F/runs/rip300k_H_97_frames12.npz $F/runs/rip300k_T_97_frames12.npz \
    $A/d162fR300ripple_bunny.obj $A1/d161smooth_bunny.obj 300000 1.5708 0.1 768 80 1.0 - $ev > $F/after/vid_ripple_whole.log 2>&1
  echo "$(date '+%m-%d %H:%M:%S') [after] ripples whole video exit $?" >> $F/status.txt ) &
( nice -n 10 $PY $T1/d161_render.py pair $F/after/vid_ripple_close $F/runs/rip300k_H_97_frames12.npz $F/runs/rip300k_T_97_frames12.npz \
    $A/d162fR300ripple_bunny.obj $A1/d161smooth_bunny.obj 300000 1.5708 0.1 768 80 2.5 1.36,-0.25,-1.16 $ev > $F/after/vid_ripple_close.log 2>&1
  echo "$(date '+%m-%d %H:%M:%S') [after] ripples close-up video exit $?" >> $F/status.txt ) &
wait
# (3) D162f band
until grep -q "R done (d162f band)" $F/status.txt 2>/dev/null; do sleep 60; done
J=$F/after/jobs_colour.txt
echo "band300k_T_97 band $F/runs/band300k_T_97_frames12.npz" > $J
echo "band300k_D_97 band $F/runs/band300k_D_97_frames12.npz twin=band300k_T_97 timeline" >> $J
nice -n 10 $PY $B/tmp/d160/d160_read.py $F/after/read_colour.jsonl $J > $F/after/read_colour.log 2>&1
echo "$(date '+%m-%d %H:%M:%S') [after] band read exit $?" >> $F/status.txt
