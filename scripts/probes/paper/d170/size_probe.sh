#!/bin/bash
# size_probe.sh -- D170 (CPU, detached): the dragon's scales and the bunny's wrinkles against the pitch at 40k and 300k, with no
# run: d170_detail.py's context (rms height, autocorrelation length, variance above one pitch) and its references (the
# detailed mesh's own N-particle samples read the same way: how much of the detail a perfect placement of N particles carries).
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d170; O=$B/output/gpu/d170; P=$O/size_probe; mkdir -p $P
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=12 D157_THREADS=12 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d166
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
cd $B/repo_d166
for n in ${NS:-40000 300000}; do
  for m in ${MS:-dragon bunny}; do
    if [ $m = dragon ]; then sm=$O/assets/d170smooth_dragon.obj; else sm=$B/output/gpu/d161/assets/d161smooth_bunny.obj; fi
    echo "{\"scene\": \"size_${m}_${n}\", \"N\": $n, \"tgt\": \"$sm\", \"smooth\": \"$sm\", \"detail\": \"$B/repo_d166/assets/$m.obj\", \"runs\": {}, \"seeds\": [97]}" > $P/scene_${m}_$n.json
    ( nice -n 10 $PY $W/d170_detail.py $P/size_${m}_$n.json $P/scene_${m}_$n.json > $P/size_${m}_$n.log 2>&1; echo "rc $?" >> $P/size_${m}_$n.log ) &
  done
done
wait
echo "$(date '+%m-%d %H:%M:%S') [size probe] done" >> $O/status.txt
