#!/bin/bash
# d162_auto29.sh -- job 29's auto-reader (CPU, detached): D161's reader on the E x3 / x7 runs (labels full3, full7) with
# D161's seed-97 twin, then output/gpu/d162/auto/29/summary.md (d162_auto.py stiff).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; W=$B/tmp/d162; T1=$B/tmp/d161; A1=$G/d161/assets; O=$G/d162s; R=$O/runs
mkdir -p $O/after $G/d162/auto/29
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 PYTHONPATH=$B/repo_d162d:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
until grep -q "R done (d162s)" $O/status.txt 2>/dev/null; do sleep 120; done
cd $B/repo_d162d
for s in coin bunny; do
  if [ $s = coin ]; then sm=$A1/d161coin.obj; rl=$A1/d161L40coin.obj; else sm=$A1/d161smooth_bunny.obj; rl=$A1/d161L40_bunny.obj; fi
  echo "{\"scene\": \"${s}40k_E\", \"N\": 40000, \"smooth\": \"$sm\", \"relief\": \"$rl\", \"runs\": {\"full3\": \"$R/${s}40k_E3_97\", \"full7\": \"$R/${s}40k_E7_97\", \"twin97\": \"$G/d161/runs40/${s}40k_twin_97\"}}" > $O/after/scene_$s.json
  nice -n 10 $PY $T1/d161_read.py $O/after/read_$s.json $O/after/scene_$s.json > $O/after/read_$s.log 2>&1 &
done
wait
$PY $W/d162_auto.py stiff $G/d162/auto/29/summary.md > $O/after/auto.log 2>&1
