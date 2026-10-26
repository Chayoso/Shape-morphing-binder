#!/bin/bash
# d161_after_b.sh -- D161b on hyde06, CPU only: once "RB done", read the eight *_pb runs (d161_read.py) and summarise.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d161; W=$B/tmp/d161; A=$O/assets; R=$O/runs40; D=$O/after_b
mkdir -p $D
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d161:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') [after_b] $*" >> $O/status.txt; }
until grep -q "RB done\|B STOP" $O/status.txt; do sleep 60; done
grep -q "B STOP" $O/status.txt && { log "B stopped; nothing to read"; exit 1; }
log "start"
cd $B/repo_d161
for s in coin bunny; do
  if [ $s = coin ]; then sm=$A/d161coin.obj; rl=$A/d161L40coin.obj; else sm=$A/d161smooth_bunny.obj; rl=$A/d161L40_bunny.obj; fi
  cat > $D/scene_$s.json << EOS
{"scene": "${s}40k_pb", "N": 40000, "smooth": "$sm", "relief": "$rl",
 "runs": {"full97": "$R/${s}40k_full_97_pb", "twin97": "$R/${s}40k_twin_97_pb", "full99": "$R/${s}40k_full_99_pb", "twin99": "$R/${s}40k_twin_99_pb"}}
EOS
  ( nice -n 10 $PY $W/d161_read.py $D/read_$s.json $D/scene_$s.json > $D/read_$s.log 2>&1; log "read $s exit $?" ) &
done
wait
$PY $W/d161_summary.py $D/summary_coin.md $D/read_coin.json > /dev/null 2>&1
$PY $W/d161_summary.py $D/summary_bunny.md $D/read_bunny.json > /dev/null 2>&1
$PY $W/d161_course_fig.py $D/course.png $D/read_coin.json $D/read_bunny.json > /dev/null 2>&1
log "done"
