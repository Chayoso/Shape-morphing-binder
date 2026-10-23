#!/bin/bash
# d162g_after.sh -- D162g's reader on hyde06's CPU (detached, from a copy): once job 26 is done, D161's reader and summary
# on H+C against D162's TC (the same placement physics), and against D161's twins for rest / shape; the window records
# (lambda, g_share, the end speed) per run.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; T1=$B/tmp/d161; A1=$G/d161/assets; O=$G/d162g; RT=$G/d162/runs40; R1=$G/d161/runs40
mkdir -p $O/after
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d162d:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
cd $B/repo_d162d
until grep -q "R done (d162g)" $O/status.txt 2>/dev/null; do sleep 60; done
echo "$(date '+%m-%d %H:%M:%S') [after] D162g start" >> $O/status.txt
for s in coin bunny; do
  if [ $s = coin ]; then sm=$A1/d161coin.obj; rl=$A1/d161L40coin.obj; else sm=$A1/d161smooth_bunny.obj; rl=$A1/d161L40_bunny.obj; fi
  for tw in TC D161; do
    if [ $tw = TC ]; then t97=$RT/${s}40k_TC_97; t99=$RT/${s}40k_TC_99; else t97=$R1/${s}40k_twin_97; t99=$R1/${s}40k_twin_99; fi
    cat > $O/after/scene_${s}_HC_$tw.json << EOS
{"scene": "${s}40k_HC_vs_$tw", "N": 40000, "smooth": "$sm", "relief": "$rl",
 "runs": {"full97": "$O/runs/${s}40k_HC_97", "twin97": "$t97", "full99": "$O/runs/${s}40k_HC_99", "twin99": "$t99"}}
EOS
    ( nice -n 10 $PY $T1/d161_read.py $O/after/read_${s}_HC_$tw.json $O/after/scene_${s}_HC_$tw.json > $O/after/read_${s}_HC_$tw.log 2>&1
      $PY $T1/d161_summary.py $O/after/summary_${s}_HC_$tw.md $O/after/read_${s}_HC_$tw.json > /dev/null 2>&1
      echo "$(date '+%m-%d %H:%M:%S') [after] D162g $s vs $tw read" >> $O/status.txt ) &
  done
done
wait
echo "$(date '+%m-%d %H:%M:%S') [after] D162g done" >> $O/status.txt
