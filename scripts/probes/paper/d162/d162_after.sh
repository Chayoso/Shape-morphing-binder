#!/bin/bash
# d162_after.sh -- D162 on hyde06, CPU only: once run list A is done, read the relief and colour scenes; once B is done
# too, the gallery; the per-window records of every run. Detached; logs in output/gpu/d162/after/.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d162; W=$B/tmp/d162; R=$O/runs40; D=$O/after
A1=$G/d161/assets; R1=$G/d161/runs40; R0=$G/d160/runs40
mkdir -p $D
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d162:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') [after] $*" >> $O/status.txt; }
until grep -q "R done (d162_runs_A.txt)" $O/status.txt; do sleep 60; done
log "start (A done)"
cd $B/repo_d162
# relief: F against D161's twins, FC against its own twin TC
for s in coin bunny; do
  if [ $s = coin ]; then sm=$A1/d161coin.obj; rl=$A1/d161L40coin.obj; else sm=$A1/d161smooth_bunny.obj; rl=$A1/d161L40_bunny.obj; fi
  for arm in F FC; do
    if [ $arm = F ]; then t97=$R1/${s}40k_twin_97; t99=$R1/${s}40k_twin_99; else t97=$R/${s}40k_TC_97; t99=$R/${s}40k_TC_99; fi
    cat > $D/scene_${s}_$arm.json << EOS
{"scene": "${s}40k_$arm", "N": 40000, "smooth": "$sm", "relief": "$rl",
 "runs": {"full97": "$R/${s}40k_${arm}_97", "twin97": "$t97", "full99": "$R/${s}40k_${arm}_99", "twin99": "$t99"}}
EOS
    ( nice -n 10 $PY $B/tmp/d161/d161_read.py $D/read_${s}_$arm.json $D/scene_${s}_$arm.json > $D/read_${s}_$arm.log 2>&1
      log "read relief $s $arm exit $?"
      $PY $B/tmp/d161/d161_summary.py $D/summary_${s}_$arm.md $D/read_${s}_$arm.json > /dev/null 2>&1 ) &
  done
done
# colour: D160's reader on F, FC and TC (twins: D160's P for F, TC for FC)
J=$D/jobs_colour.txt; : > $J
for sc in band ears; do for s in 97 99; do
  echo "bunny40k_${sc}_P$s $sc $R0/bunny40k_${sc}_P${s}_frames12.npz" >> $J
  echo "bunny40k_${sc}_TC_$s $sc $R/bunny40k_${sc}_TC_${s}_frames12.npz" >> $J
  echo "bunny40k_${sc}_F_$s $sc $R/bunny40k_${sc}_F_${s}_frames12.npz twin=bunny40k_${sc}_P$s timeline" >> $J
  echo "bunny40k_${sc}_FC_$s $sc $R/bunny40k_${sc}_FC_${s}_frames12.npz twin=bunny40k_${sc}_TC_$s timeline" >> $J
done; done
( nice -n 10 $PY $B/tmp/d160/d160_read.py $D/read_colour.jsonl $J > $D/read_colour.log 2>&1; log "read colour exit $?"
  $PY $W/d162_colour.py $D/colour.md $G/d160/read_colour.jsonl $G/d160/read_colour_H.jsonl $D/read_colour.jsonl > /dev/null 2>&1 ) &
wait
cd $R && $PY $W/d162_windows.py $D/windows_scenes.md $(ls *40k_*_9?.json | grep -v "^gal_" | sed "s/.json//" | sed "s|^|$R/|") > /dev/null 2>&1
log "scenes read"
until grep -q "R done (d162_runs_B.txt)" $O/status.txt; do sleep 60; done
cd $B/repo_d162
nice -n 10 $PY $W/d162_gallery.py $D/gallery.md $D/gallery.json > $D/gallery.log 2>&1; log "gallery exit $?: $(tail -2 $D/gallery.log | tr '\n' ' ')"
cd $R && $PY $W/d162_windows.py $D/windows_gallery.md $(ls gal_*.json | sed "s/.json//" | sed "s|^|$R/|") > /dev/null 2>&1
log "done"
