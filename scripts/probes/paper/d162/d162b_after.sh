#!/bin/bash
# d162b_after.sh -- D162b on hyde06, CPU only, detached (from a copy): once d162b_runs.txt is done, read arm S as D162 read F:
# relief (D161's reader, S against D161's twins), colour (D160's reader, S against D160's P; the table with L, H, D160d, F,
# FC beside), the colour scenes' C2 / C3 (d162_c23.py), the window reading (d162b_read.py: switches, the rest after the
# excess, the shape reading's cost) and the per-window table (d162_windows.py). Logs in output/gpu/d162b/after/.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d162b; W=$B/tmp/d162; R=$O/runs40; D=$O/after
A1=$G/d161/assets; R1=$G/d161/runs40; R0=$G/d160/runs40
mkdir -p $D
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d162b:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') [after] $*" >> $O/status.txt; }
until grep -q "R done (d162b_runs.txt)" $O/status.txt 2>/dev/null; do sleep 60; done
log "start (S runs done)"
cd $B/repo_d162b
for s in coin bunny; do
  if [ $s = coin ]; then sm=$A1/d161coin.obj; rl=$A1/d161L40coin.obj; else sm=$A1/d161smooth_bunny.obj; rl=$A1/d161L40_bunny.obj; fi
  cat > $D/scene_${s}_S.json << EOS
{"scene": "${s}40k_S", "N": 40000, "smooth": "$sm", "relief": "$rl",
 "runs": {"full97": "$R/${s}40k_S_97", "twin97": "$R1/${s}40k_twin_97", "full99": "$R/${s}40k_S_99", "twin99": "$R1/${s}40k_twin_99"}}
EOS
  ( nice -n 10 $PY $B/tmp/d161/d161_read.py $D/read_${s}_S.json $D/scene_${s}_S.json > $D/read_${s}_S.log 2>&1
    log "read relief $s S exit $?"
    $PY $B/tmp/d161/d161_summary.py $D/summary_${s}_S.md $D/read_${s}_S.json > /dev/null 2>&1 ) &
done
J=$D/jobs_colour.txt; : > $J
for sc in band ears; do for s in 97 99; do
  echo "bunny40k_${sc}_P$s $sc $R0/bunny40k_${sc}_P${s}_frames12.npz" >> $J
  echo "bunny40k_${sc}_S_$s $sc $R/bunny40k_${sc}_S_${s}_frames12.npz twin=bunny40k_${sc}_P$s timeline" >> $J
done; done
( nice -n 10 $PY $B/tmp/d160/d160_read.py $D/read_colour.jsonl $J > $D/read_colour.log 2>&1; log "read colour exit $?"
  $PY $W/d162_colour.py $D/colour.md $G/d160/read_colour.jsonl $G/d160/read_colour_H.jsonl $G/d160/read_colour_D.jsonl \
      $G/d162/after/read_colour.jsonl $D/read_colour.jsonl > /dev/null 2>&1 ) &
wait
$PY $W/d162_c23.py $D/c23_colour.md > /dev/null 2>&1
$PY $W/d162b_read.py $D/windows_d162b.md > $D/windows_d162b.log 2>&1
cd $R && $PY $W/d162_windows.py $D/windows_scenes.md $(ls *40k_*S_9?.json | sed "s/.json//" | sed "s|^|$R/|") > /dev/null 2>&1
log "scenes read"
