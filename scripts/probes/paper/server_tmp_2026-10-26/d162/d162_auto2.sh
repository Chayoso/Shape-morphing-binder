#!/bin/bash
# d162_auto2.sh -- auto-readers for the reruns (jobs 50-55, CPU, detached): the settle verdict (and D162i, job 56, on A),
# the stiffness summary, the 300k ripples / band reads and videos, the 500k plaque video, the run outcomes.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; W=$B/tmp/d162; T1=$B/tmp/d161; A1=$G/d161/assets; AU=$G/d162/auto; Q=$B/tmp/gpu1_queue
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d162d:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python; AP=$W/d162_auto.py
waitfor() { until grep -q "$2" $1 2>/dev/null; do sleep 120; done; }
cd $B/repo_d162d
for j in 23_ripple300 24_band300; do echo "# $j: died of CUDA OOM on the shared gpu1 (2026-10-22 23:20); rerun as job 55 (ripples with H+C), summary below when read" > $AU/$j/summary.md; done
echo "# 27 / 29: the settled runs crashed after settling (settle rows lacked Jmin_traj; fixed d162 wip); rerun as jobs 50 / 51" > $AU/27_settle/summary.md
cp $AU/27_settle/summary.md $AU/29/summary.md
( waitfor $G/d162h/status.txt "R done (d162h2)"
  D=$G/d162h/after; R=$G/d162h/runs; R0=$G/d160/runs40; mkdir -p $D
  for s in coin bunny; do
    if [ $s = coin ]; then sm=$A1/d161coin.obj; rl=$A1/d161L40coin.obj; else sm=$A1/d161smooth_bunny.obj; rl=$A1/d161L40_bunny.obj; fi
    echo "{\"scene\": \"${s}40k_Hs\", \"N\": 40000, \"smooth\": \"$sm\", \"relief\": \"$rl\", \"runs\": {\"full97\": \"$R/${s}40k_Hs_97\", \"twin97\": \"$G/d161/runs40/${s}40k_twin_97\", \"full99\": \"$R/${s}40k_Hs_99\", \"twin99\": \"$G/d161/runs40/${s}40k_twin_99\"}}" > $D/scene_${s}_Hs.json
    nice -n 10 $PY $T1/d161_read.py $D/read_${s}_Hs.json $D/scene_${s}_Hs.json > $D/read_${s}_Hs.log 2>&1 &
  done
  J=$D/jobs_colour.txt; : > $J
  for s in 97 99; do echo "bunny40k_band_P$s band $R0/bunny40k_band_P${s}_frames12.npz" >> $J
    echo "bunny40k_band_Hs_$s band $R/bunny40k_band_Hs_${s}_frames12.npz twin=bunny40k_band_P$s timeline" >> $J; done
  nice -n 10 $PY $B/tmp/d160/d160_read.py $D/read_colour.jsonl $J > $D/read_colour.log 2>&1 &
  wait
  v=$($PY $AP settle $AU/27_settle/summary.md 2>$D/settle.err | grep -o "VERDICT: [A-Za-z]*" | cut -d" " -f2)
  echo "$(date '+%m-%d %H:%M:%S') [auto2] D162h verdict $v" >> $G/d162h/status.txt
  if [ "$v" = A ]; then sed 's/^# 28_d162_d162i.sh/# 56_d162_d162i.sh/' $W/28_d162_d162i.sh.ready > $Q/56_d162_d162i.sh.tmp && mv $Q/56_d162_d162i.sh.tmp $Q/56_d162_d162i.sh
    echo "$(date '+%m-%d %H:%M:%S') [auto2] D162i queued (job 56)" >> $G/d162h/status.txt
    waitfor $G/d162i/status.txt "R done (d162i)"; { echo "# D162i"; grep -h "R .* exit" $G/d162i/status.txt; } > $AU/28_d162i/summary.md; fi ) &
( waitfor $G/d162s/status.txt "R done (d162s2)"; O=$G/d162s; R=$O/runs
  for s in coin bunny; do
    if [ $s = coin ]; then sm=$A1/d161coin.obj; rl=$A1/d161L40coin.obj; else sm=$A1/d161smooth_bunny.obj; rl=$A1/d161L40_bunny.obj; fi
    echo "{\"scene\": \"${s}40k_E\", \"N\": 40000, \"smooth\": \"$sm\", \"relief\": \"$rl\", \"runs\": {\"full3\": \"$R/${s}40k_E3_97\", \"full7\": \"$R/${s}40k_E7_97\", \"twin97\": \"$G/d161/runs40/${s}40k_twin_97\"}}" > $O/after/scene_$s.json
    nice -n 10 $PY $T1/d161_read.py $O/after/read_$s.json $O/after/scene_$s.json > $O/after/read_$s.log 2>&1 &
  done; wait
  $PY $AP stiff $AU/29/summary.md > $O/after/auto.log 2>&1 ) &
( F=$G/d162f; A=$F/assets; waitfor $F/status.txt "R done (300k ripple band)"
  echo "{\"scene\": \"rip300k\", \"N\": 300000, \"smooth\": \"$A1/d161smooth_bunny.obj\", \"relief\": \"$A/d162fR300ripple_bunny.obj\", \"runs\": {\"full97\": \"$F/runs/rip300k_HC_97\", \"twin97\": \"$F/runs/rip300k_T_97\"}}" > $F/after/scene_ripple.json
  nice -n 10 $PY $T1/d161_read.py $F/after/read_ripple.json $F/after/scene_ripple.json > $F/after/read_ripple.log 2>&1 &
  printf "band300k_T_97 band $F/runs/band300k_T_97_frames12.npz\nband300k_D_97 band $F/runs/band300k_D_97_frames12.npz twin=band300k_T_97 timeline\n" > $F/after/jobs_colour.txt
  nice -n 10 $PY $B/tmp/d160/d160_read.py $F/after/read_colour.jsonl $F/after/jobs_colour.txt > $F/after/read_colour.log 2>&1 &
  for v in "whole 1.0 -" "close 2.5 1.36,-0.25,-1.16"; do set -- $v
    nice -n 10 $PY $T1/d161_render.py pair $F/after/vid_ripple_$1 $F/runs/rip300k_HC_97_frames12.npz $F/runs/rip300k_T_97_frames12.npz \
      $A/d162fR300ripple_bunny.obj $A1/d161smooth_bunny.obj 300000 1.5708 0.1 768 80 $2 $3 3 > $F/after/vid_ripple_$1.log 2>&1 &
  done; wait
  $PY $AP ripple $AU/23_ripple300/summary.md $F/after/read_ripple.json $F/runs rip300k_HC_97 rip300k_T_97 $F/after/vid_ripple_whole/pair.mp4 $F/after/vid_ripple_close/pair.mp4 > /dev/null 2>&1
  $PY $AP band $AU/24_band300/summary.md $F/after/read_colour.jsonl $F/runs "(band video: d160_studio.py, not yet run)" > /dev/null 2>&1 ) &
( K=$G/d162k; waitfor $K/status.txt "R done (plq 500k)"
  bash $B/tmp/d163/d163_video.sh plq500k_HC_T $G/d163/assets/d163plaque.obj 500000 $G/d163/assets/d163R300plaque.obj $K/runs/plq500k_HC_97 $K/runs/plq500k_T_97 "dF_c + layer u + render loss (full, H+C)" "dF_c + layer u" > $K/video_plq.log 2>&1 ) &
( K=$G/d162k; waitfor $K/status.txt "R done (band 500k)"; waitfor $K/status.txt "R done (rip 500k)"
  mkdir -p $AU/82; { echo "# 500k deliverables (H+C for plaque / ripples, D160d for band; vs the twin); plaque video under output/gpu/d163/video/plq500k_HC_T"; grep -h "R .* exit\|out of memory" $K/status.txt | tail -12; } > $AU/82/summary.md ) &
wait
