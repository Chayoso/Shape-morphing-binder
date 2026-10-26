#!/bin/bash
# d162_auto.sh -- D162's auto-readers on hyde06's CPU (detached, from a copy): each part waits for its job's reads and writes
# output/gpu/d162/auto/<job>/summary.md (d162_auto.py). The settle part (job 27) also runs D161's / D160's readers on its
# runs and, if its verdict is A (premature stop), moves the ready D162i job (28) into the gpu1 queue.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; W=$B/tmp/d162; T1=$B/tmp/d161; A1=$G/d161/assets; AU=$G/d162/auto
mkdir -p $AU/22_uoff $AU/23_ripple300 $AU/24_band300 $AU/26_heldcell $AU/27_settle $AU/28_d162i $G/d162h/after
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d162d:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
AP=$W/d162_auto.py
waitfor() { until grep -q "$2" $1 2>/dev/null; do sleep 120; done; }
cd $B/repo_d162d
( waitfor $G/d162e/status.txt "D162e bunny read"; waitfor $G/d162e/status.txt "D162e coin read"
  $PY $AP relief $AU/22_uoff/summary.md "D162e: the held arm with u off (dF_c alone); D161c (u on) 0.47-0.53 coin, 0.54-0.58 bunny" \
      $G/d162e/after/summary_coin_Hu.md $G/d162e/after/summary_bunny_Hu.md ) &
( waitfor $G/d162g/status.txt "D162g done"
  $PY $AP relief $AU/26_heldcell/summary.md "D162g: held + placement at the cell (H+C) vs TC, then vs D161's twins" \
      $G/d162g/after/summary_coin_HC_TC.md $G/d162g/after/summary_bunny_HC_TC.md $G/d162g/after/summary_coin_HC_D161.md \
      $G/d162g/after/summary_bunny_HC_D161.md ) &
( waitfor $G/d162f/status.txt "ripples close-up video exit"; waitfor $G/d162f/status.txt "ripples whole video exit"
  $PY $AP ripple $AU/23_ripple300/summary.md $G/d162f/after/read_ripple.json $G/d162f/runs \
      $G/d162f/after/vid_ripple_whole/pair.mp4 $G/d162f/after/vid_ripple_close/pair.mp4 ) &
( waitfor $G/d162f/status.txt "band read exit"; waitfor $G/d162f/status.txt "job 25: band video exit"
  $PY $AP band $AU/24_band300/summary.md $G/d162f/after/read_colour.jsonl $G/d162f/runs $G/d162f/after/vid_band300k/side_by_side.mp4 ) &
( waitfor $G/d162h/status.txt "R done (d162h)"
  D=$G/d162h/after; R=$G/d162h/runs; R1=$G/d161/runs40; R0=$G/d160/runs40
  for s in coin bunny; do
    if [ $s = coin ]; then sm=$A1/d161coin.obj; rl=$A1/d161L40coin.obj; else sm=$A1/d161smooth_bunny.obj; rl=$A1/d161L40_bunny.obj; fi
    echo "{\"scene\": \"${s}40k_Hs\", \"N\": 40000, \"smooth\": \"$sm\", \"relief\": \"$rl\", \"runs\": {\"full97\": \"$R/${s}40k_Hs_97\", \"twin97\": \"$R1/${s}40k_twin_97\", \"full99\": \"$R/${s}40k_Hs_99\", \"twin99\": \"$R1/${s}40k_twin_99\"}}" > $D/scene_${s}_Hs.json
    nice -n 10 $PY $T1/d161_read.py $D/read_${s}_Hs.json $D/scene_${s}_Hs.json > $D/read_${s}_Hs.log 2>&1 &
  done
  J=$D/jobs_colour.txt; : > $J
  for s in 97 99; do echo "bunny40k_band_P$s band $R0/bunny40k_band_P${s}_frames12.npz" >> $J
    echo "bunny40k_band_Hs_$s band $R/bunny40k_band_Hs_${s}_frames12.npz twin=bunny40k_band_P$s timeline" >> $J; done
  nice -n 10 $PY $B/tmp/d160/d160_read.py $D/read_colour.jsonl $J > $D/read_colour.log 2>&1 &
  wait
  v=$($PY $AP settle $AU/27_settle/summary.md | grep -o "VERDICT: [A-Za-z]*" | cut -d" " -f2)
  echo "$(date '+%m-%d %H:%M:%S') [auto] D162h verdict $v" >> $G/d162h/status.txt
  if [ "$v" = A ]; then
    cp $W/28_d162_d162i.sh.ready $B/tmp/gpu1_queue/28_d162_d162i.sh.tmp && mv $B/tmp/gpu1_queue/28_d162_d162i.sh.tmp $B/tmp/gpu1_queue/28_d162_d162i.sh
    echo "$(date '+%m-%d %H:%M:%S') [auto] D162i queued (job 28)" >> $G/d162h/status.txt
    waitfor $G/d162i/status.txt "R done (d162i)"
    { echo "# D162i: held + settled end (ears, gallery spot check)"; echo; grep -h "R .* exit" $G/d162i/status.txt; } > $AU/28_d162i/summary.md
  fi ) &
wait
