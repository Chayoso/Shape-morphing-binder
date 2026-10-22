#!/bin/bash
# d161_after.sh -- D161 on hyde06, CPU only (no card): once d161_chain.sh R has logged "R done", read the eight 40k runs
# (d161_read.py, one process per scene) and render the side-by-side videos (d161_render.py pair: whole view and close-up,
# per scene and seed). Detached; logs in output/gpu/d161/after/.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d161; W=$B/tmp/d161; A=$O/assets; R=$O/runs40; D=$O/after
mkdir -p $D
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6
export PYTHONPATH=$B/repo_d161:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') [after] $*" >> $O/status.txt; }
until grep -q "R done" $O/status.txt; do sleep 60; done
log "start"
cd $B/repo_d161
for s in coin bunny; do
  if [ $s = coin ]; then sm=$A/d161coin.obj; rl=$A/d161L40coin.obj; else sm=$A/d161smooth_bunny.obj; rl=$A/d161L40_bunny.obj; fi
  cat > $D/scene_$s.json << EOF
{"scene": "${s}40k", "N": 40000, "smooth": "$sm", "relief": "$rl",
 "runs": {"full97": "$R/${s}40k_full_97", "twin97": "$R/${s}40k_twin_97", "full99": "$R/${s}40k_full_99", "twin99": "$R/${s}40k_twin_99"}}
EOF
  ( nice -n 10 $PY $W/d161_read.py $D/read_$s.json $D/scene_$s.json > $D/read_$s.log 2>&1; log "read $s exit $?" ) &
done
wait
# videos: coin from the front-left, the bunny's flank; whole view and a close-up on the pattern
for seed in 97 99; do
  ( nice -n 10 $PY $W/d161_render.py pair $D/vid_coin_$seed $R/coin40k_full_${seed}_frames12.npz $R/coin40k_twin_${seed}_frames12.npz \
      $A/d161L40coin.obj $A/d161coin.obj 40000 0.35 0.2 512 72 1.0 - > $D/vid_coin_$seed.log 2>&1; log "video coin $seed exit $?" ) &
  ( nice -n 10 $PY $W/d161_render.py pair $D/vid_bunny_$seed $R/bunny40k_full_${seed}_frames12.npz $R/bunny40k_twin_${seed}_frames12.npz \
      $A/d161L40_bunny.obj $A/d161smooth_bunny.obj 40000 1.5708 0.1 512 80 2.5 1.36,-0.25,-1.16 > $D/vid_bunny_$seed.log 2>&1; log "video bunny $seed exit $?" ) &
  wait
done
log "done"
