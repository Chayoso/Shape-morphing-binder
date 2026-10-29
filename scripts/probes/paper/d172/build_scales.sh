#!/bin/bash
# build_scales.sh -- D172 (D), CPU, detached: the stylized scale reliefs on the smoothed dragon (d172_mesh.py) at 40k and
# 300k, two sizes (A: spacing 4 pitches, height 0.6 pitch; B: 6 pitches, 0.8 pitch), then the representability probe on
# each (d172_detail.py, no run: the relief's own random N-samples read against the smooth canvas inside the pattern).
# The gate the coordinator set: a perfect 300k sample must read corr >= 0.25 (the bunny wrinkles' 0.27).
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d172; O=$B/output/gpu/d172; A=$O/assets; mkdir -p $A $O/size_probe
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 D157_THREADS=12 PHYSMORPH_CACHE=$B/cache/samples MPLCONFIGDIR=$B/cache/matplotlib
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
SM=$B/output/gpu/d170/assets/d170smooth_dragon.obj
log() { echo "$(date '+%m-%d %H:%M:%S') [scales] $*" >> $O/status.txt; }
cd $B/repo_d172
for v in "A 4 0.6" "B 6 0.8"; do set -- $v
  for n in 40000 300000; do
    t=${1}$((n / 1000)); o=$A/d172S${t}_dragon.obj
    echo "{\"base\": \"$SM\", \"n\": $n, \"scale\": $2, \"amp\": $3, \"edge\": 1.0, \"height\": $2, \"map\": \"flank\"}" > $A/S$t.json
    ( s=$(date +%s); nice -n 10 $PY $W/d172_mesh.py $o $A/S$t.json > $o.log 2>&1; rc=$?
      log "built $o: rc $rc ($(( $(date +%s) - s )) s) $(tail -1 $o.log | cut -c1-260)"
      [ $rc = 0 ] || exit 1
      sb=$(echo "$2 / 2" | bc -l)
      echo "{\"scene\": \"S$t\", \"N\": $n, \"tgt\": \"$SM\", \"smooth\": \"$SM\", \"detail\": \"$o\", \"runs\": {}, \"seeds\": [97], \"s_bg\": $sb}" > $O/size_probe/scene_S$t.json
      nice -n 10 $PY $W/d172_detail.py $O/size_probe/S$t.json $O/size_probe/scene_S$t.json > $O/size_probe/S$t.log 2>&1
      log "probe S$t: rc $? $(grep "^ref detail_s1" $O/size_probe/S$t.log | cut -c1-200)" ) &
  done
done
wait
log "done"
