#!/bin/bash
# build_scales3.sh -- D172 (D): the scale reliefs a third time, taller and wider (C / D read corr 0.09 / 0.14 at 300k:
# the flank map's facing ramp halves the domes' height); E = spacing 6, amp 2.0; F = 8, 2.0; G = 8, 3.0; before: with gaps (A / B, packed domes of radius 0.475 x
# spacing, blurred to a near-flat field): C = spacing 5 pitches, domes of radius 0.4 x spacing, profile (1 - r^2)^2, amp 0.8;
# D = spacing 6, the same, amp 1.0; edge 1 pitch; 40k and 300k; then the representability probe and the pick (corr >= 0.25
# at 300k, C before D) into output/gpu/d172/assets/variant.txt.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d172; O=$B/output/gpu/d172; A=$O/assets
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 D157_THREADS=12 PHYSMORPH_CACHE=$B/cache/samples MPLCONFIGDIR=$B/cache/matplotlib
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
SM=$B/output/gpu/d170/assets/d170smooth_dragon.obj
log() { echo "$(date '+%m-%d %H:%M:%S') [scales] $*" >> $O/status.txt; }
cd $B/repo_d172
for v in "E 6 2.0" "F 8 2.0" "G 8 3.0"; do set -- $v
  for n in 40000 300000; do
    t=${1}$((n / 1000)); o=$A/d172S${t}_dragon.obj
    echo "{\"base\": \"$SM\", \"n\": $n, \"scale\": $2, \"amp\": $3, \"edge\": 1.0, \"height\": $2, \"rfrac\": 0.4, \"power\": 2.0, \"map\": \"flank\"}" > $A/S$t.json
    ( s=$(date +%s); nice -n 10 $PY $W/d172_mesh.py $o $A/S$t.json > $o.log 2>&1; rc=$?
      log "built $o: rc $rc ($(( $(date +%s) - s )) s) $(tail -1 $o.log | cut -c260-560)"
      [ $rc = 0 ] || exit 1
      sb=$(echo "$2 / 2" | bc -l)
      echo "{\"scene\": \"S$t\", \"N\": $n, \"tgt\": \"$SM\", \"smooth\": \"$SM\", \"detail\": \"$o\", \"runs\": {}, \"seeds\": [97], \"s_bg\": $sb}" > $O/size_probe/scene_S$t.json
      nice -n 10 $PY $W/d172_detail.py $O/size_probe/S$t.json $O/size_probe/scene_S$t.json > $O/size_probe/S$t.log 2>&1
      log "probe S$t: rc $? $(grep "^ref detail_s1" $O/size_probe/S$t.log | cut -c1-200)" ) &
  done
done
wait
v=$($PY - << 'EOP'
import json
for v in ("E", "F", "G"):
    try:
        r = json.load(open(f"/data/relcfd/chayo/physmorph_v2/output/gpu/d172/size_probe/S{v}300.json"))["refs"]["detail_s197"]["relief"]
    except Exception:
        continue
    if (r.get("corr") or 0) >= 0.25:
        print(v)
        break
EOP
)
echo -n "$v" > $A/variant.txt
log "variant chosen: '${v}' (a perfect 300k sample reads corr >= 0.25 in the pattern; empty = none)"
