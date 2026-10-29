#!/bin/bash
# build_magC.sh -- D173 (option C), CPU, detached: the dragon's own scales magnified K = 2 / 2.5 / 3 (d173_magscale.py) at 300k,
# the representability probe of each (D172's detail reader, no run: the relief's own perfect 300k sample, s_bg = 4K pitches),
# then the pick: the smallest K whose perfect sample reads corr >= 0.25 -> output/gpu/d173/assets/magC/variant.txt.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; A=$O/assets/magC; mkdir -p $A
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 D157_THREADS=12 PHYSMORPH_CACHE=$B/cache/samples MPLCONFIGDIR=$B/cache/matplotlib
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python; SM=$B/output/gpu/d170/assets/d170smooth_dragon.obj
log() { echo "$(date "+%m-%d %H:%M:%S") [magC] $*" >> $O/status.txt; }
cd $B/repo_d173
for K in 2 2.5 3; do
  ( o=$A/d173M${K}_300_dragon.obj; s=$(date +%s); nice -n 10 $PY $W/d173_magscale.py $o $K 300000 > $o.log 2>&1; log "built K $K: rc $? ($(( $(date +%s) - s )) s) $(tail -1 $o.log | cut -c1-240)"
    sb=$(echo "4 * $K" | bc -l)
    echo "{\"scene\": \"magC$K\", \"N\": 300000, \"tgt\": \"$SM\", \"smooth\": \"$SM\", \"detail\": \"$o\", \"runs\": {}, \"seeds\": [97], \"s_bg\": $sb}" > $A/probe_$K.json
    nice -n 10 $PY $B/tmp/d172/d172_detail_fast.py $A/probe_$K.out.json $A/probe_$K.json --no_cache > $A/probe_$K.log 2>&1
    log "probe K $K: $(grep -o "\"detail_s1.*" $A/probe_$K.out.json | head -1 | cut -c1-10)" ) &
done
wait
v=$($PY - << EOP
import json
for K in ("2", "2.5", "3"):
    try:
        r = json.load(open(f"$A/probe_{K}.out.json"))["refs"]["detail_s197"]["relief"]
    except Exception:
        continue
    print(K, r["A"], r["corr"], flush=True)
EOP
)
echo "$v" > $A/probe_table.txt
pick=$(echo "$v" | awk '$3 >= 0.25 {print $1; exit}')
echo -n "$pick" > $A/variant.txt
log "probe table: $(echo $v | tr "\n" ";") -> K = ${pick:-none}"
