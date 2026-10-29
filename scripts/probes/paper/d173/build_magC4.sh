#!/bin/bash
# build_magC4.sh -- D173 (option C), CPU, detached, after build_magC3.sh. No K up to 5 passes at 300k (s_bg 4K: corr 0.088 / 0.118 /
# 0.143 / 0.164 / 0.221 for K 2 / 2.5 / 3 / 4 / 5; s_bg 4 lower, 0.10-0.12; 550k K 2 / 2.5 / 3: 0.109 / 0.149 / 0.161): the perfect
# sample's own noise (rms 0.35-0.40 pitch of the layer about the canvas) against a relief of rms 0.05-0.10 pitch. This builds and
# reads K 6 / 7 / 8 at 300k (s_bg 4K), then picks the smallest K at 300k (s_bg 4K, all tables) with corr >= 0.25 -> variant.txt.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; A=$O/assets/magC
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 D157_THREADS=12 PHYSMORPH_CACHE=$B/cache/samples MPLCONFIGDIR=$B/cache/matplotlib
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python; SM=$B/output/gpu/d170/assets/d170smooth_dragon.obj
log() { echo "$(date "+%m-%d %H:%M:%S") [magC4] $*" >> $O/status.txt; }
while kill -0 $(cat $W/build_magC3.pid) 2> /dev/null; do sleep 10; done
rm -f $A/variant.txt; cd $B/repo_d173
one() { local K=$1 o=$A/d173M$1_300_dragon.obj s=$(date +%s)
  nice -n 10 $PY $W/d173_magscale.py $o $K 300000 > $o.log 2>&1; log "built K $K: rc $? ($(( $(date +%s) - s )) s) $(tail -1 $o.log | cut -c1-120)"
  echo "{\"scene\": \"magC$1_300\", \"N\": 300000, \"tgt\": \"$SM\", \"smooth\": \"$SM\", \"detail\": \"$o\", \"runs\": {}, \"seeds\": [97], \"s_bg\": $(( 4 * K ))}" > $A/probe_$1_300.json
  nice -n 10 $PY $B/tmp/d172/d172_detail_fast.py $A/probe_$1_300.out.json $A/probe_$1_300.json --no_cache > $A/probe_$1_300.log 2>&1
  log "probe K $K: exit $?"; }
one 6 & one 7 & one 8 & wait
$PY - > $A/probe_table4.txt << EOP
import json
for kt, k in (("6", 6.0), ("7", 7.0), ("8", 8.0)):
    try:
        d = json.load(open("$A/probe_%s_300.out.json" % kt))
    except Exception:
        print(kt, k, 300, "missing"); continue
    r, c = d["refs"], list(d["ctx"].values())[0]
    s1, s2, fl = r["detail_s197"]["relief"], r["detail_s297"]["relief"], r["smooth_s297"]["relief"]
    print(kt, k, 300, "A_s1 %.3f corr_s1 %.3f corr_s2 %.3f floor_corr %.3f rmsH_pitch %.3f region %.2f" % (s1["A"], s1["corr"], s2["corr"],
          fl["corr"], c["rms_H_pitch"], c["region_share"]))
EOP
pick=$(cat $A/probe_table2.txt $A/probe_table4.txt | awk '$3 == 300 && $7 >= 0.25 {print $2, $1}' | sort -g | head -1 | awk '{print $2, $1, 4 * $1}')
echo -n "$pick" > $A/variant.txt
log "probe table (K 6-8, s_bg 4K): $(tr "\n" ";" < $A/probe_table4.txt) -> ${pick:-none}"
