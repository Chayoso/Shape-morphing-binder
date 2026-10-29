#!/bin/bash
# build_magC3.sh -- D173 (option C), CPU, detached, after build_magC2.sh. The probes so far high-passed at s_bg = 4K pitches (so the
# magnified scales stay inside the band), but D172's reader also reads its region as the canvas within 2 s_bg of the pattern: at
# 4K = 8-20 pitches that is 84-100 % of the body, mostly off the flank where H = 0 and only the sample's noise reads (corr
# 0.088 / 0.118 / 0.143 / 0.164 / 0.221 for K 2 / 2.5 / 3 / 4 / 5). Section 1's gate (the real scales) read at the reader's
# default, 4 pitches: this reads K 2 / 2.5 / 3 / 4 / 5 at 300k at s_bg = 4 pitches (the same meshes, the same perfect samples),
# then picks the smallest K with corr >= 0.25 there -> assets/magC/variant.txt = "TOKEN K S_BG" (empty: none).
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; A=$O/assets/magC
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 D157_THREADS=12 PHYSMORPH_CACHE=$B/cache/samples MPLCONFIGDIR=$B/cache/matplotlib
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python; SM=$B/output/gpu/d170/assets/d170smooth_dragon.obj
log() { echo "$(date "+%m-%d %H:%M:%S") [magC3] $*" >> $O/status.txt; }
while kill -0 $(cat $W/build_magC2.pid) 2> /dev/null; do sleep 10; done
rm -f $A/variant.txt; cd $B/repo_d173
one() { local KT=$1 o=$A/d173M$1_300_dragon.obj
  echo "{\"scene\": \"magC$1_300_s4\", \"N\": 300000, \"tgt\": \"$SM\", \"smooth\": \"$SM\", \"detail\": \"$o\", \"runs\": {}, \"seeds\": [97], \"s_bg\": 4.0}" > $A/probe_$1_300_s4.json
  nice -n 10 $PY $B/tmp/d172/d172_detail_fast.py $A/probe_$1_300_s4.out.json $A/probe_$1_300_s4.json --no_cache > $A/probe_$1_300_s4.log 2>&1
  log "probe K $1 s_bg 4: exit $?"; }
one 2 & one 25 & one 3 & wait
one 4 & one 5 & wait
$PY - > $A/probe_table3.txt << EOP
import json
for kt, k in (("2", 2.0), ("25", 2.5), ("3", 3.0), ("4", 4.0), ("5", 5.0)):
    try:
        d = json.load(open("$A/probe_%s_300_s4.out.json" % kt))
    except Exception:
        print(kt, k, 300, "missing"); continue
    r, c = d["refs"], list(d["ctx"].values())[0]
    s1, s2, fl = r["detail_s197"]["relief"], r["detail_s297"]["relief"], r["smooth_s297"]["relief"]
    print(kt, k, 300, "A_s1 %.3f corr_s1 %.3f corr_s2 %.3f floor_corr %.3f rmsH_pitch %.3f region %.2f sbg 4" % (s1["A"], s1["corr"],
          s2["corr"], fl["corr"], c["rms_H_pitch"], c["region_share"]))
EOP
pick=$(awk '$7 >= 0.25 {print $1, $2, 4; exit}' $A/probe_table3.txt)
echo -n "$pick" > $A/variant.txt
log "probe table (s_bg 4): $(tr "\n" ";" < $A/probe_table3.txt) -> ${pick:-none}"
