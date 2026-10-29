#!/bin/bash
# build_magC2.sh -- D173 (option C), CPU, detached, after build_magC.sh: its K = 2.5 mesh was named d173M2.5_300_dragon.obj and
# physmorph/sampling/orientation.py asset_name cuts the name at the first dot ("d173M2" -> orientation "id"), so that mesh was read
# turned (canvas rms H 1.56 pitch, region 0.15: its corr 0.56 is the turn, not the scales). Names now carry no dot (K 2.5 -> M25).
# K = 2 and 3 read corr 0.088 / 0.143 at 300k (< 0.25): this probes K 2.5 again and K 4 / 5 at 300k, and K 2 / 2.5 / 3 at 550k
# (information; the runs stay at 300k), then picks the smallest K at 300k whose perfect sample reads corr >= 0.25 ->
# assets/magC/variant.txt = "TOKEN K" (empty: none).
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; A=$O/assets/magC; mkdir -p $A
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 D157_THREADS=12 PHYSMORPH_CACHE=$B/cache/samples MPLCONFIGDIR=$B/cache/matplotlib
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python; SM=$B/output/gpu/d170/assets/d170smooth_dragon.obj
log() { echo "$(date "+%m-%d %H:%M:%S") [magC2] $*" >> $O/status.txt; }
cd $B/repo_d173
one() { local KT=$1 K=$2 N=$3 n=$(( $3 / 1000 )); local o=$A/d173M${KT}_${n}_dragon.obj s=$(date +%s)
  nice -n 10 $PY $W/d173_magscale.py $o $K $N > $o.log 2>&1; log "built K $K N $N: rc $? ($(( $(date +%s) - s )) s) $(tail -1 $o.log | cut -c1-200)"
  local sb=$(echo "4 * $K" | bc -l)
  echo "{\"scene\": \"magC${KT}_$n\", \"N\": $N, \"tgt\": \"$SM\", \"smooth\": \"$SM\", \"detail\": \"$o\", \"runs\": {}, \"seeds\": [97], \"s_bg\": $sb}" > $A/probe_${KT}_$n.json
  nice -n 10 $PY $B/tmp/d172/d172_detail_fast.py $A/probe_${KT}_$n.out.json $A/probe_${KT}_$n.json --no_cache > $A/probe_${KT}_$n.log 2>&1
  log "probe K $K N $N: exit $?"; }
one 25 2.5 300000 & one 4 4 300000 & one 5 5 300000 & wait
one 2 2 550000 & one 25 2.5 550000 & one 3 3 550000 & wait
$PY - > $A/probe_table2.txt << EOP
import json
rows = [("2", 2.0, 300, "probe_2.out.json"), ("25", 2.5, 300, "probe_25_300.out.json"), ("3", 3.0, 300, "probe_3.out.json"),
        ("4", 4.0, 300, "probe_4_300.out.json"), ("5", 5.0, 300, "probe_5_300.out.json"), ("2", 2.0, 550, "probe_2_550.out.json"),
        ("25", 2.5, 550, "probe_25_550.out.json"), ("3", 3.0, 550, "probe_3_550.out.json")]
for kt, k, n, f in rows:
    try:
        d = json.load(open("$A/" + f))
    except Exception:
        print(kt, k, n, "missing"); continue
    r, c = d["refs"], list(d["ctx"].values())[0]
    s1, s2, fl = r["detail_s197"]["relief"], r["detail_s297"]["relief"], r["smooth_s297"]["relief"]
    print(kt, k, n, "A_s1 %.3f corr_s1 %.3f corr_s2 %.3f floor_corr %.3f rmsH_pitch %.3f region %.2f" % (s1["A"], s1["corr"], s2["corr"],
          fl["corr"], c["rms_H_pitch"], c["region_share"]))
EOP
pick=$(awk '$3 == 300 && $7 >= 0.25 {print $1, $2; exit}' $A/probe_table2.txt)
echo -n "$pick" > $A/variant.txt
log "probe table: $(tr "\n" ";" < $A/probe_table2.txt) -> ${pick:-none}"
