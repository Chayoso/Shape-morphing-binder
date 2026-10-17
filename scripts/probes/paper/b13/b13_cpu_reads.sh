#!/bin/bash
# b13_cpu_reads.sh -- B13's CPU readings (no card), re-runnable; each step only for what has landed and is not yet read:
#   - the volume routine (tmp/d145/vol145.py) on seed 97 per mesh: L97 (d145t_b1), F97, FM97 (b13) -> b13/vol_b13.jsonl
#   - rot145.py / angL145.py on every run of the three arms with a JSON -> b13/rot_b13.jsonl, b13/angL_b13.log
source /data/relcfd/chayo/physmorph_v2/tmp/d146/cache_env.sh
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; D=$G/b13; LD=$G/d145t_b1; T=$B/tmp
source $B/repo_ts/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
cd $B
for m in bunny dragon teapot bob armadilo homer; do
  grep -q "\"$m\"" $D/vol_done.txt 2>/dev/null && continue
  fl=$LD/${m}300k_L97_frames12.npz; ff=$D/${m}300k_F97_frames12.npz; fm=$D/${m}300k_FM97_frames12.npz
  grep -q "^${m}300k_L97 sim 0 " $LD/status.txt 2>/dev/null && grep -q "^done ${m}300k_FM97 rc 0" $D/status300.txt \
    && grep -q "^done ${m}300k_F97 rc 0" $D/status300.txt && [ -f $fl ] && [ -f $ff ] && [ -f $fm ] || continue
  nice -n 10 $PY $T/d145/vol145.py $D/vol_b13.jsonl $m $G/d90/${m}_ind300k.npz L97=$fl F97=$ff FM97=$fm >> $D/vol_b13.log 2>&1 < /dev/null \
    && echo "\"$m\" $(date '+%m-%d %H:%M')" >> $D/vol_done.txt
done
specs=""
for m in bunny dragon teapot bob armadilo homer; do
  for s in 97 99 101; do
    [ -f $LD/${m}300k_L$s.json ] && specs="$specs d145t_b1/${m}300k_L$s"
    for a in F FM; do [ -f $D/${m}300k_$a$s.json ] && specs="$specs b13/${m}300k_$a$s"; done
  done
done
if [ -n "$specs" ]; then
  rm -f $D/rot_b13.jsonl
  nice -n 10 $PY $T/d145/rot145.py $D/rot_b13.jsonl $specs > $D/rot_b13.log 2>&1 < /dev/null
  nice -n 10 $PY $T/d145/angL145.py $specs > $D/angL_b13.log 2>&1 < /dev/null
fi
echo "cpu reads $(date '+%m-%d %H:%M'): vol $(grep -c . $D/vol_done.txt 2>/dev/null || echo 0) meshes; rot/angL $(echo $specs | wc -w) runs" >> $D/status300.txt
