#!/bin/bash
# ext_gpu_t1020.sh WHAT MESH [...] -- D145's methods table (2026-10-20): the card readings, one card under the agents'
# lock protocol (no lock dir, < 2000 MiB used, and idle in every one of this script's own looks over the last 10
# minutes, a look a minute; mkdir lock, info inside, released by a trap). WHAT = end: ext_end145.py on every method's end
# frame (all meshes, all seeds); WHAT = harness: harness145t.py for the listed meshes, every external method's first
# seed on the grid of the tag's L97 (read by B13 with L97 first). Outputs in output/gpu/d145_ext/.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; X=$G/d145_ext; L=$B/gpu_locks; R=$B/tmp/d145r
mkdir -p $X; what=$1; shift
declare -A idle; g=""; looks=0
while [ -z "$g" ] && [ $looks -lt 360 ]; do        # at most 6 h of looking
  for c in 0 1 2 3; do
    if [ ! -e $L/gpu$c.lock ] && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c)" -lt 2000 ]; then
      idle[$c]=$(( ${idle[$c]:-0} + 1 )); else idle[$c]=0; fi
  done
  for c in 0 1 2 3; do
    [ ${idle[$c]:-0} -ge 11 ] && mkdir $L/gpu$c.lock 2>/dev/null && { g=$c; break; }
  done
  [ -z "$g" ] && { looks=$((looks + 1)); sleep 60; }
done
[ -z "$g" ] && { echo "$what: no card idle 10 min within 6 h $(date '+%m-%d %H:%M')" >> $X/status.txt; exit 1; }
trap "rm -rf $L/gpu$g.lock" EXIT
printf "agent D145r\nrun d145_ext/$what $*\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
echo "start $what $* gpu $g $(date '+%m-%d %H:%M')" >> $X/status.txt
export PHYSMORPH_RUN_REPO=$B/repo_t CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
cd $B/repo_t
M6="bunny dragon teapot bob armadilo homer"
if [ "$what" = end ]; then
  specs=""
  for m in $M6; do
    for s in 97 99 101 103 105; do f=$G/d145t_b1/${m}300k_L${s}_end1.npz; [ -e $f ] && specs="$specs $m:ours:L$s=$f"; done
    specs="$specs $m:xu:XU97=$G/d145_b4/${m}300k_XU97_frames12.npz $m:xu:XU99=$G/d145_b4/${m}300k_XU99_end1.npz $m:xu:XU101=$G/d145_b4/${m}300k_XU101_end1.npz"
    for f in $G/d145_b6/${m}300k_OT_frames12.npz; do [ -e $f ] && specs="$specs $m:ot:OT=$f"; done
    for f in $G/d145_b11/${m}300k_ARAP_frames12.npz; do [ -e $f ] && specs="$specs $m:arap:ARAP=$f"; done
    for t in BUO BUO2 BUO3 NISE; do f=$G/d145_b5/${m}300k_${t}_frames12.npz; [ -e $f ] && specs="$specs $m:$(echo $t | tr -d 23 | tr A-Z a-z):$t=$f"; done
    for t in ISDn1 ISDn2 ISDn3 ISDr1 FDn1 FDn2 FDn3 FDr1; do f=$G/d145_b14/${m}300k_${t}_frames12.npz; [ -e $f ] && specs="$specs $m:$(echo $t | sed 's/[nr][0-9]//' | tr A-Z a-z):$t=$f"; done
    for a in LSa LSb FCa FCb; do for s in 97 99 101; do f=$G/d145_b12/${m}300k_${a}${s}_frames12.npz; [ -e $f ] && specs="$specs $m:$(echo $a | tr A-Z a-z):${a}${s}=$f"; done; done
  done
  rm -f $X/ext_end.jsonl
  $PY $R/ext_end145.py $X/ext_end.jsonl $specs > $X/ext_end.log 2>&1
else
  for m in "$@"; do
    specs="XU97=$G/d145_b4/${m}300k_XU97_frames12.npz"
    for f in $G/d145_b6/${m}300k_OT_frames12.npz; do [ -e $f ] && specs="$specs OT=$f"; done
    for f in $G/d145_b11/${m}300k_ARAP_frames12.npz; do [ -e $f ] && specs="$specs ARAP=$f"; done
    specs="$specs BUO=$G/d145_b5/${m}300k_BUO_frames12.npz NISE=$G/d145_b5/${m}300k_NISE_frames12.npz"
    specs="$specs ISDn1=$G/d145_b14/${m}300k_ISDn1_frames12.npz FDn1=$G/d145_b14/${m}300k_FDn1_frames12.npz"
    for a in LSa97 LSb97 FCa97 FCb97; do specs="$specs $a=$G/d145_b12/${m}300k_${a}_frames12.npz"; done
    rm -f $X/harness_$m.jsonl
    $PY $R/harness145t.py $X/harness_$m.jsonl $m $G/d90/${m}_ind300k.npz --grid-from=$G/d145t_b1/${m}300k_L97_frames12.npz $specs > $X/harness_$m.log 2>&1
    echo "harness $m done rc $? $(date '+%m-%d %H:%M')" >> $X/status.txt
  done
fi
echo "done $what $* gpu $g $(date '+%m-%d %H:%M')" >> $X/status.txt
