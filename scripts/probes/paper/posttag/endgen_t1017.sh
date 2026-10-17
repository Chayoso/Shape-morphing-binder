#!/bin/bash
# endgen_t1017.sh -- D145 post-tag reading 2: endgen145t.py (end silhouette IoU and chamfer against the independent 300k
# sample) on the end frames of B2, B8, B9 on the tag (d145t_*) and before it (d145_*); B9's floor runs against their
# lifted sample (ind_floor_NAME.npz, as read300.sh's MODE f). One card under the agents' lock protocol: a card with no
# lock, < 2000 MiB used and free for >= 10 min by tmp/d145/free_since; mkdir lock, info inside, released by a trap.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; E=$G/d145t_evidence; L=$B/gpu_locks; F=$B/tmp/d145/free_since
mkdir -p $E; g=""; tries=0
while [ -z "$g" ] && [ $tries -lt 240 ]; do   # wait up to 4 h for a card, a minute a look
  now=$(date +%s)
  for c in 0 1 2 3; do
    [ -e $L/gpu$c.lock ] && continue
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c)" -lt 2000 ] || continue
    [ -e $F/gpu$c ] && [ $(( now - $(cat $F/gpu$c) )) -ge 600 ] || continue
    mkdir $L/gpu$c.lock 2>/dev/null && { g=$c; break; }
  done
  [ -z "$g" ] && { tries=$((tries + 1)); sleep 60; }
done
[ -z "$g" ] && { echo "no card idle for 10 min within 4 h; nothing run $(date '+%m-%d %H:%M')" >> $E/endgen_t1017.status; exit 1; }
trap "rm -rf $L/gpu$g.lock" EXIT
printf "agent D145r\nrun d145t_evidence/endgen_t1017 (end checks B2/B8/B9, ~3 min)\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
echo "start gpu $g $(date '+%m-%d %H:%M')" >> $E/endgen_t1017.status
export PHYSMORPH_RUN_REPO=$B/repo_t CUDA_VISIBLE_DEVICES=$g; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
cd $B/repo_t
for d in d145t_b2 d145t_b8 d145t_b9 d145_b2 d145_b8 d145_b9; do
  specs=""
  for f in $G/$d/*300k_*_end1.npz; do
    n=$(basename $f _end1.npz); s="$n=$f"
    [ -e $G/$d/ind_floor_$n.npz ] && s="$s,$G/$d/ind_floor_$n.npz"
    specs="$specs $s"
  done
  rm -f $E/endgen_$d.jsonl
  $PY $B/tmp/d145r/endgen145t.py $E/endgen_$d.jsonl $specs > $E/endgen_t1017_$d.log 2>&1
done
echo "done gpu $g $(date '+%m-%d %H:%M')" >> $E/endgen_t1017.status
