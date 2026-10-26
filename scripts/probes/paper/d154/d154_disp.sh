#!/bin/bash
# d154_disp.sh -- D154 (2), the display (secondary): surface_layer_probe.py measure against the independent sample
# (repo_r77, as read300.sh) on the 8 switched runs' kept frames every 0.5 s from 4 to 12 s and the last; one card by the
# lock protocol (no lock, no other user's compute process, < 2000 MiB, idle 10 min by the pool's shared first-free time;
# GPU 2 never), released by the trap. The chain's D step had sourced the environment after cd (into repo_settled).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d154; W=$B/tmp/d154; L=$B/gpu_locks
source $B/tmp/d146/cache_env.sh
PYB=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
CARD=""
claim_card() {
  declare -A since
  while [ -z "$CARD" ]; do
    local now=$(date +%s)
    for g in 3 1 0; do
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      local uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
      local other=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | while IFS=', ' read -r u p; do
                    o=$(ps -o user= -p $p 2>/dev/null); [ -n "$o" ] && [ "$o" != chayo ] && echo x; done | wc -l)
      local mine=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
      if [ -d $L/gpu$g.lock ] || [ "$used" -gt 2000 ] || [ "$other" -gt 0 ] || [ "$mine" -gt 0 ]; then since[$g]=""; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      local fs=$(cat $B/tmp/d145/free_since/gpu$g 2>/dev/null); [ -n "$fs" ] && [ "$fs" -lt "${since[$g]}" ] && since[$g]=$fs
      if [ $((now - ${since[$g]})) -ge 600 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        CARD=$g
        printf "agent D154\nrun d154/display of the 8 switched runs (secondary reader, about 30-60 min)\npid %s\nstart %s\n" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        break
      fi
    done
    [ -z "$CARD" ] && sleep 60
  done
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "display: released gpu$CARD"; CARD=""; }
trap 'release_card' EXIT
trap 'exit 1' INT TERM HUP
log "display job start pid $$ (waiting for a card idle 10 min)"
claim_card
log "display: claimed gpu$CARD"
export PHYSMORPH_RUN_REPO=$B/repo_r77
source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=$CARD
cd $B/repo_r77
for seed in 97 99; do for mesh in bunny dragon; do for arm in Loff6 Pon6; do
  name=${mesh}300k_${arm}_$seed; f=$O/runs/${name}_frames12.npz
  case $mesh in bunny) box=1900,100,2800,700;; dragon) box=1900,60,2500,560;; esac
  raws=$($PYB -c "import numpy as np; r=[int(v) for v in np.load('$f')['raws']]; print(' '.join(str(v) for v in r if (v % 120 == 0 and 960 <= v <= 2880) or v == r[-1]))")
  $PY scripts/probes/settled/surface_layer_probe.py $f $O/runs/e3_$name $box measure ref=$G/d90/${mesh}_ind300k.npz \
    target own $raws > $O/runs/e3_$name.log 2>&1 < /dev/null
  log "display $name exit $? rows $(grep -c '^{' $O/runs/e3_$name.log)"
done; done; done
log "display done"
