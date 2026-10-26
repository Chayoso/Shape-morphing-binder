#!/bin/bash
# d156d_chain.sh STEPS -- D156d on ONE card by the agents' lock protocol (gpu_locks/gpuN.lock by mkdir with an info file,
# released by the trap; a card with no lock, no other user's compute process, < 2000 MiB, idle for 10 minutes), with
# the courtesy rule: no claim while D159, D160 or D161 (the user's top goal, ahead of D156d) are waiting for a card
# (the last line of their output/gpu/<agent>/status.txt reads "waiting").
#   C  the reachability check: d156_sfs.py on bunny, dragon and bob (a few minutes each), then the card is released.
#   S  the other six meshes (only after the check has been read and passed).
# Usage: d156d_chain.sh STEPS   (e.g. "C")
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d156d; W=$B/tmp/d156; L=$B/gpu_locks; A=$G/d156/assets_hull
mkdir -p $O
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir; mkdir -p $TMPDIR
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
STEPS=${1:-"C"}
CARD=""
priority_waiting() {
  for a in d159 d160 d161; do
    tail -1 $G/$a/status.txt 2>/dev/null | grep -q "waiting" && { echo $a; return 0; }
  done
  return 1
}
claim_card() {
  local what=$1; declare -A since; local last_report=0
  while [ -z "$CARD" ]; do
    local now=$(date +%s) reasons=""
    local pw=$(priority_waiting)
    for g in 3 1 0; do
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      local uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
      local apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
      local why=""
      [ -d $L/gpu$g.lock ] && why="lock($(grep -m1 agent $L/gpu$g.lock/info 2>/dev/null))"
      [ "$used" -gt 2000 ] && why="$why used=${used}MiB"
      [ "$apps" -gt 0 ] && why="$why apps=$apps"
      if [ -n "$why" ]; then since[$g]=""; reasons="$reasons gpu$g:$why;"; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      local fs=$(cat $B/tmp/d145/free_since/gpu$g 2>/dev/null); [ -n "$fs" ] && [ "$fs" -lt "${since[$g]}" ] && since[$g]=$fs
      if [ -n "$pw" ]; then reasons="$reasons gpu$g:free but $pw waits;"; continue; fi
      if [ $((now - ${since[$g]})) -ge 600 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        CARD=$g
        printf "agent D156\nrun d156d/%s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        break
      fi
    done
    if [ -z "$CARD" ] && [ $((now - last_report)) -ge 900 ]; then log "waiting:$reasons"; last_report=$now; fi
    [ -z "$CARD" ] && sleep 60
  done
  export CUDA_VISIBLE_DEVICES=$CARD
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "released gpu$CARD"; CARD=""; }
trap 'release_card' EXIT
trap 'exit 1' INT TERM HUP
log "chain start pid $$ steps [$STEPS]"
claim_card "chain [$STEPS]: shape from shading of the visual hull (minutes per mesh)"
log "claimed gpu$CARD"
cd $B/repo_d156
source scripts/ops/hyde06_env.sh > /dev/null 2>&1
export TMPDIR=$W/tmpdir PYTHONPATH=$B/deps/gpu_pipeline:$B/repo_d156:$W CUDA_VISIBLE_DEVICES=$CARD D156_DEV=cuda
cd $B/repo_d156
if [[ " $STEPS " == *" C "* ]]; then
  for m in bunny dragon bob; do
    $PY $W/d156_sfs.py $A 40000 96 $m > $O/sfs_$m.log 2>&1 < /dev/null
    log "C $m exit $?: $(grep -h '"loss_solid"' $O/sfs_$m.log | cut -c1-200)"
  done
fi
if [[ " $STEPS " == *" S "* ]]; then
  for m in armadilo homer cow cheburashka C A; do
    $PY $W/d156_sfs.py $A 40000 96 $m > $O/sfs_$m.log 2>&1 < /dev/null
    log "S $m exit $?"
  done
fi
log "chain done"
