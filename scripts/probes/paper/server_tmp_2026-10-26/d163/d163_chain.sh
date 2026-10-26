#!/bin/bash
# d163_chain.sh RUNS_FILE [LANES] -- D163 on hyde06: the runs of RUNS_FILE (NAME FLAGS... per line; the paths in FLAGS are
# absolute) on repo_d163s (kept frames), on ONE card by the agents' lock protocol (gpu_locks/gpuN.lock by mkdir with an
# info file "agent D163 ...", released by the trap; a card with no lock, no other user's compute process, < 2000 MiB,
# idle 10 minutes; GPU 2 never: another user's desktop; a 90-s courtesy wait once eligible). LANES runs share the card
# (each lane takes the next run not yet taken: mkdir RUN.claim). A run whose kept frames exist is skipped.
# STOP: touch $O/STOP (no new run starts).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d163; W=$B/tmp/d163; L=$B/gpu_locks
RUNS=$1; LANES=${2:-1}; OUTD=${OUTD:-$O/runs}
mkdir -p $OUTD $W/tmpdir
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
CARD=""
claim_card() {
  local what=$1; declare -A since
  local last_report=0
  while [ -z "$CARD" ]; do
    local now=$(date +%s) reasons=""
    for g in ${CARDS:-0 3 1}; do
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      local uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
      local other=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | while IFS=', ' read -r u p; do
                    o=$(ps -o user= -p $p 2>/dev/null); [ -n "$o" ] && [ "$o" != chayo ] && echo x; done | wc -l)
      local mine=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
      local util=$(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader,nounits)
      local why=""
      [ -d $L/gpu$g.lock ] && why="lock($(grep -m1 agent $L/gpu$g.lock/info 2>/dev/null))"
      [ "$used" -gt 2000 ] && why="$why used=${used}MiB"
      [ "$other" -gt 0 ] && why="$why other-user"
      [ "$mine" -gt 0 ] && why="$why apps=$mine"
      [ "$util" -gt 5 ] && why="$why util=${util}%"
      if [ -n "$why" ]; then since[$g]=""; reasons="$reasons gpu$g:$why;"; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      local fs=$(cat $W/idle_since_gpu$g 2>/dev/null); [ -n "$fs" ] && [ "$fs" -lt "${since[$g]}" ] && since[$g]=$fs
      if [ $((now - ${since[$g]})) -ge 600 ]; then
        sleep 90                                  # courtesy: a queued agent polling every 60 s goes first
        [ -d $L/gpu$g.lock ] && { log "gpu$g eligible but taken during the courtesy wait"; since[$g]=""; continue; }
        if mkdir $L/gpu$g.lock 2>/dev/null; then
          CARD=$g
          printf "agent D163\nrun d163/%s (%s lanes)\npid %s\nstart %s\n" "$what" "$LANES" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
          log "claimed gpu$g for $what"
          break
        fi
      fi
    done
    if [ -z "$CARD" ] && [ $((now - last_report)) -ge 900 ]; then
      log "waiting:$reasons"
      last_report=$now
    fi
    [ -z "$CARD" ] && sleep 60
  done
  export CUDA_VISIBLE_DEVICES=$CARD
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "released gpu$CARD"; CARD=""; }
trap 'release_card' EXIT
trap 'exit 1' INT TERM HUP
lane() {
  local k=$1
  while read -r name flags; do
    [ -z "$name" ] && continue
    case "$name" in \#*) continue;; esac
    [ -e $O/STOP ] && { log "lane $k: STOP"; break; }
    [ -e $OUTD/${name}_frames12.npz ] && continue
    mkdir $OUTD/$name.claim 2>/dev/null || continue
    log "R $name start (lane $k; $flags)"
    ( export PHYSMORPH_RUN_REPO=$B/repo_d163s PHYSMORPH_OUT=$G; source $B/repo_d163s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
      export TMPDIR=$W/tmpdir; cd $B/repo_d163s; s=$(date +%s)
      $PY scripts/pipeline_run.py $flags --out $OUTD/$name > $OUTD/$name.log 2>&1 < /dev/null
      echo "$? $(( $(date +%s) - s ))" > $OUTD/$name.rc )
    log "R $name exit/secs $(cat $OUTD/$name.rc): $(grep -m1 'ARM render_full_dt_iso_nn:' $OUTD/$name.log | cut -c1-170)"
  done < $RUNS
}
log "chain start pid $$ runs $RUNS lanes $LANES"
claim_card "$(basename $RUNS)"
for k in $(seq 1 $LANES); do lane $k & sleep 20; done
wait
release_card
log "chain done ($(basename $RUNS))"
