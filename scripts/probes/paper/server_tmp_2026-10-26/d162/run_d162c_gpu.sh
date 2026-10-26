#!/bin/bash
# d162c_gpu.sh -- D162c (B) on hyde06: the render pushes at the kept states (d162c_push.py) on ONE card by the agents' lock
# info file "agent D162 ...", released by the trap; a card with no lock, no other user's compute process, < 2000 MiB,
# idle 10 minutes by the pool's shared first-free time; GPU 2 never: another user's desktop). Courtesy: once a card is
# eligible the chain waits a further 90 s and claims it only if no other agent has (the queued agents poll every 60 s).
# The card is held only while a GPU step runs; CPU steps hold none.
#   V   the gates on repo_d162 / repo_d162bs: V1 the full suite (exit), V2 the pinned references and the merged options'
#       tests, V3 deterministic SAME at the defaults against rt1's S1, V4 four-window smoke runs of each new option
#       and their combinations with D159 / D160 / D161 (exit 0 and the log lines)
#   R   the runs of $RUNS_FILE (NAME FLAGS... per line), repo_d162bs (kept frames)
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d162c; W=$B/tmp/d162; L=$B/gpu_locks
mkdir -p $O/gates $O/runs40 $W/tmpdir
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
STEPS=${1:-"C"}
CARD=""
claim_card() {
  local what=$1; declare -A since
  # HANDOVER_FROM=PID: this agent's own chain on that card ends (its list cut short); the card passes straight to this
  # chain once that process is gone, without a gap (the card stays the agent's; another user's job may share it, as it
  # did under the chain before). If someone else's lock is there first, the normal claim below applies.
  if [ -n "$HANDOVER_FROM" ]; then
    while ps -p $HANDOVER_FROM > /dev/null 2>&1; do sleep 5; done
    for g in ${CARDS:-1}; do
      if mkdir $L/gpu$g.lock 2>/dev/null; then
        CARD=$g
        printf "agent D162\nrun d162b/%s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        log "took gpu$g over from this agent's chain pid $HANDOVER_FROM (card kept) for $what"
        HANDOVER_FROM=""; export CUDA_VISIBLE_DEVICES=$CARD; return
      fi
    done
    HANDOVER_FROM=""
  fi
  local last_report=0
  while [ -z "$CARD" ]; do
    local now=$(date +%s) reasons=""
    for g in ${CARDS:-3 1 0}; do                 # CARDS: the candidate cards, in order (default 3 1 0)
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      local uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
      local other=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | while IFS=', ' read -r u p; do
                    o=$(ps -o user= -p $p 2>/dev/null); [ -n "$o" ] && [ "$o" != chayo ] && echo x; done | wc -l)
      local mine=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
      local why=""
      [ -d $L/gpu$g.lock ] && why="lock($(grep -m1 agent $L/gpu$g.lock/info 2>/dev/null))"
      [ "$used" -gt 2000 ] && why="$why used=${used}MiB"
      [ "$other" -gt 0 ] && why="$why other-user"
      [ "$mine" -gt 0 ] && why="$why apps=$mine"
      if [ -n "$why" ]; then since[$g]=""; reasons="$reasons gpu$g:$why;"; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      local fs=$(cat $B/tmp/d145/free_since/gpu$g 2>/dev/null); [ -n "$fs" ] && [ "$fs" -lt "${since[$g]}" ] && since[$g]=$fs
      if [ $((now - ${since[$g]})) -ge 600 ]; then
        sleep 90                                  # courtesy: a queued agent polling every 60 s goes first
        [ -d $L/gpu$g.lock ] && { log "gpu$g eligible but taken during the courtesy wait"; since[$g]=""; continue; }
        if mkdir $L/gpu$g.lock 2>/dev/null; then
          CARD=$g
          printf "agent D162\nrun d162/%s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
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
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1; export TMPDIR=$W/tmpdir; cd $B/$1; }
log "chain start pid $$ steps '$STEPS' runs '${RUNS_FILE:-}'"
claim_card push
( envr repo_d162b; $PY $W/d162c_push.py $O/push ${PUSH_RUNS:-} > $O/push.log 2>&1 < /dev/null; echo $? > $O/push.rc )
log "B push exit $(cat $O/push.rc): $(grep -c 'd162c B\]' $O/push.log) runs"
release_card
( envr repo_d162b; export CUDA_VISIBLE_DEVICES=; nice -n 10 $PY $W/d162c_pushread.py $O/B.jsonl $O/push > $O/pushread.log 2>&1 < /dev/null
  echo $? > $O/pushread.rc )
log "B read exit $(cat $O/pushread.rc): $(wc -l < $O/B.jsonl) states"
