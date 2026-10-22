#!/bin/bash
# d162_chain.sh STEPS -- D162 on hyde06, on ONE card (two chains may run, one card each) by the agents' lock protocol (gpu_locks/gpuN.lock by mkdir with an
# info file "agent D162 ...", released by the trap; a card with no lock, no other user's compute process, < 2000 MiB,
# idle 10 minutes by the pool's shared first-free time; GPU 2 never: another user's desktop). Courtesy: once a card is
# eligible the chain waits a further 90 s and claims it only if no other agent has (the queued agents poll every 60 s).
# The card is held only while a GPU step runs; CPU steps hold none.
#   V   the gates on repo_d162 / repo_d162s: V1 the full suite (exit), V2 the pinned references and the merged options'
#       tests, V3 deterministic SAME at the defaults against rt1's S1, V4 four-window smoke runs of each new option
#       and their combinations with D159 / D160 / D161 (exit 0 and the log lines)
#   R   the runs of $RUNS_FILE (NAME FLAGS... per line), repo_d162s (kept frames)
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d162; W=$B/tmp/d162; L=$B/gpu_locks
mkdir -p $O/gates $O/runs40 $W/tmpdir
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
STEPS=${1:-"C"}
CARD=""
claim_card() {
  local what=$1; declare -A since
  local last_report=0
  while [ -z "$CARD" ]; do
    local now=$(date +%s) reasons=""
    for g in 3 1 0; do
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
          printf "agent D161\nrun d161/%s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
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
log "chain start pid $$ steps '$STEPS' runs '${RUNS_FILE:-}'"
A161=$G/d161/assets
if [[ " $STEPS " == *" V "* ]]; then
  V=$O/gates
  claim_card gates
  ( envr repo_d162; $PY -m pytest -q -p no:cacheprovider tests > $V/suite.log 2>&1 < /dev/null; echo $? > $V/suite.rc )
  log "V1 suite exit $(cat $V/suite.rc): $(tail -1 $V/suite.log)"
  ( envr repo_d162; $PY -m pytest -q -p no:cacheprovider tests/test_refreeze_reference.py tests/test_d162.py \
      tests/test_render_tgt.py tests/test_proximity_berth.py tests/test_appearance.py tests/test_appearance_pipeline.py \
      tests/test_render_target_shell.py tests/test_frozen_recipe.py > $V/refs.log 2>&1 < /dev/null; echo $? > $V/refs.rc )
  log "V2 references and options' tests exit $(cat $V/refs.rc): $(tail -1 $V/refs.log)"
  ( envr repo_d162s; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 --tgt assets/bunny.obj \
      --out $V/S1 > $V/S1.log 2>&1 < /dev/null; echo $? > $V/S1.rc )
  ( envr repo_d162; $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $V/S1 > $V/cmp_rt1_S1.txt 2>&1 )
  log "V3 rt1 S1 vs repo_d162s defaults (exit $(cat $V/S1.rc)): $(tail -1 $V/cmp_rt1_S1.txt)"
  smoke() { local name=$1; shift; ( envr repo_d162s; $PY scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 \
      --out $V/$name "$@" > $V/$name.log 2>&1 < /dev/null; echo $? > $V/$name.rc )
    log "V4 $name ($*) exit $(cat $V/$name.rc): $(grep -m1 'L_floor' $V/$name.log | cut -c1-140)"; }
  smoke F --tgt assets/bunny.obj --render_weight_ref floor
  smoke C --tgt assets/bunny.obj --placement_at_cell
  smoke FC_coin --tgt $A161/d161coin.obj --render_tgt $A161/d161L40coin.obj --render_weight_ref floor --placement_at_cell
  smoke F_band --tgt assets/bunny.obj --appearance band --render_weight_ref floor
  smoke F_shell --tgt assets/bunny.obj --render_target mesh_shell --render_weight_ref floor
  ok=1
  [ "$(cat $V/suite.rc)" = 0 ] || ok=0
  [ "$(cat $V/refs.rc)" = 0 ] || ok=0
  tail -1 $V/cmp_rt1_S1.txt | grep -q ": SAME$" || ok=0
  for k in F C FC_coin F_band F_shell; do [ "$(cat $V/$k.rc)" = 0 ] || ok=0; done
  for k in F FC_coin F_band F_shell; do grep -q "L_floor" $V/$k.log || ok=0; done
  if [ $ok = 1 ]; then log "V gates passed"; touch $O/gates/PASSED; else log "V STOP: a gate failed"; fi
  release_card
fi
if [[ " $STEPS " == *" R "* ]]; then
  [ -e $O/gates/PASSED ] || { log "R refused: the gates have not passed"; exit 1; }
  while read -r name flags; do
    [ -z "$name" ] && continue
    case "$name" in \#*) continue;; esac
    if [ -e $O/runs40/${name}_frames12.npz ]; then log "R $name exists, skipped"; continue; fi
    [ -n "$CARD" ] || claim_card "runs40 ${RUNS_FILE:-}"
    log "R $name start ($flags)"
    ( envr repo_d162s; $PY scripts/pipeline_run.py $flags --out $O/runs40/$name > $O/runs40/$name.log 2>&1 < /dev/null
      echo $? > $O/runs40/$name.rc )
    log "R $name exit $(cat $O/runs40/$name.rc): $(grep -m1 'ARM render_full_dt_iso_nn:' $O/runs40/$name.log | cut -c1-150)"
  done < $W/${RUNS_FILE:-d162_runs.txt}
  release_card
  log "R done (${RUNS_FILE:-d162_runs.txt})"
fi
log "chain done ($STEPS)"
