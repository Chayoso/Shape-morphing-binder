#!/bin/bash
# d164_chain.sh STEPS -- D164 on hyde06 (docs/experiments.md, D164), on ONE card by the agents' lock protocol
# (gpu_locks/gpuN.lock by mkdir with an info file "agent D164 ...", released by the trap). A card is eligible with no
# lock, no other user's compute process, < 2000 MiB, utilisation <= 5 %, for 10 minutes; then a 90-s courtesy wait.
# D162 (and D162b) go first: no claim while its chain reports waiting for a card (its status.txt's latest
# 'waiting:' line within 20 min and newer than its latest 'claimed'). Cards in the order 3 1 0 2 (GPU 2 last: another
# user's desktop runs there; D162 never takes it). D163 was dropped from the yield by the coordinator (2026-10-22 17:3x). The card is held for the chain's steps and released by the trap.
#   P    step 0, the pilot (measurement only): 40k bunny and dragon, seed 97, the full arm, repo_d164s with
#        --replay_noise start under tmp/d164/replay_probe.py -> output/gpu/d164/pilot/
#   V    the gates: V1 the full suite on repo_d164; V2 the contracts test 20x on repo_d164 (gate) and on repo_d164_start
#        (report), the two other commit_replay-prone tests 20x on both (report); V3 the pinned references and the
#        switch's tests; V4 Warp-deterministic 40k bunny seed 97 four windows on repo_d164s twice, SAME run to run and
#        SAME against rt1's S1
#   R40  the runs of tmp/d164/d164_runs40.txt (NAME FLAGS... per line) on repo_d164s -> output/gpu/d164/runs40/
#   R300 the runs of tmp/d164/d164_runs300.txt -> output/gpu/d164/runs300/, each followed by B1's readings
#        (tmp/d145/read300.sh MODE end on the card; its CPU part detached, frames kept)
# STOP: touch output/gpu/d164/STOP (no new run starts).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d164; W=$B/tmp/d164; L=$B/gpu_locks
mkdir -p $O/gates $O/pilot $O/runs40 $O/runs300 $W/tmpdir
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
STEPS=${1:-"P"}
CARD=""
render_waiting() {        # a D162 / D162b chain waiting for a card: its latest 'waiting:' within 20 min, after its latest claim
  local now=$(date +%s) f w c
  for f in $G/d162/status.txt $G/d162b/status.txt; do      # D163 dropped from the yield (the coordinator, 17:3x)
    [ -e $f ] || continue
    w=$(grep 'waiting:' $f | tail -1 | cut -c1-14); c=$(grep 'claimed gpu' $f | tail -1 | cut -c1-14)
    [ -z "$w" ] && continue
    local tw=$(date -d "2026-$w" +%s 2>/dev/null || echo 0) tc=0
    [ -n "$c" ] && tc=$(date -d "2026-$c" +%s 2>/dev/null || echo 0)
    if [ $((now - tw)) -le 1200 ] && [ $tw -gt $tc ]; then echo "$(basename $(dirname $f))"; return 0; fi
  done
  return 1
}
claim_card() {
  local what=$1; declare -A since
  local last_report=0
  while [ -z "$CARD" ]; do
    local now=$(date +%s) reasons=""
    if who=$(render_waiting); then
      reasons=" yield to $who;"; for g in 3 1 0 2; do since[$g]=""; done
    else
      for g in ${CARDS:-3 1 0 2}; do
        local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
        local util=$(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        local uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
        local other=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | while IFS=', ' read -r u p; do
                      o=$(ps -o user= -p $p 2>/dev/null); [ -n "$o" ] && [ "$o" != chayo ] && echo x; done | wc -l)
        local mine=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
        local why=""
        [ -d $L/gpu$g.lock ] && why="lock($(grep -m1 agent $L/gpu$g.lock/info 2>/dev/null))"
        [ "$used" -gt 2000 ] && why="$why used=${used}MiB"
        [ "$util" -gt 5 ] && why="$why util=${util}%"
        [ "$other" -gt 0 ] && why="$why other-user"
        [ "$mine" -gt 0 ] && why="$why apps=$mine"
        if [ -n "$why" ]; then since[$g]=""; reasons="$reasons gpu$g:$why;"; continue; fi
        [ -n "${since[$g]}" ] || since[$g]=$now
        if [ $((now - ${since[$g]})) -ge 600 ]; then
          sleep 90                                  # courtesy: a queued agent polling every 60 s goes first
          if [ -d $L/gpu$g.lock ] || render_waiting > /dev/null; then
            log "gpu$g eligible but taken or yielded during the courtesy wait"; since[$g]=""; continue; fi
          if mkdir $L/gpu$g.lock 2>/dev/null; then
            CARD=$g
            printf "agent D164\nrun d164/%s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
            log "claimed gpu$g for $what"
            break
          fi
        fi
      done
    fi
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
log "chain start pid $$ steps '$STEPS'"
if [[ " $STEPS " == *" P "* ]]; then
  claim_card pilot
  for mesh in bunny dragon; do
    name=${mesh}40k_probe97
    [ -e $O/pilot/$name.rc ] && [ "$(cat $O/pilot/$name.rc)" = 0 ] && { log "P $name done before, skipped"; continue; }
    [ -e $O/STOP ] && { log "STOP before $name"; exit 0; }
    log "P $name start"
    ( envr repo_d164s; $PY $W/replay_probe.py scripts/pipeline_run.py --tgt assets/$mesh.obj --n 40000 --seed 97 \
        --replay_noise start --out $O/pilot/$name > $O/pilot/$name.log 2>&1 < /dev/null; echo $? > $O/pilot/$name.rc )
    log "P $name exit $(cat $O/pilot/$name.rc): $(grep -c '^\[probe\]' $O/pilot/$name.log) probe lines; $(grep -m1 'ARM render_full_dt_iso_nn:' $O/pilot/$name.log | cut -c1-120)"
  done
  # the pre-registered stop rules, read mechanically before any gate: a STOP or an AMEND ends the chain here
  ( envr repo_d164; $PY $W/d164_pilot_read.py $O/pilot/bunny40k_probe97.log $O/pilot/dragon40k_probe97.log \
      > $O/pilot/pilot_read.txt 2>&1; echo $? > $O/pilot/pilot_read.rc )
  log "P read exit $(cat $O/pilot/pilot_read.rc): $(grep 'P0a\|stop rule' $O/pilot/pilot_read.txt | tr '\n' ' ' | cut -c1-400)"
  if [ "$(cat $O/pilot/pilot_read.rc)" != 0 ] || grep -q ': STOP\|: AMEND' $O/pilot/pilot_read.txt \
     || [ "$(cat $O/pilot/bunny40k_probe97.rc)" != 0 ] || [ "$(cat $O/pilot/dragon40k_probe97.rc)" != 0 ]; then
    log "P: a stop rule fired or a pilot run failed; the chain ends before the gates"; exit 1
  fi
fi
if [[ " $STEPS " == *" V "* ]]; then
  V=$O/gates; rm -f $V/PASSED $V/FAILED
  [ -n "$CARD" ] || claim_card gates
  ( envr repo_d164; $PY -m pytest -q -p no:cacheprovider tests > $V/suite.log 2>&1 < /dev/null; echo $? > $V/suite.rc )
  log "V1 suite exit $(cat $V/suite.rc): $(tail -1 $V/suite.log)"
  rep() {     # rep REPO TAG TEST: the test 20 times, -vv; prints the failure count
    local repo=$1 tag=$2 test=$3 fails=0 k
    for k in $(seq 1 20); do
      ( envr $repo; $PY -m pytest -vv -p no:cacheprovider "$test" > $V/rep_${tag}_$k.log 2>&1 < /dev/null; echo $? > $V/rep_${tag}_$k.rc )
      [ "$(cat $V/rep_${tag}_$k.rc)" = 0 ] || fails=$((fails + 1))
    done
    echo $fails
  }
  T1="tests/test_settled_contracts.py::test_null_windows_add_no_simulated_time_and_are_not_delivered"
  T2="tests/test_settled_contracts.py::test_a_nonfinite_window_cost_cannot_commit"
  T3="tests/test_volume_exact.py::test_the_commit_scores_the_accepted_candidate_on_its_own_discs"
  f1=$(rep repo_d164 nullwin "$T1"); echo $f1 > $V/v2_gate_fails.txt
  log "V2 gate: $T1 x20 on repo_d164: $f1 failures"
  log "V2 report: $T1 x20 on repo_d164_start: $(rep repo_d164_start nullwin_start "$T1") failures"
  log "V2 report: $T2 x20 on repo_d164: $(rep repo_d164 nonfinite "$T2") / repo_d164_start: $(rep repo_d164_start nonfinite_start "$T2") failures"
  log "V2 report: $T3 x20 on repo_d164: $(rep repo_d164 discs "$T3") / repo_d164_start: $(rep repo_d164_start discs_start "$T3") failures"
  ( envr repo_d164; $PY -m pytest -q -p no:cacheprovider tests/test_refreeze_reference.py tests/test_frozen_recipe.py \
      tests/test_d164_replay.py > $V/refs.log 2>&1 < /dev/null; echo $? > $V/refs.rc )
  log "V3 references and the switch's tests exit $(cat $V/refs.rc): $(tail -1 $V/refs.log)"
  for k in 1 2; do
    ( envr repo_d164s; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 --tgt assets/bunny.obj \
        --out $V/S$k > $V/S$k.log 2>&1 < /dev/null; echo $? > $V/S$k.rc )
  done
  ( envr repo_d164; $PY $B/tmp/d145/det_cmp12.py $V/S1 $V/S2 > $V/cmp_S1_S2.txt 2>&1
    $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $V/S1 > $V/cmp_rt1_S1.txt 2>&1 )
  log "V4 run to run (exit $(cat $V/S1.rc) $(cat $V/S2.rc)): $(tail -1 $V/cmp_S1_S2.txt); against rt1 S1: $(tail -1 $V/cmp_rt1_S1.txt)"
  ok=1
  [ "$(cat $V/suite.rc)" = 0 ] || ok=0
  [ "$f1" = 0 ] || ok=0
  [ "$(cat $V/refs.rc)" = 0 ] || ok=0
  tail -1 $V/cmp_S1_S2.txt | grep -q ": SAME$" || ok=0
  tail -1 $V/cmp_rt1_S1.txt | grep -q ": SAME$" || ok=0
  if [ $ok = 1 ]; then log "V gates passed"; touch $V/PASSED; else log "V STOP: a gate failed"; touch $V/FAILED; exit 1; fi
fi
runs() {     # runs LIST OUTDIR READ: the runs of LIST, skipping those whose kept frames exist
  local list=$1 od=$2 rd=$3 name flags
  [ -e $O/gates/PASSED ] || { log "runs refused: the gates have not passed"; exit 1; }
  while read -r name flags; do
    [ -z "$name" ] && continue
    case "$name" in \#*) continue;; esac
    if [ -e $od/${name}_frames12.npz ]; then log "R $name exists, skipped"; continue; fi
    [ -e $O/STOP ] && { log "STOP before $name"; exit 0; }
    [ -n "$CARD" ] || claim_card "runs $(basename $list)"
    log "R $name start ($flags)"
    ( envr repo_d164s; t0=$(date +%s); $PY scripts/pipeline_run.py $flags --out $od/$name > $od/$name.log 2>&1 < /dev/null
      echo $? > $od/$name.rc; echo $(( $(date +%s) - t0 )) > $od/$name.wall )
    log "R $name exit $(cat $od/$name.rc): $(grep -m1 'ARM render_full_dt_iso_nn:' $od/$name.log | cut -c1-150)"
    if [ "$rd" = 1 ] && [ "$(cat $od/$name.rc)" = 0 ]; then
      mesh=${name%%300k_*}
      bash $B/tmp/d145/read300.sh $CARD d164/runs300 $name $mesh end 1 > $od/read_$name.out 2>&1 < /dev/null
      log "R $name read300 (GPU part) done; CPU part detached"
    fi
  done < $list
}
[[ " $STEPS " == *" R40 "* ]] && runs $W/d164_runs40.txt $O/runs40 0 && log "R40 done"
[[ " $STEPS " == *" R300 "* ]] && runs $W/d164_runs300.txt $O/runs300 1 && log "R300 done"
log "chain done ($STEPS)"
