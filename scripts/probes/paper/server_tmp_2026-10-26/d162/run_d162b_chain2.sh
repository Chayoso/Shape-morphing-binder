#!/bin/bash
# d162b_chain.sh STEPS -- D162b (d162_chain.sh with output/gpu/d162b, repo_d162b / repo_d162bs, the shape smokes, R held until gates/GO) on hyde06, on ONE card (two chains may run, one card each) by the agents' lock protocol (gpu_locks/gpuN.lock by mkdir with an
# info file "agent D162 ...", released by the trap; a card with no lock, no other user's compute process, < 2000 MiB,
# idle 10 minutes by the pool's shared first-free time; GPU 2 never: another user's desktop). Courtesy: once a card is
# eligible the chain waits a further 90 s and claims it only if no other agent has (the queued agents poll every 60 s).
# The card is held only while a GPU step runs; CPU steps hold none.
#   V   the gates on repo_d162 / repo_d162bs: V1 the full suite (exit), V2 the pinned references and the merged options'
#       tests, V3 deterministic SAME at the defaults against rt1's S1, V4 four-window smoke runs of each new option
#       and their combinations with D159 / D160 / D161 (exit 0 and the log lines)
#   R   the runs of $RUNS_FILE (NAME FLAGS... per line), repo_d162bs (kept frames)
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d162b; W=$B/tmp/d162; L=$B/gpu_locks
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
A161=$G/d161/assets
if [[ " $STEPS " == *" V "* ]]; then
  V=$O/gates; rm -f $V/PASSED $V/FAILED
  claim_card gates
  ( envr repo_d162b; $PY -m pytest -q -p no:cacheprovider tests > $V/suite.log 2>&1 < /dev/null; echo $? > $V/suite.rc )
  log "V1 suite exit $(cat $V/suite.rc): $(tail -1 $V/suite.log)"
  ( envr repo_d162b; $PY -m pytest -q -p no:cacheprovider tests/test_refreeze_reference.py tests/test_d162.py \
      tests/test_render_tgt.py tests/test_proximity_berth.py tests/test_appearance.py tests/test_appearance_pipeline.py \
      tests/test_render_target_shell.py tests/test_frozen_recipe.py > $V/refs.log 2>&1 < /dev/null; echo $? > $V/refs.rc )
  log "V2 references and options' tests exit $(cat $V/refs.rc): $(tail -1 $V/refs.log)"
  ( envr repo_d162bs; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 --tgt assets/bunny.obj \
      --out $V/S1 > $V/S1.log 2>&1 < /dev/null; echo $? > $V/S1.rc )
  ( envr repo_d162b; $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $V/S1 > $V/cmp_rt1_S1.txt 2>&1 )
  log "V3 rt1 S1 vs repo_d162bs defaults (exit $(cat $V/S1.rc)): $(tail -1 $V/cmp_rt1_S1.txt)"
  smoke() { local name=$1; shift; ( envr repo_d162bs; $PY scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 \
      --out $V/$name "$@" > $V/$name.log 2>&1 < /dev/null; echo $? > $V/$name.rc )
    log "V4 $name ($*) exit $(cat $V/$name.rc): $(grep -m1 "b_floor" $V/$name.log | cut -c1-220)"; }
  smoke S_coin --tgt $A161/d161coin.obj --render_tgt $A161/d161L40coin.obj --render_weight_ref shape
  smoke S_bunny --tgt $A161/d161smooth_bunny.obj --render_tgt $A161/d161L40_bunny.obj --render_weight_ref shape
  smoke S_band --tgt assets/bunny.obj --appearance band --render_weight_ref shape
  smoke S_ears --tgt assets/bunny.obj --appearance ears --render_weight_ref shape
  ok=1
  [ "$(cat $V/suite.rc)" = 0 ] || ok=0
  [ "$(cat $V/refs.rc)" = 0 ] || ok=0
  tail -1 $V/cmp_rt1_S1.txt | grep -q ": SAME$" || ok=0
  for k in S_coin S_bunny S_band S_ears; do [ "$(cat $V/$k.rc)" = 0 ] || ok=0; grep -q "referenced to the shape (D162b)" $V/$k.log || ok=0; done
  if [ $ok = 1 ]; then log "V gates passed"; touch $O/gates/PASSED; else log "V STOP: a gate failed"; touch $O/gates/FAILED; fi
  :   # the card is kept for the next step; the EXIT trap releases it
fi
# T: the contract test that failed in the gates, TREPS times on each repo of TREPOS (the merge, its parts and the tag), with
#    the full stats printed (-vv), to tell a merge interaction from a flaky test before changing anything; each failure's
#    null_reason goes to the status line
if [[ " $STEPS " == *" T "* ]]; then
  [ -n "$CARD" ] || claim_card diag
  TEST=${TTEST:-"tests/test_settled_contracts.py::test_null_windows_add_no_simulated_time_and_are_not_delivered"}
  for repo in ${TREPOS:-repo_d162 repo_rt1 repo_d159 repo_d160 repo_d161}; do
    exits=""; why=""
    for k in $(seq 1 ${TREPS:-5}); do
      ( envr $repo; $PY -m pytest -vv -p no:cacheprovider "$TEST" > $O/gates/diag${TTAG}_${repo}_$k.log 2>&1 < /dev/null
        echo $? > $O/gates/diag${TTAG}_${repo}_$k.rc )
      rc=$(cat $O/gates/diag${TTAG}_${repo}_$k.rc); exits="$exits$rc "
      [ "$rc" = 0 ] || why="$why $(grep -o "'null_reason': [^,]*" $O/gates/diag${TTAG}_${repo}_$k.log | head -1)"
    done
    log "T$TTAG $repo: exits $exits| failures:$why"
  done
  :   # the card is kept for the next step; the EXIT trap releases it
fi
if [[ " $STEPS " == *" R "* ]]; then
  # D162b: the runs start on gates/GO (written with the measured cost's amendment), or end on gates/ABORT; the card is kept
  # meanwhile. A failed gate does not end the chain: it waits for the agent's decision, recorded in the amendment.
  until [ -e $O/gates/GO ] || [ -e $O/gates/ABORT ]; do sleep 30; done
  [ -e $O/gates/ABORT ] && { log "R aborted (gates/ABORT)"; exit 1; }
  [ -e $O/gates/PASSED ] || log "R on gates/GO with a failed gate: see the amendment"
  while read -r name flags; do
    [ -z "$name" ] && continue
    case "$name" in \#*) continue;; esac
    if [ -e $O/runs40/${name}_frames12.npz ]; then log "R $name exists, skipped"; continue; fi
    [ -n "$CARD" ] || claim_card "runs40 ${RUNS_FILE:-}"
    log "R $name start ($flags)"
    ( envr repo_d162bs; $PY scripts/pipeline_run.py $flags --out $O/runs40/$name > $O/runs40/$name.log 2>&1 < /dev/null
      echo $? > $O/runs40/$name.rc )
    log "R $name exit $(cat $O/runs40/$name.rc): $(grep -m1 'ARM render_full_dt_iso_nn:' $O/runs40/$name.log | cut -c1-150)"
  done < $W/${RUNS_FILE:-d162_runs.txt}
  :   # the card is kept for X; the EXIT trap releases it
  log "R done (${RUNS_FILE:-d162_runs.txt})"
fi
# X: arrangement vs shape (the coordinator's request after D156h2): per kept frame of XRUNS (default: the coin relief and
#    the colour band, seed 97, arm F and its twin TC, and coin FC), the render term on the state and on uniform resamples of
#    its solid (d162_arrange.py), on the card this chain holds; the solids and their samples are scratch, deleted by the
#    reader. XFILE / XDIR / XOUT: the runs' flags, their archives and the output prefix (the smoke reads the V4 runs)
if [[ " $STEPS " == *" X "* ]]; then
  [ -n "$CARD" ] || claim_card arrange
  mkdir -p $O/after
  XO=${XOUT:-$O/after/arrange}
  ( envr repo_d162b; $PY $W/d162_arrange.py $XO.jsonl ${XFILE:-$W/d162_runs_A1.txt} ${XDIR:-$O/runs40} \
      ${XRUNS:-coin40k_F_97 coin40k_TC_97 bunny40k_band_F_97 bunny40k_band_TC_97 coin40k_FC_97} \
      > $XO.log 2>&1 < /dev/null; echo $? > $XO.rc
    $PY $W/d162_arrange_sum.py $XO.md $XO.jsonl > /dev/null 2>&1 )
  log "X arrange exit $(cat $XO.rc): $(grep -c '"state": "frame"' $XO.jsonl) frames read ($XO)"
  :   # the EXIT trap releases the card
fi
log "chain done ($STEPS)"
