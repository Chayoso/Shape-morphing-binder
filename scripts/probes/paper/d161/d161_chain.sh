#!/bin/bash
# d161_chain.sh STEPS -- D161 on hyde06, on ONE card by the agents' lock protocol (gpu_locks/gpuN.lock by mkdir with an
# info file "agent D161 ...", released by the trap; a card with no lock, no other user's compute process, < 2000 MiB,
# idle 10 minutes by the pool's shared first-free time; GPU 2 never: another user's desktop). Courtesy: once a card is
# eligible the chain waits a further 90 s and claims it only if no other agent has (the queued agents poll every 60 s).
# The card is held only while a GPU step runs; CPU steps hold none.
#   C40 / C300  the response curve (d161_curve.py): sphere and bunny at 40k / at 300k (a claim each, released after)
#   R   the 40k prototype runs (d161_runs.txt: NAME FLAGS... per line), repo_d161s (kept frames)
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d161; W=$B/tmp/d161; L=$B/gpu_locks
mkdir -p $O/curve $O/runs40 $W/tmpdir
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
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1; export TMPDIR=$W/tmpdir; cd $B/$1; }
log "chain start pid $$ steps '$STEPS'"
LAM40=$G/d155/gate/bunny40k_V97.json
LAM300=$G/d145t_b1/bunny300k_L97.json
for C in C40 C300; do
  [[ " $STEPS " == *" $C "* ]] || continue
  if [ $C = C40 ]; then JOBS=("coin 40000 $LAM40" "bunny 40000 $LAM40" "sphere 40000 $LAM40"); else JOBS=("coin 300000 $LAM300" "bunny 300000 $LAM300"); fi
  claim_card curve_$C
  for job in "${JOBS[@]}"; do
    set -- $job
    tag=${1}$(( $2 / 1000 ))k
    { [ -e $O/curve/$tag.done ] || [ -e $O/curve/${tag}_cpu.done ]; } && { log "C $tag exists, skipped"; continue; }
    log "C $tag start"
    ( envr repo_d161; $PY $W/d161_curve.py $O/curve/$tag.jsonl $1 $2 $3 > $O/curve/$tag.log 2>&1 < /dev/null
      echo $? > $O/curve/$tag.rc )
    rc=$(cat $O/curve/$tag.rc); log "C $tag exit $rc: $(tail -1 $O/curve/$tag.log | cut -c1-160)"
    [ "$rc" = 0 ] && touch $O/curve/$tag.done
  done
  release_card
done
if [[ " $STEPS " == *" R "* ]]; then
  while read -r name flags; do
    [ -z "$name" ] && continue
    case "$name" in \#*) continue;; esac
    if [ -e $O/runs40/${name}_frames12.npz ]; then log "R $name exists, skipped"; continue; fi
    [ -n "$CARD" ] || claim_card runs40
    log "R $name start ($flags)"
    ( envr repo_d161s; $PY scripts/pipeline_run.py $flags --out $O/runs40/$name > $O/runs40/$name.log 2>&1 < /dev/null
      echo $? > $O/runs40/$name.rc )
    log "R $name exit $(cat $O/runs40/$name.rc): $(grep -m1 'ARM render_full_dt_iso_nn:' $O/runs40/$name.log | cut -c1-160)"
  done < $W/${RUNS_FILE:-d161_runs.txt}
  release_card
  log "R done (${RUNS_FILE:-d161_runs.txt})"
fi
# B: D161b (pre-registered 2026-10-21 22:51 CDT with D161): the gates on repo_d161b / repo_d161bs (wip/d161 with
# --proximity_berth, default off), then the eight 40k runs of d161_runs.txt with --proximity_berth (NAME_pb)
if [[ " $STEPS " == *" B "* ]]; then
  V=$O/gates_b; mkdir -p $V
  [ -n "$CARD" ] || claim_card d161b
  ( envr repo_d161b; $PY -m pytest -q -p no:cacheprovider tests/test_proximity_berth.py tests/test_render_tgt.py > $V/tests.log 2>&1 < /dev/null; echo $? > $V/tests.rc )
  log "B V1 tests exit $(cat $V/tests.rc): $(tail -1 $V/tests.log)"
  det() { local name=$1; shift; ( envr repo_d161bs; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 \
      --animations 4 --out $V/$name "$@" > $V/$name.log 2>&1 < /dev/null; echo $? > $V/$name.rc ); log "B det $name ($*) exit $(cat $V/$name.rc)"; }
  det S1 --tgt assets/bunny.obj
  det PB --tgt assets/bunny.obj --proximity_berth
  ( envr repo_d161b; $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $V/S1 > $V/cmp_rt1_S1.txt 2>&1 )
  log "B V2 rt1 S1 vs repo_d161bs defaults: $(tail -1 $V/cmp_rt1_S1.txt)"
  if [ "$(cat $V/tests.rc)" = 0 ] && tail -1 $V/cmp_rt1_S1.txt | grep -q ": SAME$" && [ "$(cat $V/PB.rc)" = 0 ]; then
    while read -r name flags; do
      [ -z "$name" ] && continue
      case "$name" in \#*) continue;; esac
      if [ -e $O/runs40/${name}_pb_frames12.npz ]; then log "RB ${name}_pb exists, skipped"; continue; fi
      log "RB ${name}_pb start"
      ( envr repo_d161bs; $PY scripts/pipeline_run.py $flags --proximity_berth --out $O/runs40/${name}_pb > $O/runs40/${name}_pb.log 2>&1 < /dev/null
        echo $? > $O/runs40/${name}_pb.rc )
      log "RB ${name}_pb exit $(cat $O/runs40/${name}_pb.rc): $(grep -m1 'ARM render_full_dt_iso_nn:' $O/runs40/${name}_pb.log | cut -c1-160)"
    done < $W/d161_runs.txt
    log "RB done"
  else
    log "B STOP: a gate failed; no D161b run"
  fi
  release_card
fi
log "chain done ($STEPS)"
