#!/bin/bash
# d156_chain.sh -- D156: the gates and the 40k runs, on ONE card by the agents' lock protocol (gpu_locks/gpuN.lock by
# mkdir with an info file, released by the trap; a card with no lock, no other user's compute process, < 2000 MiB, idle
# for 10 minutes by the pool's shared first-free time; GPU 2 never: another user's desktop session).
#   V  the gates, in order; any failure stops the chain before a run:
#      V1 full suite on repo_d156 (exit status), V2 the pinned digest + reference (test_refreeze_reference.py),
#      V3 Warp run-to-run deterministic 40k bunny seed 97 four windows (tmp/rt/det_run.py), repo_d156s: at the defaults
#         against rt1's S1 (output/gpu/d153/rt1/S1; det_cmp12: SAME), and with --render_tgt assets/bunny.obj (the
#         target's own mesh) against the same S1 (SAME);
#      V4 the two D156 arms four windows, deterministic (bunny: --tgt HULL --render_tgt assets/bunny.obj, and the same
#         with --render_weight_scale 0): exit 0 and the render-target log line;
#   R  the 40k runs (repo_d156s, kept frames): MESH40k_L_SEED (--tgt HULL --render_tgt assets/MESH.obj) and MESH40k_P_SEED
#      (+ --render_weight_scale 0), seed 97 for every mesh first, then seed 99.
# Usage: d156_chain.sh STEPS   (e.g. "V R")
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d156; W=$B/tmp/d156; L=$B/gpu_locks; mkdir -p $O/runs40 $O/gates
H=$O/assets_hull
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir; mkdir -p $TMPDIR
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
STEPS=${1:-"V R"}
MESHES=${D156_MESHES:-"bunny dragon armadilo homer cow bob cheburashka C A"}
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
      if [ -n "$why" ]; then [ -n "${since[$g]}" ] && log "gpu$g busy again: $why"; since[$g]=""; reasons="$reasons gpu$g:$why;"; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      local fs=$(cat $B/tmp/d145/free_since/gpu$g 2>/dev/null); [ -n "$fs" ] && [ "$fs" -lt "${since[$g]}" ] && since[$g]=$fs
      if [ $((now - ${since[$g]})) -ge 600 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        CARD=$g
        printf "agent D156\nrun d156/%s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        break
      fi
    done
    if [ -z "$CARD" ] && [ $((now - last_report)) -ge 900 ]; then
      log "waiting:$reasons free:$(for k in "${!since[@]}"; do [ -n "${since[$k]}" ] && echo -n " gpu$k since $(date -d @${since[$k]} +%H:%M:%S)"; done)"
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
fail() { log "STOP: $*"; exit 1; }
log "chain start pid $$ steps [$STEPS] meshes [$MESHES] (waiting for a card idle 10 min)"
claim_card "chain [$STEPS]: gates, 40k L/P on the visual hull with the true mesh's pictures"
log "claimed gpu$CARD"
if [[ " $STEPS " == *" V "* ]]; then
  V=$O/gates
  ( envr repo_d156; $PY -m pytest -q -p no:cacheprovider tests > $V/suite.log 2>&1 < /dev/null; echo $? > $V/suite.rc )
  log "V1 suite repo_d156 exit $(cat $V/suite.rc): $(tail -1 $V/suite.log)"
  [ "$(cat $V/suite.rc)" = 0 ] || fail "V1 suite"
  ( envr repo_d156; $PY -m pytest -q -p no:cacheprovider tests/test_refreeze_reference.py > $V/digest.log 2>&1 < /dev/null
    echo $? > $V/digest.rc )
  log "V2 pinned digest + reference exit $(cat $V/digest.rc): $(tail -1 $V/digest.log)"
  [ "$(cat $V/digest.rc)" = 0 ] || fail "V2 digest"
  det() { local name=$1; shift; ( envr repo_d156s; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 \
      --animations 4 --out $V/$name "$@" > $V/$name.log 2>&1 < /dev/null; echo $? > $V/$name.rc ); log "V det $name ($*) exit $(cat $V/$name.rc)"; }
  det S1 --tgt assets/bunny.obj
  det R1 --tgt assets/bunny.obj --render_tgt assets/bunny.obj
  ( envr repo_d156; $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $V/S1 > $V/cmp_rt1_S1.txt 2>&1
    $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $V/R1 > $V/cmp_rt1_R1.txt 2>&1 )
  log "V3 rt1 S1 vs repo_d156s defaults: $(tail -1 $V/cmp_rt1_S1.txt)"
  log "V3 rt1 S1 vs --render_tgt = the target's own mesh: $(tail -1 $V/cmp_rt1_R1.txt)"
  tail -1 $V/cmp_rt1_S1.txt | grep -q ": SAME$" || fail "V3 default path"
  tail -1 $V/cmp_rt1_R1.txt | grep -q ": SAME$" || fail "V3 render target = target"
  det HL --tgt $H/bunny_hull18_r96.obj --render_tgt assets/bunny.obj
  det HP --tgt $H/bunny_hull18_r96.obj --render_tgt assets/bunny.obj --render_weight_scale 0
  for k in HL HP; do
    [ "$(cat $V/$k.rc)" = 0 ] || fail "V4 $k exit"
    grep -q "render target (D156)" $V/$k.log || fail "V4 $k: no render-target line"
  done
  log "V4 the arms run: $(grep -m1 'render target (D156)' $V/HL.log | cut -c1-200)"
  log "V gates passed"
fi
run() {   # NAME MESH SEED FLAGS...
  local name=$1 mesh=$2 seed=$3; shift 3
  if [ -e $O/runs40/${name}_frames12.npz ]; then log "R $name exists, skipped"; return; fi
  log "R $name start ($mesh 40k seed $seed $*)"
  ( envr repo_d156s; $PY scripts/pipeline_run.py --tgt $H/${mesh}_hull18_r96.obj --render_tgt assets/$mesh.obj --n 40000 \
      --seed $seed --out $O/runs40/$name "$@" > $O/runs40/$name.log 2>&1 < /dev/null; echo $? > $O/runs40/$name.rc )
  log "R $name exit $(cat $O/runs40/$name.rc): $(grep -m1 'ARM render_full_dt_iso_nn:' $O/runs40/$name.log | cut -c1-160)"
}
if [[ " $STEPS " == *" R "* ]]; then
  for seed in 97 99; do
    for mesh in $MESHES; do
      run ${mesh}40k_L_$seed $mesh $seed
      run ${mesh}40k_P_$seed $mesh $seed --render_weight_scale 0
    done
  done
  log "R done"
fi
log "chain done"
