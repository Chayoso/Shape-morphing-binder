#!/bin/bash
# d160_chain.sh STEPS -- D160 (agent D160): the gates, the 40k prototype, its readers and its videos on ONE card.
# Lock protocol: a card among 0 / 1 / 3 (GPU 2 never: another user's desktop session) with no lock, no other user's
# compute process, < 2000 MiB and < 5 % use at TWELVE one-minute samples in a row (two more than the queued agents' ten,
# so that a card freed while one of them waits goes to it first) is claimed by mkdir with an info file and released by
# the trap.
#   V  the gates, in order; a failure stops the chain before any run:
#      V1 the full suite on repo_d160 (exit status), V2 the pinned references and D160's tests (exit status),
#      V3 Warp run-to-run deterministic 40k bunny seed 97 four windows on repo_d160s: the defaults (S1) and
#         --appearance white (WH) each against rt1's S1 (output/gpu/d153/rt1/S1), every field: SAME;
#      V4 the twin at lambda = 0 without a paint (T0) and with the band paint (TB), four windows: the trajectory SAME;
#         the band paint at the rule's weight (LB): exit 0, its appearance line, and a trajectory unlike S1's;
#   R  the 40k runs (repo_d160s): bunny40k_SCENE_{L,P}SEED for SCENE in half band ears turn and dragon40k_band_{L,P}SEED,
#      L = --appearance SCENE (dF_c + layer u + render loss), P = the same with --render_weight_scale 0 (dF_c + layer
#      u, the lambda = 0 twin); seed 97 first, then 99; each followed by D155's card readers (end frame, the yardstick
#      render_terms_probe against the independent sample, the display surface_layer_probe, gallery_ind);
#   C  d160_read.py over every run (the colour readings, the travel against the twin, the timelines), on the card;
#   M  d160_studio.py for the seed-97 pairs: the side-by-side videos and the end sheets.
# Usage: d160_chain.sh STEPS   (e.g. "V R C M")
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d160; W=$B/tmp/d160; T=$B/tmp; L=$B/gpu_locks
mkdir -p $O/runs40 $O/gates $O/video $W/tmpdir
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
STEPS=${1:-"V R C M"}
CARD=""
claim_card() {
  local what=$1; declare -A idle; local last=0
  while [ -z "$CARD" ]; do
    local now=$(date +%s) reasons=""
    for g in 3 1 0; do
      if [ -d $L/gpu$g.lock ]; then idle[$g]=0; reasons="$reasons gpu$g:lock($(grep -m1 agent $L/gpu$g.lock/info 2>/dev/null));"; continue; fi
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      local util=$(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader,nounits)
      local napps=0
      for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader); do
        [ "$(ps -o user= -p $p 2>/dev/null | tr -d ' ')" = chayo ] || napps=$((napps + 1)); done
      if [ "$used" -lt 2000 ] && [ "$util" -lt 5 ] && [ "$napps" -eq 0 ]; then idle[$g]=$(( ${idle[$g]:-0} + 1 ))
      else idle[$g]=0; reasons="$reasons gpu$g:used=${used}MiB,util=${util}%,others=$napps;"; fi
      if [ "${idle[$g]}" -ge 12 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        printf "agent D160\nrun d160/%s\npid %s\nstart %s\n" "$what" "$$" "$(date '+%Y-%m-%d %H:%M:%S %Z')" > $L/gpu$g.lock/info
        CARD=$g; break
      fi
    done
    if [ -z "$CARD" ] && [ $((now - last)) -ge 1800 ]; then log "waiting:$reasons"; last=$now; fi
    [ -z "$CARD" ] && sleep 60
  done
  export CUDA_VISIBLE_DEVICES=$CARD
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "released gpu$CARD"; CARD=""; }
trap 'release_card' EXIT
trap 'exit 1' INT TERM HUP
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1
         export TMPDIR=$W/tmpdir; cd $B/$1; }
fail() { log "STOP: $*"; exit 1; }
log "chain start pid $$ steps [$STEPS] (waiting for a card idle 12 min)"
claim_card "chain [$STEPS]: gates, 40k colour runs (L/P), readers, videos"
log "claimed gpu$CARD"

if [[ " $STEPS " == *" V "* ]]; then
  V=$O/gates
  ( envr repo_d160; $PY -m pytest -q -p no:cacheprovider tests > $V/suite.log 2>&1 < /dev/null; echo $? > $V/suite.rc )
  log "V1 suite repo_d160 exit $(cat $V/suite.rc): $(tail -1 $V/suite.log)"
  [ "$(cat $V/suite.rc)" = 0 ] || fail "V1 suite"
  ( envr repo_d160; $PY -m pytest -q -p no:cacheprovider tests/test_refreeze_reference.py tests/test_appearance.py \
      tests/test_appearance_pipeline.py > $V/pinned.log 2>&1 < /dev/null; echo $? > $V/pinned.rc )
  log "V2 pinned references + D160 tests exit $(cat $V/pinned.rc): $(tail -1 $V/pinned.log)"
  [ "$(cat $V/pinned.rc)" = 0 ] || fail "V2 pinned"
  det() { local name=$1; shift; ( envr repo_d160s; $PY $T/rt/det_run.py scripts/pipeline_run.py --tgt assets/bunny.obj \
      --n 40000 --seed 97 --animations 4 --out $V/$name "$@" > $V/$name.log 2>&1 < /dev/null; echo $? > $V/$name.rc )
      log "V det $name ($*) exit $(cat $V/$name.rc)"; [ "$(cat $V/$name.rc)" = 0 ] || fail "det $name exit"; }
  cmpd() { ( envr repo_d160; $PY $W/det_cmp_d160.py $1 $2 $3 > $4 2>&1 ); log "V cmp $1 $(basename $2) vs $(basename $3): $(tail -1 $4)"; }
  det S1
  det WH --appearance white
  cmpd full $G/d153/rt1/S1 $V/S1 $V/cmp_rt1_S1.txt
  cmpd full $G/d153/rt1/S1 $V/WH $V/cmp_rt1_WH.txt
  tail -1 $V/cmp_rt1_S1.txt | grep -q ": SAME$" || fail "V3 the default path is not rt1's"
  tail -1 $V/cmp_rt1_WH.txt | grep -q ": SAME$" || fail "V3 white is not rt1's"
  det T0 --render_weight_scale 0
  det TB --render_weight_scale 0 --appearance band
  det LB --appearance band
  cmpd traj $V/T0 $V/TB $V/cmp_T0_TB.txt
  cmpd traj $V/S1 $V/LB $V/cmp_S1_LB.txt
  tail -1 $V/cmp_T0_TB.txt | grep -q ": SAME$" || fail "V4 the twin reads the paint"
  tail -1 $V/cmp_S1_LB.txt | grep -q ": SAME$" && fail "V4 the paint does not act"
  grep -q "appearance (D160): scene band" $V/LB.log || fail "V4 no appearance line"
  log "V4: $(grep -m1 'appearance (D160)' $V/LB.log | cut -c1-260)"
  log "V gates passed"
fi

box_of() { case $1 in bunny) echo 1900,100,2800,700;; dragon) echo 1900,60,2500,560;; *) echo 0,0,3840,2160;; esac; }   # D155's
run() {   # NAME MESH SEED FLAGS...
  local name=$1 mesh=$2 seed=$3; shift 3
  local F=$O/runs40/${name}_frames12.npz IND=$G/d92/ind_${mesh}.npz
  if [ -e $O/runs40/rc_$name ] && [ "$(cat $O/runs40/rc_$name)" = 0 ]; then log "R $name exists, skipped"; return; fi
  [ -e $O/STOP ] && { log "stopped by STOP before $name"; exit 0; }
  log "R $name start ($mesh 40k seed $seed $*)"
  ( envr repo_d160s; t0=$(date +%s)
    $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n 40000 --seed $seed --out $O/runs40/$name "$@" \
        > $O/runs40/$name.log 2>&1 < /dev/null
    echo $? > $O/runs40/rc_$name; echo $(( $(date +%s) - t0 )) > $O/runs40/wall_$name )
  log "R $name exit $(cat $O/runs40/rc_$name) wall $(cat $O/runs40/wall_$name) s: $(grep -m1 'ARM render_full_dt_iso_nn:' $O/runs40/$name.log | cut -c1-170)"
  [ "$(cat $O/runs40/rc_$name)" = 0 ] || return
  ( source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; export TMPDIR=$W/tmpdir; cd $B/repo_r77
    last=$($PY $T/lastraw.py $F)
    $PY $T/d145/endframe.py $F $O/runs40/${name}_end1.npz
    $PY scripts/probes/settled/render_terms_probe.py $O/runs40/${name}_end1.npz ref=$IND > $O/runs40/t3end_$name.log 2>&1 < /dev/null
    $PY scripts/probes/settled/surface_layer_probe.py $F $O/runs40/e3end_$name $(box_of $mesh) measure ref=$IND target own $last \
        > $O/runs40/e3end_$name.log 2>&1 < /dev/null
    $PY $T/gallery_ind.py $O/runs40/ind.log "$name=$F,$IND" > $O/runs40/gind_$name.log 2>&1 < /dev/null
    rm -f $O/runs40/${name}_end1.npz )
  log "R $name card readers done"
}
if [[ " $STEPS " == *" R "* ]]; then
  for seed in 97 99; do
    for sc in half band ears turn; do
      run bunny40k_${sc}_L$seed bunny $seed --appearance $sc
      run bunny40k_${sc}_P$seed bunny $seed --appearance $sc --render_weight_scale 0
    done
    run dragon40k_band_L$seed dragon $seed --appearance band
    run dragon40k_band_P$seed dragon $seed --appearance band --render_weight_scale 0
  done
  log "R done"
fi

if [[ " $STEPS " == *" C "* ]]; then
  J=$O/jobs_read.txt; : > $J
  for seed in 97 99; do
    for mm in "bunny half" "bunny band" "bunny ears" "bunny turn" "dragon band"; do
      set -- $mm
      P=$O/runs40/${1}40k_${2}_P${seed}_frames12.npz; Lr=$O/runs40/${1}40k_${2}_L${seed}_frames12.npz
      [ -e $P ] && echo "${1}40k_${2}_P$seed $2 $P timeline" >> $J
      [ -e $Lr ] && echo "${1}40k_${2}_L$seed $2 $Lr twin=${1}40k_${2}_P$seed timeline" >> $J
    done
  done
  ( envr repo_d160; export PYTHONPATH=$B/deps/d157_lpips:$PYTHONPATH; rm -f $O/read_colour.jsonl
    $PY $W/d160_read.py $O/read_colour.jsonl $J > $O/read_colour.log 2>&1 < /dev/null; echo $? > $O/read_colour.rc )
  log "C colour readings exit $(cat $O/read_colour.rc): $(wc -l < $O/read_colour.jsonl) rows"
fi

if [[ " $STEPS " == *" M "* ]]; then
  for mm in "bunny half" "bunny band" "bunny ears" "bunny turn" "dragon band"; do
    set -- $mm
    Lr=$O/runs40/${1}40k_${2}_L97_frames12.npz; P=$O/runs40/${1}40k_${2}_P97_frames12.npz
    [ -e $Lr ] && [ -e $P ] || continue
    ( envr repo_d160; $PY $W/d160_studio.py $O/video/${1}_${2}_97 $2 $Lr $P $G/d92/ind_${1}.npz \
        > $O/video/${1}_${2}_97.log 2>&1 < /dev/null; echo $? > $O/video/${1}_${2}_97.rc )
    log "M ${1}_${2}_97 exit $(cat $O/video/${1}_${2}_97.rc)"
  done
fi
log "chain done"
