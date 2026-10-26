#!/bin/bash
# d160d_chain.sh -- D160d (agent D160; pre-registered in docs/experiments.md 02:51 CDT): the render terms on the rule's
# weight, the colour term on the render weight's first calibrated value held (--appearance_weight held), 40k, D160's
# five scenes x seeds 97 / 99, on ONE card by the lock protocol (3 / 1 / 0, never 2; no lock, no other user's process,
# < 2000 MiB, < 5 % for ten one-minute samples; mkdir + info; trap). The twins are D160's P runs.
# Card: V the gates; R the runs + D155's card readers; C d160_read.py (D with their P twins, timelines) and
# d160_surfdisp.py (D / H / L against P, and the floor); M the seed-97 videos. Then CPU (card released): pieces145,
# D155's gate reader, D157's reader on the D ends, d160_gate.py (arm D).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d160; W=$B/tmp/d160; T=$B/tmp; L=$B/gpu_locks
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') [D160d] $*" >> $S; }
CARD=""
declare -A idle
log "start pid $$ (waiting for a card idle 10 min)"
while [ -z "$CARD" ]; do
  for g in 3 1 0; do
    if [ -d $L/gpu$g.lock ]; then idle[$g]=0; continue; fi
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    util=$(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    napps=0
    for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader); do
      [ "$(ps -o user= -p $p 2>/dev/null | tr -d ' ')" = chayo ] || napps=$((napps + 1)); done
    if [ "$used" -lt 2000 ] && [ "$util" -lt 5 ] && [ "$napps" -eq 0 ]; then idle[$g]=$(( ${idle[$g]:-0} + 1 )); else idle[$g]=0; fi
    if [ "${idle[$g]}" -ge 10 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
      printf "agent D160\nrun d160d: gates + 10 x 40k (colour weight held) + readers + videos (~70 min)\npid %s\nstart %s\n" \
          "$$" "$(date '+%Y-%m-%d %H:%M:%S %Z')" > $L/gpu$g.lock/info
      CARD=$g; break
    fi
  done
  [ -z "$CARD" ] && sleep 60
done
release() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "released gpu$CARD"; CARD=""; }
trap 'release' EXIT
trap 'exit 1' INT TERM HUP
export CUDA_VISIBLE_DEVICES=$CARD
log "claimed gpu$CARD"
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1
         export TMPDIR=$W/tmpdir; cd $B/$1; }
fail() { log "STOP: $*"; exit 1; }

# ---- V: the gates ----
V=$O/gates_d; mkdir -p $V
( envr repo_d160; $PY -m pytest -q -p no:cacheprovider tests > $V/suite.log 2>&1 < /dev/null; echo $? > $V/suite.rc )
log "V1 suite exit $(cat $V/suite.rc): $(tail -1 $V/suite.log)"
[ "$(cat $V/suite.rc)" = 0 ] || fail "V1 suite"
( envr repo_d160; $PY -m pytest -q -p no:cacheprovider tests/test_refreeze_reference.py tests/test_appearance.py \
    tests/test_appearance_pipeline.py > $V/pinned.log 2>&1 < /dev/null; echo $? > $V/pinned.rc )
log "V2 pinned + D160 tests exit $(cat $V/pinned.rc): $(tail -1 $V/pinned.log)"
[ "$(cat $V/pinned.rc)" = 0 ] || fail "V2 pinned"
det() { local name=$1; shift; ( envr repo_d160s; $PY $T/rt/det_run.py scripts/pipeline_run.py --tgt assets/bunny.obj \
    --n 40000 --seed 97 --animations 4 --out $V/$name "$@" > $V/$name.log 2>&1 < /dev/null; echo $? > $V/$name.rc )
    log "V det $name ($*) exit $(cat $V/$name.rc)"; [ "$(cat $V/$name.rc)" = 0 ] || fail "det $name exit"; }
cmpd() { ( envr repo_d160; $PY $W/det_cmp_d160.py $1 $2 $3 > $4 2>&1 ); log "V cmp $1 $(basename $2) vs $(basename $3): $(tail -1 $4)"; }
det S1
det WH --appearance white
cmpd full $G/d153/rt1/S1 $V/S1 $V/cmp_rt1_S1.txt
cmpd full $G/d153/rt1/S1 $V/WH $V/cmp_rt1_WH.txt
tail -1 $V/cmp_rt1_S1.txt | grep -q ": SAME$" || fail "V3 the defaults are not rt1's"
tail -1 $V/cmp_rt1_WH.txt | grep -q ": SAME$" || fail "V3 white is not rt1's"
det T0 --render_weight_scale 0
det TH --render_weight_scale 0 --appearance band --appearance_weight held
det DH --appearance band --appearance_weight held
cmpd traj $V/T0 $V/TH $V/cmp_T0_TH.txt
cmpd traj $V/S1 $V/DH $V/cmp_S1_DH.txt
tail -1 $V/cmp_T0_TH.txt | grep -q ": SAME$" || fail "V4 the held twin reads the paint"
tail -1 $V/cmp_S1_DH.txt | grep -q ": SAME$" && fail "V4 the held colour does not act"
grep -q "\[D160d\] the colour term" $V/DH.log || fail "V4 no D160d line"
log "V4: $(grep -m1 '\[D160d\] the colour term' $V/DH.log | cut -c1-160)"
rm -f $V/*_frames12.npz
log "V gates passed"

# ---- R: the runs ----
box_of() { case $1 in bunny) echo 1900,100,2800,700;; dragon) echo 1900,60,2500,560;; *) echo 0,0,3840,2160;; esac; }
run() {   # NAME MESH SEED FLAGS...
  local name=$1 mesh=$2 seed=$3; shift 3
  local F=$O/runs40/${name}_frames12.npz IND=$G/d92/ind_${mesh}.npz
  if [ -e $O/runs40/rc_$name ] && [ "$(cat $O/runs40/rc_$name)" = 0 ]; then log "R $name exists, skipped"; return; fi
  ( envr repo_d160s; t0=$(date +%s)
    $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n 40000 --seed $seed --out $O/runs40/$name "$@" \
        > $O/runs40/$name.log 2>&1 < /dev/null
    echo $? > $O/runs40/rc_$name; echo $(( $(date +%s) - t0 )) > $O/runs40/wall_$name )
  log "R $name exit $(cat $O/runs40/rc_$name) wall $(cat $O/runs40/wall_$name) s: $(grep -m1 'ARM render_full_dt_iso_nn:' $O/runs40/$name.log | cut -c1-150)"
  [ "$(cat $O/runs40/rc_$name)" = 0 ] || return
  ( source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; export TMPDIR=$W/tmpdir; cd $B/repo_r77
    last=$($PY $T/lastraw.py $F)
    $PY $T/d145/endframe.py $F $O/runs40/${name}_end1.npz
    $PY scripts/probes/settled/render_terms_probe.py $O/runs40/${name}_end1.npz ref=$IND > $O/runs40/t3end_$name.log 2>&1 < /dev/null
    $PY scripts/probes/settled/surface_layer_probe.py $F $O/runs40/e3end_$name $(box_of $mesh) measure ref=$IND target own $last \
        > $O/runs40/e3end_$name.log 2>&1 < /dev/null
    $PY $T/gallery_ind.py $O/runs40/ind.log "$name=$F,$IND" > $O/runs40/gind_$name.log 2>&1 < /dev/null
    rm -f $O/runs40/${name}_end1.npz )
}
SC="bunny:half bunny:band bunny:ears bunny:turn dragon:band"
for seed in 97 99; do
  for ms in $SC; do m=${ms%%:*}; s=${ms##*:}; run ${m}40k_${s}_D$seed $m $seed --appearance $s --appearance_weight held; done
done
log "R done"

# ---- C: the colour readings and the surface reading ----
J=$O/jobs_read_D.txt; : > $J
for seed in 97 99; do
  for ms in $SC; do m=${ms%%:*}; s=${ms##*:}
    P=$O/runs40/${m}40k_${s}_P${seed}_frames12.npz; D=$O/runs40/${m}40k_${s}_D${seed}_frames12.npz
    [ -e $P ] && [ -e $D ] && { echo "${m}40k_${s}_P$seed $s $P" >> $J; echo "${m}40k_${s}_D$seed $s $D twin=${m}40k_${s}_P$seed timeline" >> $J; }
  done
done
( envr repo_d160; export PYTHONPATH=$B/deps/d157_lpips:$PYTHONPATH; rm -f $O/read_colour_D.jsonl
  $PY $W/d160_read.py $O/read_colour_D.jsonl $J > $O/read_colour_D.log 2>&1 < /dev/null; echo $? > $O/read_colour_D.rc )
log "C colour readings exit $(cat $O/read_colour_D.rc)"
PAIRS=""
for seed in 97 99; do
  for ms in $SC; do m=${ms%%:*}; s=${ms##*:}; P=$O/runs40/${m}40k_${s}_P${seed}_frames12.npz
    for arm in L H D; do A=$O/runs40/${m}40k_${s}_${arm}${seed}_frames12.npz; [ -e $A ] && PAIRS="$PAIRS ${m}_${s}_${arm}${seed}=$A,$P"; done
  done
  PAIRS="$PAIRS floor_bunny_${seed}=$O/runs40/bunny40k_half_P${seed}_frames12.npz,$O/runs40/bunny40k_band_P${seed}_frames12.npz"
  PAIRS="$PAIRS floor_bunny_b_${seed}=$O/runs40/bunny40k_ears_P${seed}_frames12.npz,$O/runs40/bunny40k_turn_P${seed}_frames12.npz"
  PAIRS="$PAIRS floor_dragon_${seed}=$G/d155/gate/dragon40k_W${seed}_frames12.npz,$O/runs40/dragon40k_band_P${seed}_frames12.npz"
done
( envr repo_d160; rm -f $O/surfdisp.jsonl; $PY $W/d160_surfdisp.py $O/surfdisp.jsonl $PAIRS > $O/surfdisp.log 2>&1 < /dev/null
  echo $? > $O/surfdisp.rc )
log "C surface reading exit $(cat $O/surfdisp.rc)"

# ---- M: the videos ----
for ms in $SC; do m=${ms%%:*}; s=${ms##*:}
  D=$O/runs40/${m}40k_${s}_D97_frames12.npz; P=$O/runs40/${m}40k_${s}_P97_frames12.npz
  [ -e $D ] && [ -e $P ] || continue
  ( envr repo_d160; $PY $W/d160_studio.py $O/video/${m}_${s}_97_D $s $D $P $G/d92/ind_${m}.npz > $O/video/${m}_${s}_97_D.log 2>&1 < /dev/null
    echo $? > $O/video/${m}_${s}_97_D.rc; rm -rf $O/video/${m}_${s}_97_D/frames )
  log "M ${m}_${s}_97_D exit $(cat $O/video/${m}_${s}_97_D.rc)"
done
release

# ---- CPU: pieces, D155's gate reader, D157's reader, the gate ----
export CUDA_VISIBLE_DEVICES=""
R=$(ls $O/runs40/*_D9?_frames12.npz | sed 's#.*/##; s#_frames12.npz##')
( export PYTHONPATH=$B/repo_d160; nice -n 10 $PY $T/d145r/pieces145.py $O/pieces.jsonl \
    $(for n in $R; do echo "$n=$O/runs40/${n}_frames12.npz"; done) --every 4 --threads 6 > $O/pieces_D.log 2>&1 ) &
( nice -n 10 $PY $T/d155/d155_gate_read.py $O/gate_d160D.jsonl $(for n in $R; do echo "$n=$O/runs40"; done) > $O/gate_d160D.log 2>&1 ) &
JD=$O/jobs_d157_D.txt; : > $JD
for s in 97 99; do for m in bunny dragon; do first=1
  for sc in half band ears turn; do F=$O/runs40/${m}40k_${sc}_D${s}_frames12.npz; [ -e $F ] || continue
    [ $first = 1 ] && { echo "$m $s 40k own $F tgt" >> $JD; first=0; }; echo "$m $s 40k ${sc}_D $F end" >> $JD; done
done; done
( source $T/d157/env.sh; export TMPDIR=$W/tmpdir; cd $T/d157; nice -n 10 $PY $T/d157/d157_read.py $O/d157_D.jsonl $JD > $O/d157_D.log 2>&1 ) &
wait
( export PYTHONPATH=$B/repo_d160; $PY $W/d160_gate.py $O/gate_D D > $O/gate_D.log 2>&1 )
log "gate read: $(grep -m4 '^C[1-4]\|gate ' $O/gate_D.log | tr '\n' ' ')"
log "done"
