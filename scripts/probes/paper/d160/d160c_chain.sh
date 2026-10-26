#!/bin/bash
# d160c_chain.sh -- D160c (agent D160; pre-registered in docs/experiments.md): the full arm with the render weight
# calibrated at the first window and held (--lambda_ema 0, the path before D81), 40k, the five D160 scenes x seeds 97 / 99,
# on ONE card by the lock protocol (3 / 1 / 0, never 2; no lock, no other user's process, < 2000 MiB, < 5 % for ten
# one-minute samples; mkdir + info; trap). The twins are D160's P runs (lambda = 0: the weight's rule does not enter).
# Card: R the runs + D155's card readers; C d160_read.py (H with their P twins, timelines); M the seed-97 videos.
# Then CPU (card released): pieces145, D155's gate reader, D157's reader on the H ends, d160_gate.py (arm H).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d160; W=$B/tmp/d160; T=$B/tmp; L=$B/gpu_locks
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') [D160c] $*" >> $S; }
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
      printf "agent D160\nrun d160c: 10 x 40k (lambda held) + readers + videos (~45 min)\npid %s\nstart %s\n" "$$" \
          "$(date '+%Y-%m-%d %H:%M:%S %Z')" > $L/gpu$g.lock/info
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
  for ms in $SC; do m=${ms%%:*}; s=${ms##*:}; run ${m}40k_${s}_H$seed $m $seed --appearance $s --lambda_ema 0; done
done
log "R done"
J=$O/jobs_read_H.txt; : > $J
for seed in 97 99; do
  for ms in $SC; do m=${ms%%:*}; s=${ms##*:}
    P=$O/runs40/${m}40k_${s}_P${seed}_frames12.npz; H=$O/runs40/${m}40k_${s}_H${seed}_frames12.npz
    [ -e $P ] && [ -e $H ] && { echo "${m}40k_${s}_P$seed $s $P" >> $J; echo "${m}40k_${s}_H$seed $s $H twin=${m}40k_${s}_P$seed timeline" >> $J; }
  done
done
( envr repo_d160; export PYTHONPATH=$B/deps/d157_lpips:$PYTHONPATH; rm -f $O/read_colour_H.jsonl
  $PY $W/d160_read.py $O/read_colour_H.jsonl $J > $O/read_colour_H.log 2>&1 < /dev/null; echo $? > $O/read_colour_H.rc )
log "C colour readings exit $(cat $O/read_colour_H.rc)"
for ms in $SC; do m=${ms%%:*}; s=${ms##*:}
  H=$O/runs40/${m}40k_${s}_H97_frames12.npz; P=$O/runs40/${m}40k_${s}_P97_frames12.npz
  [ -e $H ] && [ -e $P ] || continue
  ( envr repo_d160; $PY $W/d160_studio.py $O/video/${m}_${s}_97_H $s $H $P $G/d92/ind_${m}.npz > $O/video/${m}_${s}_97_H.log 2>&1 < /dev/null
    echo $? > $O/video/${m}_${s}_97_H.rc; rm -rf $O/video/${m}_${s}_97_H/frames )
  log "M ${m}_${s}_97_H exit $(cat $O/video/${m}_${s}_97_H.rc)"
done
release
export CUDA_VISIBLE_DEVICES=""
R=$(ls $O/runs40/*_H9?_frames12.npz | sed 's#.*/##; s#_frames12.npz##')
( export PYTHONPATH=$B/repo_d160; nice -n 10 $PY $T/d145r/pieces145.py $O/pieces.jsonl \
    $(for n in $R; do echo "$n=$O/runs40/${n}_frames12.npz"; done) --every 4 --threads 6 > $O/pieces_H.log 2>&1 ) &
( nice -n 10 $PY $T/d155/d155_gate_read.py $O/gate_d160H.jsonl $(for n in $R; do echo "$n=$O/runs40"; done) > $O/gate_d160H.log 2>&1 ) &
JD=$O/jobs_d157_H.txt; : > $JD
for s in 97 99; do for m in bunny dragon; do first=1
  for sc in half band ears turn; do F=$O/runs40/${m}40k_${sc}_H${s}_frames12.npz; [ -e $F ] || continue
    [ $first = 1 ] && { echo "$m $s 40k own $F tgt" >> $JD; first=0; }; echo "$m $s 40k ${sc}_H $F end" >> $JD; done
done; done
( source $T/d157/env.sh; export TMPDIR=$W/tmpdir; cd $T/d157; nice -n 10 $PY $T/d157/d157_read.py $O/d157_H.jsonl $JD > $O/d157_H.log 2>&1 ) &
wait
( export PYTHONPATH=$B/repo_d160; $PY $W/d160_gate.py $O/gate_H H > $O/gate_H.log 2>&1 )
log "gate read: $(grep -m4 '^C[1-4]\|gate ' $O/gate_H.log | tr '\n' ' ')"
log "done"
