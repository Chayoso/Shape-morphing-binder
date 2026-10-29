#!/bin/bash
# chain_gate.sh PART -- D155 option A's 40k gate on one card (PART 3: the diagnostic u_off runs, reported, not judged) (agent D155; pre-registered 2026-10-17 14:10 CDT).
# Lock protocol: a card with no lock, no other user's compute process, < 2000 MiB and < 5 % use at ten one-minute samples
# in a row is claimed by mkdir with an info file and released by the trap. GPU 2 is never used (another user's display
# session). PART 1: the default path's gates (deterministic 40k against the tag's run A, the pinned references, the full
# suite), then the switched runs and the tag's seed-97 re-runs with their card readers. PART 2: the tag's seeds 99 / 101.
# After a part the card is released; PART 2 waits 15 min more so that D147 / D154 can take a card first.
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; T=$B/tmp; G=$B/output/gpu; W=$T/d155; O=$G/d155/gate
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir; mkdir -p $TMPDIR $O
S=$O/status.txt
part=$1
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
[ "$part" = 3 ] && until grep -q "part 2 done" $S 2>/dev/null; do sleep 60; done
[ "$part" = 2 ] && until grep -q "part 1 done" $S 2>/dev/null; do sleep 60; done
[ "$part" = 2 ] && { echo "part 2 waits 15 min before looking for a card $(date '+%m-%d %H:%M:%S')" >> $S; sleep 900; }
declare -A idle
GPU=""
echo "part $part ($$) waiting for a card idle 10 min $(date '+%m-%d %H:%M:%S')" >> $S
while [ -z "$GPU" ]; do
  for g in 0 1 3; do
    if [ -e $L/gpu$g.lock ]; then idle[$g]=0; continue; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
    napps=0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g); do [ "$(ps -o user= -p $p 2>/dev/null | tr -d " ")" = chayo ] || napps=$((napps + 1)); done
    if [ "$used" -lt 2000 ] && [ "$util" -lt 5 ] && [ "$napps" -eq 0 ]; then idle[$g]=$(( ${idle[$g]:-0} + 1 )); else idle[$g]=0; fi
    if [ "${idle[$g]}" -ge 10 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
      printf "agent D155\nrun d155/gate part $part (option A's 40k gate: 40k runs of 1-3 min with their readers)\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
      GPU=$g; break
    fi
  done
  [ -z "$GPU" ] && sleep 60
done
trap "rm -rf $L/gpu$GPU.lock" EXIT
trap "exit 1" INT TERM HUP
echo "part $part claimed gpu $GPU $(date '+%m-%d %H:%M:%S')" >> $S
export CUDA_VISIBLE_DEVICES=$GPU

box_of() { case $1 in bunny) echo 1900,100,2800,700;; dragon) echo 1900,60,2500,560;; *) echo 0,0,3840,2160;; esac; }

run() {   # run NAME REPO MESH SEED [pipeline_run args]
  local name=$1 repo=$2 mesh=$3 seed=$4; shift 4
  [ -e $O/STOP ] && { echo "stopped by STOP before $name $(date '+%m-%d %H:%M:%S')" >> $S; exit 0; }
  [ -e $O/rc_$name ] && [ "$(cat $O/rc_$name)" = 0 ] && return
  (
    export PHYSMORPH_RUN_REPO=$B/$repo PHYSMORPH_OUT=$G
    source $B/$repo/scripts/ops/hyde06_env.sh > /dev/null
    export TMPDIR=$W/tmpdir
    cd $B/$repo
    t0=$(date +%s)
    $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n 40000 --seed $seed --out $O/$name "$@" > $O/$name.log 2>&1 < /dev/null
    rc=$?
    echo $rc > $O/rc_$name
    echo "$name sim rc $rc wall $(( $(date +%s) - t0 )) s $(date '+%m-%d %H:%M:%S')" >> $S
  )
  [ "$(cat $O/rc_$name)" = 0 ] || return
  local F=$O/${name}_frames12.npz IND=$G/d92/ind_${mesh}.npz
  (
    source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null
    export TMPDIR=$W/tmpdir
    cd $B/repo_r77
    last=$($PY $T/lastraw.py $F)
    $PY $T/d145/endframe.py $F $O/${name}_end1.npz
    $PY scripts/probes/settled/render_terms_probe.py $O/${name}_end1.npz ref=$IND > $O/t3end_$name.log 2>&1 < /dev/null
    $PY scripts/probes/settled/surface_layer_probe.py $F $O/e3end_$name $(box_of $mesh) measure ref=$IND target own $last > $O/e3end_$name.log 2>&1 < /dev/null
    $PY $T/gallery_ind.py $O/ind.log "$name=$F,$IND" > $O/gind_$name.log 2>&1 < /dev/null
    rm -f $O/${name}_end1.npz
    echo "$name card readers done $(date '+%m-%d %H:%M:%S')" >> $S
  )
}

gates() {
  local D=$O/det
  mkdir -p $D
  (
    export PHYSMORPH_RUN_REPO=$B/repo_d155s PHYSMORPH_OUT=$G
    source $B/repo_d155s/scripts/ops/hyde06_env.sh > /dev/null
    export TMPDIR=$W/tmpdir
    cd $B/repo_d155s
    $PY $T/rt/det_run.py scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 --animations 4 --out $D/D155 > $D/D155.log 2>&1 < /dev/null
    echo "det run rc $? $(date '+%m-%d %H:%M:%S')" >> $S
    $PY $T/d145/det_cmp12.py $G/d145t_detproof/A $D/D155 > $D/cmp_A_D155.txt 2>&1
    echo "det cmp: $(tail -1 $D/cmp_A_D155.txt) $(date '+%m-%d %H:%M:%S')" >> $S
  )
  (
    export PHYSMORPH_RUN_REPO=$B/repo_d155 PHYSMORPH_OUT=$G
    source $B/repo_d155/scripts/ops/hyde06_env.sh > /dev/null
    export TMPDIR=$W/tmpdir
    cd $B/repo_d155
    $PY -m pytest -q tests/test_refreeze_reference.py -p no:cacheprovider > $O/pytest_pinned.log 2>&1 < /dev/null
    echo "pinned references exit $? $(date '+%m-%d %H:%M:%S')" >> $S
    $PY -m pytest -q tests -p no:cacheprovider > $O/pytest_full.log 2>&1 < /dev/null
    echo "full suite exit $? ($(tail -1 $O/pytest_full.log)) $(date '+%m-%d %H:%M:%S')" >> $S
  )
}

MESHES="bunny dragon teapot armadilo homer"
if [ "$part" = 3 ]; then
  for m in bunny dragon; do
    run ${m}40k_A3U97 repo_d155s $m 97 --morph_ramp elastic --morph_ref_n 300000 --u_off
    run ${m}40k_U97 repo_t1s $m 97 --u_off
  done
elif [ "$part" = 1 ]; then
  gates
  for m in $MESHES; do
    run ${m}40k_A3L97 repo_d155s $m 97 --morph_ramp elastic --morph_ref_n 300000
    run ${m}40k_A3P97 repo_d155s $m 97 --morph_ramp elastic --morph_ref_n 300000 --render_weight_scale 0
    run ${m}40k_V97 repo_t1s $m 97
    run ${m}40k_W97 repo_t1s $m 97 --render_weight_scale 0
    run ${m}40k_A1L97 repo_d155s $m 97 --morph_ramp elastic
    run ${m}40k_A1P97 repo_d155s $m 97 --morph_ramp elastic --render_weight_scale 0
  done
else
  for s in 99 101; do
    for m in $MESHES; do
      run ${m}40k_V$s repo_t1s $m $s
      run ${m}40k_W$s repo_t1s $m $s --render_weight_scale 0
    done
  done
fi
echo "part $part done $(date '+%m-%d %H:%M:%S')" >> $S
