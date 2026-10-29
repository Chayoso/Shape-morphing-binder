#!/bin/bash
# chain_gateB.sh -- D155b on one card (agent D155; pre-registered 2026-10-20 22:58 CDT): the default path's gates (deterministic
# 40k against the tag's run A, pinned references, full suite), the exact skip's proof (deterministic, skip on / off), then
# A3G (A3 + --morph_u_gate final) L / P on five meshes with the card readers. Lock protocol as chain_gate.sh (a card idle
# 10 min, no other user's process, never GPU 2), released by the trap.
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; T=$B/tmp; G=$B/output/gpu; W=$T/d155; O=$G/d155/gate
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir; mkdir -p $TMPDIR $O
S=$O/status.txt
part=B
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
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
REPO=repo_d155bs
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
detrun() {   # detrun OUT WRAPPER [pipeline_run args]
  local out=$1 wrap=$2; shift 2
  (
    export PHYSMORPH_RUN_REPO=$B/$REPO PHYSMORPH_OUT=$G
    source $B/$REPO/scripts/ops/hyde06_env.sh > /dev/null
    export TMPDIR=$W/tmpdir
    cd $B/$REPO
    if [ -n "$wrap" ]; then
      $PY $T/rt/det_run.py $wrap scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 --animations 4 --out $out "$@" > $out.log 2>&1 < /dev/null
    else
      $PY $T/rt/det_run.py scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 --animations 4 --out $out "$@" > $out.log 2>&1 < /dev/null
    fi
    echo "det $(basename $out) rc $? $(date '+%m-%d %H:%M:%S')" >> $S
  )
}
DD=$O/det; mkdir -p $DD
detrun $DD/D155b ""
$PY $T/d145/det_cmp12.py $G/d145t_detproof/A $DD/D155b > $DD/cmp_A_D155b.txt 2>&1
echo "det default vs tag A: $(tail -1 $DD/cmp_A_D155b.txt) $(date '+%m-%d %H:%M:%S')" >> $S
MORPH="--morph_ramp elastic --morph_ref_n 300000 --morph_u_gate final"
detrun $DD/skip "" $MORPH
detrun $DD/noskip $W/run_pics.py $MORPH
$PY $T/d145/det_cmp12.py $DD/skip $DD/noskip > $DD/cmp_skip_noskip.txt 2>&1
echo "det skip vs no skip: $(tail -1 $DD/cmp_skip_noskip.txt) $(date '+%m-%d %H:%M:%S')" >> $S
(
  export PHYSMORPH_RUN_REPO=$B/repo_d155b PHYSMORPH_OUT=$G
  source $B/repo_d155b/scripts/ops/hyde06_env.sh > /dev/null
  export TMPDIR=$W/tmpdir
  cd $B/repo_d155b
  $PY -m pytest -q tests/test_refreeze_reference.py -p no:cacheprovider > $O/pytest_pinned_b.log 2>&1 < /dev/null
  echo "pinned references (b) exit $? $(date '+%m-%d %H:%M:%S')" >> $S
  $PY -m pytest -q tests -p no:cacheprovider > $O/pytest_full_b.log 2>&1 < /dev/null
  echo "full suite (b) exit $? ($(tail -1 $O/pytest_full_b.log)) $(date '+%m-%d %H:%M:%S')" >> $S
)
for m in bunny dragon teapot armadilo homer; do
  run ${m}40k_A3GL97 $REPO $m 97 $MORPH
  run ${m}40k_A3GP97 $REPO $m 97 $MORPH --render_weight_scale 0
done
echo "part B done $(date '+%m-%d %H:%M:%S')" >> $S
