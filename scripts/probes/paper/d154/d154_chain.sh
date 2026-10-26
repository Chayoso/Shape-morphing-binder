#!/bin/bash
# d154_chain.sh -- D154 (2): the render switched at 6 s, on ONE card by the agents' lock protocol (gpu_locks/gpuN.lock by
# mkdir with an info file, released by the trap; a card with no lock, no other user's compute process, < 2000 MiB, idle
# for 10 minutes by the pool's shared first-free time; GPU 2 never: another user's desktop session).
#   V  the gates, in order; any failure stops the chain before a run:
#      V1 full suite on repo_d154 (exit status), V2 the pinned digest + reference (test_refreeze_reference.py),
#      V3 Warp run-to-run deterministic 40k bunny seed 97 four windows (tmp/rt/det_run.py): repo_d154s at the defaults
#         against rt1's S1 (output/gpu/d153/rt1/S1, det_cmp12: SAME); --render_until_s 0.5 against that S1 and
#         --render_from_s 0.5 against a deterministic twin (--render_weight_scale 0), both repo_d154s
#         (det_switch_cmp.py: SAME-BEFORE, lambda 0 / > 0 after the switch);
#   R  the eight runs (repo_d154s, kept frames): MESH300k_Loff6_SEED (--render_until_s 6) and MESH300k_Pon6_SEED
#      (--render_from_s 6), bunny and dragon, seeds 97 then 99; after each, its end frame (endframe.py);
#   Q  if nothing failed: the replicates MESH300k_L_SEED / _P_SEED for seed 99 (repo_d154s at the defaults / the twin);
#   D  the display (surface_layer_probe measure against the independent sample, repo_r77, as read300.sh) at every
#      120th raw (0.5 s) and the last kept frame of every run of R and Q.
# Usage: d154_chain.sh STEPS   (e.g. "V R Q D")
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d154; W=$B/tmp/d154; L=$B/gpu_locks; mkdir -p $O/runs $O/gates
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir; mkdir -p $TMPDIR
PYB=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
STEPS=${1:-"V R Q D"}
CARD=""
claim_card() {
  local what=$1; declare -A since
  while [ -z "$CARD" ]; do
    local now=$(date +%s)
    for g in 3 1 0; do
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      local uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
      local other=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | while IFS=', ' read -r u p; do
                    o=$(ps -o user= -p $p 2>/dev/null); [ -n "$o" ] && [ "$o" != chayo ] && echo x; done | wc -l)
      local mine=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
      if [ -d $L/gpu$g.lock ] || [ "$used" -gt 2000 ] || [ "$other" -gt 0 ] || [ "$mine" -gt 0 ]; then since[$g]=""; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      local fs=$(cat $B/tmp/d145/free_since/gpu$g 2>/dev/null); [ -n "$fs" ] && [ "$fs" -lt "${since[$g]}" ] && since[$g]=$fs
      if [ $((now - ${since[$g]})) -ge 600 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        CARD=$g
        printf "agent D154\nrun d154/%s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        break
      fi
    done
    [ -z "$CARD" ] && sleep 60
  done
  export CUDA_VISIBLE_DEVICES=$CARD
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "released gpu$CARD"; CARD=""; }
trap 'release_card' EXIT
trap 'exit 1' INT TERM HUP
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1; export TMPDIR=$W/tmpdir; cd $B/$1; }
log "chain start pid $$ steps [$STEPS] (waiting for a card idle 10 min)"
claim_card "chain [$STEPS]: gates, the 8 switched 300k runs (bunny/dragon x 97/99 x off/on at 6 s), replicates, display"
log "claimed gpu$CARD"
fail() { log "STOP: $*"; exit 1; }
if [[ " $STEPS " == *" V "* ]]; then
  V=$O/gates
  ( envr repo_d154; t0=$(date +%s); $PY -m pytest -q -p no:cacheprovider tests > $V/suite.log 2>&1 < /dev/null; echo $? > $V/suite.rc )
  log "V1 suite repo_d154 exit $(cat $V/suite.rc): $(tail -1 $V/suite.log)"
  [ "$(cat $V/suite.rc)" = 0 ] || fail "V1 suite"
  ( envr repo_d154; $PY -m pytest -q -p no:cacheprovider "tests/test_refreeze_reference.py::test_the_refreeze_pipeline_is_its_pinned_digest" \
      tests/test_refreeze_reference.py > $V/digest.log 2>&1 < /dev/null; echo $? > $V/digest.rc )
  log "V2 pinned digest + reference exit $(cat $V/digest.rc): $(tail -1 $V/digest.log)"
  [ "$(cat $V/digest.rc)" = 0 ] || fail "V2 digest"
  det() { local name=$1; shift; ( envr repo_d154s; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 \
      --animations 4 --out $V/$name "$@" > $V/$name.log 2>&1 < /dev/null; echo $? > $V/$name.rc ); log "V3 det $name ($*) exit $(cat $V/$name.rc)"; }
  det S1; det U --render_until_s 0.5; det P --render_weight_scale 0; det F --render_from_s 0.5
  ( envr repo_d154; $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $V/S1 > $V/cmp_rt1_S1.txt 2>&1
    $PY $W/det_switch_cmp.py $G/d153/rt1/S1 $V/U 0.5 1 > $V/cmp_U.txt 2>&1
    $PY $W/det_switch_cmp.py $V/P $V/F 0.5 0 > $V/cmp_F.txt 2>&1 )
  log "V3 rt1 S1 vs repo_d154s defaults: $(tail -1 $V/cmp_rt1_S1.txt)"
  log "V3 until 0.5 vs rt1 S1: $(tail -1 $V/cmp_U.txt)"
  log "V3 from 0.5 vs the twin: $(tail -1 $V/cmp_F.txt)"
  tail -1 $V/cmp_rt1_S1.txt | grep -q ": SAME$" || fail "V3 default path"
  tail -1 $V/cmp_U.txt | grep -q "SAME-BEFORE$" || fail "V3 until"
  tail -1 $V/cmp_F.txt | grep -q "SAME-BEFORE$" || fail "V3 from"
  log "V gates passed"
fi
run() {   # NAME MESH SEED FLAGS...
  local name=$1 mesh=$2 seed=$3; shift 3
  if [ -e $O/runs/${name}_frames12.npz ]; then log "R $name exists, skipped"; return; fi
  log "R $name start ($mesh 300k seed $seed $*)"
  ( envr repo_d154s; $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n 300000 --seed $seed --out $O/runs/$name "$@" \
      > $O/runs/$name.log 2>&1 < /dev/null; echo $? > $O/runs/$name.rc )
  local rc=$(cat $O/runs/$name.rc)
  log "R $name exit $rc: $(grep -m1 'ARM render_full_dt_iso_nn:' $O/runs/$name.log | cut -c1-160)"
  [ "$rc" = 0 ] && $PYB $B/tmp/d145/endframe.py $O/runs/${name}_frames12.npz $O/runs/${name}_end1.npz
}
if [[ " $STEPS " == *" R "* ]]; then
  for seed in 97 99; do
    for mesh in bunny dragon; do
      run ${mesh}300k_Loff6_$seed $mesh $seed --render_until_s 6
      run ${mesh}300k_Pon6_$seed $mesh $seed --render_from_s 6
    done
  done
  log "R done"
fi
if [[ " $STEPS " == *" Q "* ]]; then
  for mesh in bunny dragon; do
    run ${mesh}300k_L_99 $mesh 99
    run ${mesh}300k_P_99 $mesh 99 --render_weight_scale 0
  done
  log "Q done"
fi
if [[ " $STEPS " == *" D "* ]]; then
  cd $B/repo_r77
  source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null 2>&1; export TMPDIR=$W/tmpdir
  for f in $O/runs/*_frames12.npz; do
    name=$(basename $f _frames12.npz); mesh=${name%%300k_*}
    [ -e $O/runs/e3_$name.log ] && continue
    case $mesh in bunny) box=1900,100,2800,700;; dragon) box=1900,60,2500,560;; esac
    raws=$($PYB -c "import numpy as np; r=[int(v) for v in np.load('$f')['raws']]; print(' '.join(str(v) for v in r if v % 120 == 0 or v == r[-1]))")
    CUDA_VISIBLE_DEVICES=$CARD $PY scripts/probes/settled/surface_layer_probe.py $f $O/runs/e3_$name $box measure \
      ref=$G/d90/${mesh}_ind300k.npz target own $raws > $O/runs/e3_$name.log 2>&1 < /dev/null
    log "D display $name exit $?"
  done
  log "D done"
fi
log "chain done"
