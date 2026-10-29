#!/bin/bash
# d174_gate.sh TAG [REPO] -- agent D174: the V3 gate of an exact speed-up (deterministic mode, tmp/rt/det_run.py; bunny 40k
# seed 97, 4 windows): REPO (default repo_d174) bit-SAME as repo_d172 on
#   G1 the defaults (reference: the tag's d153/rt1/S1, = d172/gates/S1b),
#   G2 Hu (--lambda_ema 0 --settle_windows 30 --no-settle_relax --render_split u; reference d174/gates/ref_H0 on repo_d172),
#   G3 Hu on the coin with its render target, settle 2 (D172's X2 args; reference d172/gates/X2),
#   G4 the settle with its relaxation on (--settle_windows 3; reference d174/gates/ref_R0 on repo_d172).
# Waits for a free card (0 / 1 / 3, never 2; lock gpu_locks/gpuC.lock), writes gates/TAG.ok or TAG.fail + TAG.md.
TAG=$1; RP=${2:-repo_d174}   # NOLOCK=1: under a queue runner (it holds the card and exports CUDA_VISIBLE_DEVICES)
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d174; D=$O/gates; mkdir -p $D
log() { echo "$(date '+%F %T') [gate $TAG] $*" >> $O/status.txt; }
CARD=""
[ -n "$NOLOCK" ] && CARD=${CUDA_VISIBLE_DEVICES:-0}
while [ -z "$CARD" ]; do
  for c in 0 1 3; do
    u=$(nvidia-smi -i $c --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$u" -lt 2000 ] && mkdir $B/gpu_locks/gpu$c.lock 2>/dev/null; then CARD=$c; break; fi
  done
  [ -z "$CARD" ] && sleep 60
done
LOCK=$B/gpu_locks/gpu$CARD.lock; [ -n "$NOLOCK" ] && { LOCK=$B/tmp/d174/nolock_$TAG; mkdir -p $LOCK; }
printf "agent D174\nV3 gate %s (%s)\npid %s\nstart %s\n" $TAG $RP $$ "$(date '+%F %T %Z')" > $LOCK/info
trap 'rm -rf $LOCK' EXIT
trap 'exit 1' INT TERM HUP
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1
         source $B/tmp/d146/cache_env.sh; export TMPDIR=$B/tmp/d174/tmpdir CUDA_VISIBLE_DEVICES=$CARD; cd $B/$1; }
A1=$G/d161/assets
F_coin="--tgt $A1/d161coin.obj --render_tgt $A1/d161L40coin.obj"
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"
CMP=$B/tmp/d145/det_cmp12.py
HU="--lambda_ema 0 --settle_windows 30 --no-settle_relax --render_split u"
log "start on card $CARD ($RP)"; t0=$(date +%s)
( envr repo_d172
  [ -s $D/ref_H0_frames12.npz ] || $PY $DR --tgt assets/bunny.obj $HU --out $D/ref_H0 > $D/ref_H0.log 2>&1 < /dev/null
  [ -s $D/ref_R0_frames12.npz ] || $PY $DR --tgt assets/bunny.obj --settle_windows 3 --out $D/ref_R0 > $D/ref_R0.log 2>&1 < /dev/null )
( envr $RP
  $PY $DR --tgt assets/bunny.obj --out $D/${TAG}_G1 > $D/${TAG}_G1.log 2>&1 < /dev/null
  $PY $CMP $G/d153/rt1/S1 $D/${TAG}_G1 > $D/${TAG}_cmp1.txt 2>&1
  $PY $DR --tgt assets/bunny.obj $HU --out $D/${TAG}_G2 > $D/${TAG}_G2.log 2>&1 < /dev/null
  $PY $CMP $D/ref_H0 $D/${TAG}_G2 > $D/${TAG}_cmp2.txt 2>&1
  $PY $DR $F_coin --lambda_ema 0 --settle_windows 2 --no-settle_relax --render_split u --out $D/${TAG}_G3 > $D/${TAG}_G3.log 2>&1 < /dev/null
  $PY $CMP $G/d172/gates/X2 $D/${TAG}_G3 > $D/${TAG}_cmp3.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --settle_windows 3 --out $D/${TAG}_G4 > $D/${TAG}_G4.log 2>&1 < /dev/null
  $PY $CMP $D/ref_R0 $D/${TAG}_G4 > $D/${TAG}_cmp4.txt 2>&1 )
ok=1; rows=""
for k in 1 2 3 4; do r=$(tail -1 $D/${TAG}_cmp$k.txt 2>/dev/null); echo "$r" | grep -q ": SAME$" || ok=0; rows="$rows| G$k | $r |\n"; done
v=FAIL; [ $ok = 1 ] && v=PASS
{ echo "# D174 V3 gate $TAG ($RP vs repo_d172; deterministic, bunny 40k seed 97, 4 windows)"; echo
  echo "| check | result |"; echo "|---|---|"; printf "$rows"; echo
  echo "G1 defaults vs d153/rt1/S1; G2 Hu (settle 30) vs ref_H0; G3 Hu coin (settle 2) vs d172/gates/X2; G4 settle 3 with relaxation vs ref_R0"
  echo; echo "VERDICT: $v ($(( $(date +%s) - t0 )) s)"; } > $D/$TAG.md
rm -f $D/$TAG.ok $D/$TAG.fail; [ $v = PASS ] && touch $D/$TAG.ok || touch $D/$TAG.fail
for k in 1 2 3 4; do rm -f $D/${TAG}_G${k}_frames12.npz; done     # keep the JSON and logs; the frames are re-made on demand
log "VERDICT $v"
