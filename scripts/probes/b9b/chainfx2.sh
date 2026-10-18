#!/bin/bash
# B9b forensic split (pre-registered in docs/experiments.md, B9b, 19:11 CDT, before its runs): Z5 fx_sphere0_na and
# Z6 fx_bunny0_na, Z1 / Z4 (no controls) with --assim 0: the assimilation's creep against the F smoothing's; and Z2
# fx_dragon0 again (its first run ended at 6.33 s in the loss raster's growth, which re-checks the config: zero_scene.py
# now lets iters 0 through). One card (agent B9b), claimed only after chainfx.sh released its own; tobody.py, fx_read.py.
#   setsid -f bash scripts/probes/b9b/chainfx2.sh REPO > /dev/null 2>&1 < /dev/null
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/b9b; R=$O/runs; LOG=$O/chainfx2.log
S=$REPO/scripts/probes/b9b; SP=scripts/probes/b9b/specs
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $S/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null          # $PY; cwd = $REPO
echo $$ > $O/chainfx2.pid
log "chainfx2 start (pid $$); waiting for chainfx.sh to release its card"
while ! grep -q "card released" $O/chainfx.log 2>/dev/null; do sleep 30; done
claim_card "b9b/chainfx2: 3 short 40k forensic runs without controls (~6 min)"
log "claimed gpu$CARD"
zrun() {                                        # TAG SRC TGT SPEC [pipeline flags]
  local tag=$1 src=$2 tgt=$3 spec=$4; shift 4
  $PY $S/zero_scene.py --scene $SP/$spec.json --src assets/$src.obj --tgt assets/$tgt.obj --n 40000 --seed 97 \
    --save_stride 12 --flags="$*" --out $R/$tag > $R/$tag.log 2>&1
  log "$tag exit $?: $(tail -2 $R/$tag.log | head -1 | cut -c1-200)"
}
zrun fx_sphere0_na isosphere bunny b9b_bunny --assim 0
zrun fx_bunny0_na bunny bunny b9b_bunny --assim 0
zrun fx_dragon0 dragon dragon b9b_dragon
release_card
log "card released"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
for tag in fx_sphere0_na fx_bunny0_na fx_dragon0; do
  [ -f $R/$tag.npz ] && $PY $S/tobody.py $R $tag > $R/tobody_$tag.txt 2>&1
  rm -f $R/${tag}_body12.npz $R/${tag}_frames12.npz
done
$PY $S/fx_read.py $R fx_sphere0 fx_sphere0_na fx_bunny0 fx_bunny0_na fx_dragon0 > $R/fx_read2.txt 2>&1
log "fx_read2 exit $?"
log "chainfx2 end"
