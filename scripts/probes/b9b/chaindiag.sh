#!/bin/bash
# B9b step 1b (a diagnostic of step 1; pre-registered in docs/experiments.md, B9b, "Step 1b", before its run):
# settle_diag.py on the 40k bunny L97 of the assim_load definition, windows 25 / 30 / 35, the continuations with the
# integrator as it is and with its F smoothing off (in the continuations only).
#   setsid -f bash scripts/probes/b9b/chaindiag.sh REPO > /dev/null 2>&1 < /dev/null
# ONE card idle 10 min, the lowest priority: no claim of a card D159 / D160(c) / D161 / D162 take while any of their
# scripts waits (claim.sh YIELD_RENDER); GPU 2, which none of them takes, by the idle rule alone.
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/b9b; R=$O/settle_diag; LOG=$O/chaindiag.log
S=$REPO/scripts/probes/b9b
mkdir -p $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $S/claim.sh
export YIELD_RENDER=1
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null          # $PY; cwd = $REPO
echo $$ > $O/chaindiag.pid
log "chaindiag start (pid $$; $(tail -1 $REPO/DEPLOYED_FROM)); claims yield to D159 / D160 / D161 / D162"
claim_card "b9b/chaindiag: step 1b diagnostic, 40k bunny, one run to 6 s (~8 min)"
log "claimed gpu$CARD"
$PY $S/settle_diag.py $R > $R/diag.txt 2>&1
log "settle_diag exit $?: $(grep -c '^\[diag\]' $R/diag.txt) lines; $(tail -1 $R/diag.txt | cut -c1-160)"
release_card
log "chaindiag end"
