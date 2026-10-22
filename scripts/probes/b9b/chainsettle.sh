#!/bin/bash
# B9b step 1 (the coordinator's go, 2026-10-22; pre-registered in docs/experiments.md, B9b, "Step 1: the one-window load
# effect against the settled one", before its run): settle_probe.py on the 40k bunny L97 of the assim_load definition,
# windows 4-6 s, chains of 48 windows.
#   setsid -f bash scripts/probes/b9b/chainsettle.sh REPO > /dev/null 2>&1 < /dev/null
# ONE card idle 10 min, the lowest priority: no claim of a card D159 / D160(c) / D161 / D162 take while any of their
# scripts waits (claim.sh YIELD_RENDER); GPU 2, which none of them takes, by the idle rule alone.
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/b9b; R=$O/settle; LOG=$O/chainsettle.log
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
echo $$ > $O/chainsettle.pid
log "chainsettle start (pid $$; $(tail -1 $REPO/DEPLOYED_FROM)); claims yield to D159 / D160 / D161 / D162"
claim_card "b9b/chainsettle: step 1, one-window against settled load effect, 40k bunny, one run to 6 s (~15 min)"
log "claimed gpu$CARD"
$PY $S/settle_probe.py $R 4 6 48 > $R/settle.txt 2>&1
log "settle_probe exit $?: $(grep -c '^\[settle\]' $R/settle.txt) windows measured; $(tail -1 $R/settle.txt | cut -c1-160)"
release_card
log "chainsettle end"
