#!/bin/bash
# D147: the dump runs read step by step (dump_forensic.py; the layer normals need a card): one card idle 3 min.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d147; R=$O/runs; REPO=$B/repo_d147fast; LOG=$O/dumpfx.log
log() { echo "$(date "+%F %T") $*" >> $LOG; }
source $B/tmp/d147/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null
log "start; waiting for a card idle 3 min"
CLAIM_IDLE=180 claim_card "dump forensic reading (40k dumps)"
log "claimed gpu$CARD"
for a in "dump40k_s97 11.2 300 s97_p300" "dump40k_s97 auto - s97_auto" "dump40k_s97_off auto - off_auto" \
         "dump40k_s97_off 12.2 31874 off_p31874"; do
  set -- $a
  if [ "$3" = "-" ]; then $PY $REPO/scripts/probes/d147/dump_forensic.py $R $1 1 $2 > $R/dumpfx_$4.txt 2>&1
  else $PY $REPO/scripts/probes/d147/dump_forensic.py $R $1 1 $2 $3 > $R/dumpfx_$4.txt 2>&1; fi
  log "$4 exit $?"
done
log "end"
