#!/bin/bash
# Post-process each target once its DONE marker appears (for batches launched without post_run).
# Usage: watch_post.sh <prefix> <gpu> <targets...>
source /data/relcfd/chayo/physmorph_v2/repo/scripts/ops/hyde06_env.sh
PFX=$1; GPU=$2; shift 2
UP=$(echo $PFX | tr a-z A-Z)
for i in $(seq 1 720); do
  for T in "$@"; do
    if grep -q "^$UP $T DONE" $STATUS && ! grep -q "^POST$UP $T DONE" $STATUS && [ ! -f /tmp/post_${PFX}_$T.lock ]; then
      touch /tmp/post_${PFX}_$T.lock
      bash $REPO/scripts/ops/post_run.sh $PFX $T $GPU > $OUT/post_${PFX}_$T.log 2>&1
    fi
  done
  n=0; for T in "$@"; do grep -q "^POST$UP $T DONE" $STATUS && n=$((n+1)); done
  [ $n -ge $# ] && { echo "$UP ALL POSTED $(date)" >> $STATUS; exit 0; }
  sleep 20
done
