#!/bin/bash
# rerun_carve.sh -- D170: the carve readers' fix (a frame whose region holds < 50 layer points reads None; the first carve_b
# summary crashed on it after the coin read). (1) carve_b now: the cached coin read, then the bunny-flank read, which waits
# for the ripple read (pid RIPPLE, ~95 GB) to exit (D170_WAIT_PID); (2) carve_a: when its running reader (after2 pid
# CARVE_A_AFTER, the old code) exits without a VERDICT line, read it again from the cached reads.
RIPPLE=$1; CARVE_A_AFTER=$2
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d170; O=$B/output/gpu/d170
log() { echo "$(date '+%m-%d %H:%M:%S') [rerun carve] $*" >> $O/status.txt; }
log "carve_b again (the flank read waits for pid $RIPPLE)"
D170_WAIT_PID=$RIPPLE bash $W/after2.sh carve_b 171_carve_b 171_d170_carve_b > $O/auto/171_carve_b/after_rerun.out 2>&1 &
while ps -p $CARVE_A_AFTER > /dev/null 2>&1; do sleep 60; done
if ! grep -q "^VERDICT" $O/auto/170_carve_a/summary.md 2>/dev/null; then
  log "carve_a ended without a VERDICT: read again from the cached reads"
  bash $W/after2.sh carve_a 170_carve_a 170_d170_carve_a > $O/auto/170_carve_a/after_rerun.out 2>&1
fi
wait
log "done"
