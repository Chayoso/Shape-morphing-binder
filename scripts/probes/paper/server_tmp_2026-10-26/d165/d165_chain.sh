#!/bin/bash
# D165 chain (2026-10-23, the user's top priority; the coordinator: one free card under the lock protocol, run directly):
# claim a card (no lock, under 2000 MiB), then jobs a (dense sample, decal stills, the kept-frame stamps) and b (the two
# 300k runs with their end states) side by side on it, then job c (the stamps on the end states, the videos); release;
# then the CPU follow-up (reading, close-ups, summary.md, README_ko.md).
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d165; W=$B/tmp/d165; L=$B/gpu_locks
mkdir -p $O
log() { echo "$(date '+%F %T') [chain] $*" >> $O/status.txt; }
CARD=""
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "released gpu$CARD"; CARD=""; }
trap release_card EXIT
trap "exit 143" TERM INT HUP
while [ -z "$CARD" ]; do
  for g in 3 0 2; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    [ -d $L/gpu$g.lock ] || [ "$used" -gt 2000 ] && continue
    if mkdir $L/gpu$g.lock 2>/dev/null; then
      CARD=$g
      printf "agent D147 (D165)\nrun d165/stamp test: 300k end states, decal stills, stamps, videos\npid %s\nstart %s\n" "$$" \
        "$(date '+%Y-%m-%d %H:%M:%S %Z')" > $L/gpu$g.lock/info
      break
    fi
  done
  [ -z "$CARD" ] && sleep 30
done
export CUDA_VISIBLE_DEVICES=$CARD
log "claimed gpu$CARD"
bash $W/23_d165_a_decal_smoke.sh &
A=$!
bash $W/23_d165_b_states300.sh
wait $A
bash $W/23_d165_c_stamp.sh
release_card
bash $W/d165_follow.sh
log "chain end"
