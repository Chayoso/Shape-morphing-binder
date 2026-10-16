# claim.sh (sourced): claim_card WHAT -- D147 lock protocol: a card with no lock and under 2000 MiB for 10 consecutive
# minutes (the coordinator, 2026-10-16), mkdir lock + info, released by release_card (call it from a trap).
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
CARD=""
claim_card() {
  local what=$1; declare -A since
  while [ -z "$CARD" ]; do
    local now=$(date +%s)
    for g in 3 2 1 0; do
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ -d $L/gpu$g.lock ] || [ "$used" -gt 2000 ]; then since[$g]=""; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      if [ $((now - ${since[$g]})) -ge 600 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        CARD=$g
        printf "agent D147\nrun d147/%s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        break
      fi
    done
    [ -z "$CARD" ] && sleep 30
  done
  export CUDA_VISIBLE_DEVICES=$CARD
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock; CARD=""; }
