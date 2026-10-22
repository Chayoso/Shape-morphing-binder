# claim.sh (sourced): claim_card WHAT -- the agents' lock protocol as agent B9b: a card with no lock and under 2000 MiB
# for 10 consecutive minutes (CLAIM_IDLE seconds, default 600: the agents with priority, D147 / D154 / D155, claim after
# shorter idles and so take a freed card first), mkdir lock + info, released by release_card (call it from a trap).
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
CARD=""
# YIELD_RENDER=1 (the coordinator, 2026-10-21 night: the render experiments D159 / D160 / D161 first): no claim while
# one of their chains is alive and holds no lock (it is waiting for a card, or between its card phases). Read from ps's
# list and the lock files only; nothing is signalled.
render_waiting() {
  local held=" $(cat $L/gpu*.lock/info 2>/dev/null | awk '$1 == "pid" {printf "%s ", $2}')"
  local pid args
  while read -r pid args; do
    case "$args" in
      *d159_chain.sh*|*d160_chain.sh*|*d161_chain.sh*) [[ "$held" == *" $pid "* ]] || return 0 ;;
    esac
  done < <(ps -u "$(id -un)" -o pid=,args=)
  return 1
}
claim_card() {
  local what=$1 idle=${CLAIM_IDLE:-600}; declare -A since
  while [ -z "$CARD" ]; do
    local now=$(date +%s)
    for g in 3 2 1 0; do
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ -d $L/gpu$g.lock ] || [ "$used" -gt 2000 ]; then since[$g]=""; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      if [ "${YIELD_RENDER:-0}" = 1 ] && render_waiting; then continue; fi
      if [ $((now - ${since[$g]})) -ge $idle ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        CARD=$g
        printf "agent B9b\nrun %s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        break
      fi
    done
    [ -z "$CARD" ] && sleep 30
  done
  export CUDA_VISIBLE_DEVICES=$CARD
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock; CARD=""; }
