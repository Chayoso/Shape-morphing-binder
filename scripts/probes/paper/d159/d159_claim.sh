# d159_claim.sh -- sourced: claim_card / release_card of D159's chain (gpu_locks/gpuN.lock by mkdir with an info file
# "agent D159"; a card with no lock, no other user's compute process, < 2000 MiB, free for IDLE seconds (900, behind the
# queued agents' 600), the pool's shared first-free time honoured; GPU 2 never). Needs B, O, L, S and log().
IDLE=${D159_IDLE:-900}
CARD=""
claim_card() {
  local what=$1; declare -A since
  local last_report=0
  while [ -z "$CARD" ]; do
    local now=$(date +%s) reasons=""
    for g in 3 1 0; do
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      local uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
      local other=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | while IFS=', ' read -r u p; do
                    o=$(ps -o user= -p $p 2>/dev/null); [ -n "$o" ] && [ "$o" != chayo ] && echo x; done | wc -l)
      local mine=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
      local why=""
      [ -d $L/gpu$g.lock ] && why="lock($(grep -m1 agent $L/gpu$g.lock/info 2>/dev/null))"
      [ "$used" -gt 2000 ] && why="$why used=${used}MiB"
      [ "$other" -gt 0 ] && why="$why other-user"
      [ "$mine" -gt 0 ] && why="$why apps=$mine"
      if [ -n "$why" ]; then since[$g]=""; reasons="$reasons gpu$g:$why;"; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      local fs=$(cat $B/tmp/d145/free_since/gpu$g 2>/dev/null); [ -n "$fs" ] && [ "$fs" -lt "${since[$g]}" ] && since[$g]=$fs
      if [ $((now - ${since[$g]})) -ge $IDLE ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        CARD=$g
        printf "agent D159\nrun d159/%s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        break
      fi
    done
    if [ -z "$CARD" ] && [ $((now - last_report)) -ge 900 ]; then log "waiting:$reasons"; last_report=$now; fi
    [ -z "$CARD" ] && sleep 60
  done
  export CUDA_VISIBLE_DEVICES=$CARD
  log "claimed gpu$CARD for [$what]"
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "released gpu$CARD"; CARD=""; }
