#!/bin/bash
# b12_gpu.sh WAIT_S LABEL CMD... -- B12's card claim (D145's lock protocol, the coordinator's rule for B12, the lowest
# priority): wait until one card has had no gpu_locks/gpuN.lock and < 2000 MiB in use for WAIT_S consecutive seconds
# (600 for every measured batch; a few-minute smoke test may use less), claim it by mkdir (atomic) with an info file
# (agent, run, pid, start), run CMD with CUDA_VISIBLE_DEVICES set, and release the lock by a trap.
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
WAIT=$1; LABEL=$2; shift 2
declare -A since
GPU=""
while [ -z "$GPU" ]; do
  now=$(date +%s)
  while IFS=, read -r idx mem; do
    idx=${idx// /}; mem=${mem// /}
    if [ ! -e $L/gpu$idx.lock ] && [ "$mem" -lt 2000 ]; then
      [ -z "${since[$idx]}" ] && since[$idx]=$now
      if [ $((now - since[$idx])) -ge $WAIT ] && mkdir $L/gpu$idx.lock 2>/dev/null; then
        GPU=$idx
        break
      fi
    else
      unset "since[$idx]"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  [ -z "$GPU" ] && sleep 30
done
trap 'rm -rf $L/gpu$GPU.lock; echo "$(date "+%F %T %Z") released gpu$GPU ($LABEL)"' EXIT
trap 'exit 143' INT TERM
printf "agent B12\nrun %s\npid %s\nstart %s\n" "$LABEL" "$$" "$(date '+%F %T %Z')" > $L/gpu$GPU.lock/info
echo "$(date '+%F %T %Z') claimed gpu$GPU for $LABEL (idle >= ${WAIT} s)"
CUDA_VISIBLE_DEVICES=$GPU "$@"
rc=$?
echo "$(date '+%F %T %Z') $LABEL rc $rc"
exit $rc
