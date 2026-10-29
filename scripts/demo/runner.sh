#!/bin/bash
# runner.sh GPU -- one card's demo queue: holds gpu_locks/gpuGPU.lock (info: agent DEMO / run / pid / start; released
# by the trap), takes the queue's lines (tmp/demo/queue.txt: NAME|ARCHIVE|KIND|FRAMES|BLENDER_KEYS) in order, each
# claimed by mkdir tmp/demo/claims/NAME, and runs clip.sh on them; exits when no unclaimed line is left.
gpu=$1
B=/data/relcfd/chayo/physmorph_v2; D=$B/tmp/demo; L=$B/gpu_locks/gpu$gpu.lock
mkdir -p $D/claims
if [ -d $L ] && ! grep -q "agent DEMO" $L/info 2>/dev/null; then echo "gpu$gpu is held by another agent" >&2; exit 1; fi
mkdir -p $L
printf "agent DEMO\nrun demo/blender: 4K Blender demo clips (tmp/demo/queue.txt)\npid %s\nstart %s\n" $$ "$(date '+%F %T %Z')" > $L/info
trap 'rm -rf $L; echo "$(date "+%m-%d %H:%M:%S") [gpu$gpu] runner exit, lock released" >> $D/status.txt' EXIT
echo "$(date '+%m-%d %H:%M:%S') [gpu$gpu] runner start pid $$" >> $D/status.txt
while true; do
  got=""
  while IFS='|' read -r name arc kind frames keys; do
    [ -z "$name" ] && continue
    case $name in \#*) continue;; esac
    if mkdir $D/claims/$name 2>/dev/null; then got=1; break; fi
  done < $D/queue.txt
  [ -z "$got" ] && break
  bash $D/scripts/clip.sh $gpu "$name" "$arc" "$kind" "$frames" "$keys"
done
