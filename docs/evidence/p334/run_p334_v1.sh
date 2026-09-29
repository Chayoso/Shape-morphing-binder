#!/bin/bash
set -euo pipefail
base=/data/relcfd/chayo/physmorph_v2
expected=${1:?frozen commit}
tag=${2:?unique tag}
[[ "$expected" =~ ^[0-9a-f]{40}$ && "$tag" =~ ^p334_[a-z0-9_]+$ ]] || exit 2
code=$base/work/p303/code_post_assimilation1
work=$base/work/p303
cd "$code"
[[ "$(cat VERSION)" == "$expected" ]] || exit 2
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
memory=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
((memory < 128)) || exit 77
for suffix in '' .memory.json .memory.stop .exit; do test ! -e "$work/$tag$suffix"; done
bash scripts/ops/run_p303_probe.sh post-assimilation-verify 0 "$tag" &
job=$!
printf 'Observed CUDA process PID %s\n' "$job"
/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python "$work/sample_device_memory.py" \
    --pid "$job" --gpu 0 --out "$work/$tag.memory.json" \
    --stop "$work/$tag.memory.stop" > "$work/$tag.memory.log" 2>&1 &
monitor=$!
set +e
wait "$job"
status=$?
set -e
printf '%s\n' "$status" > "$work/$tag.memory.stop"
wait "$monitor"
printf '%s\n' "$status" > "$work/$tag.exit"
printf 'CUDA process exit %s\n' "$status"
exit "$status"
