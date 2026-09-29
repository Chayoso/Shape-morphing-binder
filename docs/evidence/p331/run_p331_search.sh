#!/bin/bash
set -euo pipefail
base=/data/relcfd/chayo/physmorph_v2
expected=${1:?frozen commit}
[[ "$expected" =~ ^[0-9a-f]{40}$ ]] || exit 2
code=$base/work/p303/code_joint_withdrawal1
work=$base/work/p303
cd "$code"
[[ "$(cat VERSION)" == "$expected" ]] || exit 2
for path in "$work/p331_search1" "$work/p331_search1.memory.json" "$work/p331_search1.memory.stop"; do
    test ! -e "$path" || exit 2
done
bash scripts/ops/run_p303_probe.sh joint-withdrawal-search 0 p331_search1 &
job=$!
printf 'Observed physical process PID %s\n' "$job"
/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python "$work/sample_device_memory.py" \
    --pid "$job" --gpu 0 --out "$work/p331_search1.memory.json" \
    --stop "$work/p331_search1.memory.stop" > "$work/p331_search1.memory.log" 2>&1 &
monitor=$!
set +e
wait "$job"
status=$?
set -e
printf '%s\n' "$status" > "$work/p331_search1.memory.stop"
wait "$monitor"
printf 'Physical process exit %s\n' "$status"
exit "$status"
