#!/bin/bash
set -euo pipefail
base=/data/relcfd/chayo/physmorph_v2
export PHYSMORPH_RUN_REPO="$base/work/p303/code_window_selection2"
source "$PHYSMORPH_RUN_REPO/scripts/ops/hyde06_env.sh"
source "$PHYSMORPH_RUN_REPO/scripts/ops/gpu_env.sh"
export CUDA_VISIBLE_DEVICES=1
out="$base/work/p303/p332_independent_audit2"
test -f "$base/work/p303/p332_identity2/result.json"
exec 9>"$base/maintenance/gpu_launch.lock"
flock -x 9
now=$(date +%s)
last=0
for marker in "$base"/repro/current_pair/launch_marker_*.txt "$base/maintenance/last_gpu_launch_epoch"; do
  if test -f "$marker"; then
    value=$(cat "$marker")
    if ((value > last)); then last=$value; fi
  fi
done
if ((now-last < 50)); then
  delay=$((50-now+last))
  ((delay <= 50)) || exit 75
  sleep "$delay"
  now=$(date +%s)
fi
test -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)"
memory=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
((memory < 128)) || exit 77
used=$(du -sb "$base" | cut -f1)
((used + 100000000 < 100000000000)) || exit 76
for suffix in '' .start .log .json; do test ! -e "$out$suffix" || exit 73; done
(set -o noclobber; printf '%s\n' "$now" > "$out.start")
set -o noclobber
exec > "$out.log" 2>&1
set +o noclobber
printf '%s\n' "$now" > "$base/maintenance/last_gpu_launch_epoch"
flock -u 9
printf 'launch_utc=%s gpu=1 project_bytes=%s\n' "$(date -u +%FT%TZ)" "$used"
cd "$REPO"
exec "$PY" scripts/ops/cuda_python.py "$base/work/p303/refute_p332.py" \
  --source "$PHYSMORPH_RUN_REPO" --run "$base/work/p303/p332_identity2" \
  --expected-version 40d4bc95067629ecf82c53e6c4da5e7bbf77cd2f --out "$out.json" \
  --launcher "$base/work/p303/launch_p332_audit2.sh"
