#!/bin/bash
# P302 immutable-snapshot operator / matched prefix experiments, hyde06 only.
set -euo pipefail
base=/data/relcfd/chayo/physmorph_v2
mode=${1:?operator or zero or gs or off or compare}
gpu=${2:?GPU index}
tag=${3:?unique tag}
case "$mode" in operator|zero|gs|off|compare) ;; *) exit 2;; esac
[[ "$gpu" =~ ^[0-3]$ && "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
export PHYSMORPH_RUN_REPO=$(cd "$(dirname "$0")/../.." && pwd)
case "$PHYSMORPH_RUN_REPO" in "$base"/work/p302/*) ;; *) echo 'Use a P302 code snapshot' >&2; exit 2;; esac
source "$PHYSMORPH_RUN_REPO/scripts/ops/hyde06_env.sh"
source "$PHYSMORPH_RUN_REPO/scripts/ops/gpu_env.sh"
export CUDA_VISIBLE_DEVICES="$gpu"
exec 9>"$base/maintenance/gpu_launch.lock"
flock -x 9
now=$(date +%s); last=0
for marker in "$base"/repro/current_pair/launch_marker_*.txt "$base/maintenance/last_gpu_launch_epoch"; do
    if test -f "$marker"; then value=$(cat "$marker"); (( value <= last )) || last=$value; fi
done
(( now-last >= 50 )) || { echo 'GPU launches must be >=50s apart' >&2; exit 75; }
out="$base/work/p302/$tag"
test ! -e "$out.start" && test ! -e "$out.json" && test ! -e "$out.log"
test ! -e "$out" && test ! -e "$out.npz" && test ! -e "${out}_render_full_dt_iso_nn.npz"
used=$(du -sb "$base" | cut -f1)
(( used < 100000000000 )) || { echo 'Project exceeds 100GB; clean obsolete results first' >&2; exit 76; }
# The caller checks nvidia-smi; reserve the unique run and shared launch interval.
(set -o noclobber; printf '%s\n' "$now" > "$out.start")
set -o noclobber
exec > "$out.log" 2>&1
set +o noclobber
printf '%s\n' "$now" > "$base/maintenance/last_gpu_launch_epoch"
flock -u 9
cd "$REPO"
if [[ "$mode" == operator ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/surface_render_cuda.py --out "$out.json"
fi
if [[ "$mode" == compare ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/surface_render_compare.py \
        --control "$base/work/p302/control1" --candidate "$base/work/p302/candidate1" \
        --repeat "$base/work/p302/control_repeat1" --out "$out.json"
fi
extra=(--surface-gs-weight 0)
[[ "$mode" != gs ]] || extra=(--surface-gs-weight 1)
[[ "$mode" != off ]] || extra=(--surface-gs-weight 1 --physical)
exec "$PY" scripts/ops/cuda_python.py scripts/probes/gpu_pipeline.py \
    --windows 1 --iters 2 --archive --motion-accounting --outer-render-committed \
    --commit-pic-objective --no-shift-sub "${extra[@]}" --out "$out"
