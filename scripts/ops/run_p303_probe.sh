#!/bin/bash
# Isolated current-adjoint/no-shift physics and raster cutoff experiments.
set -euo pipefail
base=/data/relcfd/chayo/physmorph_v2
mode=${1:?baseline or raw or raw-off or raster or continuous or quality or phase or render-compare}
gpu=${2:?GPU index}
tag=${3:?unique tag}
case "$mode" in baseline|raw|raw-off|raw-no-pin|raster|continuous|live-continuous|quality|phase|render-compare|gs-zero|gs-one|gs-compare|pin-quality|pin-prefix|reference-swap|reference-verify|inner-budget|inner-verify|terminal-braking|braking-capture|braking-compensation|live-braking-compensation|running-braking-repair|remainder-braking-repair|quality-braking-repair|paired-braking-repair|silhouette-braking-repair|coverage-paths|silhouette-pixels|metric-verify|horizon-verify|reporting-verify|gradient-reporting-verify|shape-verify|baseline-shape|raw-shape|baseline-motion|raw-motion|full-baseline|full-raw|full-raw-no-layer|support-braking-repair|frozen-replay-noise|frozen-state-identity) ;; *) exit 2;; esac
[[ "$gpu" =~ ^[0-3]$ && "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
export PHYSMORPH_RUN_REPO=$(cd "$(dirname "$0")/../.." && pwd)
case "$PHYSMORPH_RUN_REPO" in "$base"/work/p303/code*) ;; *) exit 2;; esac
source "$PHYSMORPH_RUN_REPO/scripts/ops/hyde06_env.sh"
source "$PHYSMORPH_RUN_REPO/scripts/ops/gpu_env.sh"
export CUDA_VISIBLE_DEVICES="$gpu"
exec 9>"$base/maintenance/gpu_launch.lock"
flock -x 9
now=$(date +%s); last=0
for marker in "$base"/repro/current_pair/launch_marker_*.txt "$base/maintenance/last_gpu_launch_epoch"; do
    if test -f "$marker"; then value=$(cat "$marker"); (( value <= last )) || last=$value; fi
done
if (( now-last < 50 )); then
    delay=$((50-now+last))
    (( delay <= 50 )) || { echo 'Launch clock moved backwards' >&2; exit 75; }
    echo "Waiting ${delay}s for the shared GPU launch interval"
    sleep "$delay"
    now=$(date +%s)
fi
out="$base/work/p303/$tag"
for suffix in '' .start .json .log .npz _render_full_dt_iso_nn.npz; do test ! -e "$out$suffix"; done
used=$(du -sb "$base" | cut -f1)
(( used < 100000000000 )) || { echo 'Project exceeds 100GB; clean obsolete results first' >&2; exit 76; }
if [[ "$mode" == full-baseline || "$mode" == full-raw || "$mode" == full-raw-no-layer ]]; then
    (( used + 30000000000 < 100000000000 )) || { echo 'Full-horizon30GB reservation exceeds100GB; clean verified obsolete results first' >&2; exit 76; }
fi
if [[ "$mode" == baseline-shape || "$mode" == raw-shape ]]; then
    (( used + 2000000000 < 100000000000 )) || { echo 'Phase-shape2GB reservation exceeds100GB' >&2; exit 76; }
fi
(set -o noclobber; printf '%s\n' "$now" > "$out.start")
set -o noclobber
exec > "$out.log" 2>&1
set +o noclobber
printf '%s\n' "$now" > "$base/maintenance/last_gpu_launch_epoch"
flock -u 9
cd "$REPO"
if [[ "$mode" == full-baseline || "$mode" == full-raw || "$mode" == full-raw-no-layer ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/full_horizon.py --arm "${mode#full-}" --out "$out"
fi
if [[ "$mode" == baseline-motion || "$mode" == raw-motion ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/horizon_motion.py \
        --prefix "$base/work/p303/full_${mode%-motion}1" --out "$out"
fi
if [[ "$mode" == horizon-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_compute_cuda.py::test_horizon_motion_archive_matches_cpu_with_device_geometry \
        -q --junitxml="$out.xml"
fi
if [[ "$mode" == reporting-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_render_reporting_cuda.py -q --junitxml="$out.xml"
fi
if [[ "$mode" == gradient-reporting-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_gradient_reporting_cuda.py -q --junitxml="$out.xml"
fi
if [[ "$mode" == shape-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_horizon_shape_cuda.py -q --junitxml="$out.xml"
fi
if [[ "$mode" == baseline-shape || "$mode" == raw-shape ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/horizon_shape.py \
        --motion "$base/work/p303/${mode%-shape}_motion1" --out "$out"
fi
if [[ "$mode" == metric-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_compute_cuda.py::test_fixed_metric_footprint_avoids_dynamic_histogram \
        tests/test_compute_cuda.py::test_metric_summary_matches_cpu_without_cpu_neighbors \
        -q --junitxml="$out.xml"
fi
if [[ "$mode" == silhouette-pixels ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/silhouette_pixels.py --out "$out"
fi
if [[ "$mode" == support-braking-repair ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/support_braking_repair.py --out "$out"
fi
if [[ "$mode" == coverage-paths ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/coverage_paths.py --out "$out"
fi
if [[ "$mode" == silhouette-braking-repair ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/silhouette_braking_repair.py --out "$out"
fi
if [[ "$mode" == paired-braking-repair ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/paired_braking_repair.py --out "$out"
fi
if [[ "$mode" == quality-braking-repair ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/quality_braking_repair.py --out "$out"
fi
if [[ "$mode" == remainder-braking-repair ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/remainder_braking_repair.py --out "$out"
fi
if [[ "$mode" == running-braking-repair ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/running_braking_repair.py --out "$out"
fi
if [[ "$mode" == live-braking-compensation ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/live_braking_compensation.py --out "$out"
fi
if [[ "$mode" == frozen-state-identity ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/frozen_state_identity.py \
        --archive "$base/work/p303/braking_capture1/owned_window.npz" --out "$out"
fi
if [[ "$mode" == frozen-replay-noise ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/frozen_replay_noise.py \
        --archive "$base/work/p303/braking_capture1/owned_window.npz" --out "$out.json"
fi
if [[ "$mode" == braking-compensation ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/braking_compensation.py \
        --capture "$base/work/p303/braking_capture1" --snapshot "$base/work/p303/code_braking_capture1" --out "$out"
fi
if [[ "$mode" == braking-capture ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/braking_capture.py --out "$out"
fi
if [[ "$mode" == terminal-braking ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/terminal_braking.py --out "$out"
fi
if [[ "$mode" == inner-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/inner_budget_verify.py \
        --run "$base/work/p303/inner_budget2" --snapshot "$base/work/p303/code_inner_budget2" --out "$out.json"
fi
if [[ "$mode" == inner-budget ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/inner_budget.py --out "$out"
fi
if [[ "$mode" == reference-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/reference_swap_verify.py \
        --run "$base/work/p303/reference_swap2" --snapshot "$base/work/p303/code_reference2" --out "$out.json"
fi
if [[ "$mode" == reference-swap ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/reference_swap.py --out "$out"
fi
if [[ "$mode" == pin-quality ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/quality_compare.py \
        --baseline "$base/work/p303/raw_repeat24" --candidate "$base/work/p303/raw_no_pin24_schema" \
        --intervention pin_admission_off_prefix --out "$out.json"
fi
if [[ "$mode" == pin-prefix ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/pin_prefix_parity.py \
        --control "$base/work/p303/raw24a" --repeat "$base/work/p303/raw_repeat24" \
        --no-pin "$base/work/p303/raw_no_pin24_schema" --out "$out.json"
fi
if [[ "$mode" == gs-zero || "$mode" == gs-one ]]; then
    export PYTHONPATH="$base/deps/continuous_raster2:$PYTHONPATH"
    weight=0
    [[ "$mode" != gs-one ]] || weight=1
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/gpu_pipeline.py \
        --windows 1 --iters 2 --archive --motion-accounting --outer-render-committed \
        --commit-pic-objective --no-shift-sub --surface-gs-weight "$weight" \
        --surface-gs-raster continuous --out "$out"
fi
if [[ "$mode" == gs-compare ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/surface_render_compare.py \
        --control "$base/work/p303/gs_zero1" --candidate "$base/work/p303/gs_one1" \
        --repeat "$base/work/p303/gs_zero_repeat1" --out "$out.json"
fi
if [[ "$mode" == live-continuous ]]; then
    export PYTHONPATH="$base/deps/continuous_raster2:$PYTHONPATH"
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/surface_render_cuda.py \
        --backend continuous --out "$out.json"
fi
if [[ "$mode" == raster || "$mode" == continuous ]]; then
    backend=legacy
    if [[ "$mode" == continuous ]]; then
        backend=continuous
        export PYTHONPATH="$base/deps/continuous_raster2:$PYTHONPATH"
    fi
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/raster_cutoff_audit.py --backend "$backend" --out "$out.json"
fi
if [[ "$mode" == quality ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/quality_compare.py \
        --baseline "$base/work/p303/baseline24" --candidate "$base/work/p303/raw24a" \
        --intervention shared_pic_off_prefix --out "$out.json"
fi
if [[ "$mode" == phase ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/raw_phase.py \
        --baseline "$base/work/p303/baseline24" --candidate "$base/work/p303/raw24a" \
        --reference "$base/work/p303/quality24.json" --out "$out.json"
fi
if [[ "$mode" == render-compare ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/render_influence.py \
        --on "$base/work/p303/raw24a" --off "$base/work/p303/raw_off24" --out "$out.json"
fi
extra=(--commit-pic-objective)
[[ "$mode" == baseline ]] || extra=(--no-commit-pic)
[[ "$mode" != raw-off ]] || extra+=(--physical)
[[ "$mode" != raw-no-pin ]] || extra+=(--no-settle-pin)
exec "$PY" scripts/ops/cuda_python.py scripts/probes/gpu_pipeline.py \
    --windows 24 --iters 8 --archive --motion-accounting --outer-render-committed \
    --no-shift-sub "${extra[@]}" --out "$out"
