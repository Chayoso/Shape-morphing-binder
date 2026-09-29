#!/bin/bash
# Isolated current-adjoint/no-shift physics and raster cutoff experiments.
set -euo pipefail
base=/data/relcfd/chayo/physmorph_v2
mode=${1:?probe mode}
gpu=${2:?GPU index}
tag=${3:?unique tag}
case "$mode" in current-successor-preparation|current-successor-verify|post-assimilation-candidate|post-assimilation-selection-verify|post-assimilation-window-fp64|post-assimilation-window|post-assimilation-window-verify|post-assimilation-verify|assimilation-adjoint-verify|window-selection|window-selection-verify|joint-withdrawal-search|checkpoint-merit-verify|production-withdrawal|prepared-withdrawal-verify|trajectory-reporting-verify|withdrawal-adjoint-verify|fragment-adjoint-verify|full-fragment-legacy|full-fragment-retained|fragment-motion|fragment-shape|fragment-quality|fragment-phase) ;; baseline|raw|raw-off|raw-no-pin|raster|continuous|live-continuous|quality|phase|render-compare|gs-zero|gs-one|gs-compare|pin-quality|pin-prefix|reference-swap|reference-verify|inner-budget|inner-verify|terminal-braking|braking-capture|braking-compensation|live-braking-compensation|running-braking-repair|remainder-braking-repair|quality-braking-repair|paired-braking-repair|silhouette-braking-repair|coverage-paths|silhouette-pixels|metric-verify|horizon-verify|reporting-verify|gradient-reporting-verify|shape-verify|baseline-shape|raw-shape|baseline-motion|raw-motion|full-baseline|full-raw|full-raw-no-layer|support-braking-repair|frozen-replay-noise|frozen-state-identity|withdrawal-capture|withdrawal-analyze|withdrawal-verify) ;; *) exit 2;; esac
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
if [[ "$mode" == full-baseline || "$mode" == full-raw || "$mode" == full-raw-no-layer || "$mode" == withdrawal-capture || "$mode" == full-fragment-* ]]; then
    (( used + 30000000000 < 100000000000 )) || { echo 'Full-horizon30GB reservation exceeds100GB; clean verified obsolete results first' >&2; exit 76; }
fi
if [[ "$mode" == baseline-shape || "$mode" == raw-shape || "$mode" == fragment-shape ]]; then
    (( used + 2000000000 < 100000000000 )) || { echo 'Phase-shape2GB reservation exceeds100GB' >&2; exit 76; }
fi
if [[ "$mode" == withdrawal-analyze ]]; then
    (( used + 6000000000 < 100000000000 )) || { echo 'Withdrawal6GB reservation exceeds100GB' >&2; exit 76; }
fi
if [[ "$mode" == production-withdrawal || "$mode" == window-selection || "$mode" == post-assimilation-window || "$mode" == post-assimilation-window-fp64 ]]; then
    (( used + 3000000000 < 100000000000 )) || { echo 'Prepared withdrawal3GB reservation exceeds100GB' >&2; exit 76; }
fi
if [[ "$mode" == joint-withdrawal-search || "$mode" == post-assimilation-candidate ]]; then
    (( used + 12000000000 < 100000000000 )) || { echo 'Joint search12GB reservation exceeds100GB' >&2; exit 76; }
fi
if [[ "$mode" == current-successor-preparation ]]; then
    (( used + 8000000000 < 100000000000 )) || { echo 'Current preparation8GB reservation exceeds100GB' >&2; exit 76; }
fi
(set -o noclobber; printf '%s\n' "$now" > "$out.start")
set -o noclobber
exec > "$out.log" 2>&1
set +o noclobber
printf '%s\n' "$now" > "$base/maintenance/last_gpu_launch_epoch"
flock -u 9
cd "$REPO"
if [[ "$mode" == current-successor-preparation ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/current_successor_preparation.py --out "$out"
fi
if [[ "$mode" == current-successor-verify ]]; then
    export PHYSMORPH_PREPARATION_CUDA_TEST=1
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_preparation_geometry_cuda.py tests/test_current_successor_cuda.py -q -s --junitxml="$out.xml"
fi
if [[ "$mode" == post-assimilation-window-fp64 ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/post_assimilation_window.py --out "$out" --assim-fp64
fi
if [[ "$mode" == post-assimilation-window ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/post_assimilation_window.py --out "$out"
fi
if [[ "$mode" == post-assimilation-candidate ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/post_assimilation_candidate.py \
        --out "$out" --enable-candidate-search
fi
if [[ "$mode" == post-assimilation-selection-verify ]]; then
    export PHYSMORPH_POST_SELECTION_CUDA_TEST=1
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_post_selection_cuda.py -q -s --junitxml="$out.xml"
fi
if [[ "$mode" == post-assimilation-window-verify ]]; then
    export PHYSMORPH_POST_WINDOW_CUDA_TEST=1
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_post_assimilation_window_cuda.py -q -s --junitxml="$out.xml"
fi
if [[ "$mode" == post-assimilation-verify ]]; then
    export PHYSMORPH_POST_ASSIM_CUDA_TEST=1
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_post_assimilation_adjoint_cuda.py -q -s --junitxml="$out.xml"
fi
if [[ "$mode" == assimilation-adjoint-verify ]]; then
    export PHYSMORPH_ASSIMILATION_CUDA_TEST=1
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_assimilation_adjoint_cuda.py -q -s --junitxml="$out.xml"
fi
if [[ "$mode" == window-selection ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/window_selection.py --out "$out"
fi
if [[ "$mode" == window-selection-verify ]]; then
    export PHYSMORPH_CUDA_TESTS=1
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_window_selection_cuda.py -q -s --junitxml="$out.xml"
fi
if [[ "$mode" == joint-withdrawal-search ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/joint_withdrawal_search.py --out "$out"
fi
if [[ "$mode" == withdrawal-capture ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/control_withdrawal.py capture --out "$out"
fi
if [[ "$mode" == withdrawal-analyze ]]; then
    source_tag=${4:?captured run tag}
    [[ "$source_tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/control_withdrawal.py analyze \
        --source "$base/work/p303/$source_tag" --out "$out"
fi
if [[ "$mode" == withdrawal-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_withdrawal_cuda.py -q --junitxml="$out.xml"
fi
if [[ "$mode" == production-withdrawal ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/prepared_withdrawal.py --out "$out"
fi
if [[ "$mode" == prepared-withdrawal-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_frozen_withdrawal_window_cuda.py -q -s --junitxml="$out.xml"
fi
if [[ "$mode" == checkpoint-merit-verify ]]; then
    export PHYSMORPH_CUDA_TESTS=1
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_checkpoint_merit_terms_cuda.py -q -s --junitxml="$out.xml"
fi
if [[ "$mode" == withdrawal-adjoint-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_withdrawal_adjoint_cuda.py -q -s --junitxml="$out.xml"
fi
if [[ "$mode" == fragment-adjoint-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_withdrawal_adjoint_cuda.py tests/test_fragment_adjoint_observer_cuda.py -q -s --junitxml="$out.xml"
fi
if [[ "$mode" == full-fragment-legacy || "$mode" == full-fragment-retained ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/fragment_adjoint_compare.py \
        --mode "${mode#full-fragment-}" --out "$out"
fi
if [[ "$mode" == fragment-motion || "$mode" == fragment-shape ]]; then
    source_tag=${4:?source or motion tag}
    [[ "$source_tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
    if [[ "$mode" == fragment-motion ]]; then
        exec "$PY" scripts/ops/cuda_python.py scripts/probes/horizon_motion.py \
            --prefix "$base/work/p303/$source_tag" --out "$out"
    fi
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/horizon_shape.py \
        --motion "$base/work/p303/$source_tag" --out "$out"
fi
if [[ "$mode" == fragment-quality ]]; then
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/quality_compare.py \
        --baseline "$base/work/p303/p327_legacy1" --candidate "$base/work/p303/p327_retained1" \
        --intervention fragment_adjoint_retained_full --out "$out.json"
fi
if [[ "$mode" == fragment-phase ]]; then
    source_tag=${4:?quality reference tag}
    [[ "$source_tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
    exec "$PY" scripts/ops/cuda_python.py scripts/probes/raw_phase.py \
        --baseline "$base/work/p303/p327_legacy1" --candidate "$base/work/p303/p327_retained1" \
        --reference "$base/work/p303/$source_tag.json" --out "$out.json"
fi
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
if [[ "$mode" == trajectory-reporting-verify ]]; then
    exec "$PY" scripts/ops/cuda_python.py /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/pytest \
        tests/test_trajectory_reporting_cuda.py -q --junitxml="$out.xml"
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
