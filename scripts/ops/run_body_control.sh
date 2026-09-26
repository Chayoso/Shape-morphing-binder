#!/bin/bash
# Isolated baseline/candidate runs; all paths and caches remain under /data.
set -euo pipefail
if [ "$#" -lt 6 ]; then
    echo 'usage: run_body_control.sh GPU NAME TARGET N WINDOWS MODE --input-reference BUNDLE [--target-reference PATH] [--legacy-comparison]' >&2
    exit 2
fi
GPU=$1; NAME=$2; TARGET=$3; N=$4; WINDOWS=$5; MODE=$6
shift 6
TARGET_REFERENCE=${PHYSMORPH_TARGET_REFERENCE:-}
INPUT_REFERENCE=${PHYSMORPH_INPUT_REFERENCE:-}
LEGACY_COMPARISON=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --target-reference)
            [ "$#" -ge 2 ] || { echo 'Missing target reference path' >&2; exit 2; }
            TARGET_REFERENCE=$2; shift 2 ;;
        --input-reference)
            [ "$#" -ge 2 ] || { echo 'Missing input bundle path' >&2; exit 2; }
            INPUT_REFERENCE=$2; shift 2 ;;
        --legacy-comparison) LEGACY_COMPARISON=1; shift ;;
        *) echo "Unknown launcher argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$GPU" == 0 || "$GPU" == 2 ]] || { echo 'Use assigned GPU 0 or 2' >&2; exit 2; }
if [ -n "$INPUT_REFERENCE" ]; then
    [ -f "$INPUT_REFERENCE" ] || { echo 'Input reference is not a file' >&2; exit 2; }
    INPUT_REFERENCE=$(cd "$(dirname -- "$INPUT_REFERENCE")" && pwd)/$(basename -- "$INPUT_REFERENCE")
    TARGET_REFERENCE=${TARGET_REFERENCE:-$INPUT_REFERENCE}
fi
if [ "$LEGACY_COMPARISON" -eq 0 ] && [ ! -f "$TARGET_REFERENCE" ]; then
    echo 'CUDA runs require --target-reference PATH (or PHYSMORPH_TARGET_REFERENCE); no CPU fallback' >&2
    exit 2
fi
if [ -n "$TARGET_REFERENCE" ]; then
    [ -f "$TARGET_REFERENCE" ] || { echo 'Target reference is not a file' >&2; exit 2; }
    TARGET_REFERENCE=$(cd "$(dirname -- "$TARGET_REFERENCE")" && pwd)/$(basename -- "$TARGET_REFERENCE")
fi
HERE=$(cd "$(dirname "$0")/../.." && pwd)
export PHYSMORPH_RUN_REPO=$HERE
source "$HERE/scripts/ops/hyde06_env.sh"
[[ "$NAME" =~ ^c291_[a-zA-Z0-9_]+$ ]] || exit 2
[[ "$TARGET" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
mkdir -p "$OUT/c291"
if [ -e "$OUT/c291/$NAME.start" ] || [ -e "$OUT/c291/$NAME.json" ] || [ -e "$OUT/c291/$NAME.log" ]; then
    echo "Refusing to overwrite existing run evidence: $NAME" >&2
    exit 2
fi
FL=(--ctrl_rprop --ctrl_rprop_smooth --ctrl_rprop_arrived --u_rprop --u_rprop_floor 0
    --settle_pin --settle_pin_assim --settle_pin_slip)
[ -z "$INPUT_REFERENCE" ] || FL+=(--input_reference "$INPUT_REFERENCE")
if [ "$N" -ge 100000 ]; then
    FL+=(--disc_ref --shift_sub --commit_pic --render_paced --render_paced_conv
         --pace_coherent --ctrl_rprop_k 8 --ctrl_rprop_hold)
    [ "$TARGET" != bunny ] || FL+=(--settle_pin_ray)
else
    FL+=(--ctrl_rprop_hold_onset)
fi
case "$MODE" in
    baseline) ;;
    baseline_confirm) FL+=(--settle_pin_confirm) ;;
    body) FL+=(--body_ctrl) ;;
    body_no_dfc) FL+=(--body_ctrl --body_no_dfc) ;;
    force_normalized) FL+=(--body_ctrl --body_no_dfc --body_step_normalized) ;;
    force_normalized_phys) FL+=(--body_ctrl --body_no_dfc --body_step_normalized --lambda_auto 0) ;;
    force_terminal) FL+=(--body_ctrl --body_no_dfc --body_step_normalized --body_terminal_ctrl) ;;
    force_terminal_phys) FL+=(--body_ctrl --body_no_dfc --body_step_normalized --body_terminal_ctrl --lambda_auto 0) ;;
    body_terminal) FL+=(--body_ctrl --body_step_normalized --body_terminal_ctrl) ;;
    body_terminal_phys) FL+=(--body_ctrl --body_step_normalized --body_terminal_ctrl --lambda_auto 0) ;;
    body_phys) FL+=(--body_ctrl --lambda_auto 0) ;;
    *) exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=$GPU
if [ "$LEGACY_COMPARISON" -eq 1 ]; then
    COMMAND=("$PY" scripts/pipeline_run.py)
    FL+=(--compute_backend legacy)
    [ -z "$TARGET_REFERENCE" ] || FL+=(--target_reference "$TARGET_REFERENCE")
else
    COMMAND=(bash "$HERE/scripts/ops/run_gpu_pipeline.sh" "$GPU")
    FL+=(--target_reference "$TARGET_REFERENCE")
fi
date -Is > "$OUT/c291/$NAME.start"
# RECIPE is a trusted space-delimited flag list from the repository.
"${COMMAND[@]}" --arms render_full_dt_iso_nn --tgt "assets/$TARGET.obj" --n "$N" \
    $RECIPE "${FL[@]}" --stop_after_windows "$WINDOWS" --out "$OUT/c291/$NAME" > "$OUT/c291/$NAME.log" 2>&1
date -Is > "$OUT/c291/$NAME.done"
