#!/bin/bash
# Isolated baseline/candidate runs; all paths and caches remain under /data.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
export PHYSMORPH_RUN_REPO=$HERE
source "$HERE/scripts/ops/hyde06_env.sh"
GPU=$1; NAME=$2; TARGET=$3; N=$4; WINDOWS=$5; MODE=$6
[[ "$NAME" =~ ^c291_[a-zA-Z0-9_]+$ ]] || exit 2
[[ "$TARGET" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
mkdir -p "$OUT/c291"
FL=(--ctrl_rprop --ctrl_rprop_smooth --ctrl_rprop_arrived --u_rprop --u_rprop_floor 0
    --settle_pin --settle_pin_assim --settle_pin_slip)
if [ "$N" -ge 100000 ]; then
    FL+=(--disc_ref --shift_sub --commit_pic --render_paced --render_paced_conv
         --pace_coherent --ctrl_rprop_k 8 --ctrl_rprop_hold)
    [ "$TARGET" != bunny ] || FL+=(--settle_pin_ray)
else
    FL+=(--ctrl_rprop_hold_onset)
fi
case "$MODE" in
    baseline) ;;
    body) FL+=(--body_ctrl) ;;
    body_phys) FL+=(--body_ctrl --lambda_auto 0) ;;
    *) exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=$GPU
date -Is > "$OUT/c291/$NAME.start"
# RECIPE is a trusted space-delimited flag list from the repository.
$PY scripts/pipeline_run.py --arms render_full_dt_iso_nn --tgt "assets/$TARGET.obj" --n "$N" \
    $RECIPE "${FL[@]}" --stop_after_windows "$WINDOWS" --out "$OUT/c291/$NAME" > "$OUT/c291/$NAME.log" 2>&1
date -Is > "$OUT/c291/$NAME.done"
