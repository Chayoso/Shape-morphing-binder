#!/bin/bash
# Photoreal videos for finished runs: <prefix>_<T> archives -> report_<prefix>/<T>/<T>_photoreal.mp4
# (+ .components.txt QA sidecar). Usage: photoreal_batch.sh <gpu> <prefix> <targets...>
source /data/relcfd/chayo/physmorph_v2/repo/scripts/ops/hyde06_env.sh
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 EGL_PLATFORM=surfaceless
GPU=$1; PFX=$2; shift 2
ARM=render_full_dt_iso_nn
SURFACE=${SURFACE:-poisson}   # docs/method.md 10.12: the outer-layer Poisson surface; mc = the v8 level set
TRACK=${TRACK:-0}             # docs/method.md 10.15: TRACK=1 tracks the surface with the material; default per-frame again
                              # since 2026-09-23 (the g40 page's videos are per-frame; the tracked ones are under analysis)
TRACK_FLAG=""
if [ "$TRACK" = "1" ]; then TRACK_FLAG="--track"; fi
PREFETCH=${PREFETCH:-3}       # 2026-09-23: frames reconstructed ahead in parallel (2.6x per video at 6 alone; 3 per video
export PHYSMORPH_POISSON_THREADS=${PHYSMORPH_POISSON_THREADS:-8}   #   when four batches share the host, 8 Poisson threads each)
case $PFX in h150) D0=$OUT/report150 ;; *) D0=$OUT/report_${PFX} ;; esac
for T in "$@"; do
  D=$D0/$T; mkdir -p $D
  NLAB=$(grep -m1 -o "N=[0-9]*" $OUT/${PFX}_$T.log | head -1)
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/render_photoreal.py --npz $OUT/${PFX}_${T}_${ARM}.npz \
    --out $D/${T}_photoreal.mp4 --res 720 --stride 3 --surface $SURFACE $TRACK_FLAG --prefetch $PREFETCH --label "sphere -> $T, $NLAB, photoreal" \
    2>&1 | grep "saved\|Traceback\|Error" | tail -2
  echo "PHOTO${PFX^^} $T DONE $(date)" >> $STATUS
done
