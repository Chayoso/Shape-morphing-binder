#!/bin/bash
# Re-render the deliverable surfaces of finished runs with the outer-layer Poisson surface and the
# particle-median bulk (docs/method.md 10.12): <prefix>_<T> archives -> report_<prefix>/<T>/
#   <T>_photoreal.mp4 (+ .components.txt sidecar)  and  <T>_surface.gif (isosurface video, bulk fixed).
# The previous files are kept as <T>_photoreal_v8mc.mp4 / <T>_surface_v8mc.gif the first time.
# Usage: surface_rerender.sh <gpu> <prefix> <targets...>      (markers "SURFRR <prefix> <T> DONE" in $STATUS)
source /data/relcfd/chayo/physmorph_v2/repo/scripts/ops/hyde06_env.sh
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 EGL_PLATFORM=surfaceless
GPU=$1; PFX=$2; shift 2
ARM=render_full_dt_iso_nn
case $PFX in h150) D0=$OUT/report150 ;; *) D0=$OUT/report_${PFX} ;; esac
for T in "$@"; do
  D=$D0/$T; mkdir -p $D
  NLAB=$(grep -m1 -o "N=[0-9]*" $OUT/${PFX}_$T.log | head -1)
  for F in ${T}_photoreal.mp4 ${T}_photoreal.mp4.components.txt ${T}_surface.gif ${T}_surface.mp4; do
    [ -f $D/$F ] && [ ! -f $D/${F/${T}_/${T}_v8mc_} ] && cp $D/$F $D/${F/${T}_/${T}_v8mc_}
  done
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/render_photoreal.py --npz $OUT/${PFX}_${T}_${ARM}.npz \
    --out $D/${T}_photoreal.mp4 --res 720 --stride 3 --surface poisson --label "sphere -> $T, $NLAB, photoreal" \
    2>&1 | grep "saved\|Traceback\|Error" | tail -2
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/render_iso_video.py --npz $OUT/${PFX}_${T}_${ARM}.npz \
    --out $D/${T}_surface.gif --res 440 --stride 3 --label "sphere -> $T, $NLAB, isosurface" 2>&1 | grep "saved\|Traceback\|Error" | tail -2
  echo "SURFRR $PFX $T DONE $(date)" >> $STATUS
done
