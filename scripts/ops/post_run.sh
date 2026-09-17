#!/bin/bash
# Post-process one run: surface video (the object, not particles), particle GIF, PBR stills
# (delivered + target, az 35/215), scatter probe, stray census, loss curves, ARM/gates lines,
# grid-connectivity fragment count. Usage: post_run.sh <prefix> <target> [gpu]
#   -> $OUT/report_<prefix>/<target>/   (report150 for prefix h150, kept for the existing builder)
source /data/relcfd/chayo/physmorph_v2/repo/scripts/ops/hyde06_env.sh
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
PFX=$1; T=$2; GPU=${3:-0}
ARM=render_full_dt_iso_nn
R=${PFX}_${T}_${ARM}
case $PFX in h150) D=$OUT/report150/$T ;; *) D=$OUT/report_${PFX}/$T ;; esac
mkdir -p $D
export CUDA_VISIBLE_DEVICES=$GPU
NLAB=$(grep -m1 -o "N=[0-9]*" $OUT/${PFX}_$T.log | head -1)
$PY scripts/render_iso_video.py --npz $OUT/$R.npz --out $D/${T}_surface.gif --res 440 --stride 3 --label "sphere -> $T, $NLAB, isosurface" 2>&1 | grep saved
$PY scripts/render_surface_video.py --npz $OUT/$R.npz --out $D/${T}_splat.gif --res 440 --stride 3 --label "sphere -> $T, $NLAB, splats" 2>&1 | grep saved
$PY scripts/make_gif.py --npz $OUT/$R.npz --res 220 --stride 2 --label "sphere -> $T, $NLAB --ppc 8 (particles)" --out $D/${T}_particles.gif 2>&1 | grep saved
for AZ in 35 215; do
  $PY scripts/render_pbr.py --npz $OUT/$R.npz --out $D/${T}_render_pbr_az$AZ.png --azim $AZ 2>&1 | grep saved
  $PY scripts/render_pbr.py --npz $OUT/$R.npz --out $D/${T}_target_pbr_az$AZ.png --azim $AZ --target 2>&1 | grep saved
done
$PY scripts/probes/scatter_probe2.py $OUT $R 2>&1 | grep "thin_tgt" > $D/${T}_scatter.txt
$PY scripts/probes/stray_census.py $OUT $R 2>&1 | grep -v Warp > $D/${T}_census.txt
$PY scripts/probes/loss_curves.py $OUT $D/${T}_loss.png ${PFX}_$T 2>&1 | grep -v "^no\|Warp" > $D/${T}_loss.txt
grep -h "ARM render" $OUT/${PFX}_$T.log | grep chamfer > $D/${T}_arm.txt
grep -h "gates:" $OUT/${PFX}_$T.log | cut -c1-140 >> $D/${T}_arm.txt
grep -o "anim [0-9]*: fragments [0-9]*" $OUT/${PFX}_$T.log | tail -1 | grep -o "[0-9]*$" > $D/${T}_frag.txt
grep -o "([0-9.]* min)" $OUT/${PFX}_$T.log | tail -1 > $D/${T}_time.txt
echo "POST$(echo $PFX | tr a-z A-Z) $T DONE $(date)" >> $STATUS
