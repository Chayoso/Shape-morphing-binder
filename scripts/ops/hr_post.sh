#!/bin/bash
# Post-process one high-resolution example (render run + physics-only twin) on hyde06:
#   hi-res GIFs (res 320, az 0.6/2.2), PBR stills (surface splatting + GGX; final frame,
#   two azimuths; target for reference), gradient-field probe (render run: 3 commits; phys
#   run: final), scatter probe, loss curves.  Usage: hr_post.sh <target> [gpu]
cd ~/physmorph_v2
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
OUT=/data/relcfd/chayo/physmorph_v2/output
T=$1; GPU=${2:-0}
ARM=render_full_dt_iso_nn
R=hr_${T}_${ARM}; P=hr_${T}_phys_${ARM}
D=$OUT/report/$T; mkdir -p $D
export CUDA_VISIBLE_DEVICES=$GPU
echo "== $T: gifs"
$PY scripts/make_gif.py --npz $OUT/$R.npz --res 320 --stride 2 --label "sphere -> $T, 40k --ppc 8 density recipe (render)" --out $D/${T}_render.gif 2>&1 | grep saved
$PY scripts/make_gif.py --npz $OUT/$P.npz --res 320 --stride 2 --label "sphere -> $T, physics-only twin" --out $D/${T}_phys.gif 2>&1 | grep saved
echo "== $T: pbr"
for AZ in 35 215; do
  $PY scripts/render_pbr.py --npz $OUT/$R.npz --out $D/${T}_render_pbr_az$AZ.png --azim $AZ 2>&1 | grep saved
  $PY scripts/render_pbr.py --npz $OUT/$P.npz --out $D/${T}_phys_pbr_az$AZ.png --azim $AZ 2>&1 | grep saved
  $PY scripts/render_pbr.py --npz $OUT/$R.npz --out $D/${T}_target_pbr_az$AZ.png --azim $AZ --target 2>&1 | grep saved
done
echo "== $T: gradient field"
$PY scripts/probes/grad_field.py $OUT $R 0.05,0.35,1.0 2>&1 | grep -v "Warp\|CUDA Toolkit\|Devices\|\"cpu\"\|cuda:0\"\|Kernel\|Cache\|Module\|UserWarning\|detach" > $D/${T}_grad_render.txt
$PY scripts/probes/grad_field.py $OUT $P 1.0 2>&1 | grep -v "Warp\|CUDA Toolkit\|Devices\|\"cpu\"\|cuda:0\"\|Kernel\|Cache\|Module\|UserWarning\|detach" > $D/${T}_grad_phys.txt
mv $OUT/grad_field_$R.png $D/${T}_grad_render.png 2>/dev/null; mv $OUT/grad_field_$R.json $D/${T}_grad_render.json 2>/dev/null
mv $OUT/grad_field_$P.png $D/${T}_grad_phys.png 2>/dev/null; mv $OUT/grad_field_$P.json $D/${T}_grad_phys.json 2>/dev/null
echo "== $T: scatter + loss"
$PY /tmp/scatter_probe2.py $OUT $R $P 2>&1 | grep "thin_tgt" > $D/${T}_scatter.txt
$PY scripts/probes/loss_curves.py $OUT $D/${T}_loss.png hr_$T hr_${T}_phys 2>&1 | grep -v "^no\|Warp" > $D/${T}_loss.txt
grep -h "ARM render" $OUT/hr_$T.log $OUT/hr_${T}_phys.log | grep chamfer > $D/${T}_arm.txt
grep -h "gates:" $OUT/hr_$T.log $OUT/hr_${T}_phys.log | cut -c1-120 >> $D/${T}_arm.txt
cat $D/${T}_arm.txt $D/${T}_scatter.txt $D/${T}_loss.txt; tail -4 $D/${T}_grad_render.txt; tail -2 $D/${T}_grad_phys.txt
echo "POST $T DONE $(date)" >> $OUT/rcp_ladder_status.log
