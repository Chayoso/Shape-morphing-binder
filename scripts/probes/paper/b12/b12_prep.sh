#!/bin/bash
# b12_prep.sh -- D145 B12 (amendment 5): the pair data for ISD / 4Deform from B5's exported meshes, on the CPU.
# data/b12_<mesh>_<corr> in Implicit-surf-Deformation (4Deform reads the same layout; each run copies it, b12_pair.sh).
B=/data/relcfd/chayo/physmorph_v2; BL=$B/baselines; R=$BL/b12
mkdir -p $R/logs $R/cache/tmp $R/cache/xdg
export TMPDIR=$R/cache/tmp XDG_CACHE_HOME=$R/cache/xdg JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
source /home/chayo/miniforge3/etc/profile.d/conda.sh; conda activate nsd
cd $BL/Implicit-surf-Deformation
st() { echo "$(date '+%Y-%m-%d %H:%M:%S %Z') prep $*" >> $R/status.txt; }
for spec in bunny:none dragon:none teapot:none bob:none armadilo:none homer:none bunny:radial dragon:radial; do
  NAME=${spec%%:*}; CORR=${spec##*:}; IN=$BL/buonomo/inputs/$NAME; D=./data/b12_${NAME}_${CORR}
  [ -s $D/pair_info.json ] && { st "$NAME $CORR exists"; continue; }
  python $B/baselines_prep/prep_pair.py --src $IN/sphere.ply --tgt $IN/$NAME.ply --save_dir $D --corr $CORR \
    --src_name 00_sphere --tgt_name 01_$NAME > $R/logs/prep_${NAME}_${CORR}.log 2>&1 && st "$NAME $CORR done" || st "$NAME $CORR FAILED"
done
st "all done"
