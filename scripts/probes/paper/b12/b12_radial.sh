#!/bin/bash
B=/data/relcfd/chayo/physmorph_v2; BL=$B/baselines; R=$BL/b12
export TMPDIR=$R/cache/tmp XDG_CACHE_HOME=$R/cache/xdg JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
source /home/chayo/miniforge3/etc/profile.d/conda.sh; conda activate nsd
cd $BL/Implicit-surf-Deformation
for NAME in bunny dragon; do
  IN=$BL/buonomo/inputs/$NAME; D=./data/b12_${NAME}_radial; rm -rf $D; TG=$IN/${NAME}_ourframe.ply
  python $B/tmp/d145/b12/b12_radial_prep.py $IN/sphere_for_${NAME}_ourframe.ply $TG $D $NAME > $R/logs/prep_${NAME}_radial.log 2>&1 \
    && echo "$(date '+%Y-%m-%d %H:%M:%S %Z') prep $NAME radial done (b12_radial_prep.py)" >> $R/status.txt \
    || echo "$(date '+%Y-%m-%d %H:%M:%S %Z') prep $NAME radial FAILED (b12_radial_prep.py)" >> $R/status.txt
done
