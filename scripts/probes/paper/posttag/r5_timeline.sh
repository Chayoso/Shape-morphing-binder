#!/bin/bash
# r5_timeline.sh NAME MESH FRAMES12 -- D145 reading 5 (1): the display surface of a run's kept frames every 0.1 s (raw
# step 24; teapot / homer to raw 960 = 4 s, the dragon to 1440 = 6 s and every 0.05 s over raws 360-720 = 1.5-3.0 s,
# D137's tear) and at the end, meshed by r4_mesh.py into output/gpu/d145_ext/r5/display/NAME/ (display_NAME.jsonl), read
# by r4_topo.py in B11's venv (topo_NAME.jsonl); the meshes removed after the reading (logged). CPU, nice 10.
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp; G=$B/output/gpu; R=$B/tmp/d145r; X=$G/d145_ext/r5
name=$1; mesh=$2; F=$3; O=$X/display/$name
mkdir -p $O $X/tmp
source $T/d145/env_d145.sh
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6
top=960; [ "$mesh" = dragon ] && top=1440
want=$( (seq 0 24 $top; [ "$mesh" = dragon ] && seq 360 12 720) | sort -n | uniq)
raws=$($PY -c "
import numpy as np
have = set(int(r) for r in np.load('$F', mmap_mode='r')['raws'])
print(' '.join(str(r) for r in [$(echo $want | tr ' ' ',')] if r in have))")
specs="$mesh:$name:end=$F"
for r in $raws; do specs="$specs $mesh:$name:$r=$F"; done
rm -f $X/display_$name.jsonl
nice -n 10 $PY $R/r4_mesh.py $O $X/display_$name.jsonl $specs > $X/mesh_$name.log 2>&1
export PHYSMORPH_RUN_REPO=$B/repo_d145
source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
A=$B/baselines/arap
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TMPDIR=$X/tmp PYTHONPATH="$A/scripts:$PYTHONPATH"
cd $A/scripts
rm -f $X/topo_$name.jsonl
nice -n 10 $A/venv/bin/python -W ignore $R/r4_topo.py $O $X/topo_$name.jsonl > $X/topo_$name.log 2>&1
n=$(ls $O/*_mesh.npz 2>/dev/null | wc -l); sz=$(du -sh $O | cut -f1)
rm -f $O/*_mesh.npz
echo "$(date '+%Y-%m-%d %H:%M') D145r: deleted $n display meshes of $name ($sz, output/gpu/d145_ext/r5/display/$name), read into display_$name.jsonl / topo_$name.jsonl" >> $G/cleanup.log
