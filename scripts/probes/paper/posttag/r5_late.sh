#!/bin/bash
# r5_late.sh NAME MESH STEP -- D145 reading 5 (1): the display after r5_timeline.sh's window (from 4.5 s every STEP raws
# to the run's last kept frame): when the teapot's handle opens. r4_mesh.py -> display/NAME_late/, display_NAME_late.jsonl;
# r4_topo.py in B11's venv -> topo_NAME_late.jsonl; the meshes removed after the reading (logged). CPU, nice 10.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; R=$B/tmp/d145r; X=$G/d145_ext/r5
name=$1; mesh=$2; step=$3; F=$G/d145t_cell/${name}_frames12.npz; O=$X/display/${name}_late
mkdir -p $O $X/tmp
source $B/tmp/d145/env_d145.sh
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6
raws=$($PY -c "
import numpy as np
r = [int(v) for v in np.load('$F', mmap_mode='r')['raws']]
print(' '.join(str(v) for v in r if v >= 1080 and v % $step == 0))")
specs="$mesh:$name:end=$F"
for r in $raws; do specs="$specs $mesh:$name:$r=$F"; done
rm -f $X/display_${name}_late.jsonl
nice -n 10 $PY $R/r4_mesh.py $O $X/display_${name}_late.jsonl $specs > $X/mesh_${name}_late.log 2>&1
A=$B/baselines/arap
export PHYSMORPH_RUN_REPO=$B/repo_d145
source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
export OMP_NUM_THREADS=8 TMPDIR=$X/tmp PYTHONPATH="$A/scripts:$PYTHONPATH"
cd $A/scripts
rm -f $X/topo_${name}_late.jsonl
nice -n 10 $A/venv/bin/python -W ignore $R/r4_topo.py $O $X/topo_${name}_late.jsonl > $X/topo_${name}_late.log 2>&1
n=$(ls $O/*_mesh.npz 2>/dev/null | wc -l); sz=$(du -sh $O | cut -f1); rm -f $O/*_mesh.npz
echo "$(date '+%Y-%m-%d %H:%M') D145r: deleted $n display meshes of ${name}_late ($sz, output/gpu/d145_ext/r5/display/${name}_late), read into display_${name}_late.jsonl / topo_${name}_late.jsonl" >> $G/cleanup.log
echo "done late $name $(date '+%m-%d %H:%M')" >> $G/d145t_cell/status.txt
