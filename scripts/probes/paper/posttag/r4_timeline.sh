#!/bin/bash
# r4_timeline.sh -- D145 reading 4 (2, 4), observed, not gated: the display surface's internal cavities and genus over
# the early kept frames (every 24th raw = 0.1 s): teapot L97 to raw 960 (4 s), bunny L97 and its twin P97 to raw 480
# (2 s) -- r4_mesh.py -> r4/display_tl/, r4/display_tl.jsonl, then r4_topo.py -> r4/topo_tl.jsonl. CPU, nice 10.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; X=$G/d145_ext/r4; R=$B/tmp/d145r; D=$X/display_tl
mkdir -p $D $X/tmp
echo "start timeline $(date '+%m-%d %H:%M')" >> $X/status.txt
source $B/tmp/d145/env_d145.sh; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6
specs=""
for raw in $(seq 0 24 960); do specs="$specs teapot:L97:$raw=$G/d145t_b1/teapot300k_L97_frames12.npz"; done
for l in L97 P97; do
  for raw in $(seq 0 24 480); do specs="$specs bunny:$l:$raw=$G/d145t_b1/bunny300k_${l}_frames12.npz"; done
done
rm -f $X/display_tl.jsonl
nice -n 10 $PY $R/r4_mesh.py $D $X/display_tl.jsonl $specs >> $X/mesh_tl.log 2>&1
export PHYSMORPH_RUN_REPO=$B/repo_d145
source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
A=$B/baselines/arap
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TMPDIR=$X/tmp PYTHONPATH="$A/scripts:$PYTHONPATH"
cd $A/scripts
rm -f $X/topo_tl.jsonl
nice -n 10 $A/venv/bin/python -W ignore $R/r4_topo.py $D $X/topo_tl.jsonl >> $X/topo_tl.log 2>&1
echo "done timeline $(date '+%m-%d %H:%M')" >> $X/status.txt
