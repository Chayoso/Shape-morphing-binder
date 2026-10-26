#!/bin/bash
# ri_after.sh -- render influence: wait for the end maps and the dragon course, then the studio maps and the figures.
B=/data/relcfd/chayo/physmorph_v2
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 D157_THREADS=8 PYTHONPATH=$B/repo_t1
O=$B/output/gpu/render_influence
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
while kill -0 $(cat $B/tmp/render_influence/endmap2.pid) 2>/dev/null; do sleep 10; done
nice -n 10 $PY $B/tmp/render_influence/ri_endrender.py $O 0.4 bunny dragon > $O/endrender.log 2>&1
echo "$(date +%H:%M:%S) endrender rc $?" >> $O/status.txt
while kill -0 $(cat $B/tmp/render_influence/course_dragon.pid) 2>/dev/null; do sleep 10; done
nice -n 10 $PY $B/tmp/render_influence/ri_figs.py $O 0.4 > $O/figs.log 2>&1
echo "$(date +%H:%M:%S) figs rc $?" >> $O/status.txt
