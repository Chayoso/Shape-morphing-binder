#!/bin/bash
# ri_video_after.sh -- after pass 2: the hero sheet and the combined figure (CPU), then a status line.
B=/data/relcfd/chayo/physmorph_v2
O=$B/output/gpu/render_influence
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 PYTHONPATH=$B/repo_t1
until grep -q "RI pass 2 done" $O/status.txt; do sleep 30; done
nice -n 10 $PY $B/tmp/render_influence/ri_sheet.py $O/video $O/fig3_hero_sheet.png > $O/sheet.log 2>&1
echo "$(date '+%m-%d %H:%M:%S') RI sheet rc $?" >> $O/status.txt
nice -n 10 $PY $B/tmp/render_influence/ri_figs.py $O 0.4 > $O/figs.log 2>&1
echo "$(date '+%m-%d %H:%M:%S') RI figs rc $?" >> $O/status.txt
