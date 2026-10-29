#!/bin/bash
# ri_video_values.sh MESH -- render influence video, pass 1 on the CPU (nice 10)
M=$1
B=/data/relcfd/chayo/physmorph_v2
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=$B/repo_t1
O=$B/output/gpu/render_influence/video
mkdir -p $O
nice -n 10 /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python $B/tmp/render_influence/ri_video_values.py $M $O > $O/values_$M.log 2>&1
echo "$(date +%H:%M:%S) video values $M rc $?" >> $B/output/gpu/render_influence/status.txt
