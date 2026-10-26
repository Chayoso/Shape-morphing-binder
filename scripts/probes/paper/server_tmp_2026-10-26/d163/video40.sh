#!/bin/bash
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d163; V=$O/video/p40_H_T
mkdir -p $V; cd $B/repo_d163
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=4 D157_THREADS=4 PYTHONPATH=$B/repo_d163
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
s=$(date +%s)
nice -n 10 $PY $B/tmp/d163/d163_render.py video $V/frames $O/assets/d163plaque.obj 40000 $O/runs/plq40k_H_97_frames12.npz $O/runs/plq40k_T_97_frames12.npz "dF_c + layer u + render loss (full), render weight held (40k)" "dF_c + layer u (the twin, 40k)" 1920 1080 0.30 0.18 80 10 > $V/video.log 2>&1
echo "frames rc $? secs $(( $(date +%s) - s ))" >> $V/video.log
ffmpeg -loglevel error -y -framerate 20 -i $V/frames/f%05d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p $V/p40_H_T_1080p.mp4 < /dev/null
echo "ffmpeg rc $?" >> $V/video.log
