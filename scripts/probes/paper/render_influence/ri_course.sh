#!/bin/bash
# ri_course.sh MESH -- render influence (2026-10-22, CPU, nice 10): D159's R2 reader (tmp/d159/d159_course.py, unchanged)
# on the same-seed replicate of the full arm (RG97, D145 B7: the tag's path at seed 97 again, the hooks proven inert):
#   rep   = RG97 against L97 on L97's outer layer  (the run-to-run floor of "dF_c + layer u + render loss (full)")
#   full2 = RG97 against P97 on P97's outer layer  (a second full-vs-twin sample beside D159's course_tag = L97 vs P97)
M=$1
B=/data/relcfd/chayo/physmorph_v2
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=$B/repo_d159
O=$B/output/gpu/render_influence
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
G=$B/output/gpu
nice -n 10 $PY $B/tmp/d159/d159_course.py $O/course_rep_$M.jsonl $M $G/d145t_b1/${M}300k_L97_frames12.npz RG97=$G/d145t_b7/${M}300k_RG97_frames12.npz > $O/course_rep_$M.log 2>&1
echo "$(date +%H:%M:%S) course rep $M rc $?" >> $O/status.txt
nice -n 10 $PY $B/tmp/d159/d159_course.py $O/course_full2_$M.jsonl $M $G/d145t_b1/${M}300k_P97_frames12.npz RG97=$G/d145t_b7/${M}300k_RG97_frames12.npz > $O/course_full2_$M.log 2>&1
echo "$(date +%H:%M:%S) course full2 $M rc $?" >> $O/status.txt
