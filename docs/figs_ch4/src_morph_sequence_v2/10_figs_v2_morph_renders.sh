#!/bin/bash
# 10_fig_morph_renders.sh -- gpu1 queue job (agent FIGS_V2: the dissertation's Figure 4.2 v2, five rows from the sphere).
# Studio-look stills (base display, +field=zbs, fit 0, the SHOWCASE cameras of the B1 renders) of 13 kept frames of the
# tag's B1 render-arm runs at seed 97 (output/gpu/d145t_b1/*300k_L97_frames12.npz): dragon, bunny, armadillo, bob, each
# with the display rule (pieces of the display holding < 10 particles not drawn) and without it (frames_raw), and the
# per-frame piece record (pieces.jsonl). tmp/figs_v2/fig_render.py; outputs output/gpu/figs_v2/morph/MESH/.
# No lock handling (the runner holds gpu1, CUDA_VISIBLE_DEVICES=1). Expected: ~13 x 2 4K frames a mesh, < 10 GB.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; D=$G/showcase; O=$G/figs_v2/morph; T=$B/tmp/figs_v2
mkdir -p $O $T/tmpdir
export PHYSMORPH_RUN_REPO=$B/repo_t1s
source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh
export TMPDIR=$T/tmpdir PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $B/repo_t1s
KS=0,5,10,15,20,25,30,40,50,60,80,100,last
echo "$(date '+%m-%d %H:%M:%S') job 10 start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >> $O/status.txt
rc_all=0
while read -r m az el dist; do
  [ -z "$m" ] && continue
  t0=$(date +%s)
  $PY $T/fig_render.py $G/d145t_b1/${m}300k_L97_frames12.npz $D/ind/isosphere_${m}_dense15.npz $O/$m \
    +ks=$KS +az=$az +el=$el +dist=$dist +min_piece=10 +raw=1 > $O/$m.log 2>&1 < /dev/null
  rc=$?
  echo "$(date '+%m-%d %H:%M:%S') $m rc $rc $(( $(date +%s) - t0 )) s" >> $O/status.txt
  [ $rc = 0 ] || rc_all=1
done <<LIST
dragon 70 15 1.06
bunny 35 18 1.06
armadilo 25 12 1.08
bob 310 28 1.06
LIST
echo "$(date '+%m-%d %H:%M:%S') job 10 end rc $rc_all" >> $O/status.txt
exit $rc_all
