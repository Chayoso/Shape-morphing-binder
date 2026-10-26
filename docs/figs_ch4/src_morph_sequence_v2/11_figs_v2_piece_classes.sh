#!/bin/bash
# 11_figs_v2_piece_classes.sh -- gpu1 queue job (agent FIGS_V2, Figure 4.2 v2): the display-piece record again for the four
# tag B1 runs of job 10, now with each piece classed as floating (most of its particles off the body's set of D145's
# pieces) or inside/on the body; no drawing (+draw=0). Same 13 kept frames and cameras; out MESH/pieces_cls.jsonl. ~1 min a mesh.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; D=$G/showcase; O=$G/figs_v2/morph; T=$B/tmp/figs_v2
export PHYSMORPH_RUN_REPO=$B/repo_t1s
source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh
export TMPDIR=$T/tmpdir OMP_NUM_THREADS=8
cd $B/repo_t1s
KS=0,5,10,15,20,25,30,40,50,60,80,100,last
rc_all=0
for m in dragon bunny armadilo bob; do
  $PY $T/fig_render.py $G/d145t_b1/${m}300k_L97_frames12.npz $D/ind/isosphere_${m}_dense15.npz $O/$m \
    +ks=$KS +min_piece=10 +draw=0 +pieces_out=pieces_cls.jsonl > $O/${m}_cls.log 2>&1 < /dev/null
  rc=$?; echo "$(date '+%m-%d %H:%M:%S') classes $m rc $rc" >> $O/status.txt; [ $rc = 0 ] || rc_all=1
done
exit $rc_all
