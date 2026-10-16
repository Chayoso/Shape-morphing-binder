#!/bin/bash
# endgen_t1016.sh -- D145 post-tag: B5's end check (tmp/d145/endgen145.py: the last frame's silhouette IoU and chamfer
# against the independent 300k sample d90/MESH_ind300k.npz and against the run's own sample) on the end frames of the
# tag's B1 (d145t_b1), of amendment 6's 300k arms (d145t_a6) and of the pre-tag B1 (d145_b1, D135 code); a pool job
# (physmorph metrics run on the card).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; E=$G/d145t_evidence; mkdir -p $E
export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
cd $B/repo_t
for d in d145t_b1 d145t_a6 d145_b1; do
  specs=""
  for f in $G/$d/*300k_*_end1.npz; do n=$(basename $f _end1.npz); specs="$specs ${n%%300k_*}=$f"; done
  rm -f $E/endgen_$d.jsonl
  $PY $B/tmp/d145/endgen145.py $E/endgen_$d.jsonl $specs || exit 1
done
