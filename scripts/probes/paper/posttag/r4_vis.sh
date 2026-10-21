#!/bin/bash
# r4_vis.sh -- D145 reading 4 (4): visible_apart.py (D148's reader: hidden / bump / blob, the side view's) on L97's kept
# frames at the harness's p = 0.25 / 0.5 / 0.75 (B13's harness rows), each frame alone (w0 = w1 = its window), the six
# meshes; one CPU process, nice 10 -> output/gpu/d145_ext/r4/vis_mid.jsonl.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; X=$G/d145_ext/r4
source $B/tmp/d145/env_d145.sh
export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1; cd $B/repo_t
export CUDA_VISIBLE_DEVICES=
echo "start vis $(date '+%m-%d %H:%M')" >> $X/status.txt
rm -f $X/vis_mid.jsonl
for m in bunny dragon teapot bob armadilo homer; do
  for raw in $($PY -c "
import json
for s in open('$G/b13/harness/harness_$m.jsonl'):
    r = json.loads(s)
    if r['label'] == 'L97':
        print(' '.join(str(r['read'][q]['raw']) for q in ('p0.25', 'p0.5', 'p0.75') if r['read'].get(q)))"); do
    w=$($PY -c "print($raw / 40.0)")
    nice -n 10 $PY scripts/probes/paper/visible_apart.py $G/d145t_b1/${m}300k_L97_frames12.npz --label ${m}_L97_$raw \
      --w0 $w --w1 $w --jsonl $X/vis_mid.jsonl --threads 8 >> $X/vis_mid.log 2>&1
  done
done
echo "done vis $(date '+%m-%d %H:%M')" >> $X/status.txt
