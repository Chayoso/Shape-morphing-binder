#!/bin/bash
# build_assets.sh -- D170's scene assets on hyde06's CPU (nice 10, no card), detached:
#   d170L300coin.obj / d170L300_bunny.obj  D161's "PM" relief meshes at the 300k pitch (d161_mesh.py, D161's L40 specs with n = 300000)
#   d170smooth_dragon.obj                   the dragon with its scales taken out by Taubin's smoothing (D161's bunny canvas:
#                                           lambda 0.5, mu -0.51, 300 iterations; the name ends in _dragon for the orientation table)
B=/data/relcfd/chayo/physmorph_v2; A=$B/output/gpu/d170/assets; A1=$B/output/gpu/d161/assets; W=$B/tmp/d170
mkdir -p $A
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PHYSMORPH_CACHE=$B/cache/samples
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
cd $B/repo_d166
cat > $A/L300coin.json << EOS
{"base": "$A1/d161coin.obj", "n": 300000, "text": "PM", "height": 20, "amp": 1.0, "edge": 1.0, "stroke_pitch": 3.8, "map": "coin"}
EOS
cat > $A/L300bunny.json << EOS
{"base": "$A1/d161smooth_bunny.obj", "n": 300000, "text": "PM", "height": 20, "amp": 1.0, "edge": 1.0, "stroke_pitch": 3.8, "map": "flank"}
EOS
( s=$(date +%s); nice -n 10 $PY -c "
import json, trimesh, numpy as np
b = trimesh.load('assets/dragon.obj', process=True, force='mesh'); b.merge_vertices()
bs = b.copy(); trimesh.smoothing.filter_taubin(bs, lamb=0.5, nu=0.51, iterations=300)
bs.export('$A/d170smooth_dragon.obj')
d = np.linalg.norm(np.asarray(bs.vertices) - np.asarray(b.vertices), axis=1)
print(json.dumps(dict(verts=len(bs.vertices), faces=len(bs.faces), watertight=bool(bs.is_watertight),
  hull_volume_ratio=float(bs.convex_hull.volume / b.convex_hull.volume), move_raw_median=float(np.median(d)),
  move_raw_p99=float(np.quantile(d, 0.99)), edge_raw=float(np.median(b.edges_unique_length)))))
" > $A/d170smooth_dragon.log 2>&1; echo "rc $? secs $(( $(date +%s) - s ))" >> $A/d170smooth_dragon.log; [ -s $A/d170smooth_dragon.obj ] && touch $A/smooth_dragon.ok ) &
for k in coin bunny; do
  if [ $k = coin ]; then o=$A/d170L300coin.obj; else o=$A/d170L300_bunny.obj; fi
  ( s=$(date +%s); nice -n 10 $PY $B/tmp/d161/d161_mesh.py $o $A/L300$k.json > $o.log 2>&1; rc=$?
    echo "rc $rc secs $(( $(date +%s) - s ))" >> $o.log; [ $rc = 0 ] && touch $A/L300$k.ok ) &
done
wait
echo "$(date '+%m-%d %H:%M:%S') [assets] done: $(ls $A/*.ok 2>/dev/null | xargs -n1 basename | tr '\n' ' ')" >> $B/output/gpu/d170/status.txt
