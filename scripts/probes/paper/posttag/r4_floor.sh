#!/bin/bash
# r4_floor.sh -- D145 reading 4 (2, 4): the display surface of a perfect sample (the floor: D90's independent sample
# and L97's own target sample, six meshes) and of L97's frames where pieces145.py found the most particles >= 5
# pitches off the body (dragon, bob, armadilo, homer), meshed by r4_mesh.py and read by r4_topo.py (with the samples)
# -> r4/display_floor/, r4/display_floor.jsonl, r4/topo_floor.jsonl. CPU, nice 10, one process.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; X=$G/d145_ext/r4; R=$B/tmp/d145r; D=$X/display_floor
mkdir -p $D $X/tmp
echo "start floor $(date '+%m-%d %H:%M')" >> $X/status.txt
source $B/tmp/d145/env_d145.sh; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6
specs=""
for m in bunny dragon teapot bob armadilo homer; do
  e=$G/d145t_b1/${m}300k_L97_end1.npz; specs="$specs $m:L97:ind=$e $m:L97:tgt=$e"
done
specs="$specs $($PY -c "
import json, numpy as np
out = []
for s in open('$G/d145t_evidence/pieces_full_d145t_b1.jsonl'):
    r = json.loads(s)
    m = r['run'].split('300k_')[0]
    if r['run'].endswith('_L97') and m in ('dragon', 'bob', 'armadilo', 'homer') and r.get('far_frames'):
        f = '$G/d145t_b1/' + m + '300k_L97_frames12.npz'
        raws = [int(v) for v in np.load(f)['raws']]
        t = max(r['timeline'], key=lambda x: x[2])[0]
        raw = min(raws, key=lambda q: abs(q - t * 240))
        out.append(f'{m}:L97:{raw}={f}')
print(' '.join(out))")"
rm -f $X/display_floor.jsonl
nice -n 10 $PY $R/r4_mesh.py $D $X/display_floor.jsonl $specs >> $X/mesh.log 2>&1
export PHYSMORPH_RUN_REPO=$B/repo_d145
source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
A=$B/baselines/arap
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TMPDIR=$X/tmp PYTHONPATH="$A/scripts:$PYTHONPATH"
cd $A/scripts
rm -f $X/topo_floor.jsonl
nice -n 10 $A/venv/bin/python -W ignore $R/r4_topo.py $D $X/topo_floor.jsonl --sample >> $X/topo.log 2>&1
echo "done floor $(date '+%m-%d %H:%M')" >> $X/status.txt
