#!/bin/bash
# r4_cpu.sh WHAT [MESHES] -- D145 reading 4, the CPU part (nice 10; at most three such processes at a time).
#   mesh   r4_mesh.py: our display surface meshed on B1's render-arm end frames (L97/99/101, + 103/105 on bunny and
#          dragon) and on L97's kept frames at the harness's p = 0.25 / 0.5 / 0.75 (B13's harness rows) -> r4/display
#   topo   r4_topo.py in B11's venv (libigl + CGAL, trimesh; repo_d145 on PYTHONPATH for the sampler, as b12_cpu.sh)
#   fscore r4_fscore.py on every method's end frame (r4/end1, B1's L end1, our display samples) -> r4/fscore.jsonl
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; X=$G/d145_ext/r4; R=$B/tmp/d145r
mkdir -p $X/display $X/tmp; what=$1; M=${2:-"bunny dragon teapot bob armadilo homer"}
echo "start $what $M $(date '+%m-%d %H:%M')" >> $X/status.txt
if [ "$what" = mesh ]; then
  source $B/tmp/d145/env_d145.sh; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6
  for m in $M; do
    specs=""
    for s in 97 99 101 103 105; do f=$G/d145t_b1/${m}300k_L${s}_end1.npz; [ -e $f ] && specs="$specs $m:L$s:end=$f"; done
    for raw in $($PY -c "
import json
for s in open('$G/b13/harness/harness_$m.jsonl'):
    r = json.loads(s)
    if r['label'] == 'L97':
        print(' '.join(str(r['read'][q]['raw']) for q in ('p0.25', 'p0.5', 'p0.75') if r['read'].get(q)))"); do
      specs="$specs $m:L97:$raw=$G/d145t_b1/${m}300k_L97_frames12.npz"
    done
    nice -n 10 $PY $R/r4_mesh.py $X/display $X/display_${M%% *}.jsonl $specs >> $X/mesh.log 2>&1
  done
elif [ "$what" = fscore ]; then
  source $B/tmp/d145/env_d145.sh; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6
  specs=$($PY -c "
import glob, os, re
def k(l):
    for p, m in ((r'L\d+$', 'ours'), ('XU', 'xu'), ('BUO', 'buo'), ('NISE', 'nise'), ('ISDr', 'isd_r'), ('ISD', 'isd'),
                 ('FDr', 'fd_r'), ('FD', 'fd'), ('ARAP', 'arap'), ('OT', 'ot')):
        if re.match(p, l): return m
    return l[:3].lower()
out = []
for f in sorted(glob.glob('$X/end1/*_end1.npz')):
    n = os.path.basename(f)[:-9]; m, l = n.split('300k_'); out.append(f'{m}:{k(l)}:{l}={f}')
for f in sorted(glob.glob('$G/d145t_b1/*300k_L*_end1.npz')):
    n = os.path.basename(f)[:-9]; m, l = n.split('300k_')
    if re.match(r'L(97|99|101|103|105)$', l): out.append(f'{m}:ours:{l}={f}')
for f in sorted(glob.glob('$X/display/*_sample.npz')):
    n = os.path.basename(f)[:-11]; m, l = n.split('_', 1); out.append(f'{m}:ours_disp:{l}={f}')
print(' '.join(out))")
  rm -f $X/fscore.jsonl
  nice -n 10 $PY $R/r4_fscore.py $X/fscore.jsonl $specs > $X/fscore.log 2>&1
else
  source $B/tmp/d145/env_d145.sh
  export PHYSMORPH_RUN_REPO=$B/repo_d145
  source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
  A=$B/baselines/arap
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TMPDIR=$X/tmp
  export PYTHONPATH="$A/scripts:$PYTHONPATH"
  cd $A/scripts
  rm -f $X/topo.jsonl
  nice -n 10 $A/venv/bin/python -W ignore $R/r4_topo.py $X/display $X/topo.jsonl --sample >> $X/topo.log 2>&1
fi
echo "done $what $M $(date '+%m-%d %H:%M')" >> $X/status.txt
