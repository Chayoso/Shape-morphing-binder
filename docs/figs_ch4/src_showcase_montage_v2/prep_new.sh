#!/bin/bash
# prep_new.sh NAME:SRCMESH [...] -- FIGS_V2 (Figure 4.7 v2 candidates): CPU only, nice 10. Converts each candidate mesh to
# tmp/figs_v2/assets/NAME.obj (trimesh, geometry unchanged), then SHOWCASE's preparation unchanged in our own folder:
# the independent 300k sample (sc_ind.py seed 99) and the renderer's dense 1.5M sample (seed 9) of sphere -> NAME in
# output/gpu/figs_v2/ind/isosphere_NAME_{ind300k,dense15}.npz, and a camera preview strip (sc_views.py) in .../views/.
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/figs_v2; O=$B/output/gpu/figs_v2; ST=$O/prep.status
mkdir -p $T/assets $O/ind $O/views
export PHYSMORPH_RUN_REPO=$B/repo_t1s
source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd $B/repo_t1s
for spec in "$@"; do
  n=${spec%%:*}; m=${spec#*:}
  [ -e $T/assets/$n.obj ] || nice -n 10 $PY -c "import trimesh; t = trimesh.load('$m', force='mesh', process=False); t.export('$T/assets/$n.obj'); print(len(t.vertices), len(t.faces))" >> $ST 2>&1
  p=isosphere_$n
  if [ ! -e $O/ind/${p}_ind300k.npz ]; then
    t0=$(date +%s)
    nice -n 10 $PY $B/tmp/showcase/sc_ind.py assets/isosphere.obj $T/assets/$n.obj 300000 $O/ind/${p}_ind300k.npz - 99 > $O/ind/${p}_ind300k.log 2>&1 < /dev/null
    echo "$(date '+%m-%d %H:%M:%S') ind $p rc $? $(( $(date +%s) - t0 )) s" >> $ST
  fi
  [ -e $O/views/${p}.png ] || nice -n 10 $PY $B/tmp/showcase/sc_views.py $O/views/${p}.png $O/ind/${p}_ind300k.npz 0,45,90,135,180,270 15 > /dev/null 2>&1
  [ -e $O/views/${p}_top.png ] || nice -n 10 $PY $B/tmp/showcase/sc_views.py $O/views/${p}_top.png $O/ind/${p}_ind300k.npz 0,90,180,270 60 > /dev/null 2>&1
done
for spec in "$@"; do
  n=${spec%%:*}; p=isosphere_$n
  if [ ! -e $O/ind/${p}_dense15.npz ]; then
    t0=$(date +%s)
    nice -n 10 $PY $B/tmp/showcase/sc_ind.py assets/isosphere.obj $T/assets/$n.obj 1500000 $O/ind/${p}_dense15.npz - 9 300000 > $O/ind/${p}_dense15.log 2>&1 < /dev/null
    echo "$(date '+%m-%d %H:%M:%S') dense $p rc $? $(( $(date +%s) - t0 )) s" >> $ST
  fi
done
echo "$(date '+%m-%d %H:%M:%S') prep done $*" >> $ST
