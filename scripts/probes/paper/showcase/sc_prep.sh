#!/bin/bash
# sc_prep.sh SRC:TGT [...] -- SHOWCASE: the CPU preparation of each example (no card; nice 10, 8 threads; one at a time):
# the independent 300k sample of the target in the run's frame (sc_ind.py, seed 99, as D90's d90/MESH_ind300k.npz for the
# sphere), the renderer's dense 1.5M sample (seed 9, as D111's dense15_s9), both in output/gpu/showcase/ind/SRC_TGT_*.npz,
# and a camera preview strip (sc_views.py, six azimuths at elevation 18) in output/gpu/showcase/views/. Meshes outside
# assets/ are read from tmp/showcase/assets_extra/. Status in output/gpu/showcase/prep.status.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; D=$G/showcase; T=$B/tmp/showcase
mkdir -p $D/ind $D/views
export PHYSMORPH_RUN_REPO=$B/repo_t1s
source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd $B/repo_t1s
mp() { if [ -e $B/repo_t1s/assets/$1.obj ]; then echo assets/$1.obj; else echo $T/assets_extra/$1.obj; fi; }
for spec in "$@"; do
  s=${spec%%:*}; t=${spec#*:}; p=${s}_${t}
  if [ ! -e $D/ind/${p}_ind300k.npz ]; then
    t0=$(date +%s)
    nice -n 10 $PY $T/sc_ind.py $(mp $s) $(mp $t) 300000 $D/ind/${p}_ind300k.npz - 99 > $D/ind/${p}_ind300k.log 2>&1 < /dev/null
    echo "$(date '+%m-%d %H:%M:%S') ind $p rc $? $(( $(date +%s) - t0 )) s" >> $D/prep.status
  fi
  if [ ! -e $D/ind/${p}_dense15.npz ]; then
    t0=$(date +%s)
    nice -n 10 $PY $T/sc_ind.py $(mp $s) $(mp $t) 1500000 $D/ind/${p}_dense15.npz - 9 300000 > $D/ind/${p}_dense15.log 2>&1 < /dev/null
    echo "$(date '+%m-%d %H:%M:%S') dense $p rc $? $(( $(date +%s) - t0 )) s" >> $D/prep.status
  fi
  [ -e $D/views/${p}.png ] || nice -n 10 $PY $T/sc_views.py $D/views/${p}.png $D/ind/${p}_ind300k.npz > /dev/null 2>&1
done
echo "$(date '+%m-%d %H:%M:%S') prep done" >> $D/prep.status
