#!/bin/bash
# after.sh MODE OUT_DIR RUNDIR TITLE ARM:TWIN... -- one D166 / D167 job's detached CPU reader: waits for the runs it needs
# (twins / refs of other jobs too), then takes one of 4 CPU slots and writes OUT_DIR/summary.md + verdict.json (d166_read.py).
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d166
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d162d:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
mkdir -p $W/slots; cd $B/repo_d162d                  # the readers resolve the runs' assets/ paths from a repo
$PY $W/d166_read.py wait "$@"
while true; do
  for i in 1 2 3 4; do
    exec 9> $W/slots/$i.lock
    if flock -n 9; then nice -n 10 $PY $W/d166_read.py "$@"; rc=$?; echo "after: rc $rc"; exit $rc; fi
    exec 9>&-
  done
  sleep 30
done
