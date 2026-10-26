#!/bin/bash
# d163_after.sh TAG RUNS_DONE_MARK SCENE_JSON [SPREAD_READ_JSON] -- D163 on hyde06, CPU only (nice 10, no card): once the
# status line RUNS_DONE_MARK appears, the reader (d163_read.py) on SCENE_JSON, the summary and verdict (d163_summary.py),
# the per-window records of every run (d161_windows.py). Outputs in output/gpu/d163/after/TAG*.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d163; W=$B/tmp/d163; D=$O/after
TAG=$1; MARK=$2; SCENE=$3; SPREAD=$4
mkdir -p $D
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=16 D157_THREADS=16 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d163:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') [after $TAG] $*" >> $O/status.txt; }
until grep -qF "$MARK" $O/status.txt; do sleep 60; done
log "start"
cd $B/repo_d163
s=$(date +%s)
nice -n 10 $PY $W/d163_read.py $D/read_$TAG.json $SCENE > $D/read_$TAG.log 2>&1
log "read exit $? ($(( $(date +%s) - s )) s)"
nice -n 10 $PY $W/d163_summary.py $D/summary_$TAG.md $D/read_$TAG.json $SPREAD > $D/summary_$TAG.log 2>&1
log "summary exit $?"
PRE=$($PY -c "import json; print(' '.join(json.load(open('$SCENE'))['runs'].values()))")
nice -n 10 $PY $B/tmp/d161/d161_windows.py $D/windows_$TAG.md $PRE > /dev/null 2>&1
log "windows exit $?"
log "done"
