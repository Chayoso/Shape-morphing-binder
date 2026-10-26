#!/bin/bash
# d162_gal.sh -- D162's gallery after D162b (the coordinator: D162b's scenes first): the rest of list B and list A2 as one
# list (d162_runs_BA2.txt; runs with an archive are skipped) from a private copy of d162_chain.sh on one card by the
# normal claim, then the gallery read (d162_gallery.py, d162_windows.py on gal_*) on the CPU. Detached, from a copy.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d162; W=$B/tmp/d162; R=$O/runs40; D=$O/after
cat $W/d162_runs_B.full.txt $W/d162_runs_A2.txt | grep -v "^#" > $W/d162_runs_BA2.txt
C=$W/run_gal_chain.sh
cp $W/d162_chain.sh $C
RUNS_FILE=d162_runs_BA2.txt bash $C R
grep -q "R done (d162_runs_BA2.txt)" $O/status.txt || exit 1
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d162:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
cd $B/repo_d162
nice -n 10 $PY $W/d162_gallery.py $D/gallery.md $D/gallery.json > $D/gallery.log 2>&1
echo "$(date '+%m-%d %H:%M:%S') [gal] gallery exit $?: $(tail -2 $D/gallery.log | tr '\n' ' ')" >> $O/status.txt
cd $R && $PY $W/d162_windows.py $D/windows_gallery.md $(ls gal_*.json | sed "s/.json//" | sed "s|^|$R/|") > /dev/null 2>&1
echo "$(date '+%m-%d %H:%M:%S') [gal] done" >> $O/status.txt
