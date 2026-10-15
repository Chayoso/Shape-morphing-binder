#!/bin/bash
# b12_batch.sh ITEM... -- B12's GPU runs, one after another on the card b12_gpu.sh claimed (CUDA_VISIBLE_DEVICES set).
# ITEM = METHOD:MESH:ARM:SEED (METHOD ls | fc). Each run's log in logs/<method>_<mesh>_<arm>_s<seed>.log, one status
# line each in logs/status.txt. After each run its CPU readout (b12_read.sh) starts detached, without the card.
B12=/data/relcfd/chayo/physmorph_v2/baselines/b12
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
export TMPDIR=$B12/cache/tmp XDG_CACHE_HOME=$B12/cache/xdg TORCH_EXTENSIONS_DIR=$B12/cache/torch_ext
export PIP_CACHE_DIR=$B12/cache/pip PYTHONNOUSERSITE=1 PYTHONPATH= OMP_NUM_THREADS=8
export CUDA_HOME=/usr/local/cuda-12.8 LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
P=$B12/env/bin/python
S=$B12/scripts
ST=$B12/logs/status.txt
for item in "$@"; do
  IFS=: read -r method mesh arm seed <<< "$item"
  tag=${method}_${mesh}_${arm}_s${seed}
  echo "$(date '+%F %T %Z') start $tag gpu $CUDA_VISIBLE_DEVICES" >> $ST
  $P $S/b12_$method.py $mesh $arm $seed > $B12/logs/$tag.log 2>&1
  rc=$?
  echo "$(date '+%F %T %Z') done $tag rc $rc" >> $ST
  if [ $rc -eq 0 ]; then
    nohup setsid $S/b12_read.sh $method $mesh $arm $seed > $B12/logs/read_$tag.log 2>&1 < /dev/null &
  fi
done
