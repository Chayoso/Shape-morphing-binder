#!/bin/bash
# d159c_terms.sh MESH -- D159c R5 (term dumps) then the dumps deleted and logged (the readfix lacked PYTHONPATH).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d159; T=$B/tmp; W=$T/d159; R=$O/read
mesh=$1; RUN=${mesh}300k_C97; S=$O/status.txt
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
source $T/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=$B/repo_d159 OMP_NUM_THREADS=12 D159_REPO=$B/repo_d159
TD=$O/terms/$RUN
nice -n 10 $PY $W/d159_terms.py $R/terms_$mesh.jsonl $mesh C97 $TD > $R/terms_c_$mesh.log 2>&1 < /dev/null
rc=$?
if [ $rc = 0 ] && [ -d $TD ]; then
  sz=$(du -sm $TD | cut -f1); n=$(ls $TD | wc -l); rm -rf $TD
  echo "$(date "+%Y-%m-%d %H:%M") D159c: deleted the term dumps of $RUN ($n files, ${sz} MB, output/gpu/d159/terms), read into read/terms_$mesh.jsonl" >> $G/cleanup.log
fi
echo "$(date "+%m-%d %H:%M:%S") D159c terms $mesh rc $rc" >> $S
