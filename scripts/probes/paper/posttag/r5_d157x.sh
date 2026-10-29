#!/bin/bash
# r5_d157x.sh OUT_JSONL JOB|@JOBS_FILE [...] -- D145 reading 5: r5_d157x.py in D157's environment (tmp/d157/env.sh: the
# rt1 code, lpips, CPU only), from D157's directory, nice 10.
B=/data/relcfd/chayo/physmorph_v2
source $B/tmp/d157/env.sh
export D157_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
cd $B/tmp/d157
nice -n 10 $PY $B/tmp/d145r/r5_d157x.py "$@" >> $B/output/gpu/d145_ext/r5/d157x.log 2>&1
