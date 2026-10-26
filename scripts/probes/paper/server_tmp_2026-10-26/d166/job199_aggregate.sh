#!/bin/bash
# 199_d167_aggregate.sh -- queue job (agent D166): the master file output/gpu/RENDER_REST_WEIGHT_SUMMARY.md. Starts a
# detached CPU loop (no card) that rewrites it every 10 min from every D166 / D167 summary (+ D162's 51 / 53-55 when they
# appear) until all jobs in tmp/d166/expected.txt have one (36 h at most), then exits; the job itself ends at once.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d166; OUT=$B/output/gpu/RENDER_REST_WEIGHT_SUMMARY.md
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
CUDA_VISIBLE_DEVICES= setsid nohup bash -c "for k in \$(seq 1 216); do n=\$($PY $W/d166_read.py aggregate $OUT 2>>$W/aggregate.err | tail -1)
  set -- \$n; echo \"\$(date '+%m-%d %H:%M') \$n\" >> $W/aggregate.log; [ -n \"\$1\" ] && [ \"\$1\" = \"\$2\" ] && break; sleep 600; done" > /dev/null 2>&1 < /dev/null &
echo $! > $W/aggregate.pid
echo "aggregator pid $(cat $W/aggregate.pid)"
