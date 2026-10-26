#!/bin/bash
# idle_mon.sh: sample every GPU utilization + memory every 20 s for 10 min; print per-GPU max util
out=/data/relcfd/chayo/physmorph_v2/tmp/d163/idle_mon.log
: > $out
for i in $(seq 1 30); do date +%T >> $out; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> $out; sleep 20; done
echo DONE >> $out
