#!/bin/bash
S=/data/relcfd/chayo/physmorph_v2/output/gpu/d163/status.txt
until grep -q "R plq40k_TC_97 start" $S; do sleep 30; done
cd /data/relcfd/chayo/physmorph_v2/tmp/d163
USE_CARD=0 OUTD=/data/relcfd/chayo/physmorph_v2/output/gpu/d163/runs300 ./d163_chain2.sh /data/relcfd/chayo/physmorph_v2/tmp/d163/d163_runs300_T.txt 1
