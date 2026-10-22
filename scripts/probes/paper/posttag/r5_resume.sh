#!/bin/bash
# r5_resume.sh -- D145 reading 5: restart r5_gpu.sh at 17:00 CDT (the coordinator: the cards are the 17:00 demo's until then)
until [ $(date +%H%M) -ge 1700 ]; do sleep 60; done
echo "chain resumed by r5_resume.sh $(date "+%m-%d %H:%M")" >> /data/relcfd/chayo/physmorph_v2/output/gpu/d145t_cell/status.txt
exec bash /data/relcfd/chayo/physmorph_v2/tmp/d145r/r5_gpu.sh
