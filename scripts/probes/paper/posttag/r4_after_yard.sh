#!/bin/bash
# D145 reading 4: the display samples card job (r4_gpu.sh disp) after the yardstick job, one card at a time
X=/data/relcfd/chayo/physmorph_v2/output/gpu/d145_ext/r4
until grep -q "^done yard" $X/status.txt; do sleep 60; done
bash /data/relcfd/chayo/physmorph_v2/tmp/d145r/r4_gpu.sh disp
