#!/bin/bash
# d156j_chain.sh MESH -- D156j: D156i, then D156g, on the conservative hull of MESH (CPU, nice 10, one process)
B=/data/relcfd/chayo/physmorph_v2
source $B/tmp/d156/env.sh
O=$B/output/gpu/d156j
cd $B/repo_d156
m=$1
nice -n 10 $PY $B/tmp/d156/d156_sfs_i.py $O/i $O/assets_hull 40000 96 $m > $O/i/sfsi_$m.log 2>&1 < /dev/null
echo "$m i rc $? $(date "+%F %T")" >> $O/chain.state
nice -n 10 $PY $B/tmp/d156/d156_sfs_g.py $O/g $O/assets_hull 40000 96 $m > $O/g/sfsg_$m.log 2>&1 < /dev/null
echo "$m g rc $? $(date "+%F %T")" >> $O/chain.state
