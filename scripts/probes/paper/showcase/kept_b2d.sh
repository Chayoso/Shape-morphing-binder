#!/bin/bash
# kept_b2d.sh -- SHOWCASE: the moment bunny2dragon300k_L97 lands, its kept-frames path into the status file (for DEMO)
D=/data/relcfd/chayo/physmorph_v2/output/gpu/showcase; n=bunny2dragon300k_L97
until [ -e $D/rc_$n ]; do sleep 30; done
echo "$(date "+%m-%d %H:%M:%S") [kept] $n rc/wall/peak $(cat $D/rc_$n); kept frames $D/${n}_frames12.npz ($(du -h $D/${n}_frames12.npz 2>/dev/null | cut -f1); frames, raws, tgt; bunny -> dragon 300k render arm seed 97)" >> $D/status.txt
