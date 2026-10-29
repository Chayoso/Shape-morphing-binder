#!/bin/bash
# b13_hold.sh -- B13 (CPU, no card): the default arm's L99 / L101 kept frames on the four meshes whose frames D145's
# read300_cpu.sh removes after its readings (teapot, bob, armadilo, homer; KEEP 0 in the post-tag queue) are hard-linked
# into output/gpu/b13/hold as soon as they appear, so that b13_chain300k.sh can read their end against the independent
# sample; the chain removes each link after that reading (cleanup.log). Read-only on D145's files. Stops when every link
# is made or read, or after 72 h.
G=/data/relcfd/chayo/physmorph_v2/output/gpu; LD=$G/d145t_b1; H=$G/b13/hold; S=$G/b13/status300.txt
mkdir -p $H
t_end=$(( $(date +%s) + 259200 ))
while [ $(date +%s) -lt $t_end ]; do
  left=0
  for m in teapot bob armadilo homer; do
    for s in 99 101; do
      n=${m}300k_L$s
      grep -q "^L-endgen $n " $S && continue
      grep -q "^hold $n " $S && continue
      left=$((left + 1))
      if [ -f $LD/${n}_frames12.npz ] && ln $LD/${n}_frames12.npz $H/${n}_frames12.npz 2>/dev/null; then
        echo "hold $n linked $(date '+%m-%d %H:%M')" >> $S
      fi
    done
  done
  [ $left = 0 ] && break
  sleep 60
done
echo "hold watcher ends $(date '+%m-%d %H:%M')" >> $S
