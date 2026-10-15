#!/bin/bash
# b12_queue.sh MESH... -- B12's measured runs: for each mesh in turn, claim one card (unlocked and < 2 GB for 10
# consecutive minutes, b12_gpu.sh 600), run its 12 runs (Large Steps and FlexiCubes, arms a / b, view-set seeds 97 / 99
# / 101; b12_batch.sh, each run's CPU readout detached), release the card before the next mesh.
B12=/data/relcfd/chayo/physmorph_v2/baselines/b12
for mesh in "$@"; do
  items=""
  for method in ls fc; do for arm in a b; do for seed in 97 99 101; do
    items="$items $method:$mesh:$arm:$seed"
  done; done; done
  $B12/scripts/b12_gpu.sh 600 "b12/$mesh: Large Steps + FlexiCubes, 12 runs of ~1 min" $B12/scripts/b12_batch.sh $items
done
echo "$(date '+%F %T %Z') queue done: $*"
