#!/bin/bash
set -euo pipefail
B=/data/relcfd/chayo/physmorph_v2
Q=$B/tmp/gpu1_queue
D=$B/tmp/dragon_dense_checkpoint_v3
test "$(sha256sum "$D/plan.json" | cut -d' ' -f1)" = ca8dd3d2ce49ec1e6a0c7d55db2246262bca8d59539e0af9bb41c32b05d1992a
test -z "$(find "$Q" -maxdepth 1 -name '119*' -print -quit)"
test ! -e "$Q/.119_dragon_dense_checkpoint_detached.staged"
cat > "$Q/.119_dragon_dense_checkpoint_detached.staged" <<'JOB'
#!/bin/bash
set -euo pipefail
exec setsid --wait bash /data/relcfd/chayo/physmorph_v2/tmp/dragon_dense_checkpoint_v3/job_dense.sh
JOB
bash -n "$Q/.119_dragon_dense_checkpoint_detached.staged"
chmod 700 "$Q/.119_dragon_dense_checkpoint_detached.staged"
sha256sum "$Q/.119_dragon_dense_checkpoint_detached.staged" "$D/plan.json" "$D/job_dense.sh"
ln "$Q/.119_dragon_dense_checkpoint_detached.staged" "$Q/119_dragon_dense_checkpoint_detached.sh"
date
