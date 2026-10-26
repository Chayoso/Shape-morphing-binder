#!/bin/bash
set -euo pipefail
B=/data/relcfd/chayo/physmorph_v2
Q=$B/tmp/gpu1_queue
F=$B/output/gpu/fixed_rate_feasibility_v2
OLD=$B/output/gpu/fixed_rate_feasibility_v2_signal_failure_0058
test "$(realpath "$F")" = "$F"
test "$(realpath "$(dirname "$OLD")")" = "$B/output/gpu"
test ! -e "$OLD"
test ! -e "$F/result.json"
test "$(cat "$F/exit_code.txt")" = 90
if ps -p 3851858 --ppid 3851858 -o pid= | rg '[0-9]'; then
    exit 1
fi
ps -u chayo -o pid,ppid,sid,pgid,comm,args > "$B/tmp/d166/process_observation_0103.txt"
mv -- "$F" "$OLD"
for number in 117 118; do
    test -z "$(find "$Q" -maxdepth 1 -name "$number*" -print -quit)"
done
cat > "$Q/.117_fixed_rate_feasibility_detached.staged" <<'JOB'
#!/bin/bash
set -euo pipefail
exec setsid --wait bash /data/relcfd/chayo/physmorph_v2/tmp/fixed_rate_feasibility_v2/job.sh
JOB
cat > "$Q/.118_dragon_dense_checkpoint_detached.staged" <<'JOB'
#!/bin/bash
set -euo pipefail
exec setsid --wait bash /data/relcfd/chayo/physmorph_v2/tmp/dragon_dense_checkpoint_v2/job_dense.sh
JOB
bash -n "$Q/.117_fixed_rate_feasibility_detached.staged"
bash -n "$Q/.118_dragon_dense_checkpoint_detached.staged"
chmod 700 "$Q/.117_fixed_rate_feasibility_detached.staged" "$Q/.118_dragon_dense_checkpoint_detached.staged"
sha256sum "$Q/.117_fixed_rate_feasibility_detached.staged" "$Q/.118_dragon_dense_checkpoint_detached.staged" \
    "$B/tmp/fixed_rate_feasibility_v2/plan.json" "$B/tmp/fixed_rate_feasibility_v2/job.sh" \
    "$B/tmp/dragon_dense_checkpoint_v2/plan.json" "$B/tmp/dragon_dense_checkpoint_v2/job_dense.sh" \
    "$OLD/status.json" "$OLD/reservation_release.json" > "$B/tmp/d166/detached_launch_0103.sha256"
ln "$Q/.117_fixed_rate_feasibility_detached.staged" "$Q/117_fixed_rate_feasibility_detached.sh"
ln "$Q/.118_dragon_dense_checkpoint_detached.staged" "$Q/118_dragon_dense_checkpoint_detached.sh"
date
