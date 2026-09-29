#!/bin/bash
set -euo pipefail
expected=${1:?frozen commit}
archive_hash=${2:?archive hash}
[[ "$expected" =~ ^[0-9a-f]{40}$ && "$archive_hash" =~ ^[0-9a-f]{64}$ ]] || exit 2
work=/data/relcfd/chayo/physmorph_v2/work/p303
cd "$work"
printf '%s  %s\n' "$archive_hash" p337_preparation_source2.tar | sha256sum -c -
mkdir code_current_preparation2
tar -xf p337_preparation_source2.tar -C code_current_preparation2
printf '%s\n' "$expected" > code_current_preparation2/VERSION
setsid nohup bash "$work/run_p337_v2.sh" "$expected" current-successor-verify p337_cuda2 > "$work/p337_cuda2.launch.log" 2>&1 < /dev/null &
