#!/bin/bash
set -euo pipefail
work=/data/relcfd/chayo/physmorph_v2/work/p303
cd "$work"
printf '%s  %s\n' 6b3a6de0da5e052d1817833c61e86a7e180981d49881206d59a95c9d7059fcb8 p337_preparation_source1.tar | sha256sum -c -
mkdir code_current_preparation1
tar -xf p337_preparation_source1.tar -C code_current_preparation1
printf '%s\n' 7710fdc09bc380a8231fa9b93a0fb997f916f903 > code_current_preparation1/VERSION
setsid nohup bash "$work/run_p337_v1.sh" 7710fdc09bc380a8231fa9b93a0fb997f916f903 current-successor-verify p337_cuda1 > "$work/p337_cuda1.launch.log" 2>&1 < /dev/null &
