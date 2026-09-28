# Current retained results after cleanup

The requested active result set is only **before** and **corrected**, the matched
P300 cap24 pair. This is not the distinct P301 observer realization. Neither arm
establishes all-particle rest or the absence of morph-time holes.

Local results live in `output/before/` and `output/corrected/`: each contains the
original native 4K MP4, renderer JSON and run JSON. `output/index.html` opens both.
The old comparison MP4, decoded frames, thumbnails and previous experiment
folders are removed from the active output tree.

Server results use the same two directories beneath
`/data/relcfd/chayo/physmorph_v2/output/`. Each additionally retains its original
raw trajectory NPZ, compact result NPZ and run log. Filenames and historical
sidecar bytes are unchanged; their original absolute paths are provenance,
not current locations. The raw trajectories permit rerendering and the shared
inputs permit a fresh initial-state run; they are not complete endpoint resume
checkpoints because full velocity/APIC/control state is not exported.

Shared reproduction material lives under server `repro/current_pair/`: exact
original src/tgt arrays and metadata, prepared target reference, original
old/new numerical snapshots, the frozen renderer and result/QA evidence.
Each numerical snapshot has a separate `gpu_pipeline_relocated.py`; compared
with its untouched original driver, only the two input paths change.
No resampling, new mesh-volume weights or numerical changes are introduced.

`maintenance/run_current_pair.sh before|corrected GPU NEW_TAG` is the relocated
cap24 entry point. It preserves the 50 s GPU launch interval and the original
recipe: eight inner iterations per window as a budget, T = 20, dt = 1/240,
dx = 0.3062907544, loss grid = 36^3.
Check GPU occupancy and run
only on hyde06 through hyde01. The launcher reserves the new basename and opens
its own `output/NEW_TAG.log`; redirect a detached wrapper to a separate
`maintenance/NEW_TAG.launch.log`, never that reserved run log.
This launcher was syntax/configuration checked during cleanup, not run as a
new simulation. Runtime code, assets, external Python and CUDA dependencies
remain in place. The local isolated Warp 1.16 installation moved from
`output/p300/deps` to `C:/dev/physmorph_runtime/warp116`; set `PYTHONPATH` to that
directory for CPU tests and use a runtime/cache location outside `output`.

Untracked results are archived outside the workspace/server project at
`C:/dev/physmorph_archives/cleanup_keep_pair_20260928/`. Both archives have exact
file manifests, per-member SHA256 verification and durable verification receipts.
Cleanup completed on 2026-09-28: 3,624 local entries and 23,448 server entries
were removed. Server project usage fell from about 99.47 GB to 5.88 GB (decimal,
apparent bytes). The two external compressed archives occupy about 92.88 GB
locally; removing the active results does not remove these backups.
Server deletion additionally holds a kernel read
lease while rechecking each regular file and unlinking it; local deletion checks
native file IDs under handles denying writers. Both remove only listed entries
and then empty directories. Retained result copies, active source code and runtime
dependencies are excluded from the deletion scopes. Old work snapshots and the
original result paths are archived and removed after preserving the current pair.
No other project is cleaned.
