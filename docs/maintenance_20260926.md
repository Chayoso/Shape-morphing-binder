# Local cleanup and refactor, 2026-09-26

Requested by the user before continuing the thin-feature diagnosis. The existing
P291 code was pushed at `a7ec4ce` to `origin/v3-grid-gs`; `origin/main` was not used.

## Behavior-preserving extraction

`pipeline.settlement.pin_arrival_evidence` owns the pin eligibility mask and its
telemetry. Confirmed admission evaluates endpoint distances once and intersects
the result with the window-start mask. Legacy non-paced eligibility remains
explicitly labelled, with a null geometric arrival fraction. Admission and transit
protection still use the same mask. Arrival tests now live in `test_settlement.py`.

The standard and extended autograd bridges share the existing `_dfc_to_warp`
conversion. Tensor layout, storage sharing and gradient requirements are unchanged.
No physics, loss, settlement policy or control coefficients were changed.

Validation: compile checks, independent adversarial review, and the full CPU suite
with CUDA disabled: **295 passed, 8 skipped** in 72.30 seconds. This is refactor
validation, not a new morph quality result.

## Retired local reports

Archived complete `output/report_g40_page`, `output/report_g41_page`, and
`output/surface_page` packages, including their old particle/splat comparisons and
metrics, so each report can be restored intact. Also archived ten explicitly named
MC/Poisson images/videos in `output/r300_page`; its three HTML pages did not reference
these files. Current `output/c291`, the remaining r300 page, and raw audit evidence
were retained. Removed the empty root `heat_as300_mc` directory.

Archive outside the repository:
`C:/dev/physmorph_archives/local_legacy_20260926T205506Z/mc_poisson_reports.zip`.
Sibling `manifest.json` records original relative paths, sizes and SHA-256 hashes;
`deleted_paths.txt` records removed files. The ZIP contains **1,053 files**, originally
985,362,701 bytes, compressed to 759,333,890 bytes. Net storage reduction is about
226 MB before manifest/log overhead; the workspace reduction is about 985 MB.

Archive SHA-256:
`94b324a857378d09b0607e185d5191bb1148f935ac525d2e8b2716faf0f76ea1`.
Every decompressed entry and source hash was checked before deletion; an independent
review verified the archive inventory and archive hash. Source removal used native
PowerShell literal paths after allowlist, canonical-path, reparse and tracked-file
checks. Empty directories were removed non-recursively after the same checks.
No server results were deleted in this cleanup.

## Subsequent GPU request

The user expanded the refactor to move physics and rendering numerical work to
the GPU. That migration is separate from the tested extraction above. CPU remains
the host for file I/O and orchestration; GPU validation runs on hyde06 via hyde01.
The current audit found CPU covariance decomposition in the GPU-labelled splat
script, CPU kNN fallbacks, and NumPy/SciPy window preparation and commit work.

## Server headroom for the matched render-loss ablation

After the full GPU baseline archive, project usage reached99.776GB. Five completed
failed-run artifacts were copied to `C:/dev/physmorph_archives/c291/offload20260926T2206/`:
the already verified gzip archives for `force60s`, `force8s`, `norm8s`, `normphys8s`,
and the raw `fixedbody60` NPZ (all `c291_bunny_*_render_full_dt_iso_nn`).
Total7,353,173,578bytes. Each copy was SHA-256 checked and locally fsynced before
single-file server duplicate removal. Source identity, size, timestamp and SHA were
rechecked immediately before removal; durable offload receipts remain on both machines.
Logs, JSON, current mixed60/render videos and the CUDA baseline were retained.
Project usage after offload:92.426GB. The100GB watcher remains active.

The exact allowlist, sizes, hashes and local destinations are in
`output/c291/gpuwork/offload_2206_verified.json` locally and durable per-file
receipts in `/data/relcfd/chayo/physmorph_v2/archives/completed_failed_runs/`.

## P292 comparison headroom

Before the matched P292 runs, four completed superseded/failed prefix NPZ files
were offloaded to `C:/dev/physmorph_archives/p292_prefixes/`: `c291_bunny_replay20s`,
`c291_bunny_body8`, `c291_bunny_base8s`, and `c291_bunny_body8s`, each with the suffix
`_render_full_dt_iso_nn.npz`. Total size: **3,711,616,416 bytes**. The local
`manifest.json` and `verified.json` preserve paths, sizes and SHA-256 hashes.
Every local copy was hash-verified and fsynced; each remote source's identity,
size, timestamp and hash was checked again before its single-file removal.
Durable per-file `.p292_offload.json` receipts remain in the server's
`archives/completed_failed_runs/`. Logs, JSON, original mixed60, target reference
and the full CUDA render-on/off evidence were retained. Usage fell from about
95.495 GB to 91.784 GB before the new results. This also creates room before the
user's 100 GB cleanup threshold; the watcher has no remaining preauthorized
candidate after these offloads, so further headroom must be checked explicitly.

At 22:56 UTC usage reached 98.543 GB after both full P292 arms completed. The
completed earlier CUDA-on raw archive, `work/gpu_refactor/cuda_mixed60_full_render_full_dt_iso_nn.npz`,
was copied to `C:/dev/physmorph_archives/p292_previous_cuda/` (3,277,502,732 bytes).
SHA-256 `926c2a77343f8d1acef9ff32a5d91a616e5d7134b2d1c530d3a4b3997130c6bd`
matched locally and on the server immediately before removing the remote duplicate.
The same fsync/identity/receipt protocol was used. Local `manifest.json` and
`verified.json` and the server `.p292_offload.json` receipt preserve restoration
details. The earlier CUDA-on JSON and compressed endpoint/commit NPZ remain on the
server; its raw audit evidence remains recoverable from this local archive. The
original mixed60 and all current P292 inputs/results remain on the server. Usage
after removal: 95.266 GB, below the 100 GB threshold throughout this cleanup.

## Completed P292 body-RPROP raw archive

At 23:26:37 UTC, after all body-RPROP raw audits had completed, the parent-approved
single archive `work/p292/body_rprop60_render_full_dt_iso_nn.npz` was offloaded to
`C:/dev/physmorph_archives/p292_body_rprop/`. Its size is **2,946,302,700 bytes** and
SHA-256 is `ebb0bcc20162ef047b444533091aef9c755befced2bb31bc1f11393a20aded31`.
Local transfer, full-file hashing and fsync completed before removal of the remote
duplicate. The remote canonical path, device/inode, size, timestamp and hash were
rechecked against the original plan and local verified receipt. A durable server
receipt and parent-directory fsync preceded the sole authorized unlink, followed
by a source-directory fsync. The independent narrow protocol review found no blocker.

Local `manifest.json`, `verified.json` and `remote_receipt.json` retain restoration
details. The durable server receipt is
`archives/completed_failed_runs/body_rprop60_render_full_dt_iso_nn.npz.p292_offload.json`.
The body-RPROP JSON, log and compressed commit NPZ (129,868,484 bytes) remain on the
server; no local file or other remote artifact was deleted. Reproduction scripts
are retained in `output/p292/offload_body_rprop.py` and `verify_body_rprop.py` locally.

Exactly 2,946,302,700 bytes were freed. Project usage after removal was
**93,315,528,838 bytes (93.316 GB)**; the net change from the earlier 96.218 GB
measurement also includes writes by concurrent tasks. This creates headroom for
the bounded no-PIC completion test without discarding failed-experiment evidence.

## Completed P292 baseline and PIC-off raw archives

At 23:51 UTC the completed baseline and PIC-off raw archives were copied to
`C:/dev/physmorph_archives/p292_baseline/` and `p292_no_pic/`. The baseline file is
2,697,902,676 bytes, SHA-256
`212aa6723445674c8930e27824521fd7f64e2e796b430b7811233d2c070e6527`;
PIC-off is 3,360,302,740 bytes, SHA-256
`cc9010417c5188a0bed439600fc6dd05e2ddd0eb4bcb170e635c76b7351a618e`.
Both full raw audits and the separate diagnostic renderer had finished before
the remote duplicates were removed. The independent review verified that the
two procedures are exact fixed-path substitutions of the approved body-RPROP
hash/fsync/identity/receipt protocol. Local verified and remote receipts exist
alongside each archive; all compact commit NPZs, JSONs and logs remain remote.
No local result was deleted. Total raw bytes freed: 6,058,205,416. Measured project
usage after this cleanup was 90,868,927,247 bytes (90.869 GB), leaving room for
the bounded accepted-arrival handoff pair without crossing 100 GB.

## Completed P292 outer-control and arrival-handoff raw archives

At 2026-09-27 00:20 UTC, after the matched raw-quality and phase audits completed,
only `work/p292/outer_no_pic60_render_full_dt_iso_nn.npz` and
`work/p292/handoff_no_pic60_render_full_dt_iso_nn.npz` were offloaded. No renderer
consumer remained. The local archives are:

| Local directory | Raw file size, bytes | SHA-256 |
| --- | ---: | --- |
| `C:/dev/physmorph_archives/p292_outer_no_pic/` | 3,691,502,772 | `2deb73ca975898a4b940df5ddac61daebb049c621bbc2cb3e0adb422b1f722b9` |
| `C:/dev/physmorph_archives/p292_handoff_no_pic/` | 3,111,902,716 | `81e76912235dbc80387901f9b0aba89421c8855729cc8b478182ab030032860b` |

Each directory contains its named raw NPZ, `manifest.json`, `verified.json` and
`remote_receipt.json`. Both copies passed complete SHA-256 comparison and local
fsync before any unlink. The reviewed scripts are exact fixed-name substitutions
of `offload_baseline.py` and `verify_baseline.py`, retained in `output/p292/` as
`offload_outer_no_pic.py`, `verify_outer_no_pic.py`, `offload_handoff_no_pic.py` and
`verify_handoff_no_pic.py`. The independent narrow protocol gate closed before
deletion. It reviewed the procedure, not the subsequently completed transfers.

At commit, each server source was rehashed and its canonical path, device/inode,
size and timestamp rechecked against the plan and verified local receipt. A
durable per-file receipt and directory fsync preceded each sole authorized unlink;
the source directory was fsynced afterward. Both server receipts were copied back
and checked against their local verified receipts. Durable server receipt names
are `archives/completed_failed_runs/outer_no_pic60_render_full_dt_iso_nn.npz.p292_offload.json`
and `handoff_no_pic60_render_full_dt_iso_nn.npz.p292_offload.json` in that same directory.

The compact commit NPZ, JSON and log for both runs remain on the server. All local
copies remain; no other file was deleted and no GPU task was launched. Exactly
**6,803,405,488 raw-file bytes** were removed from the server. Measured project
usage afterward was **91,169,734,912 bytes (91.170 GB)**, below the user's 100 GB
threshold. Concurrent metadata/code writes may also affect the net project-size
change; the freed-byte count is the sum of the two verified payload sizes.

## Local regression-test CUDA exposure, recorded 2026-09-27 00:43 UTC

The previous-position bridge's legacy regression launcher incorrectly used
PowerShell `$env:CUDA_VISIBLE_DEVICES=''`. In this shell the empty value did not
hide the local GPU. The owned pytest process was observed in local `nvidia-smi`
while entering a CUDA-parametrized legacy test. Its PID was **39484**, and the
verified command was:

```text
C:\Users\ok429\anaconda3\python.exe -X utf8 -m pytest tests/test_ext_bridge.py tests/test_bridge_cpu.py tests/test_persistent_traj.py tests/test_body_control.py tests/test_layer_relax.py tests/test_material_bonds.py -q
```

Only that process was stopped, after checking its PID and command line with
`Get-CimInstance`; no other GPU process was touched. The unfinished run's result
was discarded. No pipeline or rendering job was launched. The exact process
start/stop timestamps and elapsed duration were not captured and are unknown;
termination completed before this 00:43 UTC entry. This was an unintended local
CUDA test branch and violated the intended CPU-only local verification scope.

The corrected invocation sets `$env:CUDA_VISIBLE_DEVICES='-1'`. Before rerunning,
the subprocess explicitly reported `CUDA_VISIBLE_DEVICES=-1` and
`torch.cuda.is_available()==False`. That complete CPU regression run passed
**40 tests with 3 CUDA skips in 7.85 seconds**. The independent reviewer also used
`-1` and passed the 25 new bridge tests plus 7 geometric-rest helper tests. Future
local test invocations require the explicit `-1` value; an empty string is not a
valid GPU exclusion on this host. CUDA captured-graph validation remains assigned
to hyde06 through the jump host.

## Five completed raw archives offloaded before the P294 full pair

At 2026-09-27 01:09 UTC, five explicitly approved completed raw archives had been
copied into `C:/dev/physmorph_archives/p294_headroom_20260927/`, fully SHA-256
verified and fsynced locally. Only their remote duplicates were then removed.
Parent confirmed that the completed audits had no remaining consumers. Before
transfer, the user process list showed no numerical job and `lsof` found no open
candidate file; its warnings concerned unrelated users' FUSE mounts.

| Original remote path below the project root | Bytes | SHA-256 |
| --- | ---: | --- |
| `work/p292/taper8_render_full_dt_iso_nn.npz` | 696,302,308 | `ceefc6fe580839f47e0fb1299467d69be490e42be73e5e5d401097f270da226c` |
| `work/p292/no_pic8_render_full_dt_iso_nn.npz` | 696,302,308 | `bc2bad0faed87ad11bd8857c2cfa1e0df9177da04fcab42f549e412405285d9c` |
| `work/p293/pic_legacy8_render_full_dt_iso_nn.npz` | 696,302,308 | `0f886be69b5eb8e7f71204a8598a309d83ae1ba007ab61f7f7e90bfd18ac3855` |
| `work/p293/pic_objective8_render_full_dt_iso_nn.npz` | 696,302,308 | `15bab0aeb17b982d83e93db9106f8d67dd242bdf38fd7e6f037b997fab2ee06a` |
| `work/gpu_refactor/cuda_phys60_full_render_full_dt_iso_nn.npz` | 2,780,702,684 | `1e3e56b4675a44c13b9fe23970f3fdddab0a5070fb60d98b5348c6b1b17e45d7` |

The local directory contains those exact five basenames, each case's
`*_manifest.json` and `*_verified.json`, and copies of all five durable server
receipts named `<archive basename>.p294_offload.json`. The downloaded server
receipts were checked for exact equality with the local verified records. Their
server originals remain in `archives/completed_failed_runs/`.

The reviewed scripts are `output/p294/offload_headroom.py` and
`output/p294/verify_headroom.py`. The remote script accepts only the five fixed
case names and unlinks one exact source per invocation. Each commit rehashed
the full source, matched canonical path and device/inode/size/mtime to its plan
and local verified receipt, wrote and fsynced its server receipt, then unlinked
that single duplicate and fsynced its parent. Independent protocol review closed
before commit; the reviewer also checked all completed local receipts, manifest
fields and file sizes without claiming a second payload rehash.

All JSON/log/compact NPZ and audit evidence remain remote, including
`render_off_full.log`, `.done`, and `render_influence_full.json`. The original
c291 source and all inputs/render deliverables/code were untouched. No local
file was deleted and no GPU task was launched for this storage operation.
Exactly **5,565,911,916 bytes** were freed. Measured project usage was
**88,628,322,583 bytes (88.628 GB)** at 01:09 UTC, down from the initial
94,191,350,462-byte inventory and below the planned 89 GB launch threshold.
The user's cleanup threshold remains 100 GB; 89 GB was the working headroom
target for the upcoming pair of full archives, not a new user requirement.
The net project-size change also includes concurrent metadata/code writes; the
freed-byte total above is the sum of the five verified payloads.
