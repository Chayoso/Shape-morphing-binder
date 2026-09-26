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
