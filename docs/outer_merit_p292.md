# P292: promoted-state outer render loss consistency

The corrected outer render record equals an independent GPU re-evaluation of
the fixed-target silhouette loss on all eight saved commits. Maximum absolute
and relative errors are both **0** in this execution. The legacy inner silhouette
record is a different quantity: on its own saved positions, the fixed-target
value is **5.85 to 9.77 times** that record. This is a loss-evaluation consistency
check, not a raw physical-quality metric, a hole measurement, or evidence of
improved settling or final shape.

## Matched inputs and numerical scope

Both `outer_legacy8` and `outer_gate8` used the immutable
`/data/relcfd/chayo/physmorph_v2/work/p292/outer_snapshot` numerical tree. The
complete `physmorph/**/*.py` digest matches both run JSONs:
`c56fdd47970d30689f858fcdbf545240706ca94ca77db7b47980e222bd327cd1`.
Their configurations and MPM parameters are equal except
`outer_render_committed=False` (absent/default in the legacy JSON) versus `True`.
The frozen driver SHA-256 is
`c1937e9ae80823bc4c29fdfb0dad1df003cdc7c366578d5a0f4291f7cc6c6859`.
The later telemetry extension and coarse-to-fine history-reset follow-up do not
enter this immutable snapshot. No coarse-to-fine event occurred in either prefix.

The source is the original mixed60 archive, whose complete file SHA-256 is
`7896e4fcb5f95020c559c3fb12a570c1e2106c062f0c2887c230986f2aee336e`.
Its source/target point-array hashes, both compact commit-NPZ hashes, and both
run-JSON hashes are retained in the diagnostic output. The frozen driver loads
this archive directly. The probe does not resample or rebuild target PBR/shading.

Discretization: N=300000, T=20, dt=1/240 s, MPM dx=0.3062907543956724 wu,
MPM/loss grid36 cubed. Each prefix contains eight accepted windows, with saved
commit indices corresponding to raw20,40,...,160. The probe checks this mapping
against history and rejects null/rejected commits or a different count.

The loss uses18 views (six azimuths at elevations0,+0.5,-0.5 radians),64x64
pixels, CIC, sil_k=1.5, w_hole=2 and w_spray=1. Extent is
`max(abs(target))*1.25 = 4.897848963737488 wu`, reproducing `build_target`.
All target alpha construction and position-to-loss evaluation run on CUDA.
NumPy is used only for archive I/O; scalar results are transferred for the report.
No Gaussian/PBR loss, render export, backward solve or full pipeline run is invoked.

## Direct check on saved commits

For each `.npz['commits'][i]`, the probe calls the frozen `d_render` directly
against silhouettes of the fixed original target. It does not call the new
`fixed_outer_render` wrapper or reuse recorded inner values as its result.
The newer record is `outer_track_version='committed_fixed_v1'`.

| Accepted commit | Legacy inner d_sil | Legacy fixed recomputation | New recorded outer = recomputation |
| --- | ---: | ---: | ---: |
| 1 | 0.02182198 | 0.12756756 | 0.12758738 |
| 2 | 0.01448807 | 0.08722506 | 0.08732551 |
| 3 | 0.00665302 | 0.06430707 | 0.06435154 |
| 4 | 0.00629313 | 0.05463647 | 0.05475247 |
| 5 | 0.00517332 | 0.04829679 | 0.04797797 |
| 6 | 0.00421252 | 0.04114148 | 0.04083489 |
| 7 | 0.00429951 | 0.03461239 | 0.03473033 |
| 8 | 0.00411129 | 0.02774037 | 0.02752200 |

All eight new outer values match at their full serialized float32 scalar values,
not merely at the table's rounded precision. The probe's absolute comparison
tolerance was2e-7; observed error was0. This single execution does not promise
bitwise repeatability of future CUDA atomic reductions.

Legacy `d_sil` can describe a paced target before the post-rollout position
corrections, while this re-evaluation describes the fixed target at the saved
promoted positions. The ratio therefore demonstrates that the two tracks are
not interchangeable; it does not attribute their difference separately to pacing,
PIC, shifting or any one operator. Small differences between the two arms'
positions/losses are not a demonstrated benefit of the corrected gate. Both
prefixes accepted all eight windows, and full convergence was not tested here.

The consistency probe launched on hyde06 GPU0 at2026-09-26 23:33:38 UTC and
reported3.184 seconds including source hashing and result assembly after Python
imports. The source pipeline runs reported148.38 seconds for legacy and149.89
seconds for the new gate. One pair is not a general performance benchmark.

## Artifacts and validation

The local result is `output/p292/outer_merit_gpu.json`; the exact executed script
is `output/p292/outer_merit_gpu_probe.py`, SHA-256
`80f7acf94b4483ab0aa7120e7c6d7a90e89486b814d0cf5b69f79385e7f59373`.
The launch script and log are beside it. Server copies are under
`work/p292/outer_merit_audit/`; the two source run JSONs and compact commit NPZs
remain under `work/p292/`. No simulation archive or delivered video was modified.

The pure helper and its runner integration were separately reviewed with15 CPU
tests passing (12 helper plus3 integration cases). They cover paced-inner-value
invariance, changed promoted positions, render-off preservation and acceptance
history behavior. The diagnostic script passed compile checking. Final
artifact/report adversarial review passed: the reviewer verified script/run-JSON
hashes, code/config provenance, all eight exact stored-versus-recomputed values,
the legacy ratio range and the reconstruction method. The reviewer did not
independently rerun GPU work or claim future bitwise repeatability.
