# Shared endpoint experiment (P293)

This opt-in prototype makes the inner spatial objective and the runner promote
the same finite-order XPIC endpoint. It does not yet establish hole-free growth,
geometric rest or improved 4K appearance. The equation and scope are in
[method section 10.40](method.md#1040-shared-finite-order-xpic-endpoint-objective-p293-prototype).

## Controlled comparison

Both arms use the original mixed60 bunny source and target, N=300000, T=20,
dt=1/240, dx=0.3062907543956724 world units, the 36-cubed loss grid, eight inner
iterations, mixed stress/body controls, commit PIC and corrected fixed-target
outer render merit. Both disable external subgrid shifting. The candidate alone
enables `commit_pic_objective`. The original 300-window schedule is retained;
one- and eight-window caps bound the integration and prefix experiments.

The immutable code lives at `/data/relcfd/chayo/physmorph_v2/work/p293/endpoint`.
Local `output/p293/endpoint_snapshot_manifest.json` identifies all 146 source
members; the independent reviewer verified every ZIP member against it.

## Operator validation on CUDA

Executed on hyde06 GPU0 at 2026-09-27 00:28:10--00:28:11.745 UTC, on the original
300000-by-3 float32 source. Source-array SHA256 is
`71eb14d38c2efb41379a12b0ce017430e093883cbce51188265ed9e2948d0e34`.
This is an operator probe, not an MPM rollout or a quality experiment.

| Check | Result |
| --- | --- |
| Forward versus legacy CUDA filter | Maximum error 9.536743e-7 wu; RMS 3.358919e-8 wu |
| H / H-transpose dot identity | Norm-product normalized error 3.709656e-11 |
| Double directional finite difference, 12 source neighbors, nonuniform masses and four pins | Absolute error 3.105161e-11 |
| Window-start pins, 17648 IDs | Exact bits preserved; zero displacement |
| Repeated applications | All declared tolerances passed; allocated memory returns after output release |

Preparation took 20.79 ms and retained 244592128 Torch bytes. A reusable forward
application took approximately 13.7 ms, with a 16731648-byte temporary Torch
allocation peak above the existing live allocations. Transpose took 13.86 ms.
The legacy measurement includes preparation, telemetry and a CuPy result copy;
it is not equivalent work and does not establish a pipeline speedup. Memory
figures are Torch allocations, not a whole-device peak. The JSON separately
records CuPy pool samples, tolerances, metadata and executed dependency hashes.

Evidence: local `output/p293/pic_endpoint_gpu.json`, with a matching remote copy.
Independent adversarial review closed the operator evidence gate.

## One-window CUDA integration

The matched pair used numerical core SHA256
`e269dc2570f4cf9d10b8e4196cc75626ddec3dfd72a91cec3afef82a4c959084`.
Both arms accepted eight inner iterations, rejected none, and promoted one
outer commit with all state guards zero. Candidate telemetry reports the owned
accepted endpoint in `promoted_xpic` space, with objective-to-commit difference
exactly zero. The legacy arm reports `raw_rollout` objective space.
This CUDA pair exercised the accepted-buffer path, not a final replay; replay
ownership and comparisons have separate CPU integration coverage.

Recorded wall times were 15.52 and 17.18 seconds; Torch allocation peaks were
7.713 and 7.950 GB. These are single integration runs on GPU0 and GPU2, not a
controlled end-to-end speed benchmark. Evidence is in local
`output/p293/pic_legacy1.json` and `pic_objective1.json`.

Launches were at 00:28:56 and 00:29:39 UTC: the latter was 43 seconds after the
former, two seconds short of the 45-second operational spacing rule. Those jobs
did not overlap. The prefix launches at 00:30:48 and 00:31:23 were also too close
(35 seconds), on separate GPUs. These scheduling deviations are recorded rather
than described as compliant; subsequent launch commands must check elapsed server
time explicitly before starting a job.

## Eight-window execution

Both bounded arms completed eight accepted outer commits in eight attempts,
each with eight accepted inner updates per window and no state guards. All
eight candidate commits report exact objective-to-commit agreement and owned
accepted-buffer promotion. Recorded times were 141.99 / 167.10 seconds and Torch
allocation peaks 8.114 / 8.362 GB, under the same cross-GPU benchmark caveat.
The candidate does not add hard pins; its existing pin policy admits 35.178%
versus the control's 36.654% by commit8. This is early transport, not settling.

The independent raw audit is described in [the prefix report](pic_endpoint_prefix_p293.md).
Execution and endpoint consistency alone do not establish transport quality.
