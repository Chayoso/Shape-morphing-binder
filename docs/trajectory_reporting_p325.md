# P325: one host packet for final deformation observations

The final accepted-rollout report previously transferred each step's minimum
determinant and then the count of particles ever inverted: T+1 scalar transfers.
`pipeline/trajectory_reporting.trajectory_health` preserves each native float32
determinant, per-step minimum and per-ID inversion OR. It packs the completed
T minima and count into one FP64 vector, transfers that vector once, and applies
the original ordered Python minimum. At T20 this replaces21 transfers with1.

This deliberately preserves the old NaN-step-minimum and signed-zero semantics;
it is not a new health rule or a state repair. The integer count is exact at
the production particle counts. Optimizer state, objective, gradient, acceptance
and rendered appearance are unchanged. No end-to-end speedup or removal of the
remaining balancer/line-search host decisions is claimed.

CPU tests compare with the original expressions, including mixed dtypes,
noncontiguous/flattened matrices, inversion unions, nonfinite values and signed
zero. CUDA tests additionally require exact parity on a side stream, one FP64
packet transfer and no individual scalar extraction. Independent verification:
16 CPU cases pass, plus39 existing CPU pipeline cases. Frozen d349d58 on hyde06:
all4 actual CUDA cases pass (no skips) in1.38s. These small input cases validate
parity and transfer behavior, not full-run performance. Logs/XML and executed
source hashes are retained in `docs/evidence/p325`.
