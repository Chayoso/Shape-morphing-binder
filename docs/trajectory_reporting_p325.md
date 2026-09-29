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
packet transfer and no individual scalar extraction. Server validation is pending.
