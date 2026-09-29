# P318: batch accepted-update render telemetry into one host copy

`accepted_render_step` previously extracted each norm/dot/loss separately. With
three control channels and both image losses present that is21 CUDA scalar
reads; the render-direction dot accepted-control delta was reduced twice.
The function now keeps the original native-dtype GPU reductions, reuses that
identical dot sum, promotes ONLY the completed scalar packet and copies18 scalar
values to the host once. Python arithmetic, payload fields, component references
and report interpretation remain unchanged. FP64 observations must not be
silently rounded to FP32. Missing direction/loss and empty-channel semantics
remain compatible; no gradient/state/optimizer or loss policy is changed.

Independent static review and53 CPU cases pass. Four server-only CUDA cases
compare the complete report exactly against the frozen legacy implementation,
forbid per-scalar CUDA extraction and count exactly one CUDA-to-CPU vector copy.
They cover a side stream, mixed dtypes, missing directions/losses and CPU losses
with CUDA directions. CUDA validation and timing results must be stated
separately; fewer transfers do not by themselves establish end-to-end speedup.

This change concerns observational telemetry. The optimization acceptance and
line search still use host scalar decisions; KNN adaptive support and Gaussian
raster dynamic bin allocation also retain host decisions. In particular the
raster reads its overlap count before host allocation/CUB sizing/backward
layout; cudaMemcpyAsync alone cannot remove that dependency. A possible fixed
capacity/sentinel implementation would require overflow rejection, exact
image/gradient parity, stream/lifetime tests and realistic4K memory gates.
It is not implemented or promoted by this telemetry change. The P316 frozen
physical and P317 motion runs precede this change and remain untouched.
