# P321: batch three raw gradient observations into one host copy

The split-gradient optimizer records the unprojected physical norm, render norm
and their dot product before PCGrad/balancing. Those independent observations
previously each synchronized a scalar. `gradient_reporting.raw_direction_observations`
preserves the native-dtype reductions and Python's ordered tensor sums exactly,
then stacks the three completed scalars and copies one vector to the host.
FP64 observations are preserved. No gradient combination, lambda update,
line-search decision, loss or physical state changes.

This removes two host synchronizations per active split-gradient iteration.
It does not remove the remaining balancer/line-search/raster host decisions,
and no end-to-end speedup is inferred. P318's accepted-step reporting is separate.
The helper changes none of the interpretation limits of rendering influence.

Independent CPU verification passes11 new cases plus53 existing report cases.
They exercise mixed dtypes, order-sensitive cancellation/norm accumulation,
FP64 precision, ownership and zero/nonfinite semantics. The four server CUDA
cases additionally require exact legacy parity on a side stream, no individual
CUDA scalar extraction, and exactly one three-element device-to-host copy.
Those actual CUDA results are pending; local CUDA cases are deliberately skipped.
