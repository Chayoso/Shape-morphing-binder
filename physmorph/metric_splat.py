"""Fixed-size binary-metric footprint counts; not a rendering/loss operator.

CUDA bincount reads the maximum index on the host even with minlength. Boolean
index compaction also needs an output size. Keep all input rows, using a zero
weight for excluded centers, and scatter into the known image allocation.
"""
from .compute import array_api as np


def fixed_footprint_counts(ij, valid, res):
    """Count the existing clipped 3x3 footprint, including duplicate edge taps.

    Projection/floor/center inclusion remain the caller's responsibility. No
    data-dependent array size, scalar device read or host conversion occurs here.
    Float64 integer counts agree exactly with the existing metric histogram.
    """
    if (not isinstance(res, int) or res <= 0 or ij.ndim != 2 or ij.shape[1] != 2
            or valid.shape != (len(ij),) or valid.dtype != np.bool_
            or ij.dtype != np.int64):
        raise ValueError('Invalid footprint layout or resolution')
    flat = np.zeros(res * res, np.float64)
    weights = valid.astype(np.float64)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            i2 = np.clip(ij[:, 0] + ox, 0, res - 1)
            j2 = np.clip(ij[:, 1] + oy, 0, res - 1)
            np.add.at(flat, i2 * res + j2, weights)
    return flat.reshape(res, res)
