"""Opt-in appearance experiment; physical state and live opacity are untouched."""
import math

import torch


def reference_thickness_covariance(rotation, tangent_sigma, spacing):
    """Keep the supplied tangent axes; set normal std to reference spacing / 4.

    rotation has orthonormal columns (tangent1, tangent2, normal), as supplied by
    the historical photoreal exporter. This is a world-space covariance rule,
    not a projected pixel cap, mass-preserving density, or physical hole repair.
    The caller must validate normal/covariance health before evaluating images.
    """
    if rotation.shape != (len(tangent_sigma), 3, 3) or tangent_sigma.ndim != 1:
        raise ValueError('Expected rotation[N,3,3] and tangent_sigma[N]')
    if rotation.device != tangent_sigma.device or rotation.dtype != tangent_sigma.dtype:
        raise ValueError('Rotation and radii must share device and dtype')
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('Reference spacing must be finite and positive')
    variance = torch.stack((tangent_sigma.square(), tangent_sigma.square(),
                            torch.full_like(tangent_sigma, (spacing / 4) ** 2)), dim=1)
    return (rotation * variance[:, None]) @ rotation.transpose(1, 2)
