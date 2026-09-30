"""The surface roughness measurement: zero for the target itself and for a smooth offset of it, large for a
particle-scale alternation of the same size."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def ball(sp=.1, r=1.):
    g = np.arange(-r, r + 1e-9, sp)
    P = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    return P[np.linalg.norm(P, axis=1) <= r]


def test_roughness_is_zero_for_the_target_and_small_for_a_smooth_offset():
    from physmorph.surface import surface_roughness
    from physmorph.thin import outer_mask
    import physmorph.gpu as gpu
    T = ball()
    assert surface_roughness(T, T)["surf_rough"] == 0.
    smooth = surface_roughness(1.03 * T, T)["surf_rough"]                     # the surface 0.3 spacings out, alike
    outer = outer_mask(gpu.tensor(T), .1).cpu().numpy()
    radial = T / np.linalg.norm(T, axis=1, keepdims=True).clip(1e-9)
    idx = np.round(T / .1).astype(int).sum(1) % 2                                # lattice parity: particle scale
    shift = np.where(outer[:, None], .03 * radial * np.where(idx[:, None] == 0, 1., -1.), 0.)
    alt = surface_roughness(T + shift, T)["surf_rough"]
    assert smooth < .05
    assert alt > 3 * smooth and alt > .1                                         # 0.035 against 0.144 here
