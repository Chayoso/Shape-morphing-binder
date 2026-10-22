"""D161b: the surface proximity's free radius at the berth (--proximity_berth, default off).

The default path builds the frozen floor K(sp) / 2; with berth_k the floor is K(berth): a body particle exactly at the
berth from a target point leaves it covered (penalty zero), one a little farther does not; the default does not read
berth_k at all."""
import math

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="SurfaceProximity's KD-tree is on the device")


def _cloud(n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, generator=g, dtype=torch.float64) * 2.0 - 1.0


@cuda
def test_default_floor_is_the_frozen_one():
    from physmorph.losses.support import SurfaceProximity
    sp = SurfaceProximity(_cloud())
    assert sp.floor == pytest.approx(0.5 * math.exp(-sp.spacing ** 2 / (2 * sp.h * sp.h)), rel=0, abs=0)


@cuda
def test_berth_floor_covers_within_the_berth():
    from physmorph.losses.support import SurfaceProximity
    from physmorph import gpu
    k = 1.97
    sp = SurfaceProximity(_cloud(), berth_k=k)
    assert sp.floor == pytest.approx(math.exp(-(k * sp.spacing) ** 2 / (2 * sp.h * sp.h)))
    y = sp.y[:1].to(gpu.DEVICE)
    for f, covered in ((0.99, True), (1.05, False)):
        x = torch.cat([y + torch.tensor([[f * k * sp.spacing, 0.0, 0.0]], device=y.device, dtype=y.dtype),
                       y + 50.0], 0)
        sp.y = y
        pen = float(sp.penalty_per_point(x)[0])
        assert (pen == 0.0) == covered
