"""D162: the render weight referenced to the render's own floor (render_weight_ref "floor") and the placement terms at
the loss cell (placement_at_cell); both default off, the frozen path unchanged."""
import math

import pytest
import torch

from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.render_loss import floor_referenced_weight

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the device KD-tree")


def test_defaults_are_the_frozen_path():
    c = PipelineConfig()
    assert c.render_weight_ref == "residual" and c.placement_at_cell is False


def test_floor_needs_the_exterior_and_a_known_reference():
    with pytest.raises(ValueError):
        PipelineConfig(render_weight_ref="floor")                 # render_exterior off: no exterior render term
    with pytest.raises(ValueError):
        PipelineConfig(render_exterior=True, render_weight_ref="excess")
    PipelineConfig(render_exterior=True, render_weight_ref="floor")


def test_floor_referenced_weight():
    lam0 = 0.3
    assert floor_referenced_weight(lam0, 1.0, 0.0) == pytest.approx(lam0)            # no floor: the first weight
    assert floor_referenced_weight(lam0, 2e-3, 1e-3) == pytest.approx(0.15)          # half the term is excess
    assert floor_referenced_weight(lam0, 1e-3, 1e-3) == 0.0                          # at the floor: no push
    assert floor_referenced_weight(lam0, 5e-4, 1e-3) == 0.0                          # below it: none, never negative
    assert floor_referenced_weight(lam0, 0.0, 1e-3) == 0.0
    for L in (1.1e-3, 3e-3, 1e-1, 10.0):                                             # bounded by the first weight,
        w = floor_referenced_weight(lam0, L, 1e-3)                                    # growing with the excess
        assert 0.0 < w < lam0
    assert floor_referenced_weight(lam0, 3e-3, 1e-3) > floor_referenced_weight(lam0, 2e-3, 1e-3)


@cuda
def test_proximity_free_radius_is_one_loss_cell():
    from physmorph import gpu
    from physmorph.losses.support import SurfaceProximity
    g = torch.Generator().manual_seed(0)
    cloud = torch.rand(4000, 3, generator=g, dtype=torch.float64) * 2.0 - 1.0
    base = SurfaceProximity(cloud)
    assert base.floor == 0.5 * math.exp(-base.spacing ** 2 / (2 * base.h * base.h))   # the frozen floor, untouched
    r = 4.0 * base.spacing
    sp = SurfaceProximity(cloud, free_radius=r)
    assert sp.floor == pytest.approx(math.exp(-r ** 2 / (2 * sp.h * sp.h)))
    y = sp.y[:1].to(gpu.DEVICE)
    sp.y = y
    for f, covered in ((0.98, True), (1.05, False)):
        x = torch.cat([y + torch.tensor([[f * r, 0.0, 0.0]], device=y.device, dtype=y.dtype), y + 50.0], 0)
        assert (float(sp.penalty_per_point(x)[0]) == 0.0) == covered


@cuda
def test_isolation_gate_in_loss_cells():
    from physmorph import gpu
    from physmorph.losses.volumetric import isolation_gate
    g = torch.Generator().manual_seed(1)
    x = (torch.rand(5000, 3, generator=g) * 2.0 - 1.0).to(gpu.DEVICE)
    x = torch.cat([x, torch.tensor([[5.0, 5.0, 5.0]], device=x.device)], 0)          # one particle far out
    a = isolation_gate(x)                                                             # the frozen path
    b = isolation_gate(x, scale=None)
    assert torch.equal(a, b)
    c = isolation_gate(x, scale=1.0)                                                 # in units of 1.0 (a "loss cell")
    assert float(c[-1]) == 1.0 and float(c[:-1].max()) == 0.0                        # only the far particle is isolated
