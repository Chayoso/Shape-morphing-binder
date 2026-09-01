"""Local-global surface pass (docs/rationale.md §5): band construction, pinned-interior
invariant, render-residual descent. Torch CPU."""
import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.render_loss import make_views, target_silhouettes
from physmorph.pipeline.runner import build_target
from physmorph.pipeline.surface_local import SurfaceLocal, surface_local_pass

DEV = "cpu"


@pytest.fixture(scope="module")
def prm():
    # production resolution (dx=0.5): with the coarse test grid (dx=1.0) the one-cell
    # band's corner nodes reach the whole body and nothing is fully pinned
    return MPMParams(dx=0.5, nx=64, ny=64, nz=64)


def _ball(n=6000, seed=0, r=3.0):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, (n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (v * (rng.uniform(0, 1, (n, 1)) ** (1 / 3)) * r).astype(np.float32)


def test_band_is_a_shell_and_interior_is_pinned(prm):
    x = _ball()
    sl = SurfaceLocal(x, prm.grid_min, prm.dx, (prm.nx, prm.ny, prm.nz),
                      2e3, 0.2, DEV)
    assert sl.A > 0
    # the band must NOT cover the body: a substantial fraction of particles is fully
    # pinned (all 8 corners), and those particles cannot move under ANY u
    fully_pinned = sl.pinned.all(1)
    assert float(fully_pinned.float().mean()) > 0.2
    # fully-pinned particles sit deeper than the band particles on average
    r = torch.as_tensor(x, device=DEV).norm(dim=1)
    assert float(r[fully_pinned].mean()) < float(r[~fully_pinned].mean())
    u = torch.randn(sl.A, 3) * 0.1
    disp, _ = sl.kinematics(u)
    assert float(disp[fully_pinned].abs().max()) < 1e-6


def test_surface_pass_reduces_render_residual(prm):
    """The local pass must descend lambda*D_render + elastic against a squashed target
    while leaving the deep interior untouched (the global anchor)."""
    x = _ball()
    tgt_x = (_ball(seed=1) * np.array([1.0, 0.85, 1.0])).astype(np.float32)
    cfg = PipelineConfig(lambda_auto=0.5, render_views=3, render_elevs=(0.0, 0.5),
                        render_res=32, lg_sweeps=6, device=DEV)
    pack = build_target(tgt_x, prm, cfg)
    N = len(x)
    F = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    out = surface_local_pass(x, F, F.copy(), pack, cfg, lam_r=200.0, prm=prm)
    assert out is not None
    x2, F2, tele = out
    assert tele["lg_E1"] <= tele["lg_E0"]            # monotone energy
    assert tele["lg_move"] > 1e-4                    # it actually moved the shell
    # global anchor intact: fully-pinned particles are bit-still
    sl = SurfaceLocal(x, prm.grid_min, prm.dx, (prm.nx, prm.ny, prm.nz), 2e3, 0.2, DEV)
    anchor = sl.pinned.all(1).cpu().numpy()
    assert anchor.any()
    assert float(np.abs(x2[anchor] - x[anchor]).max()) < 1e-6
    assert np.isfinite(F2).all() and (np.linalg.det(F2) > 0).all()
