"""VBD-MPM quasi-static grid solver (docs/method.md §7). Torch CPU."""
import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig
from physmorph.pipeline.runner_vbd import run_vbd_pipeline
from physmorph.vbd.solver import QuasiStaticGrid, psi_snh

DEV = "cpu"


@pytest.fixture(scope="module")
def prm():
    return MPMParams(dx=1.0, nx=32, ny=32, nz=32)


def _cloud(n=400, seed=0, scale=1.5):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, (n, 3)).astype(np.float32)


def test_psi_snh_zero_at_identity_positive_for_stretch():
    I = torch.eye(3).unsqueeze(0)
    assert abs(float(psi_snh(I, 1.0, 1.0))) < 1e-12
    S = torch.diag(torch.tensor([1.4, 1.0, 0.8])).unsqueeze(0)
    assert float(psi_snh(S, 1.0, 1.0)) > 0


def test_kinematics_translation_and_affine(prm):
    x = _cloud()
    # w_min=0: no pinned fringe nodes -> CIC kinematics must be EXACT for linear fields
    # (the default w_min trades an O(w_min) fringe error for solver robustness, R12)
    qs = QuasiStaticGrid(x, prm.grid_min, prm.dx, (prm.nx, prm.ny, prm.nz),
                         1.4e5, 0.2, device=DEV, w_min=0.0)
    # uniform translation: disp = c, grad u = 0
    u = torch.tensor([0.3, -0.2, 0.1]).expand(qs.A, 3).clone()
    disp, Ap = qs.kinematics(u)
    assert torch.allclose(disp, torch.tensor([0.3, -0.2, 0.1]).expand_as(disp), atol=1e-5)
    assert torch.allclose(Ap, torch.eye(3).expand_as(Ap), atol=1e-5)
    # affine node field u_g = M pos_g -> grad u == M (CIC is exact for linear fields)
    M = torch.tensor([[0.02, 0.01, 0.0], [0.0, -0.015, 0.005], [0.0, 0.0, 0.01]])
    ny, nz = prm.ny, prm.nz
    lin = qs.active
    pos = torch.stack([lin // (ny * nz), (lin // nz) % ny, lin % nz], 1).double()
    pos = pos * prm.dx + torch.tensor(prm.grid_min, dtype=torch.float64)
    u = (pos.float() @ M.T)
    _, Ap = qs.kinematics(u)
    gradu = Ap - torch.eye(3)
    assert torch.allclose(gradu, M.expand_as(gradu), atol=1e-4)


def test_solve_descends_and_gates_convergence(prm):
    """Pull the cloud toward a translated copy through a quadratic data term."""
    x = _cloud()
    tgt = torch.tensor(x + np.array([0.4, 0.0, 0.0], np.float32))
    qs = QuasiStaticGrid(x, prm.grid_min, prm.dx, (prm.nx, prm.ny, prm.nz),
                         1.4e5, 0.2, device=DEV)
    Fe0 = torch.eye(3).expand(len(x), 3, 3).clone()

    def energy(u):
        E_el, _ = qs.elastic(u, Fe0)
        disp, _ = qs.kinematics(u)
        return 1e-4 * E_el + ((qs.x0 + disp) - tgt).pow(2).sum()

    u, info = qs.solve(energy, sweeps=80, tol=1e-2, step=0.9)
    E0 = float(energy(torch.zeros_like(u)))
    E1 = info["energy"][-1]
    assert E1 < 0.2 * E0                          # substantial descent
    assert info["energy"] == sorted(info["energy"], reverse=True)   # monotone
    disp, _ = qs.kinematics(u)
    err = float((qs.x0 + disp - tgt).norm(dim=1).mean())
    assert err < 0.15                              # moved most of the 0.4 gap


def test_run_vbd_pipeline_smoke(prm):
    rng = np.random.default_rng(7)
    src = _cloud(300, seed=1)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.2, 0.85, 1.0])).astype(np.float32)
    cfg = PipelineConfig(animations=2, loss_res=12, render_views=2,
                         render_elevs=(0.0, 0.5), render_res=24, lambda_auto=0.5,
                         vbd_sweeps=8, device=DEV, patience=2)
    res = run_vbd_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert len(res["frames"]) == len(res["F_frames"])
    assert set(res["guards"]) == {"clamped", "nan_x", "nan_state", "F_reset", "F_flip",
                                  "F_invert_steps"}
    recs = [h for h in res["history"] if "d_vol" in h]
    assert recs and recs[-1]["d_render"] is not None and recs[-1]["lambda"] > 0
    assert np.isfinite(res["frames"][-1]).all()
    assert recs[-1]["Jmin"] > 0
